"""ReLU SAE trainer with neuron resampling (used for LM SAEs).

Ports the dead-latent resampling loop from
``saprmarks/dictionary_learning/trainers/standard.py:resample_neurons``.
Every ``resample_steps`` training steps, latents that have been silent
for at least ``resample_steps // 2`` steps are reinitialised:

  * encoder column sampled from a high-loss recent token
    (multinomial weighted by per-token recon loss),
    scaled to ``alive_norm * 0.2`` (saprmarks heuristic);
  * decoder row set to the same direction, unit-normalised;
  * encoder bias for resampled indices zeroed;
  * Adam ``exp_avg`` / ``exp_avg_sq`` zeroed at those indices, so stale
    optimizer state from the dead phase does not push them back to zero.

Reference: Bricken et al. 2023 "Towards Monosemanticity" §III.B
("Resampling"); Gao et al. 2024 §C.2 ("Removing dead latents").
"""

from __future__ import annotations

import collections

import torch

from consdl.training.trainer import SAETrainer


class ResamplingSAETrainer(SAETrainer):
    """``SAETrainer`` + periodic dead-latent resampling.

    Resampling runs from :meth:`_log_wandb_if_needed`, which fastsae calls
    once per step after ``optimizer.step()``.

    ``_z_buffer`` / ``_loss_buffer`` hold recent training rows and their
    reconstruction losses; revived latents take the directions of rows
    sampled in proportion to loss.
    """

    def __init__(
        self,
        *args,
        resample_steps: int = 25_000,
        resample_buffer_size: int = 8_192,
        resample_tail_skip_frac: float = 0.25,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_sae = int(self.wrapped.sae.d_sae)
        self.resample_steps = int(resample_steps)
        self.resample_buffer_size = int(resample_buffer_size)
        # No resampling in the final ``resample_tail_skip_frac`` of
        # training: latents revived near the end of the LR schedule cannot
        # settle and, with ``b_enc = 0``, fire on most tokens.
        self.resample_tail_skip_frac = float(resample_tail_skip_frac)
        # Per-latent inactivity counter in training steps (the resample
        # period is a step count).
        self.steps_since_active = torch.zeros(
            d_sae, dtype=torch.long, device=self.wrapped.device,
        )
        # Rolling buffer of recent training rows and their losses, on CPU.
        self._z_buffer: collections.deque = collections.deque(
            maxlen=self.resample_buffer_size,
        )
        self._loss_buffer: collections.deque = collections.deque(
            maxlen=self.resample_buffer_size,
        )
        self._resampled_count_total = 0

    # ── Per-step bookkeeping ──────────────────────────────────────────

    def loss_fn(self, z, a, z_hat):
        with torch.no_grad():
            firing_dims = tuple(range(a.ndim - 1))
            fired = (a > 0).any(dim=firing_dims)
            if self.steps_since_active.device != fired.device:
                self.steps_since_active = self.steps_since_active.to(fired.device)
            self.steps_since_active[fired] = 0
            self.steps_since_active[~fired] += 1

            # Buffer rows and per-row losses for the resampler.
            z_flat = z.reshape(-1, z.shape[-1]).detach().float().cpu()
            z_hat_flat = z_hat.reshape(-1, z_hat.shape[-1]).detach().float().cpu()
            per_row_loss = (z_flat - z_hat_flat).pow(2).sum(dim=-1)
            self._z_buffer.extend(z_flat.unbind(0))
            self._loss_buffer.extend(per_row_loss.unbind(0))

        return super().loss_fn(z, a, z_hat)

    # ── Periodic resampling (called after optimizer.step()) ───────────

    def _log_wandb_if_needed(self, loss_outputs, metrics_outputs, weighted_loss_sum):
        # Run fastsae's logging path first so wandb sees the pre-resample
        # state of the current step.
        super()._log_wandb_if_needed(loss_outputs, metrics_outputs, weighted_loss_sum)
        # Resample every ``resample_steps`` steps (``_n_training_steps``
        # is already incremented here).
        if not (
            self.resample_steps > 0
            and self._n_training_steps > 0
            and self._n_training_steps % self.resample_steps == 0
        ):
            return
        total = int(getattr(self, "total_n_training_samples", 0) or 0)
        seen = int(getattr(self, "_n_training_samples", 0) or 0)
        if total > 0 and seen >= total * (1.0 - self.resample_tail_skip_frac):
            return
        self._resample_dead()

    @torch.no_grad()
    def _resample_dead(self) -> None:
        if not self._z_buffer:
            return
        # Dead = silent for more than half the resample period.
        dead_mask = self.steps_since_active > (self.resample_steps // 2)
        if not bool(dead_mask.any()):
            return
        dead_indices = dead_mask.nonzero(as_tuple=True)[0]
        n_dead = int(dead_indices.numel())

        sae = self.wrapped.sae
        device = sae.W_enc.device
        dtype = sae.W_enc.dtype

        # ── Pick replacement encoder directions ───────────────────────
        z_buf = torch.stack(list(self._z_buffer), dim=0)          # (B, d_in)
        loss_buf = torch.stack(list(self._loss_buffer), dim=0)    # (B,)
        n_buf = z_buf.shape[0]
        n_sample = min(n_dead, n_buf)
        if n_sample == 0:
            return
        # Sample rows in proportion to loss.
        weights = loss_buf.clamp_min(1e-12)
        sampled_idx = torch.multinomial(weights, n_sample, replacement=False)
        sampled_vecs = z_buf[sampled_idx].to(device=device, dtype=dtype)
        # With more dead latents than buffered rows, only the first
        # ``n_sample`` are revived this round.
        dead_to_reset = dead_indices[:n_sample]

        # Scale new encoder directions to 0.2x the mean alive encoder norm.
        alive_mask = ~dead_mask
        if bool(alive_mask.any()):
            alive_norm = sae.W_enc.data[:, alive_mask].norm(dim=0).mean()
        else:
            # Every latent is dead: use the sampled vectors' norm.
            alive_norm = sampled_vecs.norm(dim=-1).mean()

        unit = sampled_vecs / sampled_vecs.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        new_enc_dirs = unit * alive_norm * 0.2                    # (n_sample, d_in)

        # ── Overwrite weights ────────────────────────────────────────
        # ``W_enc`` is ``(d_in, d_sae)``; ``W_dec`` is ``(d_sae, d_in)``.
        sae.W_enc.data[:, dead_to_reset] = new_enc_dirs.T
        sae.W_dec.data[dead_to_reset, :] = unit
        sae.b_enc.data[dead_to_reset] = 0.0

        # ── Reset Adam state for the resampled slices ────────────────
        for p, slot_idx in [
            (sae.W_enc, (slice(None), dead_to_reset)),
            (sae.W_dec, (dead_to_reset, slice(None))),
            (sae.b_enc, (dead_to_reset,)),
        ]:
            state = self.optimizer.state.get(p, None)
            if state is None:
                continue
            for key in ("exp_avg", "exp_avg_sq"):
                if key in state and isinstance(state[key], torch.Tensor):
                    state[key][slot_idx] = 0.0

        self.steps_since_active[dead_to_reset] = 0
        self._resampled_count_total += int(n_sample)

        print(
            f"[resample] step={self._n_training_steps} "
            f"dead={n_dead} resampled={n_sample} "
            f"total_resamples={self._resampled_count_total}",
            flush=True,
        )
