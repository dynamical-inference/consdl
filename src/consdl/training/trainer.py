"""``SAETrainer``: fastsae's trainer with ConSDL's additions.

Base class for every trainer in this package. On top of fastsae's
``SAETrainer`` it:

  * adds co-active similarity regularization as an opt-in loss term:
    ``trainer.set_co_active_sim(coefficient=γ, ...)`` adds ``γ · L_cas``
    (see :mod:`consdl.training.co_active_sim`); off by default;
  * honours ``use_ghost_grads=False`` (fastsae computes the ghost-grad
    term regardless of the flag);
  * floors the per-sample reconstruction normalizer (see :meth:`loss_fn`);
  * logs ``metrics/l0_per_token`` (fraction of (token, latent) entries with
    ``|a| > eps``, matching ``stats/interpreter.py``), ``metrics/l1_per_token``
    and a periodic decoder nearest-neighbour cosine diagnostic to wandb;
  * supports mid-run resume (:mod:`consdl.training.resume`), a graceful
    checkpoint-and-exit on SIGUSR1/SIGTERM, an optional L0 early abort,
    and optional autocast.
"""

from __future__ import annotations

import json
import pathlib
import signal
from typing import NamedTuple

import torch
import wandb
from fastsae.training.trainer import LossOutput
from fastsae.training.trainer import SAETrainer as _FastSAETrainer

from consdl.training.co_active_sim import (
    CO_ACTIVE_SIM_ACTIVATION_EPS_DEFAULT,
    CO_ACTIVE_SIM_BARRIER_DELTA_DEFAULT,
    CO_ACTIVE_SIM_BARRIER_MARGIN_DEFAULT,
    CO_ACTIVE_SIM_CACHE_BLOCK_SIZE_DEFAULT,
    CO_ACTIVE_SIM_R_DEFAULT,
    CO_ACTIVE_SIM_REFRESH_STEPS_DEFAULT,
    co_active_sim_loss,
)


class CoActiveSimLossOutput(NamedTuple):
    """``LossOutput`` plus the co-active similarity term."""
    recon_loss: torch.Tensor
    sparsity_loss: torch.Tensor
    auxiliary_loss: torch.Tensor
    co_active_sim_loss: torch.Tensor

# ── Graceful stop: checkpoint before the scheduler kills us ───────────
#
# fastsae only checkpoints at fixed thresholds and at the end, so a job
# killed in between loses its progress. On SIGUSR1 (e.g. sent by a scheduler
# before its time limit) or SIGTERM, the trainer checkpoints at the next step
# boundary, where model and optimizer state are consistent, and exits.
_STOP = {"requested": False, "signum": None}


def _handle_stop(signum, _frame):  # pragma: no cover - signal path
    _STOP["requested"] = True
    _STOP["signum"] = int(signum)
    print(f"\n[graceful-stop] signal {signum} received — will checkpoint at "
          f"the next step boundary and exit", flush=True)


def install_graceful_stop_handler() -> None:
    """Latch SIGUSR1/SIGTERM so the next step boundary checkpoints and exits.

    Idempotent, and a no-op where the signal cannot be installed (a
    non-main thread). Call once from the training entry point.
    """
    for sig in (signal.SIGUSR1, signal.SIGTERM):
        try:
            signal.signal(sig, _handle_stop)
        except (ValueError, OSError) as e:  # noqa: PERF203
            print(f"[graceful-stop] could not install handler for {sig}: {e}")


def stop_requested() -> bool:
    return bool(_STOP["requested"])


class GracefulStop(Exception):
    """Raised out of the train loop once a resumable checkpoint is on disk.

    Carries the token count that was saved, so the caller can report it and
    exit 0 — a checkpointed stop is a successful partial run, not a failure.
    """

    def __init__(self, n_samples: int, path: str):
        super().__init__(f"checkpointed at {n_samples:,} samples -> {path}")
        self.n_samples = int(n_samples)
        self.path = str(path)


# Same active-vs-zero cutoff as ``stats/interpreter.py``'s ``ZERO_EPS``.
_TOKEN_EPS = 1e-6

# Floor on the per-sample recon-loss normaliser ``√(Σz²)`` — see
# ``SAETrainer.loss_fn``.
_RECON_NORM_EPS = 1e-6

# ── Periodic Babel μ₁ diagnostic ─────────────────────────────────────
# Babel μ₁ = mean over decoder atoms of ``max_{j≠i} |cos(D_i, D_j)|``.
# Computed blockwise over the full Gram every ``_BABEL_LOG_EVERY`` steps.
_BABEL_LOG_EVERY = 500
_BABEL_BLOCK = 2048
_BABEL_DUP_THRS = (0.6, 0.7, 0.8, 0.9)


class L0AbortError(RuntimeError):
    """Raised when a run's L0 stays outside its usable band.

    A badly chosen sparsity setting can drive L0 to zero or leave it far
    too high; this is visible early and does not recover, so the run is
    stopped. It is an exception so the run is not mistaken for a
    completed one.
    """


def _l0_count(a: "torch.Tensor") -> float:
    """Active latents per token — the COUNT, matching ``metrics/l0``.

    Not ``metrics/l0_per_token``, which is a fraction of all (token,
    latent) entries; the abort thresholds are in latents.
    """
    return float((a.detach() > 0).float().sum(-1).mean())


@torch.no_grad()
def _babel_mu1_and_dup(W_dec: torch.Tensor) -> dict[str, float]:
    """Return ``{"babel/mu1_mean", "babel/n_dup_ge_<t>", …}`` plus the
    ``signed_`` companions, for the full normalised decoder gram
    (block-computed, diag masked out).

    Duplicates are reported as atom counts, not fractions.
    """
    import torch.nn.functional as F  # noqa: PLC0415
    W = F.normalize(W_dec.detach().float(), dim=-1, eps=1e-12)
    n = W.shape[0]
    # Two nearest-neighbour readings from one pass: ``abs`` (|cos|) and
    # ``signed`` (largest same-direction cosine). With non-negative codes,
    # anti-parallel atoms are not duplicates, which only ``signed`` reflects.
    nn_abs = torch.zeros(n, device=W.device)
    nn_sgn = torch.zeros(n, device=W.device)
    for i in range(0, n, _BABEL_BLOCK):
        raw = W[i : i + _BABEL_BLOCK] @ W.T           # (b, n) SIGNED
        # Mask self-cosine.
        rows = torch.arange(raw.shape[0], device=W.device)
        cols = rows + i
        raw[rows, cols] = -1.0
        nn_sgn[i : i + raw.shape[0]] = raw.max(dim=1).values
        blk = raw.abs()
        blk[rows, cols] = -1.0
        nn_abs[i : i + raw.shape[0]] = blk.max(dim=1).values
    nn_abs.clamp_(min=0.0)
    nn_sgn.clamp_(min=0.0)
    out: dict[str, float] = {
        "babel/mu1_mean": float(nn_abs.mean().item()),
        "babel/signed_mu1_mean": float(nn_sgn.mean().item()),
    }
    for t in _BABEL_DUP_THRS:
        out[f"babel/n_dup_ge_{t:g}"] = float((nn_abs >= t).sum().item())
        out[f"babel/signed_n_dup_ge_{t:g}"] = float((nn_sgn >= t).sum().item())
    return out


class SAETrainer(_FastSAETrainer):
    # ── L0 early abort ───────────────────────────────────────────────
    # Off unless both bounds are set. ``warmup_frac`` skips the start of
    # training; ``patience`` is the number of consecutive out-of-band log
    # intervals required.
    # Plain, un-annotated class attributes set via ``set_l0_abort()``:
    # ``config_field``s declared on a subclass are not accepted as
    # constructor kwargs by config_dataclass.
    l0_abort_min = 0.0
    l0_abort_max = 0.0
    l0_abort_warmup_frac = 0.10
    l0_abort_patience = 5

    def set_l0_abort(self, *, minimum: float, maximum: float,
                     warmup_frac: float = 0.10, patience: int = 5):
        """Arm the L0 guard. Returns self so it can be chained onto a
        constructor call: ``Trainer(...).set_l0_abort(...).train()``."""
        self.l0_abort_min = float(minimum)
        self.l0_abort_max = float(maximum)
        self.l0_abort_warmup_frac = float(warmup_frac)
        self.l0_abort_patience = int(patience)
        return self

    @staticmethod
    def _num(value, default):
        """Read a config field that may never have been materialised.

        A ``config_field`` declared on a subclass may still hold the raw
        ``Field`` object; treat that (or any non-number) as unset.
        """
        return value if isinstance(value, (int, float)) else default

    def _check_l0_abort(self, a) -> None:
        lo = float(self._num(self.l0_abort_min, 0.0))
        hi = float(self._num(self.l0_abort_max, 0.0))
        if lo <= 0.0 or hi <= 0.0 or hi <= lo:
            return
        every = max(1, int(self._num(self.wandb_log_frequency, 1)))
        if (self._n_training_steps + 1) % every:
            return
        # Warmup is measured in samples, not steps, so it means the same
        # amount of data regardless of batch size.
        seen = int(getattr(self, "_n_training_samples", 0))
        total = int(self._num(self.total_n_training_samples, 0) or 0)
        warm = float(self._num(self.l0_abort_warmup_frac, 0.10))
        if total and seen < total * warm:
            return
        l0 = _l0_count(a)
        if lo <= l0 <= hi:
            self._l0_out_of_band = 0
            return
        self._l0_out_of_band = getattr(self, "_l0_out_of_band", 0) + 1
        if self._l0_out_of_band < int(self._num(self.l0_abort_patience, 5)):
            return
        raise L0AbortError(
            f"L0={l0:.1f} outside ({lo:g}, {hi:g}) for "
            f"{self._l0_out_of_band} consecutive log intervals at step "
            f"{self._n_training_steps + 1} ({seen:,}/{total:,} samples)"
        )

    # ── Co-active similarity regularization (ConSDL) ─────────────────
    # Plain class attributes set via ``set_co_active_sim()``;
    # ``coefficient == 0`` (default) leaves the loss untouched.
    co_active_sim_coefficient = 0.0
    co_active_sim_r = CO_ACTIVE_SIM_R_DEFAULT
    co_active_sim_refresh_steps = CO_ACTIVE_SIM_REFRESH_STEPS_DEFAULT
    co_active_sim_barrier_delta = CO_ACTIVE_SIM_BARRIER_DELTA_DEFAULT
    co_active_sim_barrier_margin = CO_ACTIVE_SIM_BARRIER_MARGIN_DEFAULT
    co_active_sim_activation_eps = CO_ACTIVE_SIM_ACTIVATION_EPS_DEFAULT
    co_active_sim_cache_block_size = CO_ACTIVE_SIM_CACHE_BLOCK_SIZE_DEFAULT

    def set_co_active_sim(self, *, coefficient: float,
                          r: int = CO_ACTIVE_SIM_R_DEFAULT,
                          refresh_steps: int = CO_ACTIVE_SIM_REFRESH_STEPS_DEFAULT,
                          barrier_delta: float = CO_ACTIVE_SIM_BARRIER_DELTA_DEFAULT,
                          barrier_margin: float = CO_ACTIVE_SIM_BARRIER_MARGIN_DEFAULT,
                          activation_eps: float = CO_ACTIVE_SIM_ACTIVATION_EPS_DEFAULT,
                          cache_block_size: int = CO_ACTIVE_SIM_CACHE_BLOCK_SIZE_DEFAULT):
        """Enable co-active similarity regularization with weight
        ``coefficient`` (γ; 0 disables). Returns self so it can be
        chained onto a constructor call."""
        self.co_active_sim_coefficient = float(coefficient)
        self.co_active_sim_r = int(r)
        self.co_active_sim_refresh_steps = int(refresh_steps)
        self.co_active_sim_barrier_delta = float(barrier_delta)
        self.co_active_sim_barrier_margin = float(barrier_margin)
        self.co_active_sim_activation_eps = float(activation_eps)
        self.co_active_sim_cache_block_size = int(cache_block_size)
        # Neighbour cache consumed by ``co_active_sim_loss`` (empty ⇒
        # refresh on the first call).
        self._co_active_sim_cache = {}
        return self

    def _with_co_active_sim(self, out: LossOutput, a) -> LossOutput:
        """Append the co-active similarity term to ``out`` when enabled."""
        if self.co_active_sim_coefficient <= 0.0:
            return out
        cas = co_active_sim_loss(
            self.wrapped.sae.W_dec,
            a,
            self._co_active_sim_cache,
            step=int(self._n_training_steps),
            r=self.co_active_sim_r,
            refresh_steps=self.co_active_sim_refresh_steps,
            activation_eps=self.co_active_sim_activation_eps,
            barrier_delta=self.co_active_sim_barrier_delta,
            barrier_margin=self.co_active_sim_barrier_margin,
            cache_block_size=self.co_active_sim_cache_block_size,
        )
        return CoActiveSimLossOutput(*out, co_active_sim_loss=cas)

    def _log_co_active_sim_diagnostics(self, loss_outputs) -> None:
        """``r_L`` (loss-scale ratio) = ``γ * L_cas / L_rec``, on the
        wandb cadence (the ``float(...)`` is a host sync)."""
        if not getattr(self, "log_to_wandb", False):
            return
        if (self._n_training_steps + 1) % self.wandb_log_frequency != 0:
            return
        recon_loss = loss_outputs.recon_loss.mean()
        cas_loss = loss_outputs.co_active_sim_loss.mean()
        r_L = float((self.co_active_sim_coefficient * cas_loss)
                    / recon_loss.detach().clamp_min(1e-12))
        wandb.log({"co_active_sim/r_L": r_L}, step=int(self._n_training_steps))

    def _mse_ghost_loss(self, z, z_hat, a, z_mse_loss):
        if not self.use_ghost_grads:
            return torch.zeros(
                (), dtype=self.wrapped.dtype, device=self.wrapped.device,
            )
        return super()._mse_ghost_loss(z, z_hat, a, z_mse_loss)

    def loss_fn(self, z, a, z_hat):
        """fastsae's ``SAETrainer.loss_fn`` with an ε-floored recon
        normaliser.

        fastsae normalizes the per-sample reconstruction MSE by the input
        norm ``√(Σz²)`` with no floor, so an all-zero input row produces
        an infinite loss and NaN gradients. The floor prevents this. The
        co-active similarity term is attached here when enabled.
        """
        denom = (z ** 2).sum(dim=-1, keepdim=True).sqrt().clamp_min(_RECON_NORM_EPS)
        z_mse_loss = torch.pow((z_hat - z.float()), 2) / denom
        a_l1_loss = torch.abs(a).sum(dim=-1).mean(dim=(0,))
        return self._with_co_active_sim(LossOutput(
            recon_loss=z_mse_loss.mean(),
            sparsity_loss=a_l1_loss.mean(),
            auxiliary_loss=self._compute_auxiliary_loss(z, a, z_hat, z_mse_loss),
        ), a)

    def _compute_auxiliary_loss(self, z, a, z_hat, z_mse_loss):
        """The ``auxiliary_loss`` term, as a subclass hook.

        Default = fastsae's ghost-grad revival (already zeroed when
        ``use_ghost_grads`` is off). Subclasses that carry their own
        single auxiliary term (Top-K dead-latent residual, MP dead-atom
        revival) override *this* instead of re-implementing ``loss_fn``.
        """
        return self._mse_ghost_loss(z, z_hat, a, z_mse_loss).mean()

    def metric_fn(self, z, a, z_hat):
        # Compute per-token L0/L1 here, where ``a`` is available; they are
        # logged in ``_log_wandb_if_needed``.
        a_abs = a.detach().abs()
        self._token_extras = {
            "metrics/l0_per_token": float((a_abs > _TOKEN_EPS).float().mean()),
            "metrics/l1_per_token": float(a_abs.sum(dim=-1).mean()),
        }
        # Checked here so the abort works without wandb logging.
        self._check_l0_abort(a)
        return super().metric_fn(z, a, z_hat)

    def _log_wandb_if_needed(self, loss_outputs, metrics_outputs, weighted_loss_sum):
        super()._log_wandb_if_needed(loss_outputs, metrics_outputs, weighted_loss_sum)
        if (
            self.log_to_wandb
            and (self._n_training_steps + 1) % self.wandb_log_frequency == 0
        ):
            extras = getattr(self, "_token_extras", None)
            # Babel μ₁ at least ``_BABEL_LOG_EVERY`` steps apart. Uses
            # "steps since last" rather than divisibility, since log ticks
            # are already sub-sampled by ``wandb_log_frequency``.
            _last_babel = getattr(self, "_last_babel_step", -_BABEL_LOG_EVERY)
            if self._n_training_steps - _last_babel >= _BABEL_LOG_EVERY:
                W = getattr(self.wrapped.sae, "W_dec", None)
                if isinstance(W, torch.Tensor):
                    extras = dict(extras or {})
                    extras.update(_babel_mu1_and_dup(W))
                    self._last_babel_step = self._n_training_steps
            if extras:
                wandb.log(extras, step=self._n_training_steps)

    # ── Optimizer + scheduler state persistence ──────────────────────
    # Trainer state is saved and restored by ``consdl.training.resume``.

    _TRAINER_STATE_FILENAME = "trainer_state.pt"

    # ── total loss ────────────────────────────────────────────────────

    def compute_total_loss(self, loss_outputs):
        weighted_sum = (
            self.recon_coefficient * loss_outputs.recon_loss
            + self.sparsity_coefficient * loss_outputs.sparsity_loss
            + self.auxiliary_coefficient * loss_outputs.auxiliary_loss
        )
        if self.co_active_sim_coefficient > 0.0:
            if not hasattr(loss_outputs, "co_active_sim_loss"):
                raise RuntimeError(
                    f"{type(self).__name__}.loss_fn does not go through "
                    "SAETrainer.loss_fn, so co-active similarity "
                    "regularization cannot be applied to it."
                )
            self._log_co_active_sim_diagnostics(loss_outputs)
            weighted_sum = (weighted_sum
                            + self.co_active_sim_coefficient
                            * loss_outputs.co_active_sim_loss)
        return weighted_sum.mean()

    # ── mid-run resume ────────────────────────────────────────────────
    # Hooked here (not in fastsae) so every trainer in this package gets it.

    def _save_sae_if_needed(self):
        """Save trainer state next to fastsae's threshold checkpoints, and
        handle a requested graceful stop."""
        prev = list(getattr(self, "_checkpoint_thresholds", []) or [])
        super()._save_sae_if_needed()
        if prev and list(self._checkpoint_thresholds) != prev:
            from consdl.training.resume import (  # noqa: PLC0415
                save_trainer_state,
            )
            try:
                save_trainer_state(
                    self, f"{self.checkpoint_path}/{self._n_training_samples}",
                )
            except Exception as e:  # noqa: BLE001 - never kill a run over this
                print(f"[resume] state save failed: {type(e).__name__}: {e}")

        # ── graceful stop ────────────────────────────────────────────
        # This per-step callback runs at a step boundary, where weights,
        # optimizer state and counters are consistent. The checkpoint uses
        # the same format as threshold checkpoints, so it resumes the same
        # way.
        if not stop_requested() or self.checkpoint_path is None:
            return
        n = int(self._n_training_samples)
        path = pathlib.Path(self.checkpoint_path) / str(n)
        self.wrapped.sae.save(str(path))
        try:
            from consdl.training.resume import (  # noqa: PLC0415
                save_trainer_state,
            )
            save_trainer_state(self, str(path))
        except Exception as e:  # noqa: BLE001 - a scoreable checkpoint beats none
            print(f"[graceful-stop] trainer state unavailable "
                  f"({type(e).__name__}: {e}) — checkpoint is scoreable but "
                  f"NOT cleanly resumable")
        (path / "resume_state.json").write_text(json.dumps({
            "n_training_samples": n,
            "n_training_steps": int(getattr(self, "_n_training_steps", 0)),
            "signum": _STOP["signum"],
        }, indent=2))
        print(f"[graceful-stop] checkpointed {n:,} samples -> {path}", flush=True)
        raise GracefulStop(n, str(path))

    # ── mixed precision ───────────────────────────────────────────────
    # Set to ``torch.bfloat16`` (``--bf16``) to run the whole training
    # step under autocast; ``None`` means full fp32. Wrapping
    # ``super().train()`` covers fastsae's forward, loss and backward.
    autocast_dtype: "torch.dtype | None" = None

    def train(self, *args, **kwargs):
        """Restore from the newest checkpoint, then run fastsae's loop.

        A fresh run (no ``trainer_state.pt``) is unaffected. ``SAE_RESUME=0``
        forces a clean start.
        """
        from consdl.training.resume import (  # noqa: PLC0415
            try_resume_full,
        )
        try:
            self._resumed_samples = try_resume_full(self)
        except Exception as e:  # noqa: BLE001
            self._resumed_samples = 0
            print(f"[resume] skipped: {type(e).__name__}: {e}")
        if self.autocast_dtype is None:
            return super().train(*args, **kwargs)
        # No GradScaler is needed for bf16 (it has fp32's exponent range).
        # Parameters and optimizer state stay fp32.
        print(f"[amp] autocast enabled: {self.autocast_dtype}")
        with torch.autocast(device_type="cuda", dtype=self.autocast_dtype):
            return super().train(*args, **kwargs)

    def initialize_b_dec(self, *args, **kwargs):
        """Skip the geometric-median b_dec init when resuming.

        fastsae's ``train()`` calls this after the restore, and would
        otherwise overwrite the restored ``b_dec``.
        """
        if getattr(self, "_resumed_samples", 0) > 0:
            print("[resume] keeping checkpointed b_dec "
                  "(skipping geometric-median init)")
            return None
        return super().initialize_b_dec(*args, **kwargs)

    def load_trainer_state(self, load_dir: str) -> dict:
        """Restore optimizer / scheduler / counters from ``load_dir``.

        Delegates to :mod:`consdl.training.resume`. Returns the loaded
        state dict, or ``{}`` when the directory has no state.
        """
        import pathlib  # noqa: PLC0415

        from consdl.training.resume import (
            STATE_FILENAME, try_resume,
        )

        src = pathlib.Path(load_dir)
        if not (src / STATE_FILENAME).is_file():
            return {}
        try_resume(self, src.parent, verbose=False)
        return torch.load(str(src / STATE_FILENAME), map_location="cpu",
                          weights_only=False)
