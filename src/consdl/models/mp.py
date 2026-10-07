"""Matching-Pursuit SAE.

Mirrors the Kempner Institute's ``overcomplete.MpSAE`` but inherits
from fastsae's ``VanillaSAE`` to reuse save / load, unit-norm decoder
projection, and the parallel-gradient-removal hook. :meth:`encode`
replaces the
standard ``relu(W_enc · z + b_enc)`` projection with ``k`` greedy
matching-pursuit rounds against the decoder dictionary ``W_dec``.

Sparsity is structural (one new atom per round, ≤ ``k`` non-zero
codes per token) so no L1 penalty is needed; an optional revival aux
loss that nudges dead atoms back toward the
unexplained residual — see :class:`MPRevivalLossTrainer`.

References
----------
Mallat & Zhang (1993) "Matching Pursuits with Time-Frequency
Dictionaries"; the SAE training-time variant is described in
"From Flat to Hierarchical: Extracting Sparse Representations
with Matching Pursuit" (https://arxiv.org/abs/2506.03093) and
the Kempner Institute's ``overcomplete.MpSAE``.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
from config_dataclass import config_field, torch_dataclass
from fastsae.models.sae import VanillaSAE
from torch import Tensor

from consdl.training.trainer import SAETrainer


@torch_dataclass
class MPSAE(VanillaSAE):
    """SAE that obtains sparse codes by greedy matching pursuit.

    The decoder ``W_dec`` (rows are atoms) is used both for encoding
    and decoding. ``W_enc`` / ``b_enc`` are kept for VanillaSAE
    save / load compatibility but frozen, since ``encode`` never uses
    them. Only ``W_dec`` and ``b_dec`` are trained.

    Training-time encoder runs ``k_iters`` matching-pursuit rounds.
    Each round:
      1. Compute ``corr = (z − b_dec − Σ already-explained) @ W_decᵀ``
         — the dot product of every atom with the current residual.
      2. Pick the atom with the largest correlation per token (or
         the largest non-negative correlation when ``mp_nonneg=True``).
      3. Add ``val · 1_{idx}`` to the running code, scaled by
         ``1 / ‖atom‖²`` for least-squares alignment when atoms
         aren't strictly unit-norm.
      4. Subtract the picked atom's contribution from the residual.

    Inference-time round count is *separate* from training and
    selected by ``inference_mode``:

      * ``"fixed"``    — run exactly ``inference_iters`` rounds.
                         Defaults to ``k_iters``.
      * ``"adaptive"`` — unroll until the relative residual norm drops
                         below ``residual_thr`` or the support stops
                         growing for ``patience`` consecutive rounds,
                         capped at ``max_iters``. No fixed sparsity
                         target is needed at eval time.

    During training, ``mp_dropout`` (when > 0) randomly drops a
    fraction of dictionary atoms each forward; disabled at eval.
    """

    architecture: str = config_field(default="mp")
    k_iters: int = config_field(default=1)
    mp_nonneg: bool = config_field(default=True)
    mp_dropout: float = config_field(default=0.0)
    # Inference-time settings, separate from the training budget.
    inference_mode: str = config_field(default="fixed")
    inference_iters: int = config_field(default=0)   # 0 ⇒ mirror k_iters
    residual_thr: float = config_field(default=0.05)
    max_iters: int = config_field(default=0)         # 0 ⇒ d_sae (set in lazy init)
    patience: int = config_field(default=2)

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        # MP doesn't use W_enc / b_enc.
        self.W_enc.requires_grad_(False)
        self.b_enc.requires_grad_(False)
        if int(self.k_iters) <= 0:
            raise ValueError(f"k_iters must be > 0, got {self.k_iters}")
        if self.inference_mode not in ("fixed", "adaptive"):
            raise ValueError(
                f"inference_mode must be 'fixed' or 'adaptive', "
                f"got {self.inference_mode!r}"
            )
        if int(self.inference_iters) <= 0:
            self.inference_iters = int(self.k_iters)
        if int(self.max_iters) <= 0:
            self.max_iters = int(self.d_sae)

    def _atom_inv_sq(self, W: Tensor) -> Tensor:
        """``1 / ‖atom_i‖²`` per row of ``W`` (the dictionary)."""
        return 1.0 / (W ** 2).sum(dim=-1).clamp_min(1e-30)

    def _mp_step(self, residual: Tensor, codes: Tensor, W: Tensor,
                 inv_sq: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """One matching-pursuit round."""
        corr = residual @ W.t()                              # (B, d_sae)
        if self.mp_nonneg:
            corr_for_argmax = corr.clamp_min(0.0)
        else:
            corr_for_argmax = corr.abs()
        _, idx = torch.max(corr_for_argmax, dim=1)           # (B,)
        picked = corr.gather(1, idx.unsqueeze(1)).squeeze(1)
        if self.mp_nonneg:
            picked = picked.clamp_min(0.0)
        picked = picked * inv_sq[idx]
        update = torch.zeros_like(codes)
        update.scatter_(1, idx.unsqueeze(1), picked.unsqueeze(1))
        return residual - update @ W, codes + update, idx

    def encode(self, z: Tensor) -> Tensor:
        """Run matching pursuit on each row of ``z``.

        Round-count rule depends on the mode:

          * ``self.training``        → fixed ``k_iters`` rounds (also
                                       applies dropout if configured).
          * eval, ``inference_mode = 'fixed'``   → ``inference_iters``
                                                    rounds.
          * eval, ``inference_mode = 'adaptive'`` → unroll until
                                                     residual norm is
                                                     small enough or
                                                     support stabilises.
        """
        flat = z.reshape(-1, z.shape[-1])
        W = self.W_dec                                       # (d_sae, d_in)
        if self.training and float(self.mp_dropout) > 0.0:
            # Bernoulli dropout on whole atoms, resampled every forward.
            keep = torch.bernoulli(torch.full(
                (W.shape[0], 1), 1.0 - float(self.mp_dropout),
                device=W.device, dtype=W.dtype,
            ))
            W = W * keep
        inv_sq = self._atom_inv_sq(W)                        # (d_sae,)
        residual = flat - self.b_dec                         # (B, d_in)
        codes = torch.zeros(flat.shape[0], self.d_sae,
                            device=flat.device, dtype=flat.dtype)

        if self.training:
            n_iters = int(self.k_iters)
            for _ in range(n_iters):
                residual, codes, _ = self._mp_step(residual, codes, W, inv_sq)
            return codes.reshape(*z.shape[:-1], self.d_sae)

        if self.inference_mode == "fixed":
            n_iters = int(self.inference_iters)
            for _ in range(n_iters):
                residual, codes, _ = self._mp_step(residual, codes, W, inv_sq)
            return codes.reshape(*z.shape[:-1], self.d_sae)

        # Adaptive: stop on residual or support stabilisation.
        z_norm = flat.norm(dim=-1).clamp_min(1e-12)
        prev_support_size = 0
        no_grow_count = 0
        thr = float(self.residual_thr)
        max_iters = int(self.max_iters)
        for _ in range(max_iters):
            rel_res = (residual.norm(dim=-1) / z_norm).max().item()
            if rel_res < thr:
                break
            residual, codes, _ = self._mp_step(residual, codes, W, inv_sq)
            support_size = int((codes != 0).any(dim=0).sum().item())
            if support_size == prev_support_size:
                no_grow_count += 1
                if no_grow_count >= int(self.patience):
                    break
            else:
                prev_support_size = support_size
                no_grow_count = 0
        return codes.reshape(*z.shape[:-1], self.d_sae)

    def forward(
        self,
        z: Tensor,
        steer_cfg: Optional[Dict[str, Any]] = None,
        return_act: bool = True,
        return_pre_act: bool = False,
    ) -> Tuple[Tensor, Tensor] | Tensor:
        a = self.encode(z)
        if steer_cfg is not None:
            a = self.steer(a, steer_cfg)
        z_hat = self.decode(a)
        if return_pre_act:
            # MP has no separate pre-activation; return ``a`` twice.
            return (z_hat, a, a)
        return (z_hat, a) if return_act else z_hat

    def aux_losses(self, a: Tensor) -> Dict[str, Tensor]:
        # Sparsity comes from the MP loop; the revival loss lives on the
        # trainer because it needs ``z_hat``.
        return {}


class MPRevivalLossTrainer(SAETrainer):
    """fastsae ``SAETrainer`` + MP-SAE dead-atom revival aux loss.

    Follows the ``overcomplete`` recipe:

        revive_mask = (z.amax(dim=0) < dead_thr).detach()    # shape: (d_sae,)
        if revive_mask.sum() > min_dead:
            projected   = residual @ W_decᵀ                  # shape: (B, d_sae)
            revive_term = projected[:, revive_mask].mean()
            aux_loss    = -revive_coef * revive_term         # subtract → MAXIMISE alignment
                                                             # of dead atoms with the residual

    Maximizing the projection moves dead atoms toward the unexplained
    residual so they can win future argmax rounds, counteracting MP's
    rich-get-richer dynamics. ``use_ghost_grads`` is disabled since this
    replaces fastsae's ghost-grad revival.
    """

    def __init__(
        self,
        *args,
        revive_coef: float = 1e-2,
        dead_threshold: float = 1e-2,
        min_dead: int = 10,
        **kwargs,
    ):
        self.revive_coef = float(revive_coef)
        self.dead_threshold = float(dead_threshold)
        self.min_dead = int(min_dead)
        kwargs["use_ghost_grads"] = False
        super().__init__(*args, **kwargs)
        if not isinstance(self.wrapped.sae, MPSAE):
            raise ValueError(
                "MPRevivalLossTrainer requires an MPSAE wrapped module "
                f"(got {type(self.wrapped.sae).__name__})."
            )

    def _revival_aux_loss(
        self, z: torch.Tensor, a: torch.Tensor, z_hat: torch.Tensor,
    ) -> torch.Tensor:
        if self.revive_coef == 0.0:
            return z.new_tensor(0.0)
        sae = self.wrapped.sae
        # Latents whose peak activation in the batch stays below
        # ``dead_threshold`` are revival candidates.
        peak = a.reshape(-1, a.shape[-1]).amax(dim=0)
        revive_mask = (peak < self.dead_threshold).detach()
        if int(revive_mask.sum().item()) <= self.min_dead:
            return z.new_tensor(0.0)
        # Not detached: gradient reaches W_dec through both the residual
        # and the projection.
        residual = z - z_hat
        projected = residual.reshape(-1, sae.d_in) @ sae.W_dec.t()  # (B, d_sae)
        revive_term = projected[:, revive_mask].mean()
        return -self.revive_coef * revive_term

    def _compute_auxiliary_loss(self, z, a, z_hat, z_mse_loss) -> torch.Tensor:
        return self._revival_aux_loss(z, a, z_hat)


# ── Eval-time MP inference helpers ─────────────────────────────────────
#
#   * Train-time max-iters resolution from a per-token sparsity cap.
#   * Eval-time per-mode overrides: ``gtk`` (fixed at ``k_active``),
#     ``k<N>`` (fixed at N) and ``r2`` (adaptive, residual-driven).

import argparse  # noqa: E402  — local to the helper block
import re  # noqa: E402
from typing import Mapping  # noqa: E402

MP_MODES: tuple[str, ...] = ("gtk", "r2")
"""Per-mode metric files land at ``...mp_<mode>.json`` for each entry.

These are the two NAMED modes.  A third family, ``k<N>`` (``k32``,
``k64``, ``k128``, …), is accepted anywhere a mode is: see
:func:`parse_mp_mode`.  It is not listed here because it is unbounded —
``MP_MODES`` stays the enumerable set that a UI or a default can loop
over."""

_K_MODE = re.compile(r"^k([1-9][0-9]*)$")


def parse_mp_mode(mode: str) -> int | None:
    """``"k64"`` → ``64``; any other label → ``None``.

    MP's sparsity is not fixed by the weights: one checkpoint can run
    any number of rounds, one atom per round, so ``inference_iters`` is
    the L0. A single trained checkpoint can therefore be evaluated at
    several L0 values (``k32``/``k64``/``k128``) without retraining.
    """
    m = _K_MODE.match(str(mode))
    return int(m.group(1)) if m else None


def validate_mp_mode(mode: str) -> str:
    """argparse ``type=`` hook: accept a named mode or ``k<N>``."""
    if mode in MP_MODES or parse_mp_mode(mode) is not None:
        return mode
    raise argparse.ArgumentTypeError(
        f"unknown MP eval mode {mode!r} — expected one of "
        f"{', '.join(MP_MODES)} or k<N> (e.g. k32, k64, k128)"
    )

DEFAULT_MP_MODES: tuple[str, ...] = ("gtk",)
"""Modes that auto-eval triggers by default after MP training.  ``gtk``
matches the ground-truth support size per token (k_active = 60) and is
directly comparable to fixed-K methods, so the paper figures default to
it.  ``r2`` (adaptive, residual-driven) is still selectable via
``--mp-modes r2`` when its threshold-stopping behaviour is the subject
of the comparison."""

DEFAULT_R2_SPARSITY_CAP: float = 0.1
"""Default per-token sparsity cap (fraction of ``d_sae``) for both
train-time max_iters resolution and r2-mode eval-time max_iters."""

DEFAULT_R2_RESIDUAL_THR: float = 0.05
"""Default relative-residual stopping threshold; matches MPSAE's class
default and the trainer-side ``--mp-residual-thr``."""


def resolve_max_iters(
    *,
    d_sae: int,
    explicit: int = 0,
    sparsity_cap: float = 0.0,
) -> int | None:
    """Resolve ``mp_max_iters`` for the train-time MP encoder / SAE.

    Order:
      1. ``explicit > 0``  → user override wins.
      2. ``sparsity_cap > 0`` → ``ceil(cap × d_sae)``.  Each MP round
         adds at most one atom, so capping iterations directly bounds
         per-token L0.
      3. ``None`` — caller falls back to the encoder default
         (typically ``d_sae``).
    """
    if int(explicit) > 0:
        return int(explicit)
    if float(sparsity_cap) > 0:
        return max(1, int(float(sparsity_cap) * int(d_sae)))
    return None


def mode_override(
    mode: str,
    *,
    k_active: int,
    d_sae: int,
    cfg_residual_thr: float = DEFAULT_R2_RESIDUAL_THR,
    residual_thr_override: float = 0.0,
    r2_sparsity_cap: float = DEFAULT_R2_SPARSITY_CAP,
) -> dict[str, Any]:
    """Map an MP eval-mode label to encoder-attribute overrides.

    ``gtk`` — fixed mode at exactly ``k_active`` rounds (the ground-truth
    support size per token).

    ``k<N>`` — fixed mode at ``N`` rounds, so one checkpoint can be
    measured on several L0 bands (see :func:`parse_mp_mode`).

    ``r2`` — adaptive mode keyed off the per-token relative residual
    norm.  Threshold resolution: ``residual_thr_override`` when > 0
    else ``cfg_residual_thr`` (which the caller pulls from each run's
    saved training cfg).  Sparsity cap resolution:
    ``r2_sparsity_cap × d_sae`` → ``max_iters`` (when > 0).

    The returned keys are :class:`MPSAE` attribute names, so they can be
    passed to a constructor or applied to a loaded model via
    :func:`apply_mp_override`.
    """
    if mode == "gtk":
        return {
            "inference_mode": "fixed",
            "inference_iters": int(k_active),
        }
    # ``k<N>``: fixed mode at a chosen L0 (``gtk`` equals ``k<k_active>``).
    k_n = parse_mp_mode(mode)
    if k_n is not None:
        return {
            "inference_mode": "fixed",
            "inference_iters": int(k_n),
        }
    if mode == "r2":
        thr = (
            float(residual_thr_override)
            if float(residual_thr_override) > 0
            else float(cfg_residual_thr)
        )
        out: dict[str, Any] = {
            "inference_mode": "adaptive",
            "residual_thr": thr,
        }
        if float(r2_sparsity_cap) > 0 and int(d_sae) > 0:
            out["max_iters"] = max(1, int(float(r2_sparsity_cap) * int(d_sae)))
        return out
    raise ValueError(
        f"unknown mp eval mode {mode!r} (expected one of {MP_MODES})"
    )


def apply_mp_override(sae, override: Mapping[str, Any]) -> None:
    """Mutate a loaded MP SAE's inference attributes in place.

    Switches a loaded SAE between eval modes without rebuilding it. Keys
    the SAE does not have are ignored, so non-MP models pass through.
    """
    for key, value in override.items():
        if hasattr(sae, key):
            setattr(sae, key, value)


def add_mp_train_args(
    parser: argparse.ArgumentParser,
    *,
    group_title: str = "MP train options",
) -> None:
    """Register ``--mp-sparsity-cap`` on a trainer's parser.

    Other MP training flags are registered by each training CLI.
    """
    g = parser.add_argument_group(group_title)
    g.add_argument(
        "--mp-sparsity-cap",
        type=float,
        default=DEFAULT_R2_SPARSITY_CAP,
        help=(
            "Train-time per-token sparsity cap for MP.  Resolves "
            "``mp_max_iters = ceil(cap × d_sae)`` when "
            "``--mp-max-iters`` is left at 0.  Default "
            f"{DEFAULT_R2_SPARSITY_CAP} (10%% of d_sae).  Pass "
            "``0`` to disable the cap (max_iters falls back to d_sae)."
        ),
    )


def add_mp_eval_args(
    parser: argparse.ArgumentParser,
    *,
    group_title: str = "MP eval options",
    legacy_alias: bool = True,
) -> None:
    """Register the canonical eval-time MP flags.

    Canonical names: ``--mp-modes``, ``--mp-residual-thr``,
    ``--mp-r2-sparsity-cap``.  Variables on the parsed namespace use
    the matching ``snake_case`` form.

    When ``legacy_alias`` is set, a hidden alias ``--mp-reeval-modes``
    also writes into ``mp_modes``.
    """
    g = parser.add_argument_group(group_title)
    g.add_argument(
        "--mp-modes",
        nargs="+",
        type=validate_mp_mode,
        default=list(DEFAULT_MP_MODES),
        help=(
            "Inference rules to evaluate for ``--architecture=mp`` "
            "checkpoints.  Each mode writes its own metrics file "
            "(``..._mp_<mode>.json``):\n"
            "  * ``gtk`` — fixed mode at ``inference_iters = k_active``.\n"
            "  * ``r2``  — adaptive mode (residual ‖r‖/‖z‖ threshold).\n"
            "  * ``k<N>`` — fixed mode at N rounds, so ONE checkpoint "
            "can be measured at several L0 bands (``k32 k64 k128``).\n"
            "Non-mp runs ignore this flag."
        ),
    )
    if legacy_alias:
        g.add_argument(
            "--mp-reeval-modes",
            nargs="+",
            type=validate_mp_mode,
            dest="mp_modes",
            help=argparse.SUPPRESS,
        )
    g.add_argument(
        "--mp-residual-thr",
        type=float,
        default=0.0,
        help=(
            "Override the relative residual norm threshold used by "
            "the ``r2`` (adaptive) mode at eval time.  ``0`` (default) "
            "→ use ``mp_residual_thr`` from each run's training cfg "
            f"(typically {DEFAULT_R2_RESIDUAL_THR})."
        ),
    )
    g.add_argument(
        "--mp-r2-sparsity-cap",
        type=float,
        default=DEFAULT_R2_SPARSITY_CAP,
        help=(
            "Per-token sparsity cap for the ``r2`` mode at eval time. "
            "Translates to ``max_iters = ceil(cap × d_sae)``.  Default "
            f"{DEFAULT_R2_SPARSITY_CAP} (10%% of d_sae); pass ``0`` "
            "to disable the cap."
        ),
    )
