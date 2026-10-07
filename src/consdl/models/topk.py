"""Top-K SAE: enforces sparsity via per-token top-k selection.

Inherits from fastsae's ``VanillaSAE`` to reuse save/load, unit-norm
decoder projection and the parallel-gradient-removal hook; only
``encode`` / ``forward`` / ``aux_losses`` are overridden.

Importing this module registers ``TopKSAE`` with ``config_dataclass``.
Use :func:`load_sae` to reload a TopK checkpoint; fastsae's ``SAE.load``
does not know this class. Also provides :class:`TopKAuxLossTrainer`
(AuxK dead-latent loss).

Reference: Gao et al. 2024, "Scaling and evaluating sparse autoencoders"
— https://arxiv.org/abs/2406.04093.
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Tuple

import torch
from config_dataclass import config_field, torch_dataclass
from fastsae.models.sae import VanillaSAE
from torch import Tensor

from consdl.training.trainer import SAETrainer


@torch_dataclass
class TopKSAE(VanillaSAE):
    """SAE that enforces sparsity via per-token top-k selection.

    Keeps the ``top_k`` largest activations per token (last dim),
    zeroing out the rest.  Every token independently contributes
    exactly ``top_k`` non-zero latents.

    The inference-time budget ``top_k_inference`` can differ from the
    training budget; ``0`` (default) uses ``top_k``. Top-k is applied
    along the last dim.
    """

    architecture: str = config_field(default="topk")
    top_k: int = config_field(default=64)
    top_k_inference: int = config_field(default=0)   # 0 ⇒ mirror top_k

    def _resolved_top_k(self) -> int:
        """Train budget in training mode; inference budget at eval
        (falling back to ``top_k`` when unset)."""
        if self.training or int(self.top_k_inference) <= 0:
            return int(self.top_k)
        return int(self.top_k_inference)

    def _apply_topk(self, a: Tensor) -> Tensor:
        """Zero out all but the top-k activations per token (last dim)."""
        k = self._resolved_top_k()
        topk_result = torch.topk(a, k, dim=-1)
        return torch.zeros_like(a).scatter(
            -1, topk_result.indices, topk_result.values,
        )

    def encode(self, z: Tensor) -> Tensor:
        a_pre = self._encode(z)
        return self._apply_topk(torch.relu(a_pre))

    def forward(
        self,
        z: Tensor,
        steer_cfg: Optional[Dict[str, Any]] = None,
        return_act: bool = True,
        return_pre_act: bool = False,
    ) -> Tuple[Tensor, Tensor] | Tensor:
        a_pre = self._encode(z)
        a = self._apply_topk(torch.relu(a_pre))
        if steer_cfg is not None:
            a = self.steer(a, steer_cfg)
        z_hat = self.decode(a)
        if return_pre_act:
            return (z_hat, a, a_pre)
        return (z_hat, a) if return_act else z_hat

    def aux_losses(self, a: Tensor) -> Dict[str, Tensor]:
        # Sparsity is enforced by top-k; no L1 penalty needed.
        return {}


# ── Dead-latent threshold ────────────────────────────────────────────
# ``num_tokens_since_fired`` counts rows (tokens for LM, samples for
# vision / synthetic) since a latent last fired; it is "dead" once the
# counter reaches ``dead_feature_threshold``.
#
# The common absolute default (10M tokens, ~2% of a 500M-token run) is
# never reached on shorter runs, which silently disables AuxK. The
# threshold is therefore set to the same fraction of the training budget.
DEAD_THRESHOLD_BUDGET_FRAC = 0.02
DEAD_THRESHOLD_FLOOR = 10_000


def resolve_dead_feature_threshold(explicit: int | None, *, total_samples: int) -> int:
    """``explicit`` when > 0, else ``2 %`` of the training budget.

    ``0`` / ``None`` means auto. Floored at :data:`DEAD_THRESHOLD_FLOOR`
    so very short runs do not mark every latent dead on step 1.
    """
    if explicit and int(explicit) > 0:
        return int(explicit)
    return max(DEAD_THRESHOLD_FLOOR, int(int(total_samples) * DEAD_THRESHOLD_BUDGET_FRAC))


class TopKAuxLossTrainer(SAETrainer):
    """fastsae ``SAETrainer`` + dead-mask-gated AuxK auxiliary loss
    (Gao et al. 2024 §B.1).

    Tracks a per-latent ``num_tokens_since_fired`` counter; a latent that
    has not fired for ``dead_feature_threshold`` tokens is dead and
    eligible for the aux loss. At every step:

        post = relu(W_enc · z + b_enc)               # (..., d_sae)
        dead = num_tokens_since_fired >= threshold   # (d_sae,)
        auxk_latents = where(dead, post, -inf)       # mask non-dead OUT
        vals, idx = auxk_latents.topk(k_aux)          # k_aux = d_in // 2
        auxk_acts = zeros.scatter(idx, vals)
        aux_recon = auxk_acts @ W_dec                # no bias
        residual  = (z - z_hat).detach()             # gradient only via aux path
        aux_loss  = (aux_recon - residual)² / ||residual||²  (normalised)
        loss      = main_l2 + alpha * aux_loss       # alpha defaults to 1/32

    Notes:

    * Only latents marked dead by the counter can be selected.
    * ``k_aux = d_in // 2`` (Gao et al. Appendix B.1).
    * The loss is normalized by residual energy, so it is invariant to
      reconstruction scale.
    * The residual is detached, so aux gradients reach only the dead
      latents' encoder and decoder weights.
    * Default coefficient 1/32 (Gao et al. Appendix A.2).

    ``use_ghost_grads`` is disabled since this replaces fastsae's
    ghost-grad revival.
    """

    def __init__(
        self,
        *args,
        aux_loss_coefficient: float = 1.0 / 32.0,
        dead_feature_threshold: int = 10_000_000,
        top_k_aux: int | None = None,
        **kwargs,
    ):
        self.aux_loss_coefficient = float(aux_loss_coefficient)
        self.dead_feature_threshold = int(dead_feature_threshold)
        # ``None`` resolves to ``d_in // 2`` (see :attr:`k_aux`).
        self._top_k_aux_override: int | None = (
            int(top_k_aux) if top_k_aux is not None else None
        )
        kwargs["use_ghost_grads"] = False
        super().__init__(*args, **kwargs)
        if not hasattr(self.wrapped.sae, "_resolved_top_k"):
            raise ValueError(
                "TopKAuxLossTrainer requires a top-k-style SAE "
                "(must expose ``_resolved_top_k``); got "
                f"{type(self.wrapped.sae).__name__}."
            )
        # Dead-latent counter, kept on the trainer (not in the checkpoint).
        d_sae = int(self.wrapped.sae.d_sae)
        self.register_buffer_or_attr(
            "num_tokens_since_fired",
            torch.zeros(d_sae, dtype=torch.long),
        )
        # Diagnostics for logging.
        self._last_dead_count = 0
        self._last_num_dead_picked = 0

    # fastsae's SAETrainer is not an ``nn.Module``, so "buffers" are
    # plain attributes.
    def register_buffer_or_attr(self, name: str, tensor: torch.Tensor) -> None:
        setattr(self, name, tensor)

    @property
    def k_aux(self) -> int:
        """Resolve ``top_k_aux``: override > ``d_in // 2`` heuristic."""
        if self._top_k_aux_override is not None:
            return self._top_k_aux_override
        d_in = int(self.wrapped.sae.d_in)
        return max(1, d_in // 2)

    # ── Aux loss kernel ───────────────────────────────────────────────

    def _topk_aux_loss(
        self, z: torch.Tensor, a: torch.Tensor, z_hat: torch.Tensor,
    ) -> torch.Tensor:
        if self.aux_loss_coefficient == 0.0:
            return z.new_tensor(0.0)

        sae = self.wrapped.sae
        d_sae = int(sae.d_sae)

        # 1) Update the tokens-since-fired counter.
        with torch.no_grad():
            firing_dims = tuple(range(a.ndim - 1))
            did_fire = (a > 0).any(dim=firing_dims)         # (d_sae,)
            num_tokens_in_step = int(a.numel() // a.shape[-1])
            if self.num_tokens_since_fired.device != a.device:
                self.num_tokens_since_fired = (
                    self.num_tokens_since_fired.to(a.device)
                )
            self.num_tokens_since_fired += num_tokens_in_step
            self.num_tokens_since_fired[did_fire] = 0

            # 2) Dead-latent set.
            dead_mask = (
                self.num_tokens_since_fired >= self.dead_feature_threshold
            )
            num_dead = int(dead_mask.sum())
            self._last_dead_count = num_dead
            self._last_num_dead_picked = min(num_dead, self.k_aux)
        if num_dead == 0:
            return z.new_tensor(0.0)
        k_aux = min(num_dead, self.k_aux)

        # 3) Top-k of dead-mask-gated post-ReLU activations.
        a_pre = sae._encode(z)                              # (..., d_sae)
        post_relu = torch.relu(a_pre)                       # (..., d_sae)
        flat = post_relu.reshape(-1, d_sae)
        masked = torch.where(
            dead_mask[None, :], flat,
            torch.full_like(flat, float("-inf")),
        )
        vals, idx = masked.topk(k_aux, dim=-1, sorted=False)
        # Replace any ``-inf`` picks with zeros.
        vals = torch.where(torch.isfinite(vals), vals, torch.zeros_like(vals))
        auxk_acts = torch.zeros_like(flat).scatter_(-1, idx, vals)
        auxk_acts = auxk_acts.reshape_as(post_relu)

        # 4) Reconstruct the residual without ``b_dec`` (the residual
        # already excludes it).
        aux_recon = auxk_acts @ sae.W_dec                   # (..., d_in)
        residual = (z - z_hat).detach()
        aux_l2 = (aux_recon - residual).pow(2).sum(dim=-1).mean()
        loss_denom = (residual.pow(2).sum(dim=-1).mean()).clamp_min(1e-12)
        normalized = (aux_l2 / loss_denom).nan_to_num(0.0)
        return self.aux_loss_coefficient * normalized

    # ── Auxiliary-loss hook ───────────────────────────────────────────

    def _compute_auxiliary_loss(self, z, a, z_hat, z_mse_loss) -> torch.Tensor:
        return self._topk_aux_loss(z, a, z_hat)


def load_sae(load_dir: str | pathlib.Path):
    """Load a VanillaSAE / TopKSAE checkpoint by reading its ``class_type``.

    Falls back to ``fastsae.models.SAE.load`` for anything that is not
    TopK (e.g. ReLU checkpoints).
    """
    path = pathlib.Path(load_dir)
    cfg = None
    for json_file in sorted(path.glob("*.json")):
        try:
            data = json.loads(json_file.read_text())
        except Exception:
            continue
        if isinstance(data, dict) and "d_in" in data and (
            "class_type" in data or "architecture" in data
        ):
            cfg = data
            break
    if cfg is None:
        raise FileNotFoundError(f"Could not locate SAE config JSON in {path}")

    class_type = cfg.get("class_type")
    architecture = (cfg.get("architecture") or "").lower()
    if class_type == "TopKSAE" or architecture == "topk":
        return TopKSAE.load(path)
    # Delegate to fastsae for other classes.
    import fastsae.models
    return fastsae.models.SAE.load(str(path))
