"""iSAE / iSAE-ME: identifiable SAEs (AbsTopK + aRIP regularization).

Port of the ``rsae`` reference (third_party/isaes), "Toward Identifiable
Sparse Autoencoders". Subclasses :class:`TopKSAE` to reuse save/load, the
unit-norm decoder, geometric-median ``b_dec`` init, the tied bias and the
parallel-gradient-removal hook. Adds the sparsity op (magnitude AbsTopK
instead of ReLU+TopK), the optional multi-step encoder, and the aRIP
regularization loss.

Two architectures:
  * ``isae``    — AbsTopK + aRIP (rip_weight 0.1), single-step encoder.
  * ``isae_me`` — same, but a ``num_encode_steps``-step iterative encoder (T=5).

Reference algorithm (third_party/isaes/rsae/):
  * AbsTopK           — utils/topk.py::apply_topk_and_scatter(use_abstopk=True)
  * aRIP loss         — rip_loss.py::compute_rip_loss (decoder-only, weighted=False)
  * multi-step encode — rip_topk.py:183 (z_topk @ step_size + pre_codes, re-topk)
"""

from __future__ import annotations

import json
import pathlib
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
from config_dataclass import config_field, torch_dataclass
from torch import Tensor

from fastsae.training.trainer import LossOutput

from consdl.models.topk import TopKSAE, TopKAuxLossTrainer


def abstopk_scatter(a_pre: Tensor, k: int, use_abstopk: bool) -> Tensor:
    """Top-k by magnitude keeping SIGN (use_abstopk), else ReLU+top-k."""
    if use_abstopk:
        idx = torch.topk(a_pre.abs(), k, dim=-1).indices
        return torch.zeros_like(a_pre).scatter(-1, idx, a_pre.gather(-1, idx))
    pos = torch.relu(a_pre)
    res = torch.topk(pos, k, dim=-1)
    return torch.zeros_like(pos).scatter(-1, res.indices, res.values)


def compute_rip_loss(
    preact: Tensor, dictionary: Tensor, k: int,
    weighted: bool = False, multiplier: int = 2, use_abstopk: bool = True,
) -> Tensor:
    """aRIP loss: mean squared deviation-from-1 of the eigenvalues of the
    k'=k*multiplier Gram submatrices of the (unit-norm) decoder. Decoder-only.

    ``dictionary`` is ``W_dec`` of shape [d_sae, d_in] (unit-norm rows).
    Port of third_party/isaes/rsae/rip_loss.py::compute_rip_loss.
    """
    rip_dim = k * multiplier
    code = abstopk_scatter(preact, rip_dim, use_abstopk)
    if not weighted:  # replace values with N(0,1) on the same support
        code = torch.randn_like(code) * (code != 0.0).float()
    projections = code @ dictionary                      # [b, d_in]
    projections2 = (projections @ dictionary.T) * (code != 0.0)  # [b, d_sae]
    htr = torch.sum(projections ** 2, dim=1).mean()
    htr2 = torch.sum(projections2 ** 2, dim=1).mean()
    return htr2 / htr ** 2 * rip_dim - 1.0


@torch_dataclass
class IsaeSAE(TopKSAE):
    """Identifiable SAE: AbsTopK sparsity + aRIP decoder regularization."""

    architecture: str = config_field(default="isae")
    top_k: int = config_field(default=40)
    use_abstopk: bool = config_field(default=True)
    num_encode_steps: int = config_field(default=1)
    initial_step_size: float = config_field(default=0.9)
    rip_weight: float = config_field(default=0.1)
    rip_multiplier: int = config_field(default=2)
    rip_weighted: bool = config_field(default=False)
    l1_weight: float = config_field(default=0.01)
    #: Train and encode on per-sample mean-centred, unit-L2 inputs.
    #:
    #: The iSAE reference normalizes every batch this way (init, training,
    #: eval and inference). It matters because the reference's ``l1_weight``
    #: was tuned for ‖x‖ = 1, while raw activations can be much larger.
    #: Off by default.
    input_norm: bool = config_field(default=False)

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        # Learned step-size matrix for the multi-step encoder, parameterized as
        # step = I - F (initial_step_size 0.9 ⇒ F = 0.1·I). Only for T>1 (iSAE-ME).
        if int(self.num_encode_steps) > 1:
            f = torch.eye(self.d_sae, dtype=self.dtype) * (1.0 - float(self.initial_step_size))
            self._step_size_factor = nn.Parameter(f)
            self.to(self.device)

    # ── (optional) per-sample input normalization ────────────────────────
    @staticmethod
    def _input_stats(z: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """``(z_normalized, mean, norm)`` -- ``(z - mean(z)) / ‖z - mean(z)‖``.

        Matches the reference's ``BaseTrainer._normalize_data``, including its
        zero-norm guard. Reduction is over the feature axis, per sample.
        """
        mean = z.mean(dim=-1, keepdim=True)
        centered = z - mean
        norm = torch.linalg.vector_norm(centered, dim=-1, keepdim=True).clamp_min(1e-8)
        return centered / norm, mean, norm

    def normalize_input(self, z: Tensor):
        """``(z', mean, norm)``; a no-op returning ``(z, None, None)`` when off.

        When ``input_norm`` is on, the stats are cached so a following
        :meth:`decode` can undo them.
        """
        if not bool(self.input_norm):
            return z, None, None
        z_n, mean, norm = self._input_stats(z)
        self._norm_cache = (mean, norm)
        return z_n, mean, norm

    def decode(self, a: Tensor) -> Tensor:
        """Decode to RAW activation space, undoing the input normalization.

        Callers compare ``sae.decode(sae.encode(z))`` against the raw ``z``,
        so with ``input_norm`` on the reconstruction is mapped back to raw
        space here.

        This makes decode stateful (as in the reference's inference wrapper):
        it must follow an ``encode``/``forward`` on the same batch. The shape
        check catches decoding codes from a different batch. With
        ``input_norm`` off this is just ``super().decode``.
        """
        z_hat = super().decode(a)
        if not bool(self.input_norm):
            return z_hat
        cache = getattr(self, "_norm_cache", None)
        if cache is None:
            raise RuntimeError(
                "IsaeSAE.decode() called with input_norm=True but no preceding "
                "encode()/forward() on this module — the per-sample "
                "normalization stats are needed to return a raw-space "
                "reconstruction and cannot be recovered from the codes alone. "
                "Call encode() (or forward()) on the batch first."
            )
        mean, norm = cache
        if mean.shape[:-1] != z_hat.shape[:-1]:
            raise RuntimeError(
                f"IsaeSAE.decode(): cached normalization stats have batch shape "
                f"{tuple(mean.shape[:-1])} but the codes have "
                f"{tuple(z_hat.shape[:-1])}. decode() must follow encode() on "
                f"the SAME batch; interleaving batches silently mixes them."
            )
        return z_hat * norm + mean

    @torch.no_grad()
    def init_b_dec(self, z, init_method):
        """Geometric median on the same inputs the encoder will actually see.

        As in the reference, the init sample is normalized before the median
        so ``b_dec`` lives in the same space as the training inputs.
        """
        z_n, _, _ = self.normalize_input(z)
        return super().init_b_dec(z_n, init_method)

    # ── sparsity + (optional) iterative encoder ──────────────────────────
    def _abstopk(self, a_pre: Tensor) -> Tensor:
        return abstopk_scatter(a_pre, self._resolved_top_k(), bool(self.use_abstopk))

    def _get_step_size(self) -> Tensor:
        eye = torch.eye(self.d_sae, device=self.W_dec.device, dtype=self.W_dec.dtype)
        return eye - self._step_size_factor

    def _encode_iter(self, z: Tensor) -> Tuple[Tensor, Tensor]:
        """Return (final_pre_activations, codes). Single-step unless T>1."""
        a_pre0 = self._encode(z)
        a = self._abstopk(a_pre0)
        if int(self.num_encode_steps) <= 1:
            return a_pre0, a
        step = self._get_step_size()
        a_pre = a_pre0
        for _ in range(int(self.num_encode_steps) - 1):
            a_pre = a @ step + a_pre0            # pre_step = z_topk @ step + pre_codes
            a = self._abstopk(a_pre)
        return a_pre, a

    def encode(self, z: Tensor) -> Tensor:
        # Normalizes internally, so callers can pass raw activations.
        return self._encode_iter(self.normalize_input(z)[0])[1]

    def preactivations(self, z: Tensor) -> Tensor:
        return self._encode_iter(self.normalize_input(z)[0])[0]

    def forward(
        self,
        z: Tensor,
        steer_cfg: Optional[Dict[str, Any]] = None,
        return_act: bool = True,
        return_pre_act: bool = False,
    ) -> Tuple[Tensor, Tensor] | Tensor:
        z_n, _, _ = self.normalize_input(z)
        a_pre, a = self._encode_iter(z_n)
        if steer_cfg is not None:
            a = self.steer(a, steer_cfg)
        # ``decode`` returns raw-space reconstructions so metrics are
        # comparable across architectures; ``IsaeAuxRipTrainer.loss_fn``
        # computes the loss in normalized space, as the reference does.
        z_hat = self.decode(a)
        if return_pre_act:
            return (z_hat, a, a_pre)
        return (z_hat, a) if return_act else z_hat

    # ── aRIP loss ────────────────────────────────────────────────────────
    def compute_rip_loss(self, preact: Tensor) -> Tensor:
        return compute_rip_loss(
            preact, self.W_dec, self._resolved_top_k(),
            weighted=bool(self.rip_weighted), multiplier=int(self.rip_multiplier),
            use_abstopk=bool(self.use_abstopk),
        )


@torch_dataclass
class IsaeMESAE(IsaeSAE):
    """iSAE-ME: iSAE with a multi-step (T=5) iterative encoder."""

    architecture: str = config_field(default="isae_me")
    num_encode_steps: int = config_field(default=5)


class IsaeAuxRipTrainer(TopKAuxLossTrainer):
    """TopK AuxK trainer + aRIP regularization, with AbsTopK-aware dead-latent
    detection. Total auxiliary term = AuxK(dead-latent recon) + rip_weight·aRIP.
    (L1 is applied by the base trainer via ``sparsity_coefficient``.)

    Also overrides ``loss_fn`` to put the reconstruction term on the
    reference's scale, so the paper's regularizer weights apply unchanged.
    """

    # Un-annotated so it is not a config field: the reference-convention MSE
    # normalizer, set from the first batch.
    _mse_norm = None

    def loss_fn(self, z, a, z_hat):
        """Reference-convention reconstruction loss.

        ``SAETrainer.loss_fn`` normalizes the squared error by the per-sample
        norm ``‖z‖₂``; the iSAE reference normalizes by a constant data
        variance ``E[(z-z̄)²]`` computed once from the first batch. The two
        can differ by an order of magnitude, which would rescale every
        regularizer weight relative to reconstruction, so the reference
        convention is used here.
        """
        sae = self.wrapped.sae
        if bool(getattr(sae, "input_norm", False)):
            # ``IsaeSAE.forward`` returns a raw-space reconstruction; map it
            # back to normalized space, where the reference computes the loss.
            z_n, mean, norm = sae._input_stats(z.float())
            z_hat = (z_hat - mean) / norm
            z = z_n
        if self._mse_norm is None:
            with torch.no_grad():
                flat = z.reshape(-1, z.shape[-1]).float()
                var = ((flat - flat.mean(dim=0, keepdim=True)) ** 2).mean()
                self._mse_norm = float(var.clamp_min(1e-8))
        z_mse_loss = torch.pow(z_hat - z.float(), 2) / self._mse_norm
        a_l1_loss = torch.abs(a).sum(dim=-1).mean(dim=(0,))
        return LossOutput(
            recon_loss=z_mse_loss.mean(),
            sparsity_loss=a_l1_loss.mean(),
            auxiliary_loss=self._compute_auxiliary_loss(z, a, z_hat, z_mse_loss),
        )

    # AbsTopK-aware firing: a latent "fired" if its code is non-zero (signed),
    # and dead-latent candidates are ranked by |pre-activation|.
    def _topk_aux_loss(self, z, a, z_hat):  # noqa: C901 (mirrors parent)
        if self.aux_loss_coefficient == 0.0:
            return z.new_tensor(0.0)
        sae = self.wrapped.sae
        use_abs = bool(getattr(sae, "use_abstopk", False))
        d_sae = int(sae.d_sae)

        with torch.no_grad():
            firing_dims = tuple(range(a.ndim - 1))
            did_fire = (a != 0).any(dim=firing_dims) if use_abs else (a > 0).any(dim=firing_dims)
            num_tokens_in_step = int(a.numel() // a.shape[-1])
            if self.num_tokens_since_fired.device != a.device:
                self.num_tokens_since_fired = self.num_tokens_since_fired.to(a.device)
            self.num_tokens_since_fired += num_tokens_in_step
            self.num_tokens_since_fired[did_fire] = 0
            dead_mask = self.num_tokens_since_fired >= self.dead_feature_threshold
            num_dead = int(dead_mask.sum())
            self._last_dead_count = num_dead
            self._last_num_dead_picked = min(num_dead, self.k_aux)
        if num_dead == 0:
            return z.new_tensor(0.0)
        k_aux = min(num_dead, self.k_aux)

        a_pre = sae._encode(z)
        # Rank dead candidates by magnitude (AbsTopK) or ReLU value.
        score = a_pre.abs() if use_abs else torch.relu(a_pre)
        flat_score = score.reshape(-1, d_sae)
        flat_pre = a_pre.reshape(-1, d_sae)
        masked = torch.where(dead_mask[None, :], flat_score,
                             torch.full_like(flat_score, float("-inf")))
        _, idx = masked.topk(k_aux, dim=-1, sorted=False)
        # Scatter the SIGNED pre-activation at the selected dead latents.
        vals = flat_pre.gather(-1, idx)
        auxk_acts = torch.zeros_like(flat_pre).scatter_(-1, idx, vals).reshape_as(a_pre)

        aux_recon = auxk_acts @ sae.W_dec
        residual = (z - z_hat).detach()
        # Normalize by the centered residual variance, matching the reference
        # (rsae/rip_loss.py::compute_auxk_loss).
        res_flat = residual.reshape(-1, residual.shape[-1])
        aux_l2 = (aux_recon - residual).pow(2).mean()
        loss_denom = ((res_flat - res_flat.mean(dim=0, keepdim=True)) ** 2).mean().clamp_min(1e-12)
        normalized = (aux_l2 / loss_denom).nan_to_num(0.0)
        return self.aux_loss_coefficient * normalized

    def _compute_auxiliary_loss(self, z, a, z_hat, z_mse_loss):
        aux = self._topk_aux_loss(z, a, z_hat)
        sae = self.wrapped.sae
        rip_w = float(getattr(sae, "rip_weight", 0.0))
        if rip_w > 0.0:
            # ``_encode_iter``, not ``preactivations``: ``loss_fn`` has already
            # normalized ``z``.
            preact = sae._encode_iter(z)[0].detach()
            aux = aux + rip_w * sae.compute_rip_loss(preact)
        return aux


def load_isae(load_dir: str | pathlib.Path):
    """Load an IsaeSAE / IsaeMESAE checkpoint by its ``class_type``/architecture."""
    path = pathlib.Path(load_dir)
    for jf in sorted(path.glob("*.json")):
        try:
            cfg = json.loads(jf.read_text())
        except Exception:
            continue
        if isinstance(cfg, dict) and "d_in" in cfg:
            ct = cfg.get("class_type"); arch = (cfg.get("architecture") or "").lower()
            if ct == "IsaeMESAE" or arch == "isae_me":
                return IsaeMESAE.load(path)
            if ct == "IsaeSAE" or arch == "isae":
                return IsaeSAE.load(path)
    raise FileNotFoundError(f"No iSAE config JSON in {path}")
