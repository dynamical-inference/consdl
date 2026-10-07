"""BatchTopK and Matryoshka-BatchTopK SAEs.

**BatchTopK** (Bussmann, Leask & Nanda 2024, arXiv:2412.06410) replaces
top-k's per-token budget with a per-batch one: keep the largest
``k * n_rows`` activations of the flattened batch, so mean L0 is still
``k`` but individual tokens may use more or fewer latents. Inference has
no batch to rank against, so training tracks an EMA of the smallest
surviving activation and eval thresholds against it (a JumpReLU at eval).

**Matryoshka** splits the dictionary into nested groups and requires every
prefix to reconstruct the input on its own, pushing early groups towards
general features. Its sparsity is BatchTopK's plus an extra loss.

Both follow ``bartbussmann/matryoshka_sae`` and SAELens, with two
departures that keep them comparable with the other architectures here:

* the per-prefix reconstruction error uses the input-norm normalization
  of ``SAETrainer.loss_fn`` rather than a raw ``.pow(2).mean()``;
* group sizes are fractions of ``d_sae`` rather than absolute sizes, so
  the number of groups does not change with the expansion factor.

Reference: https://github.com/bartbussmann/matryoshka_sae
           https://github.com/decoderesearch/SAELens
"""

from __future__ import annotations

import torch
from config_dataclass import config_field, torch_dataclass
from torch import Tensor

from consdl.models.topk import TopKAuxLossTrainer, TopKSAE

# EMA rate for the inference threshold. 0.01 is the value both reference
# implementations use and neither exposes as a knob.
THRESHOLD_LR = 0.01

# Nested group sizes as fractions of ``d_sae``, as in the reference. The
# first two groups are intentionally the same size.
MATRYOSHKA_GROUP_FRACTIONS = (1 / 16, 1 / 16, 1 / 8, 1 / 4, 1 / 2)


def group_sizes_for(d_sae: int, fractions=MATRYOSHKA_GROUP_FRACTIONS) -> list[int]:
    """Split ``d_sae`` by ``fractions``, giving the remainder to the last.

    Assigning the remainder (rather than dropping it) guarantees the
    groups cover the full dictionary.
    """
    sizes = [int(d_sae * f) for f in fractions[:-1]]
    sizes.append(int(d_sae) - sum(sizes))
    if any(s <= 0 for s in sizes):
        raise ValueError(
            f"d_sae={d_sae} is too small for {len(fractions)} matryoshka "
            f"groups: got {sizes}. Use a larger expansion factor."
        )
    return sizes


@torch_dataclass
class BatchTopKSAE(TopKSAE):
    """Top-k over the flattened batch instead of per token.

    Training keeps the ``k * n_rows`` largest activations in the batch.
    Evaluation thresholds against an EMA of the smallest activation that
    survived training-time selection, so eval L0 is only approximately
    ``k``. ``top_k_inference`` is ignored.
    """

    architecture: str = config_field(default="batchtopk")

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        # A buffer so the threshold is saved with the checkpoint.
        if not hasattr(self, "threshold"):
            # Use the module's own device: this lazy init can run after
            # ``.to(device)``.
            _ref = next(self.parameters(), None)
            self.register_buffer(
                "threshold",
                torch.zeros((), dtype=torch.float32,
                            device=None if _ref is None else _ref.device),
            )

    @torch.no_grad()
    def _update_threshold(self, a: Tensor) -> None:
        # Outside autocast: a bf16 EMA over many steps accumulates error.
        with torch.autocast(a.device.type, enabled=False):
            pos = a > 0
            if bool(pos.any()):
                lo = a[pos].min().to(self.threshold.dtype)
                self.threshold.mul_(1.0 - THRESHOLD_LR).add_(THRESHOLD_LR * lo)

    def _apply_topk(self, a: Tensor) -> Tensor:
        if not self.training:
            return torch.where(a > self.threshold.to(a.dtype),
                               a, torch.zeros_like(a))
        n_rows = int(a.numel() // a.shape[-1])
        k_total = int(self._resolved_top_k()) * n_rows
        flat = a.reshape(-1)
        k_total = min(k_total, flat.numel())
        top = torch.topk(flat, k_total, dim=-1)
        out = (torch.zeros_like(flat)
               .scatter(-1, top.indices, top.values)
               .reshape(a.shape))
        self._update_threshold(out)
        return out


@torch_dataclass
class MatryoshkaBatchTopKSAE(BatchTopKSAE):
    """BatchTopK whose dictionary is a nest of self-sufficient prefixes.

    This class only stores the group split; the matryoshka loss lives in
    :class:`MatryoshkaAuxLossTrainer`.
    """

    architecture: str = config_field(default="matryoshka")
    # Stored as a comma-separated string so it round-trips through
    # ``config_dataclass`` cleanly.
    group_sizes_csv: str = config_field(default="")

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if not self.group_sizes_csv:
            self.group_sizes_csv = ",".join(
                str(s) for s in group_sizes_for(int(self.d_sae)))

    @property
    def group_sizes(self) -> list[int]:
        return [int(s) for s in self.group_sizes_csv.split(",") if s]

    @property
    def group_bounds(self) -> list[int]:
        """Cumulative prefix ends, e.g. ``[192, 384, 768, 1536, 3072]``."""
        out, run = [], 0
        for s in self.group_sizes:
            run += s
            out.append(run)
        return out


class BatchTopKAuxLossTrainer(TopKAuxLossTrainer):
    """AuxK dead-latent revival, unchanged, for a BatchTopK SAE.

    BatchTopK's ``_resolved_top_k`` is the mean per-token budget, which is
    what AuxK's ``k_aux`` heuristic needs.
    """


class MatryoshkaAuxLossTrainer(BatchTopKAuxLossTrainer):
    """AuxK plus the nested prefix reconstruction losses.

    ``recon_loss`` becomes the mean over prefixes of each prefix's own
    reconstruction error, including the empty prefix (``b_dec`` alone).
    Dividing by ``len(prefixes) + 1`` (as the reference does) keeps the
    term on the scale of a plain reconstruction loss.
    """

    def loss_fn(self, z, a, z_hat):
        base = super().loss_fn(z, a, z_hat)
        sae = self.wrapped.sae
        bounds = getattr(sae, "group_bounds", None)
        if not bounds:
            return base

        z32 = z.float()
        denom = (z32 ** 2).sum(dim=-1, keepdim=True).sqrt().clamp_min(1e-6)

        def err(recon: Tensor) -> Tensor:
            return (torch.pow(recon - z32, 2) / denom).mean()

        # Accumulate incrementally: each prefix is the previous plus one group.
        recon = sae.b_dec.float().expand_as(z32)
        total = err(recon)
        start = 0
        for end in bounds:
            recon = recon + a[..., start:end].float() @ sae.W_dec[start:end].float()
            total = total + err(recon)
            start = end

        # ``_replace`` keeps any fields a subclass added to LossOutput.
        return base._replace(recon_loss=total / (len(bounds) + 1))
