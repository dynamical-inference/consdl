"""Identity backbone that passes synthetic ``z`` through unchanged.

The trainer only calls ``backbone.forward_to_z(batch)`` and
``backbone.save(path)``, so this provides just that surface for a
dataloader that already yields the target tensor ``z``.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn


class IdentityBackbone(nn.Module):
    def __init__(self, d_in: int, device: str | torch.device = "cuda",
                 dtype: torch.dtype = torch.float32):
        super().__init__()
        self._d_in = int(d_in)
        self._hook_target_dim = int(d_in)
        self.device = torch.device(device)
        self.dtype = dtype
        # Frozen dummy Parameter so ``next(self.parameters())`` (used to pick
        # the autocast device) does not raise StopIteration. A buffer would
        # not show up in ``.parameters()``.
        self._anchor = nn.Parameter(torch.zeros(1, dtype=dtype), requires_grad=False)

    # ── Interface the trainer touches ──────────────────────────────────────

    @property
    def hook_target_dim(self) -> int:
        return self._hook_target_dim

    @property
    def d_in(self) -> int:
        return self._d_in

    def forward_to_z(self, x: torch.Tensor) -> torch.Tensor:
        return x.to(device=self.device, dtype=self.dtype, non_blocking=True)

    def forward(self, x, return_hooked_out: bool = False):
        z = self.forward_to_z(x)
        if return_hooked_out:
            return z, z
        return z

    def save(self, path):
        p = Path(path); p.mkdir(parents=True, exist_ok=True)
        (p / "backbone_config.json").write_text(json.dumps({
            "kind": "IdentityBackbone", "d_in": self._d_in,
        }))

    # ── nn.Module bookkeeping ──────────────────────────────────────────────

    def to(self, device):  # type: ignore[override]
        self.device = torch.device(device)
        return super().to(device)
