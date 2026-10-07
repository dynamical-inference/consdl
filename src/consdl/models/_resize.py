"""Exact-``d_sae`` support for fastsae's ``VanillaSAE`` (and subclasses).

fastsae stores the SAE width as ``d_sae = d_in * d_sae_factor`` with an
integer factor. To allow arbitrary ``d_sae`` this module provides:

* :func:`resize_sae_in_place`, which rebuilds ``W_enc`` / ``W_dec`` /
  ``b_enc`` at a target ``d_sae`` using VanillaSAE's own init convention.
* A patch on ``VanillaSAE._load_additional`` that resizes the SAE to the
  saved ``W_enc`` shape before loading, so every loader works without
  knowing the width was overridden. It also loads ``model.safetensors``
  checkpoints.

Importing this module installs the patch exactly once.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from fastsae.models.sae import VanillaSAE


def _is_archetypal(sae: VanillaSAE) -> bool:
    """Return True for ``ArchetypalSAE`` / ``TopKArchetypalSAE`` instances.

    Detects by parameter presence to avoid importing the archetypal module
    (a circular import at patch-install time).
    """
    return (
        "W_archetypal" in dict(sae.named_parameters())
        and "W_dec" not in dict(sae.named_parameters())
    )


def resize_sae_in_place(
    sae: VanillaSAE,
    target_d_sae: int,
    seed: int | None = None,
) -> None:
    """Rebuild ``W_enc`` / ``W_dec`` / ``b_enc`` at a new ``d_sae``.

    Mirrors VanillaSAE's ``__lazy_post_init__``: kaiming-uniform
    ``W_enc``, ``W_dec = W_enc.T``, zero ``b_enc``; ``b_dec`` is untouched.
    ``d_sae_factor`` is left as-is so the saved config stays valid under
    fastsae's schema; the load-time patch resizes back on reload.

    For archetypal SAEs ``W_dec`` is a derived property, so instead
    ``W_archetypal`` and ``Relax`` are resized as placeholders that
    ``init_archetypal`` later re-initializes.
    """
    if int(target_d_sae) == int(sae.d_sae):
        return
    if seed is not None:
        torch.manual_seed(int(seed))
    device = sae.W_enc.device
    dtype = sae.W_enc.dtype
    d_in = int(sae.d_in)
    target = int(target_d_sae)

    new_W_enc = torch.empty(d_in, target, dtype=dtype, device=device)
    nn.init.kaiming_uniform_(new_W_enc)
    sae.W_enc = nn.Parameter(new_W_enc)
    sae.b_enc = nn.Parameter(torch.zeros(target, dtype=dtype, device=device))

    if _is_archetypal(sae):
        nb = int(getattr(sae, "nb_candidates", 0)) or int(sae.W_archetypal.shape[1])
        W_arch = torch.zeros(target, max(nb, 1), dtype=dtype, device=device)
        if nb > 0:
            W_arch.fill_(1.0 / nb)
        sae.W_archetypal = nn.Parameter(W_arch)
        sae.Relax = nn.Parameter(torch.zeros(target, d_in, dtype=dtype, device=device))
    else:
        sae.W_dec = nn.Parameter(
            new_W_enc.data.clone().transpose(0, 1).contiguous(),
        )

    sae.d_sae = target


# ── Load-path monkey-patch ───────────────────────────────────────────

_INSTALL_FLAG = "_consdl_resize_patched"


def _install_load_resize_patch() -> None:
    """Install the load-time resize patch on ``VanillaSAE`` once."""
    if getattr(VanillaSAE, _INSTALL_FLAG, False):
        return
    orig_load_additional = VanillaSAE._load_additional

    def _load_additional_with_resize(self: VanillaSAE, path: Path) -> Any:
        sd_file = Path(path) / "state_dict.pt"
        # The base hook only reads ``state_dict.pt``; also accept
        # ``model.safetensors``.
        st_file = Path(path) / "model.safetensors"
        from_safetensors = not sd_file.exists() and st_file.exists()
        sd = None
        if from_safetensors:
            from safetensors.torch import load_file  # noqa: PLC0415
            sd = load_file(str(st_file), device="cpu")
        elif sd_file.exists():
            try:
                sd = torch.load(
                    sd_file, map_location="cpu", weights_only=False,
                )
            except Exception:
                sd = None
        if isinstance(sd, dict):
            W_enc = sd.get("W_enc")
            if (
                isinstance(W_enc, torch.Tensor)
                and W_enc.dim() == 2
                and int(W_enc.shape[1]) != int(self.d_sae)
            ):
                resize_sae_in_place(self, int(W_enc.shape[1]))
        if from_safetensors:
            self.load_state_dict(sd)
            return None
        return orig_load_additional(self, path)

    VanillaSAE._load_additional = _load_additional_with_resize  # type: ignore[assignment]
    setattr(VanillaSAE, _INSTALL_FLAG, True)


_install_load_resize_patch()
