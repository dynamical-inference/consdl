"""SAE and decoder loading for the viewers.

MP's sparsity is its inference round count, not its weights (the released
MP cells of one expansion share one checkpoint), so an MP panel of level
``l0_<N>`` is encoded with exactly N rounds, as ``code_metrics --mp-k``
scores it. Fixed rounds are per token, so MP codes do not depend on the
batch.
"""
from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from consdl.viewers.layout import level_l0


def unit_decoder(step_dir: Path) -> np.ndarray:
    """``(d_sae, d_in)`` decoder with unit rows (not cached)."""
    from consdl.io.checkpoint import load_raw_decoder  # noqa: PLC0415

    W = load_raw_decoder(Path(step_dir))
    if W is None:
        raise FileNotFoundError(f"no readable decoder in {step_dir}")
    W = np.asarray(W, dtype=np.float32)
    return W / np.linalg.norm(W, axis=1, keepdims=True).clip(1e-12)


def mp_override(arch: str, level: str) -> dict | None:
    l0 = level_l0(level)
    if arch != "mp" or l0 is None:
        return None
    return {"inference_mode": "fixed", "inference_iters": int(l0)}


def load_sae(step_dir: Path, *, arch: str, level: str, device):
    """One seed's SAE on ``device``, in eval mode (MP pinned to its level's L0)."""
    from consdl.io.checkpoint import load_sae_device_safe  # noqa: PLC0415

    with contextlib.redirect_stdout(io.StringIO()):
        sae = load_sae_device_safe(Path(step_dir))
    ov = mp_override(arch, level)
    if ov:
        from consdl.models.mp import apply_mp_override  # noqa: PLC0415
        apply_mp_override(sae, ov)
    sae = sae.to(device)
    if hasattr(sae, "device"):
        sae.device = device
    return sae.eval()


def encode(sae, z):
    """Signed codes ``(N, d_sae)`` of ``z`` ``(N, d_in)``; no grad."""
    import torch  # noqa: PLC0415

    with torch.no_grad():
        a = sae.encode(z.to(next(sae.parameters()).device, dtype=next(sae.parameters()).dtype))
        a = a[0] if isinstance(a, tuple) else a
        return a.reshape(-1, a.shape[-1])
