"""Cached loaders for SAE decoders and per-seed stats tensors, and
:func:`load_sae`, which picks the SAE class from a checkpoint's
``config.json``.
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import pathlib
from pathlib import Path

import numpy as np
import torch

from consdl.io.paths import DEFAULT_DATASET, DEFAULT_SPLIT, stats_dir

_decoder_cache: dict[str, np.ndarray] = {}
_mean_act_cache: dict[str, np.ndarray] = {}
_mean_act_per_token_cache: dict[str, np.ndarray] = {}


def _load_state(step_dir: Path) -> dict | None:
    """A checkpoint's state dict from ``state_dict.pt`` or
    ``model.safetensors``, or ``None`` when neither is readable."""
    pt = Path(step_dir) / "state_dict.pt"
    st = Path(step_dir) / "model.safetensors"
    try:
        if pt.exists():
            return torch.load(pt, map_location="cpu")
        if st.exists():
            from safetensors.torch import load_file  # noqa: PLC0415
            return load_file(str(st), device="cpu")
    except (RuntimeError, EOFError, OSError) as e:
        # Truncated or corrupt checkpoint: warn and let the caller treat
        # it as missing instead of failing the whole pass.
        print(f"\n[checkpoint] CORRUPT checkpoint, skipping: {step_dir}\n"
              f"  ({type(e).__name__}: {e})", file=sys.stderr)
    return None


def load_raw_decoder(step_dir: Path) -> np.ndarray | None:
    """Raw (non-normalised) ``W_dec`` ``(d_sae, d_in)`` as numpy."""
    state = _load_state(step_dir)
    if state is None:
        return None
    for k in ("W_dec", "decoder.weight"):
        if k in state:
            return state[k].float().numpy()
    for k, v in state.items():
        if k.endswith("W_dec"):
            return v.float().numpy()
    return None


def load_decoder(step_dir: Path) -> np.ndarray | None:
    """Return L2-normalised decoder ``(d_sae, d_in)`` or ``None``. Cached by path."""
    key = str(step_dir)
    if key in _decoder_cache:
        return _decoder_cache[key]
    if not (Path(step_dir) / "config.json").exists():
        return None
    D = load_raw_decoder(step_dir)
    if D is None:
        return None
    norms = np.linalg.norm(D, axis=1, keepdims=True)
    D = (D / np.clip(norms, 1e-12, None)).astype(np.float32)
    _decoder_cache[key] = D
    return D


def load_stat(step_dir: Path, name: str, dataset: str = DEFAULT_DATASET,
              split: str = DEFAULT_SPLIT) -> np.ndarray | None:
    """``<stats_dir>/<name>.pt`` as numpy, or None."""
    p = stats_dir(step_dir, dataset, split) / f"{name}.pt"
    if not p.exists():
        return None
    try:
        return torch.load(p, map_location="cpu").numpy()
    except Exception:
        return None


def load_mean_act(step_dir: Path, dataset: str = DEFAULT_DATASET,
                  split: str = DEFAULT_SPLIT) -> np.ndarray | None:
    """Mean activation ``(d_sae,)``; cached by path.

    fastsae's ``mean_act.pt`` (per image, conditional on firing). Use
    ``load_mean_act_per_token`` for the per-token ``E[|a|]`` over all
    positions.
    """
    key = str(step_dir)
    if key in _mean_act_cache:
        return _mean_act_cache[key]
    val = load_stat(step_dir, "mean_act", dataset, split)
    if val is not None:
        _mean_act_cache[key] = val
    return val


def load_mean_act_per_token(step_dir: Path, dataset: str = DEFAULT_DATASET,
                            split: str = DEFAULT_SPLIT) -> np.ndarray | None:
    """Per-token mean activation ``(d_sae,)``; cached by path.

    ``E_token[|a_k|]`` over all positions (zeros included), written by
    :class:`consdl.stats.interpreter.SAEInterpreter`. Falls back to
    ``mean_act.pt`` when ``mean_act_per_token.pt`` is missing.
    """
    key = str(step_dir)
    if key in _mean_act_per_token_cache:
        return _mean_act_per_token_cache[key]
    val = load_stat(step_dir, "mean_act_per_token", dataset, split)
    if val is None:
        val = load_mean_act(step_dir, dataset, split)
    if val is not None:
        _mean_act_per_token_cache[key] = val
    return val


def load_top_idx(step_dir: Path, dataset: str = DEFAULT_DATASET,
                 split: str = DEFAULT_SPLIT) -> np.ndarray | None:
    """Top activating sample indices ``(d_sae, K)``."""
    return load_stat(step_dir, "top_idx", dataset, split)


@contextlib.contextmanager
def _lock_free_if_readonly(path: pathlib.Path):
    """Drop ``config_dataclass``'s load lock on a directory we cannot write.

    ``config_dataclass`` takes a ``FileLock`` on ``<ckpt>/lock.lock`` even
    when loading, which fails with ``PermissionError`` on a read-only
    checkpoint directory. The lock only serialises a concurrent writer,
    which cannot exist for a directory we cannot write, so it is replaced
    by a no-op there. On writable directories the lock is unchanged.
    """
    import os  # noqa: PLC0415

    if os.access(path, os.W_OK):
        yield
        return
    from config_dataclass import core as _cd_core  # noqa: PLC0415

    class _NullLock:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def acquire(self, *_args, **_kwargs) -> None:
            pass

        def release(self, *_args, **_kwargs) -> None:
            pass

    original = _cd_core.FileLock
    _cd_core.FileLock = _NullLock
    try:
        yield
    finally:
        _cd_core.FileLock = original


def load_sae(load_dir: str | pathlib.Path):
    """Load an SAE checkpoint (``state_dict.pt`` or ``model.safetensors``).

    Tolerates read-only trees. See :func:`_load_sae_dispatch` for the
    per-architecture resolution order.
    """
    path = pathlib.Path(load_dir)
    with _lock_free_if_readonly(path):
        return _load_sae_dispatch(path)


def _load_sae_dispatch(load_dir: str | pathlib.Path):
    """Load an SAE of any architecture in this package.

    The first ``class_type`` / ``architecture`` in ``config.json`` that
    matches, in this order, wins; otherwise ``fastsae.models.SAE.load``
    is used (ReLU):

      1. FISTA-SAE  (``FistaSAE``,              arch ``"fista"`` — ConSDL
                     and its ablations)
      2. MP         (``MPSAE``,                 arch ``"mp"``)
      3. iSAE       (``IsaeMESAE`` /            arch ``"isae_me"`` /
                     ``IsaeSAE``,               ``"isae"``)
      4. Matryoshka / BatchTopK / TopK
      5. fastsae    (ReLU)

    Subclasses are checked before their bases (``IsaeSAE`` and
    ``BatchTopKSAE`` subclass ``TopKSAE``): the state dicts are
    compatible, so a base-class match would load silently with the wrong
    activation.

    Per-class imports are local to keep this module cheap to import.
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

    # The paper's architectures. ReLU (relu) falls through to the
    # fastsae loader below.
    if class_type == "FistaSAE" or architecture == "fista":
        # ConSDL and its ablations (sparse coding, +refitting,
        # +co-active similarity) are all FistaSAE checkpoints.
        from consdl.models.fista import FistaSAE
        return FistaSAE.load(path)
    if class_type == "MPSAE" or architecture == "mp":
        from consdl.models.mp import MPSAE
        return MPSAE.load(path)
    if class_type == "IsaeMESAE" or architecture == "isae_me":
        from consdl.models.isae import IsaeMESAE
        return IsaeMESAE.load(path)
    if class_type == "IsaeSAE" or architecture == "isae":
        from consdl.models.isae import IsaeSAE
        return IsaeSAE.load(path)
    # Subclasses of ``TopKSAE`` before ``TopKSAE`` itself (see docstring).
    if class_type == "MatryoshkaBatchTopKSAE" or architecture == "matryoshka":
        from consdl.models.batchtopk import MatryoshkaBatchTopKSAE
        return MatryoshkaBatchTopKSAE.load(path)
    if class_type == "BatchTopKSAE" or architecture == "batchtopk":
        from consdl.models.batchtopk import BatchTopKSAE
        return BatchTopKSAE.load(path)
    if class_type == "TopKSAE" or architecture == "topk":
        from consdl.models.topk import TopKSAE
        return TopKSAE.load(path)
    import fastsae.models
    # Importing the package installs the load hook (exact d_sae resize,
    # ``model.safetensors``) that every branch above gets via its class.
    import consdl.models  # noqa: F401, PLC0415
    return fastsae.models.SAE.load(str(path))


def quiet_load_sae(step_dir: Path):
    """Load an SAE with stdout suppressed (hides 'Seed has been set').

    Wraps :func:`load_sae`.
    """
    with contextlib.redirect_stdout(io.StringIO()):
        return load_sae(step_dir)


def load_sae_device_safe(final_dir: Path):
    """Load an SAE, tolerating a checkpoint saved with ``device: "cuda"``
    on a CPU-only session.

    ``config.json`` records the training device, which fails to
    construct on a machine without that device. The field is patched in
    a staging directory that symlinks the real weights; the checkpoint
    itself is never modified (it may be read-only or still being
    written).
    """
    final_dir = Path(final_dir)
    cfg_path = final_dir / "config.json"
    cfg = json.loads(cfg_path.read_text())
    want = "cuda" if torch.cuda.is_available() else "cpu"
    if cfg.get("device") in (None, want):
        return quiet_load_sae(final_dir)

    cfg["device"] = want
    # Staged once per (checkpoint, device).
    key = f"{final_dir.resolve()}|{want}"
    stage = _DEVICE_STAGE.get(key)
    if stage is None or not (stage / "config.json").exists():
        stage = Path(tempfile.mkdtemp(prefix="sae_cfg_"))
        for item in final_dir.iterdir():
            if item.name == "config.json":
                continue
            # Skip lock files: they belong to the checkpoint's writer, and
            # a lock that is itself a symlink breaks the staged load.
            if item.name.endswith(".lock"):
                continue
            try:
                # ``resolve()`` so a symlinked source is linked to its
                # TARGET rather than to another link.
                (stage / item.name).symlink_to(item.resolve())
            except OSError:
                pass
        (stage / "config.json").write_text(json.dumps(cfg, indent=4))
        _DEVICE_STAGE[key] = stage
    return quiet_load_sae(stage)


#: ``"<resolved checkpoint>|<device>" -> staging dir``. Process-lifetime;
#: the directories are small (one JSON plus symlinks) and go with /tmp.
_DEVICE_STAGE: dict[str, Path] = {}
