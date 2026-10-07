"""Mid-run resume for SAE training.

fastsae's periodic checkpoints contain only the SAE weights. This module
saves and restores the rest of the training state so an interrupted run
can continue. :class:`consdl.training.trainer.SAETrainer` calls into it,
so every trainer built on it supports resume.

Saved next to the SAE weights in ``<ckpt>/<step>/``:

``trainer_state.pt``
    optimizer + scheduler ``state_dict``; ``_n_training_samples`` /
    ``_n_training_steps`` / ``_n_frac_active_tokens``; the dead-latent
    trackers ``_n_forward_passes_since_fired`` / ``_a_freq_scores``; the
    remaining ``_checkpoint_thresholds``; and torch/numpy/python RNG
    states.

**Data position.** The counters are restored, so training stops at the
right total, but the dataloader restarts at the top of its stream. For a
shuffled map-style loader this is harmless; for a streaming loader the
first samples are seen again. ``resume_state()`` returns the sample count
so a caller can skip ahead if needed.

Without a ``trainer_state.pt``, :func:`try_resume` is a no-op.
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

STATE_FILENAME = "trainer_state.pt"


def _checkpoint_dirs(ckpt_root: Path) -> list[Path]:
    """Saved step dirs (``<ckpt>/<n_samples>/``) newest-first.

    ``final/`` is excluded: fastsae writes it from a ``finally`` block even
    when a run is killed, so it is not a reliable resume point.
    """
    if not ckpt_root.is_dir():
        return []
    numeric = [
        p for p in ckpt_root.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / STATE_FILENAME).is_file()
    ]
    return sorted(numeric, key=lambda p: int(p.name), reverse=True)


def save_trainer_state(trainer: Any, save_dir: str | Path) -> None:
    """Persist everything needed to continue ``trainer`` from this point.

    Called right after the SAE weights are written for the same step, so
    the pair in ``save_dir`` is self-consistent.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "version": 1,
        "n_training_samples": int(trainer._n_training_samples),
        "n_training_steps": int(trainer._n_training_steps),
        "n_frac_active_tokens": int(trainer._n_frac_active_tokens),
        "checkpoint_thresholds": list(trainer._checkpoint_thresholds),
        "n_forward_passes_since_fired": trainer._n_forward_passes_since_fired.detach().cpu(),
        "a_freq_scores": trainer._a_freq_scores.detach().cpu(),
        "rng_torch": torch.get_rng_state(),
        "rng_numpy": np.random.get_state(),
        "rng_python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    opt = getattr(trainer, "optimizer", None)
    if opt is not None:
        state["optimizer"] = opt.state_dict()
    sched = getattr(trainer, "scheduler", None)
    if sched is not None and hasattr(sched, "state_dict"):
        state["scheduler"] = sched.state_dict()
    # Write via temp file + rename so an interrupted write cannot leave a
    # truncated state.
    tmp = save_dir / (STATE_FILENAME + ".tmp")
    torch.save(state, tmp)
    tmp.replace(save_dir / STATE_FILENAME)


def resume_state(ckpt_root: str | Path) -> tuple[Path, dict] | None:
    """Newest resumable ``(step_dir, state)`` under ``ckpt_root``, or None.

    Unreadable states are skipped, falling back to an older checkpoint.
    """
    for step_dir in _checkpoint_dirs(Path(ckpt_root)):
        try:
            return step_dir, torch.load(step_dir / STATE_FILENAME, map_location="cpu")
        except Exception as e:  # noqa: BLE001 - any unpickling failure
            print(f"[resume] ignoring unreadable {step_dir / STATE_FILENAME}: "
                  f"{type(e).__name__}: {e}")
    return None


def try_resume(trainer: Any, ckpt_root: str | Path | None,
               *, verbose: bool = True) -> int:
    """Restore ``trainer`` in place from the newest checkpoint.

    Returns the number of samples already trained (0 when there is nothing
    to resume from, i.e. a fresh run). The SAE weights are **not** loaded
    here — the caller builds the SAE from the same step dir before the
    trainer is constructed, because fastsae wires the optimizer to the
    parameter objects at ``__init__``.
    """
    if ckpt_root is None:
        return 0
    found = resume_state(ckpt_root)
    if found is None:
        return 0
    step_dir, state = found

    trainer._n_training_samples = int(state["n_training_samples"])
    trainer._n_training_steps = int(state["n_training_steps"])
    trainer._n_frac_active_tokens = int(state["n_frac_active_tokens"])
    trainer._checkpoint_thresholds = list(state["checkpoint_thresholds"])
    dev = trainer.device
    trainer._n_forward_passes_since_fired = (
        state["n_forward_passes_since_fired"].to(dev)
    )
    trainer._a_freq_scores = state["a_freq_scores"].to(dev)

    opt = getattr(trainer, "optimizer", None)
    if opt is not None and "optimizer" in state:
        opt.load_state_dict(state["optimizer"])
    sched = getattr(trainer, "scheduler", None)
    if sched is not None and "scheduler" in state:
        sched.load_state_dict(state["scheduler"])

    try:
        torch.set_rng_state(state["rng_torch"])
        np.random.set_state(state["rng_numpy"])
        random.setstate(state["rng_python"])
        if torch.cuda.is_available() and "rng_cuda" in state:
            torch.cuda.set_rng_state_all(state["rng_cuda"])
    except Exception as e:  # noqa: BLE001 - RNG shape can differ across boxes
        print(f"[resume] RNG state not restored ({type(e).__name__}: {e}); "
              f"continuing with the current streams")

    if verbose:
        print(f"[resume] continuing from {step_dir}: "
              f"{trainer._n_training_samples:,} / "
              f"{trainer.total_n_training_samples:,} samples "
              f"({trainer._n_training_steps:,} steps), "
              f"{len(trainer._checkpoint_thresholds)} checkpoints left")
    return int(trainer._n_training_samples)


def try_resume_full(trainer: Any, *, verbose: bool = True) -> int:
    """Resume ``trainer`` end-to-end: SAE weights **and** trainer state.

    Called from ``SAETrainer.train()``. Returns samples already trained
    (0 = fresh run).

    Weights are loaded with ``load_state_dict``, which copies in place, so
    the optimizer built against these parameter objects stays valid.

    Disable with ``SAE_RESUME=0`` to force a clean start.
    """
    import os

    if os.environ.get("SAE_RESUME", "1") == "0":
        return 0
    ckpt_root = getattr(trainer, "checkpoint_path", None)
    if ckpt_root is None:
        return 0
    step_dir = resume_sae_dir(ckpt_root)
    if step_dir is None:
        return 0

    sae = trainer.wrapped.sae
    sd_path = step_dir / "state_dict.pt"
    if sd_path.is_file():
        try:
            sd = torch.load(sd_path, map_location=trainer.device)
            missing, unexpected = sae.load_state_dict(sd, strict=False)
            if verbose and (missing or unexpected):
                print(f"[resume] state_dict: missing={list(missing)} "
                      f"unexpected={list(unexpected)}")
        except Exception as e:  # noqa: BLE001
            print(f"[resume] could NOT load {sd_path} "
                  f"({type(e).__name__}: {e}) — starting fresh")
            return 0
    else:
        print(f"[resume] no state_dict.pt in {step_dir} — starting fresh")
        return 0

    return try_resume(trainer, ckpt_root, verbose=verbose)


def resume_sae_dir(ckpt_root: str | Path | None) -> Path | None:
    """Step dir whose SAE weights pair with the newest resumable state.

    Callers load the SAE from here *before* constructing the trainer.
    """
    if ckpt_root is None:
        return None
    found = resume_state(ckpt_root)
    return None if found is None else found[0]


__all__ = [
    "STATE_FILENAME",
    "resume_sae_dir",
    "resume_state",
    "save_trainer_state",
    "try_resume",
]
