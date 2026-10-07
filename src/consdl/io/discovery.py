"""Enumerate checkpoint directories, seeds, and steps under a sweep root."""

from __future__ import annotations

import os
from pathlib import Path


def _walk_dirs(root: Path):
    """Every directory under ``root``, following symlinks.

    ``Path.rglob`` does not descend into symlinked directories, so seeds
    that are links would be silently skipped. Each directory is visited
    once by real path (links can form cycles), but the walk yields the
    path it arrived by, so callers write beside the link, not the target.
    """
    seen: set[Path] = set()
    for d, dirs, _files in os.walk(root, followlinks=True):
        p = Path(d)
        try:
            real = p.resolve()
        except OSError:                                       # broken link
            dirs[:] = []
            continue
        if real in seen:
            dirs[:] = []                                      # already walked
            continue
        seen.add(real)
        yield p


def discover_checkpoints(root_dir: Path, steps: list[str] | None = None) -> list[Path]:
    """Find checkpoint step directories under *root_dir*.

    *steps* can be ``["final"]``, a list of specific step names, or
    ``["all"]`` (every numeric dir + ``final``). Defaults to ``["final"]``.
    A valid step dir has a sibling ``backbone/`` directory and at least
    one ``.json`` file.
    """
    if steps is None:
        steps = ["final"]

    def _is_step_dir(d: Path) -> bool:
        return d.is_dir() and (d.parent / "backbone").is_dir() and any(d.glob("*.json"))

    want = None if steps == ["all"] else set(steps)
    out: list[Path] = []
    for d in _walk_dirs(Path(root_dir)):
        if want is None:
            if (d.name == "final" or d.name.isdigit()) and _is_step_dir(d):
                out.append(d)
        elif d.name in want and _is_step_dir(d):
            out.append(d)
    return sorted(set(out))


def discover_seeds(sweep_dir: Path) -> list[str]:
    """List ``seedN`` subdirectories under a sweep directory."""
    return sorted(p.name for p in Path(sweep_dir).iterdir()
                  if p.is_dir() and p.name.startswith("seed"))


def discover_steps(seed_dir: Path, include_final: bool = True) -> list[str]:
    """Numeric training steps available for a single seed, sorted ascending.

    ``"final"`` (if present) is appended last when *include_final* is True.
    """
    numeric = sorted(
        (p.name for p in Path(seed_dir).iterdir() if p.is_dir() and p.name.isdigit()),
        key=int,
    )
    if include_final and (Path(seed_dir) / "final").is_dir():
        return [*numeric, "final"]
    return numeric
