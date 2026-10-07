"""Filesystem conventions: where checkpoints, results and stats live.

    <repo>/var/checkpoints/<rel>/seed<S>/final/      weights (training writes)
    <repo>/var/results/<rel>/...                     everything derived from them

A seed's firing statistics live in ``stats.<dataset>.split_<split>.seed_no``,
either beside the weights or under the mirrored ``var/results`` path
(read first). Released checkpoints keep weights and stats directly in
``seed<S>/``.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

#: The repository checkout (installed with ``pip install -e .``): ``var/``
#: (checkpoints, results, caches), ``data/`` and ``.env`` live here.
REPO_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_DATASET = "imagenet"
DEFAULT_SPLIT = "val"

#: A seed's weights: ``state_dict.pt`` as training writes it, or
#: ``model.safetensors`` as the released checkpoints ship.
WEIGHT_FILES = ("state_dict.pt", "model.safetensors")


def stats_leaf(dataset: str = DEFAULT_DATASET, split: str = DEFAULT_SPLIT) -> str:
    """``stats.<dataset>.split_<split>.seed_no`` -- the stats directory name."""
    return f"stats.{dataset}.split_{split}.seed_no"


def auto_repo_root(path: Path) -> Path:
    """The first ancestor of ``path`` holding ``var/checkpoints``, else the cwd.

    The literal path is walked before the resolved one so a symlinked
    checkpoint dir still maps to THIS repo.
    """
    lit = Path(os.path.abspath(path))
    for parent in (*lit.parents, *lit.resolve().parents):
        if (parent / "var" / "checkpoints").is_dir():
            return parent
    return Path.cwd()


def _checkpoint_rel(path: Path, repo_root: Path) -> Path | None:
    """``path`` relative to ``<repo_root>/var/checkpoints``, or ``None``."""
    lit = Path(os.path.abspath(path))
    base = Path(os.path.abspath(repo_root)) / "var" / "checkpoints"
    for cand in (lit, lit.resolve()):
        for b in (base, base.resolve()):
            try:
                return cand.relative_to(b)
            except ValueError:
                continue
    return None


def to_results_path(checkpoint_path: Path, repo_root: Path) -> Path:
    """Mirror ``<repo>/var/checkpoints/<rel>`` → ``<repo>/var/results/<rel>``.

    A path outside ``var/checkpoints`` (e.g. a downloaded release cell)
    maps to ``var/results/external/<its absolute path>``, so two such
    cells never collide.
    """
    repo = Path(os.path.abspath(repo_root))
    rel = _checkpoint_rel(checkpoint_path, repo)
    if rel is not None:
        return repo / "var" / "results" / rel
    lit = Path(os.path.abspath(checkpoint_path))
    return repo / "var" / "results" / "external" / lit.relative_to(lit.anchor)


def _results_mirror(step_dir: Path) -> Path | None:
    repo = auto_repo_root(step_dir)
    rel = _checkpoint_rel(step_dir, repo)
    return None if rel is None else repo / "var" / "results" / rel


def stats_search_paths(step_dir: Path, dataset: str = DEFAULT_DATASET,
                       split: str = DEFAULT_SPLIT) -> list[Path]:
    """Where a seed's stats may live, in read order: ``var/results`` first,
    the checkpoint-adjacent copy last."""
    leaf = stats_leaf(dataset, split)
    mirror = _results_mirror(step_dir)
    out = [] if mirror is None else [mirror / leaf]
    out.append(Path(step_dir) / leaf)
    return out


def stats_dir(step_dir: Path, dataset: str = DEFAULT_DATASET,
              split: str = DEFAULT_SPLIT) -> Path:
    """The seed's stats dir: the first that exists, else the
    checkpoint-adjacent path."""
    cands = stats_search_paths(step_dir, dataset, split)
    return next((p for p in cands if p.is_dir()), cands[-1])


def stats_glob(step_dir: Path, pattern: str = "stats.*") -> list[Path]:
    """Stats dirs matching ``pattern`` under ``var/results`` and beside the
    weights, ``var/results`` first, deduplicated by real path.

    Two DISTINCT matches mean two different measurements (e.g. a
    subsampled pass beside a full-split one); callers that need exactly
    one check the length.
    """
    roots = [r for r in (_results_mirror(step_dir), Path(step_dir)) if r is not None]
    out: list[Path] = []
    seen: set = set()
    for root in roots:
        if not root.is_dir():
            continue
        for p in sorted(root.glob(pattern)):
            if p.is_dir() and p.resolve() not in seen:
                seen.add(p.resolve())
                out.append(p)
    return out


def stats_write_dir(step_dir: Path, dataset: str = DEFAULT_DATASET,
                    split: str = DEFAULT_SPLIT, *, to_results: bool = True) -> Path:
    """Where a stats pass writes: the ``var/results`` mirror, or beside the
    weights when ``to_results`` is False or the checkpoint is outside
    ``var/checkpoints``."""
    leaf = stats_leaf(dataset, split)
    mirror = _results_mirror(step_dir) if to_results else None
    if mirror is None:
        if to_results:
            print(f"[paths] {step_dir} is not under var/checkpoints/ -- writing "
                  f"stats beside the checkpoint.", file=sys.stderr, flush=True)
        return Path(step_dir) / leaf
    return mirror / leaf


# ── Seeds and cells on disk ──────────────────────────────────────────


def has_weights(d: Path) -> bool:
    """Whether ``d`` holds an SAE checkpoint."""
    return any((Path(d) / f).is_file() for f in WEIGHT_FILES)


def seed_step_dirs(tag_dir: Path, step: str = "final") -> list[tuple[str, Path]]:
    """``(seed name, checkpoint dir)`` for every trained seed of a cell.

    Training writes ``seed<N>/<step>/``; the released checkpoints keep
    the weights directly in ``seed<N>/``. Both are accepted.
    """
    out = []
    for seed in sorted(Path(tag_dir).glob("seed*")):
        sd = seed / step if (seed / step).is_dir() else seed
        if has_weights(sd):
            out.append((seed.name, sd))
    return out


def find_tags(tree: Path, step: str = "final", min_seeds: int = 2) -> list[Path]:
    """Cell directories under ``tree`` with at least ``min_seeds`` trained seeds.

    ``os.walk(followlinks=True)``: ``glob``'s ``**`` does not descend into
    a symlinked directory.
    """
    cells: set[Path] = set()
    for d, _dirs, _files in os.walk(tree, followlinks=True):
        p = Path(d)
        if p.name.startswith("seed") and (has_weights(p / step) or has_weights(p)):
            cells.add(p.parent)
    return [t for t in sorted(cells) if len(seed_step_dirs(t, step)) >= min_seeds]
