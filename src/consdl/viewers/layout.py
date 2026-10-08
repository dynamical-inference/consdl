"""Checkpoint trees, cell names and output paths shared by the viewers.

Input is a release-style tree (``python -m consdl.cli.sweep`` or
``consdl.io.hub.download``)::

    <root>/<modality>/<backbone>_layer<L>/<arch>/x<E>/<level>/seed<S>/

A *cell* is one ``(expansion, level)`` of one arch; its id is
``x<E>_<level>`` (``x16_l0_128``) and a panel key is ``<arch>/<cell>/<seed>``.
Everything the viewers read is written under
``var/results/viewers/<modality>/<backbone>_layer<L>/``.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from consdl.io.paths import REPO_ROOT, seed_step_dirs

#: Where the precompute writes and the servers read.
DEFAULT_OUT_ROOT = REPO_ROOT / "var" / "results" / "viewers"

ARCH_ORDER = ("relu", "topk", "batchtopk", "matryoshka", "mp", "isae", "consdl")
ARCH_LABEL = {"relu": "ReLU", "topk": "TopK", "batchtopk": "BatchTopK",
              "matryoshka": "Matryoshka", "mp": "MP", "isae": "iSAE",
              "consdl": "ConSDL", "sc": "SC", "refit": "SC+refit"}
#: Archs whose atom sign is unidentified: every comparison uses |cos|.
ABS_ARCHES = frozenset({"isae"})
#: The arch whose seed-0 clusters colour the arch views.
REFERENCE_ARCH = "consdl"

BACKBONE_LABEL = {"clip-vit-b-16": "CLIP ViT-B/16", "clip-vit-l-14": "CLIP ViT-L/14",
                  "dinov2-b": "DINOv2-B/14", "dinov2-l": "DINOv2-L/14",
                  "siglip-b": "SigLIP-B/16", "siglip-l": "SigLIP-L/16",
                  "pythia-70m": "Pythia-70M", "pythia-160m": "Pythia-160M",
                  "pythia-160m-nondedup": "Pythia-160M (non-dedup)",
                  "pythia-410m": "Pythia-410M", "gpt2-small": "GPT-2 Small",
                  "gemma-2-2b": "Gemma-2-2B"}


def arch_rank(arch: str) -> int:
    return ARCH_ORDER.index(arch) if arch in ARCH_ORDER else len(ARCH_ORDER)


def level_l0(level: str) -> int | None:
    """Target L0 of a level name (``l0_128`` -> 128), else ``None``."""
    m = re.fullmatch(r"l0_(\d+)", level)
    return int(m.group(1)) if m else None


def cell_id(expansion: int, level: str) -> str:
    return f"x{int(expansion)}_{level}"


def split_cell(cell: str) -> tuple[int, str]:
    head, _, level = cell.partition("_")
    return int(head[1:]), level


def cell_label(cell: str) -> str:
    """``x16_l0_128`` -> ``x16 · L0≈128``."""
    e, level = split_cell(cell)
    l0 = level_l0(level)
    return f"x{e} · " + (f"L0≈{l0}" if l0 is not None else level)


def cell_sort_key(cell: str) -> tuple:
    e, level = split_cell(cell)
    l0 = level_l0(level)
    return (e, l0 if l0 is not None else 1 << 30, level)


# ── backbones ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class BackboneSpec:
    """One ``<backbone>_layer<L>`` directory: which model and hook it reads."""
    name: str                # dir name, e.g. clip-vit-b-16_layer10
    short: str               # registry key, e.g. clip-vit-b-16
    layer: int
    modality: str            # "vision" | "lm"
    hf_name: str
    hook: str

    @property
    def label(self) -> str:
        return f"{BACKBONE_LABEL.get(self.short, self.short)} · layer {self.layer}"


def backbone_spec(name: str) -> BackboneSpec | None:
    """Spec of a ``<backbone>_layer<L>`` name, or ``None`` if unknown."""
    from consdl.data.lm_backbones import LM_BACKBONE_MAP  # noqa: PLC0415

    short, _, layer = name.rpartition("_layer")
    if not layer.isdigit():
        return None
    L = int(layer)
    if short in LM_BACKBONE_MAP:
        c = LM_BACKBONE_MAP[short]
        return BackboneSpec(name, short, L, "lm", c["hf_name"], c["hook_layer_fmt"].format(L))
    from consdl.data.vision_backbones import BACKBONE_MAP  # noqa: PLC0415
    if short in BACKBONE_MAP:
        c = BACKBONE_MAP[short]
        return BackboneSpec(name, short, L, "vision", c["hf_name"], c["hook_layer_fmt"].format(L))
    return None


def find_backbone_dirs(root: Path, names: list[str] | None = None) -> list[Path]:
    """``<backbone>_layer<L>`` dirs at ``root``, one or two levels below it.

    ``names`` filters by full dir name or by backbone short name
    (``clip-vit-b-16`` matches ``clip-vit-b-16_layer10``).
    """
    root = Path(root)
    cands = [root, *sorted(root.glob("*")), *sorted(root.glob("*/*"))]
    out = [p for p in cands if p.is_dir() and backbone_spec(p.name) is not None]
    if names:
        rank = {n: i for i, n in enumerate(names)}
        out = [p for p in out if p.name in rank or backbone_spec(p.name).short in rank]
        out.sort(key=lambda p: rank.get(p.name, rank.get(backbone_spec(p.name).short)))
    seen, uniq = set(), []
    for p in out:
        if p.name not in seen:
            seen.add(p.name)
            uniq.append(p)
    return uniq


def out_dir(bb: BackboneSpec, out_root: Path = DEFAULT_OUT_ROOT) -> Path:
    return Path(out_root) / bb.modality / bb.name


# ── cells ─────────────────────────────────────────────────────────────


@dataclass
class Cell:
    arch: str
    expansion: int
    level: str
    seeds: dict[str, Path] = field(default_factory=dict)   # seed name -> weights dir

    @property
    def name(self) -> str:
        return cell_id(self.expansion, self.level)

    def seed_names(self) -> list[str]:
        return sorted(self.seeds, key=lambda s: int(s[4:]) if s[4:].isdigit() else 0)


def discover(bb_dir: Path, *, archs=None, expansions=None, levels=None,
             n_seeds: int | None = None) -> dict[str, list[Cell]]:
    """``{arch: [Cell, ...]}`` under one backbone dir, cells in grid order."""
    out: dict[str, list[Cell]] = {}
    for adir in sorted(Path(bb_dir).iterdir()):
        if not adir.is_dir() or (archs and adir.name not in archs):
            continue
        cells = []
        for xdir in sorted(adir.glob("x*")):
            if not xdir.name[1:].isdigit():
                continue
            e = int(xdir.name[1:])
            if expansions and e not in expansions:
                continue
            for ldir in sorted(xdir.iterdir()):
                if not ldir.is_dir() or (levels and ldir.name not in levels):
                    continue
                seeds = dict(seed_step_dirs(ldir))
                if not seeds:
                    continue
                c = Cell(adir.name, e, ldir.name, seeds)
                names = c.seed_names()[:n_seeds] if n_seeds else c.seed_names()
                c.seeds = {s: c.seeds[s] for s in names}
                cells.append(c)
        if cells:
            out[adir.name] = sorted(cells, key=lambda c: cell_sort_key(c.name))
    return dict(sorted(out.items(), key=lambda kv: arch_rank(kv[0])))
