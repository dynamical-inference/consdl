"""Recover the ``(backbone, layer)`` of a real-data sweep from its path."""

from __future__ import annotations

import re
from pathlib import Path

from consdl.data.vision_backbones import BACKBONE_MAP


def resolve_backbone_layer(sweep_dir: Path) -> tuple[str, int] | None:
    """Recover ``(backbone_short, layer)`` from a sweep-dir path.

    Accepts both a separate ``<backbone>/.../layer<L>`` layout and a
    combined ``<backbone>_layer<L>`` segment.

    Returns ``None`` when the path carries no recognisable backbone or
    layer segment (e.g. a synthetic sweep).
    """
    backbone: str | None = None
    layer: int | None = None
    for seg in Path(sweep_dir).resolve().parts:
        if seg in BACKBONE_MAP:
            backbone = seg
            continue
        # Combined ``<backbone>_layer<N>`` segment.
        matched_combo = False
        for bk in sorted(BACKBONE_MAP, key=len, reverse=True):
            pref = f"{bk}_layer"
            if seg.startswith(pref) and seg[len(pref):].isdigit():
                backbone = bk
                layer = int(seg[len(pref):])
                matched_combo = True
                break
        if matched_combo:
            continue
        # Bare ``layer<N>`` segment.
        m = re.fullmatch(r"layer(\d+)", seg)
        if m is not None:
            layer = int(m.group(1))
    if backbone is None or layer is None:
        return None
    return backbone, layer
