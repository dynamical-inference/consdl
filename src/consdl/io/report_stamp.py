"""The ``results/.reports_dirty`` stamp.

Every metrics-json writer touches it, marking that results have changed
since tables derived from them were last rebuilt.
"""
from __future__ import annotations

import time
from pathlib import Path


def mark_reports_dirty(out_path: Path) -> None:
    """Record that a metrics json landed under ``results/``.

    Never raises: a read-only or full filesystem must not fail a run
    that has already written its result.
    """
    try:
        results_root = next(
            (p for p in Path(out_path).parents if p.name == "results"), None)
        if results_root is None:
            return
        (results_root / ".reports_dirty").write_text(
            f"{time.time():.0f}\n{out_path}\n")
    except OSError:
        pass
