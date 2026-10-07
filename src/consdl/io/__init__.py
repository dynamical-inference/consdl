"""Checkpoint, stats and path I/O.

Import-light: the names below are re-exported lazily (PEP 562
``__getattr__``), so ``from consdl.io import load_decoder`` imports torch
only when a loader is actually requested, and path helpers stay cheap.
"""

_LAZY = {
    "load_decoder": "consdl.io.checkpoint",
    "load_raw_decoder": "consdl.io.checkpoint",
    "load_mean_act": "consdl.io.checkpoint",
    "load_stat": "consdl.io.checkpoint",
    "load_top_idx": "consdl.io.checkpoint",
    "quiet_load_sae": "consdl.io.checkpoint",
    "stats_dir": "consdl.io.paths",
    "to_results_path": "consdl.io.paths",
    "discover_checkpoints": "consdl.io.discovery",
    "discover_seeds": "consdl.io.discovery",
    "discover_steps": "consdl.io.discovery",
}

__all__ = list(_LAZY)


def __getattr__(name: str):
    """Resolve a re-exported name on first use, not on import."""
    mod = _LAZY.get(name)
    if mod is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}")
    import importlib
    value = getattr(importlib.import_module(mod), name)
    globals()[name] = value          # subsequent lookups skip this path
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
