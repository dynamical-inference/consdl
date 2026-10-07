"""Single-model quality scalars (reconstruction and sparsity) and
liveness.

Callers pass already-summed SSE/SST plus the activation matrix and get the
same per-seed dict for real and synthetic data. Liveness for #dead is read
from the full-split firing statistics when available (see
:func:`firing_rate_from_stats`).
"""

from __future__ import annotations

import numpy as np


DEAD_THRESHOLD = 1e-6


def alive_mask(acts: np.ndarray, dead_threshold: float = DEAD_THRESHOLD) -> np.ndarray:
    """Per-latent liveness, the exact complement of ``dead_rate``.

    Uses ``|a| > 0`` as :func:`quality_scalars` does, so sign-preserving
    codes whose latent fires only negative count as alive. Consumers that
    exclude dead latents use this so they agree with ``dead_rate``.
    """
    firing = (np.abs(np.asarray(acts)) > 0).astype(np.float32).mean(axis=0)
    return firing > float(dead_threshold)


def quality_scalars(
    *,
    sse: float,
    sst: float,
    n_elems: int,
    acts: np.ndarray,
    live_threshold: float = 0.1,
    dead_threshold: float = DEAD_THRESHOLD,
    dense_threshold: float = 0.2,
) -> dict:
    """Per-seed reconstruction + sparsity scalars.

    ``sse`` / ``sst`` are the summed squared error / total sum of squares of
    the reconstruction (``recon_r2 = 1 - sse/sst``); ``n_elems`` is the total
    element count (``n_tokens * d_in``) for the per-element L2 reading.
    ``acts`` is the ``(n_tokens, d_sae)`` activation matrix. A latent counts
    as *live* iff ``|a|`` crosses ``live_threshold`` at least once across the
    pool.

    * ``dead_rate``      = fraction of latents whose firing rate
                           ``(|acts| > 0).mean(axis=0)`` is
                           ``<= dead_threshold``.
    * ``dense_rate_0.2`` = fraction of latents whose firing rate is
                           ``>= dense_threshold``.
    """
    abs_acts = np.abs(acts)
    # Support is |a| > 0, not a > 0: identical for non-negative codes, but
    # AbsTopK (``isae``) keeps the sign of the top-k by magnitude, so about
    # half its active latents are negative.
    firing = (abs_acts > 0).astype(np.float32).mean(axis=0)  # (d_sae,)
    out = {
        "recon_r2": float(1.0 - sse / max(sst, 1e-30)),
        "recon_l2": float(sse / max(n_elems, 1)),
        "avg_active_per_token": float((abs_acts > 0).sum(axis=1).mean()),
        "l1_per_token": float(abs_acts.sum(axis=1).mean()),
        "mean_abs_act": float(abs_acts.mean()),
        "live_latent_frac": float((abs_acts.max(axis=0) > live_threshold).mean()),
        "live_threshold": float(live_threshold),
        "dead_rate": float((firing <= dead_threshold).mean()),
        "dense_rate_0.2": float((firing >= dense_threshold).mean()),
    }
    return out


def firing_rate_from_stats(step_dir, dataset: str, split: str):
    """Per-latent firing rate off ``compute_stats``, or ``None``.

    Reads ``<stats dir>/firing_rate_per_latent.pt``, a ``(d_sae,)`` vector
    computed by the stats pass over the whole split.

    Liveness comes from these full-split statistics rather than the small
    activation pool the metrics encode: on a pool of ~20k tokens the
    smallest non-zero firing rate is ~5e-5, so ``DEAD_THRESHOLD`` never
    binds and "dead" degenerates to "did not fire in the sample". That
    over-counts dead latents for architectures with a heavy tail of
    rarely-firing latents, which biases comparisons between them.

    Returns ``None`` when the file is absent, so callers fall back to
    the pool.
    """
    from pathlib import Path  # noqa: PLC0415

    from consdl.io.paths import stats_dir  # noqa: PLC0415

    #: The same vector under two file names; the synthetic stats pass
    #: writes ``sparsity_per_token.pt``.
    NAMES = ("firing_rate_per_latent.pt", "sparsity_per_token.pt")

    def _first(d):
        for n in NAMES:
            q = d / n
            if q.is_file():
                return q
        return None

    p = _first(stats_dir(Path(step_dir), dataset, split))
    if p is None:
        # Fall back to whichever stats dir exists: the (dataset, split)
        # naming differs between vision and LM.
        from consdl.io.paths import stats_glob  # noqa: PLC0415
        cands = stats_glob(Path(step_dir), "stats.*.seed_no")
        # Different names are different measurements (e.g. a subsample
        # beside a full-split pass), so refuse to guess. Copies of one
        # name under several roots are the same measurement; take the
        # first, as ``stats_glob`` orders roots by precedence.
        if len({c.name for c in cands}) != 1:
            return None
        p = next((q for c in cands if (q := _first(c)) is not None), None)
        if p is None:
            return None
    try:
        import numpy as _np  # noqa: PLC0415
        import torch  # noqa: PLC0415

        v = torch.load(p, map_location="cpu", weights_only=True)
        return _np.asarray(v, dtype=_np.float64).reshape(-1)
    except Exception as e:  # noqa: BLE001
        print(f"    [stats] unreadable {p}: {e}", flush=True)
        return None


def alive_mask_from_stats(step_dir, dataset: str, split: str,
                          dead_threshold: float = DEAD_THRESHOLD):
    """``firing_rate_from_stats`` thresholded, or ``None``.

    Same ``firing > dead_threshold`` rule as :func:`alive_mask`, so the
    two sources stay comparable and only the EVIDENCE differs.
    """
    fr = firing_rate_from_stats(step_dir, dataset, split)
    return None if fr is None else (fr > float(dead_threshold))


def stats_metrics(step_dir, dataset: str, split: str,
                  d_sae: int | None = None) -> dict:
    """Scalars from the seed's ``compute_stats`` dir, or ``{}``.

    Measured over the whole split, unlike the metrics' own activation
    pool, which matters for ``dead_rate`` and, to a lesser degree, L0.

    Reads ``metrics.json`` (vision, synthetic: L0 as a fraction of
    ``d_sae``) and ``l0_per_token.json`` (LM: L0 as a count), merged and
    normalised by :func:`consdl.stats.scalars.canonical_scalars`.

    MP is handled by the caller: its sparsity is set by the inference
    round count, not the weights.

    ``(dataset, split)`` is only the first guess; a seed with a single
    ``stats.*.seed_no`` name resolves through that instead.
    """
    import json  # noqa: PLC0415
    from pathlib import Path  # noqa: PLC0415

    from consdl.io.paths import stats_dir, stats_glob  # noqa: PLC0415

    d = stats_dir(Path(step_dir), dataset, split)
    if not d.is_dir():
        cands = stats_glob(Path(step_dir), "stats.*.seed_no")
        # Refuse only when candidate names differ (different
        # measurements, e.g. a subsample beside a full-split pass). Copies
        # of one name under several roots are the same measurement;
        # ``stats_glob`` orders them by precedence, so take the first.
        names = {p.name for p in cands}
        if not cands or len(names) != 1:
            return {}
        d = cands[0]

    out: dict = {}
    for fn in ("metrics.json", "l0_per_token.json"):
        p = d / fn
        if not p.is_file():
            continue
        try:
            blob = json.loads(p.read_text())
        except (OSError, ValueError) as e:
            print(f"    [stats] unreadable {p}: {e}", flush=True)
            continue
        if isinstance(blob, dict):
            out.update(blob)
            out.setdefault("_files", []).append(fn)
    if not out:
        return {}
    from consdl.stats.scalars import canonical_scalars  # noqa: PLC0415

    return canonical_scalars(out, d_sae)
