"""Dictionary metrics, computed from decoder matrices and firing rates only.

Per seed: L0, recon R^2 (from the seed's stats pass), #dead and dense
atoms (from its firing rate over the whole split) and #dup (atoms in
complete-linkage clusters of the live decoder atoms). Per seed pair:
``DCS``, the mean Hungarian-matched decoder cosine. Synthetic cells add
``DCS`` against the ground-truth dictionary.

Inputs:

    <tag>/seed<N>/final/                                   the decoder
    <tag>/seed<N>/final/stats.*/firing_rate_per_latent.pt   liveness

No SAE forward pass is needed, so the same code serves every modality.

Thresholds. Threshold-dependent counts are stored at each of
``THRESHOLDS``; the summary reports the mean over ``REPORT_BAND``
(tau in {0.6, ..., 0.9}). Per-threshold values are kept so another band
can be computed without rerunning.

Sign. Intra-dictionary cosines are evaluated both signed and as
``|cos|``. For a non-negative code, ``w`` and ``-w`` fire on different
inputs, so an anti-parallel pair is not a duplicate; for the AbsTopK
family (``(w, +a)`` == ``(-w, -a)``) it is. Both readings are stored and
the architecture determines which one applies.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from consdl.metrics.clustering import (
    LINKAGE_DEFAULT,
    complete_refine,
    edges,
    labels_at,
)
from consdl.metrics.kernels import decoder_match
from consdl.metrics.kernels import unit_rows as _unit_rows
from consdl.metrics.quality import (
    DEAD_THRESHOLD,
    firing_rate_from_stats,
    stats_metrics,
)
from consdl.io.report_stamp import mark_reports_dirty as _mark_reports_dirty
from consdl.io.checkpoint import load_decoder

#: Output file schema version.
SCHEMA = 3

#: Firing rate at or above which a latent counts as dense (active on one
#: token in ten). Independent of ``quality_scalars``' ``dense_rate_0.2``.
DENSE_THRESHOLD = 0.1

#: Every threshold stored. Lower cosines do not indicate duplication
#: (at low tau whole dictionaries merge into one component) and make
#: clustering expensive. 0.5 is kept because the co-active similarity
#: margin (m=0.5) concentrates atom pairs just below it.
THRESHOLDS: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9)

#: The band the summary reports (the paper's tau in {0.6, ..., 0.9}).
#: ``per_threshold`` values are unaffected by it.
REPORT_BAND: tuple[float, ...] = (0.6, 0.7, 0.8, 0.9)


# ── Duplicate clustering (see ``consdl.metrics.clustering``) ──────────


def _band_mean(per_t: dict, band: tuple[float, ...] = REPORT_BAND):
    """Mean over ``band`` of a ``{threshold_str: value}`` map, or None."""
    vals = [per_t.get(f"{t:g}") for t in band]
    if any(v is None for v in vals):
        return None
    return float(np.mean(vals))


def _dup_stats(D: np.ndarray, *, absolute: bool) -> dict:
    """Per-threshold duplicate counts from one dictionary.

    ``n_dup`` counts ATOMS with at least one sibling, not pairs: a
    K-atom cluster is K(K-1)/2 pairs but only K wasted atoms, and it is
    the atoms that cost dictionary capacity.

    ``largest`` is the biggest cluster, which distinguishes many small
    duplicate families from a collapsed dictionary.
    """
    U = _unit_rows(D)
    G = U @ U.T
    np.fill_diagonal(G, 0.0)
    if absolute:
        G = np.abs(G)
    n = int(G.shape[0])
    ei, ej, ew = edges(G, min(THRESHOLDS))
    del G
    out: dict = {"n_dup": {}, "n_clusters": {}, "largest": {}}
    for t in THRESHOLDS:
        lab, n_comp = labels_at(ei, ej, ew, n, float(t))
        if LINKAGE_DEFAULT == "complete":
            # Under complete linkage an atom can have a neighbour above
            # tau yet be a singleton: with A-B 0.75, A-C 0.90 and B-C 0.50,
            # A and C merge and B cannot join them.
            lab, n_comp = complete_refine(U, lab, n_comp, float(t),
                                          absolute=absolute)
        sizes = np.bincount(lab, minlength=n_comp)
        dup = sizes[sizes > 1]
        k = f"{t:g}"
        out["n_dup"][k] = int(dup.sum())
        out["n_clusters"][k] = int(dup.size)
        out["largest"][k] = int(sizes.max()) if sizes.size else 0
    return out


def _babel_mu1(D: np.ndarray, *, absolute: bool) -> float:
    """Mean nearest-neighbour cosine — how close the dictionary packs.

    Uses the same sign convention as the duplicate counts.
    """
    U = _unit_rows(D)
    G = U @ U.T
    np.fill_diagonal(G, -np.inf)
    if absolute:
        G = np.abs(np.where(np.isfinite(G), G, 0.0))
        np.fill_diagonal(G, -np.inf)
    return float(G.max(axis=1).mean())


def seed_metrics(dec: np.ndarray, alive: "np.ndarray | None",
                 stats: dict | None = None, mp: bool = False,
                 firing: "np.ndarray | None" = None) -> dict:
    """Single-dictionary metrics for one seed.

    ``stats`` is the seed's stats-pass scalars (``stats_metrics``). L0 and
    recon R^2 come from there rather than from an activation pass, as
    liveness does, because the stats pass covers the whole split.
    """
    st = stats or {}
    d_sae = int(dec.shape[0])
    out: dict = {"d_sae": d_sae, "d_in": int(dec.shape[1])}
    # MP's sparsity is set by the inference round count, not the
    # weights, so the stats-pass L0 / R^2 describe one particular round
    # count. They are left None; MP's L0 and R^2 come from the code
    # metrics evaluated at each target round count.
    _l0 = None if mp else st.get("l0_abs")
    out["l0"] = None if _l0 is None else int(round(float(_l0)))
    # Unrounded L0, kept so the cross-seed std is not quantised by the
    # rounding of ``l0``.
    out["l0_raw"] = None if _l0 is None else float(_l0)
    out["recon_r2"] = None if mp else st.get("r2_score")
    out["l1_per_token"] = None if mp else st.get("l1_per_token")
    out["stats_source"] = ("mp-excluded" if mp
                           else ("stats" if st else "none"))
    if alive is None:
        # No fallback to a small pool: it cannot resolve rare firing,
        # so #dead is left unknown.
        out["n_dead"] = None
        out["dead_rate"] = None
        out["alive_source"] = "none"
    else:
        out["n_dead"] = int((~alive).sum())
        out["dead_rate"] = float((~alive).mean())
        out["alive_source"] = "stats"
    # Dense latents: active (|a| > 0) on >= DENSE_THRESHOLD of tokens,
    # from the same full-split firing-rate vector as ``n_dead``.
    if firing is None:
        out["n_dense"] = None
        out["dense_rate"] = None
    else:
        out["n_dense"] = int((firing >= DENSE_THRESHOLD).sum())
        out["dense_rate"] = float((firing >= DENSE_THRESHOLD).mean())
    out["dense_threshold"] = float(DENSE_THRESHOLD)
    # Dead atoms are excluded from the duplicate count, so #dead and #dup
    # are disjoint (dead atoms often sit near each other).
    #
    # ``n_dup`` counts every atom in a duplicate cluster; ``n_dup_removed``
    # counts the atoms deduplication would delete (cluster size minus one
    # representative per cluster).
    _live = dec if alive is None else dec[np.asarray(alive, dtype=bool)]
    for suf, absolute in (("signed", False), ("abs", True)):
        d = _dup_stats(_live, absolute=absolute) if _live.shape[0] > 1 else {
            "n_dup": {f"{t:g}": 0 for t in THRESHOLDS},
            "n_clusters": {f"{t:g}": 0 for t in THRESHOLDS},
            "largest": {f"{t:g}": 0 for t in THRESHOLDS}}
        out[f"n_dup_{suf}"] = d["n_dup"]
        out[f"n_dup_clusters_{suf}"] = d["n_clusters"]
        out[f"n_dup_removed_{suf}"] = {
            k: int(v) - int(d["n_clusters"][k]) for k, v in d["n_dup"].items()}
        out[f"largest_dup_{suf}"] = d["largest"]
        out[f"babel_mu1_{suf}"] = _babel_mu1(_live, absolute=absolute)
    out["dup_excludes_dead"] = True
    return out


def pair_metrics(D_a: np.ndarray, D_b: np.ndarray) -> dict:
    """Cross-seed decoder consistency for one pair: ``DCS``, the mean
    cosine of the Hungarian-matched atoms over the whole dictionary."""
    m = decoder_match(D_a, D_b)
    return {k: m[k] for k in ("dcs_signed", "dcs_abs", "frac_sign_flipped")}


def gt_metrics(D_gt: np.ndarray, D_S: np.ndarray) -> dict:
    """Recovery of a ground-truth dictionary by one SAE: ``DCS`` of the
    Hungarian match against the ground truth, under ``gt_*`` so it cannot
    be confused with the cross-seed reading."""
    out = {f"gt_{k}": v for k, v in pair_metrics(D_gt, D_S).items()}
    out["gt_n_atoms"] = int(D_gt.shape[0])
    return out


def _mean_or_none(vals):
    vals = [v for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def _std_or_none(vals):
    """SAMPLE std (ddof=1) across seeds/pairs, or None below two values.

    The seeds are a sample of training-noise outcomes, hence ddof=1. A
    single value (e.g. the one pair of a 2-seed cell) gives None rather
    than 0.0, which would suggest a measured zero spread.
    """
    vals = [v for v in vals if v is not None]
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else None


def _band_per_source(recs, name: str, band: tuple[float, ...]):
    """Each seed's/pair's own band mean, for the spread across them.

    The summary's band value averages across sources first, so the
    cross-source spread of the band needs each source banded first.
    """
    out = []
    for v in recs:
        per_t = (v or {}).get(name) or {}
        vals = [per_t.get(f"{t:g}") for t in band]
        if all(x is not None for x in vals):
            out.append(float(np.mean(vals)))
    return out


def summarise(per_seed: dict, per_pair: dict,
              band: tuple[float, ...] = REPORT_BAND) -> dict:
    """Cross-seed rollup, plus the band means the tables print.

    Both the per-threshold maps (``*_per_threshold``) and the band means
    (``*_band``) are kept.
    """
    out: dict = {"n_seeds": len(per_seed), "n_pairs": len(per_pair),
                 "report_band": [float(t) for t in band]}
    # Counts (``n_dead``, ``n_dense``, ``l0``) are reported as rounded
    # integer means. Every ``_mean`` has a ``_std`` (cross-seed or
    # cross-pair spread), computed from unrounded values.
    for _k in ("n_dead", "n_dense", "l0"):
        _v = _mean_or_none([v.get(_k) for v in per_seed.values()])
        out[f"{_k}_mean"] = None if _v is None else int(round(_v))
        # ``l0`` is continuous but rounded per seed; its std uses
        # ``l0_raw``.
        _src = "l0_raw" if _k == "l0" else _k
        _raw = [v.get(_src) for v in per_seed.values()]
        if _k == "l0" and all(x is None for x in _raw):
            _raw = [v.get("l0") for v in per_seed.values()]  # no l0_raw
        out[f"{_k}_std"] = _std_or_none(_raw)
    for _k in ("recon_r2", "l1_per_token"):
        out[f"{_k}_mean"] = _mean_or_none(
            [v.get(_k) for v in per_seed.values()])
        out[f"{_k}_std"] = _std_or_none(
            [v.get(_k) for v in per_seed.values()])
    for suf in ("signed", "abs"):
        out[f"babel_mu1_{suf}_mean"] = _mean_or_none(
            [v.get(f"babel_mu1_{suf}") for v in per_seed.values()])
        out[f"babel_mu1_{suf}_std"] = _std_or_none(
            [v.get(f"babel_mu1_{suf}") for v in per_seed.values()])
        for name in ("n_dup", "n_dup_clusters", "largest_dup"):
            per_t = {
                f"{t:g}": _mean_or_none(
                    [v.get(f"{name}_{suf}", {}).get(f"{t:g}")
                     for v in per_seed.values()])
                for t in THRESHOLDS
            }
            out[f"{name}_{suf}_per_threshold"] = per_t
            out[f"{name}_{suf}_band"] = _band_mean(per_t, band)
            out[f"{name}_{suf}_band_std"] = _std_or_none(
                _band_per_source(per_seed.values(), f"{name}_{suf}", band))
        out[f"dcs_{suf}_mean"] = _mean_or_none(
            [v.get(f"dcs_{suf}") for v in per_pair.values()])
        out[f"dcs_{suf}_std"] = _std_or_none(
            [v.get(f"dcs_{suf}") for v in per_pair.values()])
    srcs = {v.get("alive_source") for v in per_seed.values()}
    out["alive_source"] = "/".join(sorted(s for s in srcs if s)) or None
    _ds = {v.get("d_sae") for v in per_seed.values()}
    _ds = sorted(v for v in _ds if v)
    out["d_sae"] = int(_ds[0]) if len(_ds) == 1 else None
    return out


def summarise_gt(per_gt: dict) -> dict:
    """Ground-truth recovery averaged over seeds, each scored against the
    same fixed ground truth."""
    out: dict = {"gt_n_seeds": len(per_gt)}
    out["gt_n_atoms"] = _mean_or_none(
        [v.get("gt_n_atoms") for v in per_gt.values()])
    for suf in ("signed", "abs"):
        out[f"gt_dcs_{suf}_mean"] = _mean_or_none(
            [v.get(f"gt_dcs_{suf}") for v in per_gt.values()])
    return out


def compute_tag(tag_dir: Path, step: str = "final",
                gt: "np.ndarray | None" = None) -> dict | None:
    """Every seed and every seed-pair under one tag directory.

    ``gt`` (synthetic only) is a unit-norm ``(d_sae_gt, d)`` ground-truth
    dictionary, passed in by the caller so this module stays
    modality-agnostic.
    """
    # MP is identified by its architecture directory, not the tag name.
    mp = "mp" in tag_dir.parts
    from consdl.io.paths import seed_step_dirs  # noqa: PLC0415
    seeds = seed_step_dirs(tag_dir, step)
    per_seed: dict = {}
    decs: dict = {}
    alives: dict = {}
    for name, sd in seeds:
        dec = load_decoder(sd)
        if dec is None:
            continue
        # (dataset, split) is only the first guess; the stats readers fall
        # back to the seed's single ``stats.*.seed_no`` directory, so LM
        # needs no branch here.
        st = stats_metrics(sd, "imagenet", "val", int(dec.shape[0]))
        # Read the raw firing rate once: it gives both dead and dense.
        fr = firing_rate_from_stats(sd, "imagenet", "val")
        if fr is not None and fr.size != dec.shape[0]:
            print(f"    [dict] {name}: firing_rate {fr.size} vs decoder "
                  f"{dec.shape[0]} — liveness dropped", flush=True)
            fr = None
        av = None if fr is None else (fr > DEAD_THRESHOLD)
        decs[name], alives[name] = dec, av
        per_seed[name] = seed_metrics(dec, av, st, mp=mp, firing=fr)
    if not per_seed:
        return None
    names = sorted(per_seed)
    per_pair = {
        f"{a}↔{b}": pair_metrics(decs[a], decs[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    per_gt = ({n: gt_metrics(gt, decs[n]) for n in names}
              if gt is not None else {})
    out = {
        "schema": SCHEMA, "step": step,
        "linkage": LINKAGE_DEFAULT,
        "thresholds": [float(t) for t in THRESHOLDS],
        "seeds": names,
        "per_seed": per_seed, "per_pair": per_pair,
        "summary": summarise(per_seed, per_pair),
    }
    if per_gt:
        out["per_gt"] = per_gt
        out["summary"].update(summarise_gt(per_gt))
    return out


def write_tag(tag_dir: Path, out_path: Path, step: str = "final",
              gt: "np.ndarray | None" = None) -> bool:
    payload = compute_tag(tag_dir, step, gt=gt)
    if payload is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    _mark_reports_dirty(out_path)
    return True


#: Summary keys grouped by meaning: single-model ("training") vs
#: cross-seed consistency. The files are split by cost instead (decoder +
#: stats here, activations in ``code_metrics``), so the grouping is
#: declared explicitly. ``consistency_code`` keys live in
#: ``code_metrics``.
GROUPS: dict[str, tuple[str, ...]] = {
    "training": ("l0", "recon_r2", "l1_per_token", "n_dead", "n_dense",
                 "n_dup", "n_dup_clusters", "largest_dup", "babel_mu1"),
    "consistency_decoder": ("dcs",),
    "consistency_code": (),
}


def backfill_stats(tag_dir: Path, out_path: Path,
                   step: str = "final") -> str:
    """Add the stats-only scalars to an EXISTING ``dict_metrics`` file.

    ``l0``, ``recon_r2``, ``l1_per_token`` and the dense counts come from
    each seed's stats directory and nothing else in the file depends on
    them, so they can be updated without redoing the decoder work.

    Returns ``"ok"``, ``"skip"`` (nothing to add) or a reason string.
    """
    if not out_path.is_file():
        return "no-file"
    try:
        payload = json.loads(out_path.read_text())
    except (OSError, ValueError) as e:
        return f"unreadable: {e}"
    per_seed = payload.get("per_seed") or {}
    if not per_seed:
        return "no-per-seed"

    mp = "mp" in tag_dir.parts
    changed = False
    for name, rec in per_seed.items():
        sd = tag_dir / name / step
        d_sae = int(rec.get("d_sae") or 0)
        st = stats_metrics(sd, "imagenet", "val", d_sae)
        _l0 = st.get("l0_abs")
        vals = {
            "l0": None if (mp or _l0 is None) else int(round(float(_l0))),
            "l0_raw": None if (mp or _l0 is None) else float(_l0),
            "recon_r2": None if mp else st.get("r2_score"),
            "l1_per_token": None if mp else st.get("l1_per_token"),
            "stats_source": ("mp-excluded" if mp
                             else ("stats" if st else "none")),
        }
        fr = firing_rate_from_stats(sd, "imagenet", "val")
        if fr is not None and d_sae and fr.size != d_sae:
            fr = None
        vals["n_dense"] = (None if fr is None
                           else int((fr >= DENSE_THRESHOLD).sum()))
        vals["dense_rate"] = (None if fr is None
                              else float((fr >= DENSE_THRESHOLD).mean()))
        vals["dense_threshold"] = float(DENSE_THRESHOLD)
        for k, v in vals.items():
            if rec.get(k) != v:
                rec[k] = v
                changed = True
    if not changed:
        return "skip"
    for _k in ("l0", "recon_r2", "l1_per_token", "n_dense"):
        _v = _mean_or_none([v.get(_k) for v in per_seed.values()])
        if _k in ("l0", "n_dense") and _v is not None:
            _v = int(round(_v))
        payload.setdefault("summary", {})[f"{_k}_mean"] = _v
    # An unwritable results file is reported, not raised, so a backfill
    # over many tags continues.
    try:
        out_path.write_text(json.dumps(payload, indent=1))
    except OSError as e:
        return f"unwritable: {e}"
    _mark_reports_dirty(out_path)
    return "ok"
