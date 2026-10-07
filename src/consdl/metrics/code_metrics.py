"""Code metrics: cross-seed consistency measured on activations.

Companion to ``dict_metrics``, which works from decoder matrices and
firing rates alone. These metrics run each seed's SAE over a shared
activation pool:

    per seed   l0, recon_r2, l1_per_token
    per pair   mcc, input_r2, act_iou

``mcc`` is the mean Pearson correlation between the codes of atoms paired
by the decoder's Hungarian match (``kernels.decoder_match``), stored
signed and as ``|r|``. Synthetic cells add the same readings against the
ground-truth codes (``gt_*``).

``input_r2`` is the R^2 of one seed's reconstruction against the
other's. ``compute_tag_stream`` accumulates everything in one pass over
batches (``TagStream``), so memory is bounded by the batch, not the pool.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from consdl.metrics.dict_metrics import (
    REPORT_BAND,
    THRESHOLDS,
    _mean_or_none,
    _std_or_none,
)
from consdl.metrics.kernels import decoder_match
from consdl.metrics.quality import alive_mask_from_stats
from consdl.io.paths import seed_step_dirs
from consdl.io.report_stamp import mark_reports_dirty as _mark_reports_dirty

SCHEMA = 1


def seed_metrics(qual: dict) -> dict:
    """Per-seed scalars from ``seed_encode``'s quality dict.

    ``dead_rate`` is omitted: a small pool cannot resolve firing rates
    near ``DEAD_THRESHOLD``. #dead comes from ``dict_metrics``, which
    reads full-split statistics.
    """
    return {
        "l0": qual.get("avg_active_per_token"),
        "recon_r2": qual.get("recon_r2"),
        "l1_per_token": qual.get("l1_per_token"),
    }


def pair_metrics(D_a, D_b, acts_a, acts_b, zhat_a, zhat_b,
                 alive_a, alive_b) -> dict:
    """Cross-seed code agreement for one pair, over the whole pool.

    Treated as a one-batch stream, so it shares its accumulators with
    ``compute_tag_stream``.
    """
    z_a = np.asarray(zhat_a)
    st_a, st_b = _SeedStream(z_a.shape[1]), _SeedStream(z_a.shape[1])
    sup_a = np.abs(np.asarray(acts_a)) > 0
    sup_b = np.abs(np.asarray(acts_b)) > 0
    # Pair readings need only the code and zhat sums, so zhat stands in
    # for the reconstruction target and the resulting ``sse`` is unused.
    st_a.update(zhat_a, acts_a, zhat_a, sup_a)
    st_b.update(zhat_b, acts_b, zhat_b, sup_b)
    ps = _PairStream(D_a, D_b, alive_a, alive_b)
    ps.update(acts_a, acts_b, sup_a, sup_b, zhat_a, zhat_b)
    return ps.result(st_a, st_b)


def gt_metrics(D_gt, D_S, Z_gt, acts_S, z_gt, zhat_S, alive_S) -> dict:
    """Code recovery of a ground-truth dictionary by one SAE.

    ``z_gt`` is the generative input (which the ground truth reconstructs
    exactly), so ``gt_input_r2`` is how much of the true signal the SAE's
    reconstruction recovers.
    """
    n_gt = int(D_gt.shape[0])
    av_T = np.ones(n_gt, dtype=bool)
    av_S = (np.ones(int(D_S.shape[0]), dtype=bool)
            if alive_S is None else alive_S)
    out = pair_metrics(D_gt, D_S, Z_gt, acts_S, z_gt, zhat_S, av_T, av_S)
    out = {f"gt_{k}": v for k, v in out.items()}
    out["gt_n_atoms"] = n_gt
    out["gt_alive_source"] = "none" if alive_S is None else "stats"
    return out


def summarise_gt(per_gt: dict) -> dict:
    """Ground-truth recovery averaged over SEEDS: each entry is one SAE
    scored against the same fixed ground truth."""
    out: dict = {"gt_n_seeds": len(per_gt)}
    out["gt_n_atoms"] = _mean_or_none(
        [v.get("gt_n_atoms") for v in per_gt.values()])
    out["gt_input_r2_mean"] = _mean_or_none(
        [v.get("gt_input_r2") for v in per_gt.values()])
    for suf in ("signed", "abs"):
        out[f"gt_mcc_{suf}_mean"] = _mean_or_none(
            [v.get(f"gt_mcc_{suf}") for v in per_gt.values()])
    srcs = {v.get("gt_alive_source") for v in per_gt.values()}
    out["gt_alive_source"] = "/".join(sorted(s for s in srcs if s)) or None
    return out


def summarise(per_seed: dict, per_pair: dict,
              band: tuple[float, ...] = REPORT_BAND) -> dict:
    out: dict = {"n_seeds": len(per_seed), "n_pairs": len(per_pair),
                 "report_band": [float(t) for t in band]}
    # ``_std`` beside every ``_mean``, as in dict_metrics: sample std
    # (ddof=1) across seeds or across pairs, None below two values.
    for k in ("l0", "recon_r2", "l1_per_token"):
        _v = [v.get(k) for v in per_seed.values()]
        out[f"{k}_mean"] = _mean_or_none(_v)
        out[f"{k}_std"] = _std_or_none(_v)
    _ir = [v.get("input_r2") for v in per_pair.values()]
    out["input_r2_mean"] = _mean_or_none(_ir)
    out["input_r2_std"] = _std_or_none(_ir)
    for suf in ("signed", "abs"):
        _m = [v.get(f"mcc_{suf}") for v in per_pair.values()]
        out[f"mcc_{suf}_mean"] = _mean_or_none(_m)
        out[f"mcc_{suf}_std"] = _std_or_none(_m)
    _iou = [v.get("act_iou") for v in per_pair.values()]
    out["act_iou_mean"] = _mean_or_none(_iou)
    out["act_iou_std"] = _std_or_none(_iou)
    out["act_iou_n_pairs"] = _mean_or_none(
        [v.get("act_iou_n_pairs") for v in per_pair.values()])
    srcs = {v.get("alive_source") for v in per_pair.values()}
    out["alive_source"] = "/".join(sorted(s for s in srcs if s)) or None
    return out


class _SeedStream:
    """Per-seed batch accumulators.

    Anything that depends on one seed only (support, column sums of
    ``a``, ``a^2`` and ``zhat``) is accumulated here once rather than in
    each of the S-1 pairs the seed takes part in.

    Reductions pass ``dtype=np.float64`` instead of casting the operand,
    which avoids a float64 copy of each ``(batch, d_sae)`` array.
    """

    __slots__ = ("n", "n_elems", "nnz", "l1", "sse",
                 "sa", "saa", "ssup", "sh", "shh")

    def __init__(self, d_in: int):
        self.n = 0
        self.n_elems = 0
        self.nnz = 0
        self.l1 = 0.0
        self.sse = 0.0
        self.sa = None          # column sums of the code
        self.saa = None         # column sums of the squared code
        self.ssup = None        # per-column active counts
        self.sh = np.zeros(int(d_in), dtype=np.float64)   # sums of zhat
        self.shh = np.zeros(int(d_in), dtype=np.float64)

    def update(self, z, acts, zhat, sup) -> None:
        a = np.asarray(acts)
        zh = np.asarray(zhat)
        if self.sa is None:
            w = a.shape[1]
            self.sa = np.zeros(w, dtype=np.float64)
            self.saa = np.zeros(w, dtype=np.float64)
            self.ssup = np.zeros(w, dtype=np.int64)
        self.n += a.shape[0]
        self.n_elems += int(z.size)
        self.nnz += int(sup.sum(dtype=np.int64))
        self.l1 += float(np.abs(a).sum(dtype=np.float64))
        self.sse += float(((z - zh) ** 2).sum(dtype=np.float64))
        self.sa += a.sum(0, dtype=np.float64)
        self.saa += (a * a).sum(0, dtype=np.float64)
        self.ssup += sup.sum(0, dtype=np.int64)
        self.sh += zh.sum(0, dtype=np.float64)
        self.shh += (zh * zh).sum(0, dtype=np.float64)

    def result(self, sst: float) -> dict:
        if self.n == 0:
            return {"l0": None, "recon_r2": None, "l1_per_token": None}
        return {
            "l0": self.nnz / self.n,
            "recon_r2": float(1.0 - self.sse / max(sst, 1e-30)),
            "l1_per_token": self.l1 / self.n,
        }


class _PairStream:
    """Pairwise accumulators for one seed pair.

    Three accumulators (two ``(n_matched,)`` and a scalar):

      ``sab``   sum of the matched-column products, for Pearson;
      ``both``  per-column count of inputs active on BOTH sides;
      ``sse``   squared error between the two reconstructions.

    Everything else is read from the per-seed streams at ``result``
    time::

      sum b   = sum a_b[cols]
      sum b^2 = sum a_b^2[cols]
      either  = ssup_a + ssup_b[cols] - both
      SST     = the partner's own sh / shh
    """

    def __init__(self, D_a, D_b, alive_a, alive_b):
        self.cols = decoder_match(D_a, D_b)["cols"]
        self.has_alive = alive_a is not None and alive_b is not None
        self.sab = np.zeros(len(self.cols), dtype=np.float64)
        self.both = np.zeros(len(self.cols), dtype=np.int64)
        self.sse = 0.0
        self.n = 0

    def update(self, acts_a, acts_b, sup_a, sup_b, zhat_a, zhat_b) -> None:
        A = np.asarray(acts_a)
        Bp = np.asarray(acts_b)[:, self.cols]
        self.n += A.shape[0]
        self.sab += (A * Bp).sum(0, dtype=np.float64)
        self.both += (sup_a & sup_b[:, self.cols]).sum(0, dtype=np.int64)
        self.sse += float(((np.asarray(zhat_a) - np.asarray(zhat_b)) ** 2)
                          .sum(dtype=np.float64))

    @staticmethod
    def _pearson(n, sa, saa, sb, sbb, sab):
        cov = sab - sa * sb / n
        va = saa - sa * sa / n
        vb = sbb - sb * sb / n
        den = np.sqrt(np.clip(va, 0.0, None) * np.clip(vb, 0.0, None))
        r = np.zeros_like(cov)
        ok = den > 1e-12
        r[ok] = cov[ok] / den[ok]
        return np.clip(r, -1.0, 1.0)

    def result(self, sa_st, sb_st) -> dict:
        c = self.cols
        n = float(self.n)
        out: dict = {"alive_source": "stats" if self.has_alive else "none"}
        if n < 2 or sa_st.sa is None or sb_st.sa is None:
            out.update({"mcc_signed": None, "mcc_abs": None,
                        "act_iou": None, "act_iou_n_pairs": 0,
                        "input_r2": None})
            return out
        r = self._pearson(n, sa_st.sa, sa_st.saa,
                          sb_st.sa[c], sb_st.saa[c], self.sab)
        out["mcc_signed"] = float(r.mean()) if r.size else None
        out["mcc_abs"] = float(np.abs(r).mean()) if r.size else None
        either = sa_st.ssup + sb_st.ssup[c] - self.both
        live = either > 0
        out["act_iou"] = (float((self.both[live] / either[live]).mean())
                          if live.any() else None)
        out["act_iou_n_pairs"] = int(live.sum())
        sst = float((sa_st.shh - sa_st.sh * sa_st.sh / n).sum())
        out["input_r2"] = float(1.0 - self.sse / max(sst, 1e-30))
        return out


class TagStream:
    """Every seed and pair under one tag, accumulated over batches.

    Work per batch is ``S`` seed passes plus ``S(S-1)/2`` pairs of two
    reductions each. The input's own sums are taken once.

    Encode each batch with every seed and pass them together, since pair
    readings need both sides on the same rows::

        st = TagStream(names, decs, alives, d_in)
        for z in batches:
            st.update(z, {n: (acts_n, zhat_n) for n in names})
        st.result()
    """

    def __init__(self, names, decs, alives, d_in):
        self.names = list(names)
        self.d_in = int(d_in)
        self.seeds = {n: _SeedStream(d_in) for n in self.names}
        self.pairs = {
            f"{a}↔{b}": _PairStream(decs[a], decs[b], alives[a], alives[b])
            for i, a in enumerate(self.names) for b in self.names[i + 1:]
        }
        self.n_tokens = 0
        self.zn = 0
        self.sz = np.zeros(self.d_in, dtype=np.float64)
        self.szz = np.zeros(self.d_in, dtype=np.float64)

    def update(self, z, enc: dict) -> None:
        z = np.asarray(z)
        nb = z.shape[0]
        self.n_tokens += nb
        self.zn += nb
        self.sz += z.sum(0, dtype=np.float64)
        self.szz += (z * z).sum(0, dtype=np.float64)
        sup: dict = {}
        for n in self.names:
            a, zh = enc[n]
            sup[n] = np.abs(a) > 0
            self.seeds[n].update(z, a, zh, sup[n])
        for i, a in enumerate(self.names):
            for b in self.names[i + 1:]:
                self.pairs[f"{a}↔{b}"].update(
                    enc[a][0], enc[b][0], sup[a], sup[b],
                    enc[a][1], enc[b][1])

    def result(self, *, step: str = "final") -> dict:
        sst = (float((self.szz - self.sz * self.sz / self.zn).sum())
               if self.zn else 0.0)
        per_seed = {n: s.result(sst) for n, s in self.seeds.items()}
        per_pair = {}
        for k, p in self.pairs.items():
            a, b = k.split("↔", 1)
            per_pair[k] = p.result(self.seeds[a], self.seeds[b])
        return {
            "schema": SCHEMA, "step": step,
            "n_tokens": int(self.n_tokens),
            "thresholds": [float(t) for t in THRESHOLDS],
            "seeds": list(self.names),
            "per_seed": per_seed, "per_pair": per_pair,
            "summary": summarise(per_seed, per_pair),
        }


def compute_tag(tag_dir: Path, *, z_pool: np.ndarray, encode, load_decoder_fn,
                step: str = "final", gt: "tuple | None" = None) -> dict | None:
    """Every seed and pair under one tag, given a prepared pool.

    ``encode(step_dir) -> (acts, z_hat, quality) | None`` is injected so
    this module never imports a modality.

    ``gt`` (synthetic only) is ``(D_gt, Z_gt, z_gt)`` -- the ground-truth
    dictionary, its generative codes and the input those codes produced.
    Absent, the ground-truth block is simply not written.
    """
    per_seed, acts, zhat, decs, alives = {}, {}, {}, {}, {}
    for name, sd in seed_step_dirs(tag_dir, step):
        enc = encode(sd)
        dec = load_decoder_fn(sd)
        if enc is None or dec is None:
            continue
        a, zh, qual = enc
        av = alive_mask_from_stats(sd, "imagenet", "val")
        if av is not None and av.size != dec.shape[0]:
            print(f"    [code] {name}: firing_rate {av.size} vs decoder "
                  f"{dec.shape[0]} — liveness dropped", flush=True)
            av = None
        per_seed[name] = seed_metrics(qual)
        acts[name], zhat[name], decs[name], alives[name] = a, zh, dec, av
    if not per_seed:
        return None
    names = sorted(per_seed)
    per_pair = {
        f"{a}↔{b}": pair_metrics(
            decs[a], decs[b], acts[a], acts[b], zhat[a], zhat[b],
            alives[a], alives[b])
        for i, a in enumerate(names) for b in names[i + 1:]
    }
    per_gt = {}
    if gt is not None:
        D_gt, Z_gt, z_gt = gt
        per_gt = {
            n: gt_metrics(D_gt, decs[n], Z_gt, acts[n], z_gt, zhat[n],
                          alives[n])
            for n in names
        }
    out = {
        "schema": SCHEMA, "step": step,
        "n_tokens": int(z_pool.shape[0]),
        "thresholds": [float(t) for t in THRESHOLDS],
        "seeds": names,
        "per_seed": per_seed, "per_pair": per_pair,
        "summary": summarise(per_seed, per_pair),
    }
    if per_gt:
        out["per_gt"] = per_gt
        out["summary"].update(summarise_gt(per_gt))
    return out


def compute_tag_stream(tag_dir: Path, *, batches, encode_batch,
                       open_seed=None,
                       load_decoder_fn, step: str = "final",
                       alive_dataset: str = "imagenet",
                       alive_split: str = "val") -> dict | None:
    """``compute_tag`` over a pool too large to hold in memory.

    ``compute_tag`` materialises ``(n_tokens, d_sae)`` per seed;
    ``TagStream`` keeps only length-``d_sae`` accumulators, so peak memory
    is set by the batch.

    ``batches`` yields backbone features ``z`` of shape ``(batch, d_in)``,
    typically from a precomputed activation cache.

    ``open_seed(step_dir) -> handle | None`` is called once per seed and
    ``encode_batch(handle, z) -> (acts, zhat) | None`` once per batch, so
    each SAE is loaded once. The loop is batch-outer, seeds-inner: every
    seed sees the same rows in the same order, as pair readings require.
    """
    names, decs, alives, dirs = [], {}, {}, {}
    for name, sd in seed_step_dirs(tag_dir, step):
        dec = load_decoder_fn(sd)
        if dec is None:
            continue
        av = alive_mask_from_stats(sd, alive_dataset, alive_split)
        if av is not None and av.size != dec.shape[0]:
            print(f"    [code] {name}: firing_rate {av.size} vs decoder "
                  f"{dec.shape[0]} -- liveness dropped", flush=True)
            av = None
        names.append(name)
        decs[name], alives[name], dirs[name] = dec, av, sd
    if not names:
        return None
    st = None
    handles = None
    for z in batches:
        z = np.asarray(z)
        if st is None:
            # d_in is taken from the data so a mismatch fails here
            # rather than as a silent broadcast later.
            st = TagStream(names, decs, alives, int(z.shape[1]))
        enc = {}
        if handles is None:
            handles = {n: (dirs[n] if open_seed is None else open_seed(dirs[n]))
                       for n in names}
        for n in names:
            h = handles.get(n)
            if h is None:
                return None
            e = encode_batch(h, z)
            if e is None:
                return None
            enc[n] = e
        st.update(z, enc)
    return None if st is None else st.result(step=step)


def write_tag(tag_dir: Path, out_path: Path, **kw) -> bool:
    payload = compute_tag(tag_dir, **kw)
    if payload is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=1))
    _mark_reports_dirty(out_path)
    return True


#: Counterpart of :data:`dict_metrics.GROUPS`: consistency measured on
#: codes, which needs the activation pool.
GROUPS: dict[str, tuple[str, ...]] = {
    "training": (),
    "consistency_decoder": (),
    "consistency_code": ("mcc", "input_r2"),
}
