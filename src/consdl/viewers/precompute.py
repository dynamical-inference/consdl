"""Decoder clusters and joint UMAP views: everything the viewers read except
the top inputs (``consdl.viewers.tops``).

Stages, per backbone (output under ``var/results/viewers/<modality>/<backbone>``):

  kmeans     per arch × cell: spherical k-means of seed 0's unit decoder rows
             at ``m`` atoms per cluster (K = d_sae / m); every other seed and a
             Haar-rotated seed 0 (the no-shared-direction floor) are assigned
             to seed 0's centroids.  -> <arch>/within/<cell>/centroids_m<m>.npz,
             metrics.json
  figures    cluster-size curves per arch (seed 0's cluster sizes and the
             other seeds' counts on the same clusters).  -> <arch>/cluster_sizes_m<m>.png
  views      per arch: one ``seed`` view per cell (its seeds, Hungarian + NN
             matches) and one ``cross`` view (seed 0 of every cell, NN), each a
             joint UMAP-2D of all its panels; plus the panel keys.
             -> views/seed__<arch>__<cell>.{npz,json}, views/cross__<arch>.*,
                views/keys__<arch>.json
  archviews  per cell: seed 0 of every arch (NN).  -> views/arch__<cell>.*

Colour = nearest centroid of the view's anchor: the cell's own seed 0 on a
seed view, the arch's largest cell on a cross view, ConSDL's seed 0 on an
arch view. iSAE's atom sign is unidentified, so any comparison touching an
iSAE atom uses |cos|, and for the UMAP each iSAE atom is flipped to agree
with its assigned centroid.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from consdl.viewers import layout as lay
from consdl.viewers.models import unit_decoder

M_COLOUR = 48          # atoms per cluster of every colour anchor
N_NEIGHBORS, MIN_DIST = 15, 0.1


def _log(msg: str) -> None:
    print(f"[viewer-precompute] {msg}", flush=True)


def write_backbone_json(bb: lay.BackboneSpec, root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "backbone.json").write_text(json.dumps(
        {"name": bb.name, "short": bb.short, "layer": bb.layer, "modality": bb.modality,
         "hf_name": bb.hf_name, "hook": bb.hook, "label": bb.label}, indent=1))


# ── spherical k-means ─────────────────────────────────────────────────


def _assign_t(X, C, absolute: bool):
    S = X @ C.T
    if not absolute:
        best, lab = S.max(dim=1)
        return best, lab, None
    best, lab = S.abs().max(dim=1)
    sgn = S.gather(1, lab[:, None]).squeeze(1).sign()
    sgn[sgn == 0] = 1
    return best, lab, sgn


def skmeans(X, K: int, *, seed: int, restarts: int, absolute: bool, max_iters: int = 100):
    """Best-of-``restarts`` spherical k-means (k-means++ init) of unit rows ``X``.

    Returns ``(C (K, d), labels (n,))`` as torch tensors on ``X.device``.
    """
    import torch  # noqa: PLC0415

    g = torch.Generator(device="cpu").manual_seed(int(seed))
    n = X.shape[0]
    best = None
    for _ in range(int(restarts)):
        idx = [int(torch.randint(n, (1,), generator=g))]
        sim = (lambda v: (X @ v).abs()) if absolute else (lambda v: X @ v)
        dmin = (1.0 - sim(X[idx[0]])).clamp_min(0)
        for _k in range(1, K):
            p = (dmin ** 2).cpu().double()
            j = int(torch.multinomial(p / p.sum(), 1, generator=g)) if p.sum() > 0 else int(
                torch.randint(n, (1,), generator=g))
            idx.append(j)
            dmin = torch.minimum(dmin, (1.0 - sim(X[j])).clamp_min(0))
        C = X[idx].clone()
        lab = None
        for _it in range(max_iters):
            fit, new, sgn = _assign_t(X, C, absolute)
            if lab is not None and torch.equal(new, lab):
                break
            lab = new
            Xs = X if sgn is None else X * sgn[:, None]
            C = torch.zeros_like(C).index_add_(0, lab, Xs)
            empty = torch.nonzero(torch.bincount(lab, minlength=K) == 0).flatten()
            if empty.numel():
                C[empty] = X[torch.argsort(fit)[: empty.numel()]]
            C = C / C.norm(dim=1, keepdim=True).clamp_min(1e-12)
        fit, lab, _ = _assign_t(X, C, absolute)
        inertia = float((1.0 - fit).sum())
        if best is None or inertia < best[0]:
            best = (inertia, C, lab)
    return best[1], best[2]


def assign(X: np.ndarray, C: np.ndarray, absolute: bool):
    """``(label, sign)`` of each row of ``X`` against centroids ``C`` (numpy)."""
    S = X @ C.T
    lab = (np.abs(S) if absolute else S).argmax(1)
    sgn = np.sign(S[np.arange(len(X)), lab]) if absolute else np.ones(len(X), np.float32)
    sgn[sgn == 0] = 1
    return lab.astype(np.int32), sgn.astype(np.float32)


def _haar(d: int, seed: int = 1234) -> np.ndarray:
    rng = np.random.default_rng(seed)
    Q, R = np.linalg.qr(rng.standard_normal((d, d)))
    return (Q * np.sign(np.diag(R))).astype(np.float32)


def _anchor_rows(C0, lab0, Y, absolute: bool) -> tuple[dict, np.ndarray]:
    """How the atoms ``Y`` fall onto seed 0's centroids: scalars and sizes."""
    from scipy.stats import spearmanr  # noqa: PLC0415

    K = len(C0)
    lab, sgn = assign(Y, C0, absolute)
    fit = (Y * C0[lab]).sum(1) * (sgn if absolute else 1.0)
    size0 = np.bincount(lab0, minlength=K)
    size = np.bincount(lab, minlength=K)
    Cy = np.zeros_like(C0)
    np.add.at(Cy, lab, Y * sgn[:, None])
    occ = size > 0
    Cy[occ] /= np.linalg.norm(Cy[occ], axis=1, keepdims=True).clip(1e-12)
    drift = (C0[occ] * Cy[occ]).sum(1)
    return {"fit_mean_cos": float(fit.mean()), "occupancy": float(occ.mean()),
            "drift_mean_cos": float(drift.mean()),
            "size_spearman": float(spearmanr(size0, size).statistic)}, size


def run_kmeans(cells_by_arch: dict, root: Path, *, m: int = M_COLOUR, restarts: int = 3,
               device: str = "cpu", force: bool = False) -> None:
    import torch  # noqa: PLC0415

    for arch, cells in cells_by_arch.items():
        absolute = arch in lay.ABS_ARCHES
        for c in cells:
            out = root / arch / "within" / c.name
            f = out / f"centroids_m{m}.npz"
            if f.is_file() and not force:
                continue
            t0 = time.time()
            seeds = c.seed_names()
            D = {s: unit_decoder(c.seeds[s]) for s in seeds}
            X0 = D[seeds[0]]
            K = max(2, X0.shape[0] // m)
            C, lab = skmeans(torch.from_numpy(X0).to(device), K, seed=K, restarts=restarts,
                             absolute=absolute)
            C0, lab0 = C.cpu().numpy().astype(np.float32), lab.cpu().numpy().astype(np.int32)
            arrs = {"C0": C0, "lab_seed0": lab0, "size_seed0": np.bincount(lab0, minlength=K)}
            fit0 = (X0 * C0[lab0]).sum(1)
            res = {"arch": arch, "cell": c.name, "m": m, "K": K,
                   "gauge": "abs" if absolute else "signed",
                   "seeds": {s: str(c.seeds[s]) for s in seeds},
                   "seed0_self_fit": float((np.abs(fit0) if absolute else fit0).mean()), "rows": {}}
            targets = [(s, D[s]) for s in seeds[1:]] + [("rotated_seed0_floor", X0 @ _haar(X0.shape[1]))]
            for name, Y in targets:
                row, size = _anchor_rows(C0, lab0, Y, absolute)
                res["rows"][name] = row
                arrs[f"size_{name}"] = size
            out.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(f, **arrs)
            (out / "metrics.json").write_text(json.dumps(res, indent=1))
            _log(f"kmeans {arch}/{c.name}  K={K}  {time.time() - t0:.0f}s")


def anchor_of(root: Path, arch: str, cell: str, m: int = M_COLOUR):
    z = np.load(root / arch / "within" / cell / f"centroids_m{m}.npz")
    return z["C0"].astype(np.float32), z["size_seed0"].astype(np.int32)


# ── cluster-size curves ───────────────────────────────────────────────


def run_figures(cells_by_arch: dict, root: Path, *, m: int = M_COLOUR) -> None:
    """Seed 0's cluster sizes ranked descending, and the other seeds' atom
    counts on the same clusters; grey band = 5-95% of the rotated floor."""
    import matplotlib  # noqa: PLC0415
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"]
    for arch, cells in cells_by_arch.items():
        exps = sorted({c.expansion for c in cells})
        levels = sorted({c.level for c in cells}, key=lambda lv: lay.cell_sort_key(f"x1_{lv}"))
        fig, axs = plt.subplots(len(exps), len(levels), figsize=(4.2 * len(levels), 3.2 * len(exps)),
                                squeeze=False)
        for c in cells:
            f = root / arch / "within" / c.name / f"centroids_m{m}.npz"
            ax = axs[exps.index(c.expansion)][levels.index(c.level)]
            if not f.is_file():
                ax.set_axis_off()
                continue
            z = np.load(f)
            s0 = z["size_seed0"]
            order = np.argsort(-s0, kind="stable")
            x = np.arange(1, len(s0) + 1)
            lo, hi = np.percentile(z["size_rotated_seed0_floor"], [5, 95])
            ax.axhspan(lo, hi, color="#9a9a9a", alpha=0.22, lw=0, label="rotated seed 0 (5-95%)")
            for n, s in enumerate(c.seed_names()):
                y = s0[order] if n == 0 else z[f"size_{s}"][order]
                ax.plot(x, y, color=cols[n % len(cols)], lw=2 if n == 0 else 1.3, label=s,
                        zorder=3 if n == 0 else 2)
            ax.set_title(lay.cell_label(c.name), fontsize=10, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
        axs[0][0].legend(fontsize=7, frameon=False)
        fig.suptitle(f"{lay.ARCH_LABEL.get(arch, arch)}: atoms per seed-0 cluster (m={m}), "
                     "ranked by seed 0's size", fontsize=11)
        fig.tight_layout()
        out = root / arch / f"cluster_sizes_m{m}.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=120)
        plt.close(fig)
        _log(f"figure {out}")


# ── views ─────────────────────────────────────────────────────────────


def _gram(A, B, absolute: bool) -> np.ndarray:
    S = A @ B.T
    return np.abs(S) if absolute else S


def nn_match(A, B, absolute: bool):
    S = _gram(A, B, absolute)
    j = S.argmax(1)
    return j.astype(np.int32), S[np.arange(len(A)), j].astype(np.float32)


def hung_match(A, B, absolute: bool):
    from scipy.optimize import linear_sum_assignment  # noqa: PLC0415

    S = _gram(A, B, absolute)
    r, c = linear_sum_assignment(-S)
    fwd = np.full(len(A), -1, np.int32)
    fwd[r] = c
    bwd = np.full(len(B), -1, np.int32)
    bwd[c] = r
    cf = np.zeros(len(A), np.float32)
    cf[r] = S[r, c]
    cb = np.zeros(len(B), np.float32)
    cb[c] = S[r, c]
    return (fwd, cf), (bwd, cb)


def umap2d(blocks: list[np.ndarray]) -> list[np.ndarray]:
    import umap  # noqa: PLC0415

    Z = np.concatenate(blocks).astype(np.float32)
    E = umap.UMAP(n_components=2, metric="cosine", n_neighbors=N_NEIGHBORS, min_dist=MIN_DIST,
                  n_jobs=-1).fit_transform(Z).astype(np.float32)
    out, o = [], 0
    for b in blocks:
        out.append(E[o:o + len(b)])
        o += len(b)
    return out


def save_view(views: Path, name: str, *, kind, title, keys, labels, anchor_key, xy, clu,
              size_anchor, matches, rules, meta, bb: lay.BackboneSpec) -> None:
    d = {"kind": np.array(kind), "title": np.array(title), "keys": np.array(keys),
         "labels": np.array(labels), "anchor": np.array(anchor_key),
         "rules": np.array(rules), "size_anchor": size_anchor}
    for i in range(len(keys)):
        d[f"xy_{i}"], d[f"clu_{i}"] = xy[i], clu[i]
    for (rule, i, j), (m, c) in matches.items():
        d[f"m_{rule}_{i}_{j}"], d[f"c_{rule}_{i}_{j}"] = m, np.round(c, 3)
    np.savez_compressed(views / f"{name}.npz", **d)
    (views / f"{name}.json").write_text(json.dumps(
        {"id": name, "kind": kind, "title": title, "backbone": bb.name,
         "modality": bb.modality, **meta}, indent=1))


def _panel_view(views, name, *, kind, title, keys, labels, decs, absolutes, C0, size0,
                anchor_key, rules, meta, bb, colour_abs) -> None:
    """One joint view: assign every panel to ``C0``, embed, match all pairs."""
    labs, blocks = [], []
    for X, ab in zip(decs, colour_abs):
        lab, sgn = assign(X, C0, ab)
        labs.append(lab)
        blocks.append(X * sgn[:, None])
    xy = umap2d(blocks)
    matches = {}
    n = len(decs)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            ab = absolutes[i] or absolutes[j]
            if "hung" in rules and i < j:
                (f, cf), (bw, cb) = hung_match(decs[i], decs[j], ab)
                matches[("hung", i, j)], matches[("hung", j, i)] = (f, cf), (bw, cb)
            matches[("nn", i, j)] = nn_match(decs[i], decs[j], ab)
    save_view(views, name, kind=kind, title=title, keys=keys, labels=labels,
              anchor_key=anchor_key, xy=xy, clu=labs, size_anchor=size0,
              matches=matches, rules=rules, meta=meta, bb=bb)


def run_views(cells_by_arch: dict, root: Path, bb: lay.BackboneSpec, *, m: int = M_COLOUR) -> None:
    views = root / "views"
    views.mkdir(parents=True, exist_ok=True)
    for arch, cells in cells_by_arch.items():
        t0 = time.time()
        ab = arch in lay.ABS_ARCHES
        lab_a = lay.ARCH_LABEL.get(arch, arch)
        keys_meta = {}
        D = {}
        for c in cells:
            seeds = c.seed_names()
            D[c.name] = {s: unit_decoder(c.seeds[s]) for s in seeds}
            for s in seeds:
                keys_meta[f"{arch}/{c.name}/{s}"] = {
                    "arch": arch, "cell": c.name, "expansion": c.expansion, "level": c.level,
                    "seed": s, "step": str(c.seeds[s]), "gauge": "abs" if ab else "signed",
                    "backbone": bb.name, "modality": bb.modality,
                    "anchor": s == seeds[0], "tops": f"tops__{arch}__{c.name}.npz"}
            C0, size0 = anchor_of(root, arch, c.name, m)
            keys = [f"{arch}/{c.name}/{s}" for s in seeds]
            _panel_view(views, f"seed__{arch}__{c.name}", kind="seed",
                        title=f"{lab_a} · {bb.label} · {lay.cell_label(c.name)} — seeds",
                        keys=keys, labels=seeds, decs=[D[c.name][s] for s in seeds],
                        absolutes=[ab] * len(seeds), colour_abs=[ab] * len(seeds),
                        C0=C0, size0=size0, anchor_key=keys[0],
                        rules=["hung", "nn"] if len(seeds) > 1 else ["nn"],
                        meta={"arch": arch, "cell": c.name, "gauge": "abs" if ab else "signed",
                              "colour": f"{keys[0]} k-means, m={m} (K={len(size0)})"}, bb=bb)
            _log(f"seed view {arch}/{c.name}  {time.time() - t0:.0f}s")
        (views / f"keys__{arch}.json").write_text(json.dumps(keys_meta, indent=1))
        if len(cells) > 1:
            anchor_cell = max((c.name for c in cells), key=lay.cell_sort_key)
            C0, size0 = anchor_of(root, arch, anchor_cell, m)
            names = [c.name for c in cells]
            first = {c.name: c.seed_names()[0] for c in cells}
            keys = [f"{arch}/{n}/{first[n]}" for n in names]
            _panel_view(views, f"cross__{arch}", kind="cross",
                        title=f"{lab_a} · {bb.label} — seed 0 of {len(names)} cells",
                        keys=keys, labels=[lay.cell_label(n) for n in names],
                        decs=[D[n][first[n]] for n in names], absolutes=[ab] * len(names),
                        colour_abs=[ab] * len(names), C0=C0, size0=size0,
                        anchor_key=f"{arch}/{anchor_cell}/{first[anchor_cell]}", rules=["nn"],
                        meta={"arch": arch, "gauge": "abs" if ab else "signed",
                              "colour": f"{arch}/{anchor_cell} seed 0 k-means, m={m} (K={len(size0)})"},
                        bb=bb)
            _log(f"cross view {arch}  {time.time() - t0:.0f}s")


def run_archviews(cells_by_arch: dict, root: Path, bb: lay.BackboneSpec, *, m: int = M_COLOUR) -> None:
    views = root / "views"
    views.mkdir(parents=True, exist_ok=True)
    cell_names = sorted({c.name for cs in cells_by_arch.values() for c in cs}, key=lay.cell_sort_key)
    for cell in cell_names:
        t0 = time.time()
        have = {a: next(c for c in cs if c.name == cell)
                for a, cs in cells_by_arch.items() if any(c.name == cell for c in cs)}
        if len(have) < 2:
            continue
        archs = list(have)
        ref = lay.REFERENCE_ARCH if lay.REFERENCE_ARCH in have else archs[0]
        first = {a: have[a].seed_names()[0] for a in archs}
        decs = [unit_decoder(have[a].seeds[first[a]]) for a in archs]
        C0, size0 = anchor_of(root, ref, cell, m)
        ref_abs = ref in lay.ABS_ARCHES
        _panel_view(views, f"arch__{cell}", kind="arch",
                    title=f"{bb.label} · {lay.cell_label(cell)} — seed 0 of {len(archs)} architectures",
                    keys=[f"{a}/{cell}/{first[a]}" for a in archs],
                    labels=[lay.ARCH_LABEL.get(a, a) for a in archs], decs=decs,
                    absolutes=[a in lay.ABS_ARCHES for a in archs],
                    colour_abs=[ref_abs or a in lay.ABS_ARCHES for a in archs],
                    C0=C0, size0=size0, anchor_key=f"{ref}/{cell}/{first[ref]}", rules=["nn"],
                    meta={"cell": cell, "gauge": "|cos| for any pair with iSAE, else signed",
                          "colour": f"{ref}/{cell} seed 0 k-means, m={m} (K={len(size0)})"}, bb=bb)
        _log(f"arch view {cell}  {time.time() - t0:.0f}s")
