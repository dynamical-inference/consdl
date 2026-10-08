"""Region carve of one input: patches (or tokens) clustered by how an SAE
decomposes them.

Each patch's ``|a|`` profile is floored at ``eps * max`` (relative, per
patch) and L1-normalised; the distance between two patches is half the L1
distance of their profiles (``1 - overlap`` of activation mass). One
average-linkage tree per panel serves every k (each k is a ``maxclust``
cut). Ids are relabelled onto panel 0 for display only; ARI is computed on
the unmatched labels.

Text adds two things: position 0 (the attention sink) can be left out, and
``contiguous=True`` lets only neighbouring tokens merge, so every region
is a span.
"""
from __future__ import annotations

import numpy as np

EPS_REL = 0.01                      # relative support floor, |a| >= eps * max per patch
K_RANGE = tuple(range(2, 21))
NOISE = -1
HUE_MIN_SEP = 30.0                  # touching regions differ by at least this hue


# ── clustering ────────────────────────────────────────────────────────


def profiles(A, *, eps_rel: float = EPS_REL) -> np.ndarray:
    """``(P, m)`` L1-normalised ``|a|`` profiles over the latents active anywhere."""
    A = np.abs(np.asarray(A, dtype=np.float64))
    mx = A.max(axis=1, keepdims=True)
    A[A < float(eps_rel) * mx] = 0.0
    p = A / np.maximum(A.sum(axis=1, keepdims=True), 1e-12)
    return p[:, np.flatnonzero(p.any(axis=0))]


def _mass_distance(A, eps_rel: float) -> np.ndarray:
    """Condensed ``0.5 * L1`` distances between patch profiles."""
    from scipy.spatial.distance import pdist  # noqa: PLC0415
    return 0.5 * pdist(profiles(A, eps_rel=eps_rel), metric="cityblock")


def _relabel_by_size(lab) -> np.ndarray:
    """Labels renumbered largest-first (stable); ``NOISE`` stays."""
    lab = np.asarray(lab)
    pos = lab >= 0
    out = np.full(lab.size, NOISE, dtype=np.int64)
    if pos.any():
        uniq, inv = np.unique(lab[pos], return_inverse=True)
        order = np.argsort(-np.bincount(inv, minlength=uniq.size), kind="stable")
        remap = np.empty(order.size, dtype=np.int64)
        remap[order] = np.arange(order.size)
        out[pos] = remap[inv]
    return out


def _cut(Z, k: int) -> np.ndarray:
    from scipy.cluster.hierarchy import fcluster  # noqa: PLC0415
    return _relabel_by_size(fcluster(Z, t=int(k), criterion="maxclust") - 1)


def _tree(A, *, eps_rel: float, connectivity=None):
    """Average-linkage tree (scipy linkage matrix) over the rows of ``A``."""
    from scipy.cluster.hierarchy import linkage  # noqa: PLC0415

    d = _mass_distance(A, eps_rel)
    if connectivity is None:
        return linkage(d, method="average")
    from scipy.spatial.distance import squareform  # noqa: PLC0415
    from sklearn.cluster import AgglomerativeClustering  # noqa: PLC0415

    m = AgglomerativeClustering(n_clusters=None, distance_threshold=0, linkage="average",
                                metric="precomputed", connectivity=connectivity,
                                compute_full_tree=True).fit(squareform(d))
    n = len(m.labels_)
    counts = np.zeros(m.children_.shape[0])
    for i, merge in enumerate(m.children_):
        counts[i] = sum(1 if c < n else counts[c - n] for c in merge)
    # connectivity can make merge heights non-monotone; maxclust needs them monotone
    h = np.maximum.accumulate(np.asarray(m.distances_, dtype=np.float64))
    return np.column_stack([m.children_, h, counts]).astype(np.float64)


def match_to_reference(labels, ref) -> np.ndarray:
    """Relabel ``labels`` onto ``ref``'s ids by greedy one-to-one Jaccard (display only).

    Clusters without a free counterpart get fresh ids past ``ref``'s range.
    """
    lab, r = np.asarray(labels), np.asarray(ref)
    keep = lab >= 0
    if not keep.any():
        return np.full(lab.size, NOISE, dtype=np.int64)
    nl, nr = int(lab[keep].max()) + 1, int(max(r.max(), 0)) + 1
    ls = [np.flatnonzero(lab == c) for c in range(nl)]
    rs = [np.flatnonzero(r == c) for c in range(nr)]
    J = np.zeros((nl, nr))
    for i, a in enumerate(ls):
        for j, b in enumerate(rs):
            u = len(np.union1d(a, b))
            J[i, j] = len(np.intersect1d(a, b)) / u if u else 0.0
    out = np.full(nl, -1, dtype=np.int64)
    taken: set[int] = set()
    for flat in np.argsort(-J, axis=None):
        i, j = int(flat // nr), int(flat % nr)
        if J[i, j] > 0 and out[i] < 0 and j not in taken:
            out[i] = j
            taken.add(j)
    nxt = nr
    for i in range(nl):
        if out[i] < 0:
            out[i], nxt = nxt, nxt + 1
    res = np.full(lab.size, NOISE, dtype=np.int64)
    res[keep] = out[lab[keep]]
    return res


# ── colours ───────────────────────────────────────────────────────────


def _hsl_hex(h: float, s: float, ll: float) -> str:
    h = float(h) % 360.0
    c = (1.0 - abs(2.0 * ll - 1.0)) * s
    x = c * (1.0 - abs((h / 60.0) % 2.0 - 1.0))
    m = ll - c / 2.0
    r, g, b = [(c, x, 0.0), (x, c, 0.0), (0.0, c, x),
               (0.0, x, c), (x, 0.0, c), (c, 0.0, x)][int(h // 60) % 6]
    return "#" + "".join(f"{int(round((v + m) * 255)):02x}" for v in (r, g, b))


def _separate(hue: np.ndarray, adj: set, min_sep: float) -> np.ndarray:
    """Push touching regions apart in hue until they differ by ``min_sep``."""
    def sep(x, y):
        d = abs((x - y) % 360.0)
        return min(d, 360.0 - d)

    for _ in range(64):
        worst = 0.0
        for u, v in adj:
            d = sep(hue[u], hue[v])
            if d < min_sep:
                push = (min_sep - d) / 2.0 + 1e-6
                s = 1.0 if (hue[v] - hue[u]) % 360.0 <= 180.0 else -1.0
                hue[u] -= s * push
                hue[v] += s * push
                worst = max(worst, min_sep - d)
        if worst == 0.0:
            break
    return hue


def patch_colors(labels, *, side: int, min_sep: float = HUE_MIN_SEP) -> list[str]:
    """One colour per region: hue = angle of its centroid about the image
    centre, saturation/lightness from its radius."""
    lab = np.asarray(labels)
    n = int(lab.max()) + 1 if lab.size and lab.max() >= 0 else 0
    if n <= 0:
        return []
    rr, cc = np.divmod(np.arange(lab.size), side)
    mid = (side - 1) / 2.0
    rmax = np.hypot(mid, mid) or 1.0
    hue, sat, lig = np.zeros(n), np.full(n, 0.55), np.full(n, 0.62)
    for c in range(n):
        m = lab == c
        if not m.any():
            continue
        dy, dx = rr[m].mean() - mid, cc[m].mean() - mid
        hue[c] = (np.degrees(np.arctan2(dy, dx)) + 360.0) % 360.0
        rad = float(min(np.hypot(dy, dx) / rmax, 1.0))
        sat[c], lig[c] = 0.62 + 0.33 * rad, 0.56 - 0.14 * rad
    g = lab.reshape(side, side)
    adj = set()
    for a, b in ((g[:, :-1], g[:, 1:]), (g[:-1, :], g[1:, :])):
        d = a != b
        for u, v in zip(a[d].ravel(), b[d].ravel()):
            if u >= 0 and v >= 0:
                adj.add((min(int(u), int(v)), max(int(u), int(v))))
    hue = _separate(hue, adj, min_sep)
    return [_hsl_hex(hue[c], sat[c], lig[c]) for c in range(n)]


def span_colors(labels, *, min_sep: float = HUE_MIN_SEP) -> list[str]:
    """One colour per token region: hue = its mean position along the text."""
    lab = np.asarray(labels)
    keep = lab >= 0
    if not keep.any():
        return []
    n = int(lab[keep].max()) + 1
    pos = np.arange(lab.size, dtype=np.float64)
    span = max(lab.size - 1, 1)
    hue = np.array([300.0 * pos[lab == c].mean() / span if (lab == c).any() else 0.0
                    for c in range(n)])
    adj = {(min(int(a), int(b)), max(int(a), int(b)))
           for a, b in zip(lab[:-1], lab[1:]) if a != b and a >= 0 and b >= 0}
    hue = _separate(hue, adj, min_sep)
    return [_hsl_hex(hue[c], 0.70, 0.62) for c in range(n)]


# ── per-panel carves ──────────────────────────────────────────────────


def patch_labels(codes, *, ks=K_RANGE, eps_rel: float = EPS_REL) -> list[dict[int, np.ndarray]]:
    """``[{k: (P,) labels}]``, one per panel; ``codes`` are ``(P+1, d_sae)``
    with CLS as row 0 (dropped here)."""
    out = []
    for A in codes:
        Z = _tree(np.asarray(A)[1:], eps_rel=eps_rel)
        out.append({int(k): _cut(Z, k) for k in ks})
    return out


def _chain(n: int):
    from scipy.sparse import csr_matrix, diags  # noqa: PLC0415
    if n < 2:
        return csr_matrix((n, n))
    one = np.ones(n - 1)
    return diags([one, one], [-1, 1], shape=(n, n), format="csr")


def token_carve(A, *, ks=K_RANGE, eps_rel: float = EPS_REL, contiguous: bool = False,
                exclude_first: bool = True) -> dict[int, np.ndarray]:
    """``{k: (T,) labels}`` for one panel's ``(T, d_sae)`` codes."""
    A = np.asarray(A)
    T = A.shape[0]
    off = 1 if exclude_first else 0
    if T - off < 2:
        raise ValueError(f"need at least 2 tokens to carve, got {T - off}")
    Z = _tree(A[off:], eps_rel=eps_rel,
              connectivity=_chain(T - off) if contiguous else None)
    out = {}
    for k in ks:
        lab = np.full(T, NOISE, dtype=np.int64)
        lab[off:] = _cut(Z, min(int(k), T - off))
        out[int(k)] = lab
    return out


def token_labels(codes, *, ks=K_RANGE, eps_rel: float = EPS_REL,
                 contiguous: bool = False, exclude_first: bool = True) -> dict:
    """Every panel carved alone at every k, relabelled onto panel 0.

    Returns ``{ks, contiguous, exclude_first, labels: [{k: list}],
    colors: [{k: list}], ari: {k: {"i|j": float}}}``.
    """
    from sklearn.metrics import adjusted_rand_score  # noqa: PLC0415

    codes = [np.asarray(A) for A in codes]
    if len({A.shape[0] for A in codes}) != 1:
        raise ValueError("panels disagree on the token count")
    raw = [token_carve(A, ks=ks, eps_rel=eps_rel, contiguous=contiguous,
                       exclude_first=exclude_first) for A in codes]
    off = 1 if exclude_first else 0
    out = {"ks": [int(k) for k in ks], "contiguous": bool(contiguous),
           "exclude_first": bool(exclude_first),
           "labels": [{} for _ in codes], "colors": [{} for _ in codes], "ari": {}}
    for k in ks:
        k = int(k)
        ref = raw[0][k]
        for n in range(len(codes)):
            lab = ref if n == 0 else match_to_reference(raw[n][k], ref)
            out["labels"][n][k] = np.asarray(lab, dtype=np.int64).tolist()
            out["colors"][n][k] = span_colors(lab)
        out["ari"][k] = {f"{i}|{j}": round(float(adjusted_rand_score(raw[i][k][off:], raw[j][k][off:])), 4)
                         for i in range(len(codes)) for j in range(i + 1, len(codes))}
    return out
