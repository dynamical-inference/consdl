"""Duplicate-atom clusters of one dictionary, at several cosine thresholds.

``dict_metrics`` counts an atom as duplicated (#dup) when it falls in a
complete-linkage cluster of more than one atom at threshold ``t``. This is
computed exactly in two steps:

1. Edge list. Off-diagonal cosines of a large dictionary concentrate near
   zero (std ~ 1/sqrt(d)), so a low cut such as ``cos > 0.1`` keeps a tiny
   fraction of pairs. They are extracted once; each threshold is then a
   mask over that edge list.
2. Complete linkage within connected components. Connected components
   (single linkage) chain; each component is split so every pair inside a
   cluster clears ``t``.
"""
from __future__ import annotations

import numpy as np


def edges(G: np.ndarray, t_min: float):
    """Upper-triangle edges of ``G`` above ``t_min`` as ``(i, j, w)``.

    ``G`` is used as given: the caller applies the sign convention. With
    signed cosines, anti-parallel pairs are not edges, since for a
    non-negative code an atom and its negation are different features.
    """
    iu = np.triu_indices(G.shape[0], k=1)
    w = G[iu]
    keep = w > float(t_min)
    return iu[0][keep], iu[1][keep], w[keep]


def labels_at(edge_i, edge_j, edge_w, n: int, t: float):
    """Connected components of the ``> t`` subgraph of an edge list."""
    from scipy.sparse import coo_matrix  # noqa: PLC0415
    from scipy.sparse.csgraph import connected_components  # noqa: PLC0415

    m = edge_w > float(t)
    if not m.any():
        return np.arange(n), n
    g = coo_matrix(
        (np.ones(int(m.sum()), dtype=np.int8), (edge_i[m], edge_j[m])),
        shape=(n, n))
    n_comp, labels = connected_components(g, directed=False)
    return labels, int(n_comp)


#: How duplicate groups are formed.
#:
#: Connected components are single linkage: ``A-B`` and ``B-C`` above tau
#: put A and C in one group however dissimilar A and C are, and for
#: duplicate-heavy dictionaries a component can contain near anti-parallel
#: atoms. Complete linkage requires every pair inside a group to clear the
#: cut. It only refines the components (atoms in different components are
#: below tau by construction), so it runs per component and never needs
#: the full Gram matrix.
LINKAGE_DEFAULT: str = "complete"


def complete_refine(U: np.ndarray, labels: np.ndarray, n_lab: int,
                    t: float, *, absolute: bool) -> tuple[np.ndarray, int]:
    """Split each component so every pair inside it clears ``t``.

    ``U`` is the unit-row dictionary the labels index; each component's
    similarity submatrix is recomputed from its rows, which avoids holding
    the full Gram matrix across the threshold loop.

    Returns ``(labels, n_labels)`` in the same shape ``labels_at`` does,
    so callers swap one for the other.
    """
    from scipy.cluster.hierarchy import fcluster, linkage  # noqa: PLC0415
    from scipy.spatial.distance import squareform          # noqa: PLC0415

    out = labels.copy()
    nxt = int(n_lab)
    order = np.argsort(labels, kind="stable")
    bounds = np.flatnonzero(np.diff(labels[order])) + 1
    for m in np.split(order, bounds):
        if m.size < 3:
            continue          # a pair is already a complete cluster
        S = U[m] @ U[m].T
        if absolute:
            S = np.abs(S)
        D = np.clip(1.0 - S, 0.0, None)
        np.fill_diagonal(D, 0.0)
        sub = fcluster(
            linkage(squareform(D, checks=False), method="complete"),
            t=1.0 - float(t), criterion="distance")
        pieces = np.unique(sub)
        if pieces.size < 2:
            continue          # the component was already complete
        for k in pieces[1:]:  # the first keeps the component's own id
            out[m[sub == k]] = nxt
            nxt += 1
    return out, nxt
