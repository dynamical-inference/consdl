"""Build a synthetic ground-truth dictionary ``D_gt``.

Geometries (default ``d=768`` rows, ``d_sae=3072`` unit-norm columns):

- ``laplacian_er``  : low-frequency eigenvectors of the graph Laplacian of a
                       sparse Erdős–Rényi graph on ``d_sae`` nodes. Produces a
                       spectral embedding where adjacent nodes end up close in
                       R^d — mean cosine near the frame bound, but a heavy
                       right tail from graph neighbours.
- ``laplacian_sbm`` : same construction on a community / stochastic-block graph,
                       giving block structure and moderate within-block cosine.
- ``laplacian_er_cofire`` / ``laplacian_sbm_cofire`` : the same ``D_gt`` as
                       the base geometry under the same seed, plus the ``d``
                       smallest normalised-Laplacian eigenvalues saved as
                       ``corr_eigvals``. These drive the dataset's
                       ``sampling="correlated"`` mode (a Gaussian copula whose
                       covariance uses the Laplacian's pseudo-inverse spectrum).
- ``gaussian``      : i.i.d. columns from ``N(0, I_d / d)``, unit-normalised.
                       No forced near-neighbours; moderate typical cosine,
                       light tail.
- ``gaussian_cofire``: same ``D_gt`` as ``gaussian`` plus a random low-rank
                       co-firing factor ``F`` (shape ``(d_sae, cofire_rank)``,
                       entries ``N(0, cofire_scale²)``), following the
                       sae_lens ``LowRankCorrelationConfig`` recipe: the
                       Gaussian copula uses ``F`` directly (no graph, no
                       dictionary tie-in), so co-firing groups are
                       arbitrary random directions in latent space rather
                       than graph neighbours.  Defaults
                       ``cofire_rank=50, cofire_scale=0.1`` give a per-
                       latent variance budget ``E[‖F[i]‖²] = rank·scale²
                       ≈ 0.5`` that matches ``laplacian_er_cofire`` at
                       ``corr_strength≈0.7``.
- ``frame``         : random partial orthonormal frame (polar factor of a
                       ``(d, d_sae)`` Gaussian, column-normalised). Column
                       cosines concentrate tightly at the frame bound
                       ``sqrt((d_sae-d)/(d(d_sae-1)))`` — the theoretical
                       upper bound on pairwise orthogonality for ``d_sae``
                       unit vectors in R^d.

Usage
-----
    python -m consdl.cli.build_dictionary --geometry laplacian_er --out D_gt_laplacian_er.pt
    python -m consdl.cli.build_dictionary --geometry laplacian_sbm --out D_gt_laplacian_sbm.pt --n-blocks 16
    python -m consdl.cli.build_dictionary --geometry gaussian_cofire --out D_gt_gaussian_cofire.pt

To build every geometry at the default location, use
``python -m consdl.cli.build_all_dictionaries``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _laplacian_eigvecs(A: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """Low-frequency normalised-Laplacian eigenpairs.

    Returns ``(U_d, lam_d)`` — ``U_d`` shape ``(d_sae, d)`` (the ``d``
    smallest-eigenvalue eigenvectors, columns ordered by eigenvalue
    ascending), ``lam_d`` shape ``(d,)`` (matching eigenvalues, also
    ascending). The eigenvalues are used only by the ``_cofire``
    geometries.
    """
    deg = A.sum(axis=1)
    deg[deg == 0] = 1.0
    d_inv_sqrt = 1.0 / np.sqrt(deg)
    # L_sym = I - D^{-1/2} A D^{-1/2}
    L = np.eye(A.shape[0]) - (d_inv_sqrt[:, None] * A * d_inv_sqrt[None, :])
    # Symmetric eigendecomp; eigvals ascending.
    eigvals, U = np.linalg.eigh(L)
    return U[:, :d], eigvals[:d]


def _er_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    A = (rng.random((n, n)) < p).astype(np.float64)
    A = np.triu(A, k=1)
    A = A + A.T
    return A


def _sbm_adjacency(n: int, n_blocks: int, p_in: float, p_out: float,
                   rng: np.random.Generator) -> np.ndarray:
    """Symmetric stochastic-block-model adjacency."""
    blocks = np.array_split(np.arange(n), n_blocks)
    A = np.zeros((n, n), dtype=np.float64)
    for bi, Bi in enumerate(blocks):
        for bj, Bj in enumerate(blocks):
            if bj < bi:
                continue
            p = p_in if bi == bj else p_out
            sub = (rng.random((len(Bi), len(Bj))) < p).astype(np.float64)
            if bi == bj:
                sub = np.triu(sub, k=1); sub = sub + sub.T
            A[np.ix_(Bi, Bj)] = sub
            if bi != bj:
                A[np.ix_(Bj, Bi)] = sub.T
    return A


def build_D_T(
    geometry: str,
    d: int = 768,
    d_sae: int = 3072,
    seed: int = 0,
    er_avg_degree: float = 10.0,
    n_blocks: int = 16,
    sbm_p_in: float = 0.20,
    sbm_p_out: float = 0.01,
    cofire_rank: int = 50,
    cofire_scale: float = 0.1,
) -> tuple[torch.Tensor, np.ndarray | None, np.ndarray | None]:
    """Return ``(D_gt, corr_eigvals, corr_factor)``.

    * ``D_gt`` is ``(d, d_sae)`` with unit-norm columns.
    * ``corr_eigvals`` is the ``(d,)`` ascending vector of the
      low-frequency normalised-Laplacian eigenvalues — populated only
      for the ``laplacian_*_cofire`` geometries (consumed by the
      dataset's ``sampling="correlated"`` mode); ``None`` everywhere
      else.
    * ``corr_factor`` is a precomputed low-rank co-firing factor of
      shape ``(d_sae, cofire_rank)`` — populated only for
      ``gaussian_cofire``. The dataset's ``sampling="correlated"``
      mode uses it directly as the copula factor ``F`` (scaled by
      ``corr_strength``).

    Each ``_cofire`` geometry reproduces its base geometry's ``D_gt``
    exactly under the same seed (same RNG consumption order).
    """
    rng = np.random.default_rng(seed)
    corr_eigvals: np.ndarray | None = None
    corr_factor: np.ndarray | None = None

    if geometry == "laplacian_er":
        p = er_avg_degree / d_sae
        A = _er_adjacency(d_sae, p, rng)
        U, _ = _laplacian_eigvecs(A, d)         # eigenvalues discarded
        D = U.T                                 # (d, d_sae)

    elif geometry == "laplacian_sbm":
        A = _sbm_adjacency(d_sae, n_blocks, sbm_p_in, sbm_p_out, rng)
        U, _ = _laplacian_eigvecs(A, d)
        D = U.T

    elif geometry == "laplacian_er_cofire":
        # Same RNG-consumption order as ``laplacian_er`` → identical D_gt.
        p = er_avg_degree / d_sae
        A = _er_adjacency(d_sae, p, rng)
        U, lam = _laplacian_eigvecs(A, d)
        D = U.T
        corr_eigvals = lam.astype(np.float32)

    elif geometry == "laplacian_sbm_cofire":
        # Same RNG-consumption order as ``laplacian_sbm`` → identical D_gt.
        A = _sbm_adjacency(d_sae, n_blocks, sbm_p_in, sbm_p_out, rng)
        U, lam = _laplacian_eigvecs(A, d)
        D = U.T
        corr_eigvals = lam.astype(np.float32)

    elif geometry == "gaussian":
        D = rng.standard_normal((d, d_sae)) / np.sqrt(d)

    elif geometry == "gaussian_cofire":
        # Same D_gt draw as ``gaussian``. The co-firing factor
        # F ~ N(0, scale²·I) at rank ``cofire_rank`` is drawn afterwards
        # and, unlike the ``laplacian_*_cofire`` branches, is unrelated to
        # D_gt's columns.
        D = rng.standard_normal((d, d_sae)) / np.sqrt(d)
        r = max(1, int(cofire_rank))
        s_F = float(cofire_scale)
        corr_factor = (s_F * rng.standard_normal((d_sae, r))).astype(np.float32)

    elif geometry == "frame":
        # Polar factor of a random (d, d_sae) Gaussian — its rows are an
        # orthonormal system in R^{d_sae}, so columns are a random partial
        # orthonormal frame. After per-column unit normalisation below, all
        # pairwise cosines concentrate tightly near the frame bound.
        A = rng.standard_normal((d, d_sae))
        U, _, Vt = np.linalg.svd(A, full_matrices=False)  # U: (d,d), Vt: (d, d_sae)
        D = U @ Vt  # (d, d_sae), D @ D.T = I_d

    else:
        raise ValueError(f"unknown geometry: {geometry}")

    # Unit-normalise columns.
    D = D / (np.linalg.norm(D, axis=0, keepdims=True) + 1e-12)
    return torch.from_numpy(D).float(), corr_eigvals, corr_factor


def _report_cosines(D: torch.Tensor, n_pairs: int = 20000, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    n = D.shape[1]
    i = rng.integers(0, n, n_pairs); j = rng.integers(0, n, n_pairs)
    mask = i != j
    i, j = i[mask], j[mask]
    c = (D[:, i] * D[:, j]).sum(0).abs().numpy()
    print(f"  |cos| summary on {len(c)} pairs: "
          f"mean={c.mean():.4f}  p50={np.median(c):.4f}  "
          f"p95={np.quantile(c, 0.95):.4f}  max={c.max():.4f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--geometry", required=True,
                   choices=["laplacian_er", "laplacian_sbm",
                            "laplacian_er_cofire", "laplacian_sbm_cofire",
                            "gaussian", "gaussian_cofire", "frame"])
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--d-sae", type=int, default=3072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--er-avg-degree", type=float, default=10.0)
    p.add_argument("--n-blocks", type=int, default=16)
    p.add_argument("--sbm-p-in", type=float, default=0.20)
    p.add_argument("--sbm-p-out", type=float, default=0.01)
    p.add_argument("--cofire-rank", type=int, default=50,
                   help="(gaussian_cofire) rank of the random low-rank "
                        "co-firing factor F.  Default 50 matches the "
                        "per-latent variance budget of laplacian_er_cofire "
                        "at corr_strength≈0.7 when paired with "
                        "--cofire-scale 0.1.")
    p.add_argument("--cofire-scale", type=float, default=0.1,
                   help="(gaussian_cofire) entry-wise std of F: "
                        "F ~ N(0, scale²·I).  See --cofire-rank.")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    D, corr_eigvals, corr_factor = build_D_T(
        args.geometry, d=args.d, d_sae=args.d_sae, seed=args.seed,
        er_avg_degree=args.er_avg_degree, n_blocks=args.n_blocks,
        sbm_p_in=args.sbm_p_in, sbm_p_out=args.sbm_p_out,
        cofire_rank=args.cofire_rank, cofire_scale=args.cofire_scale,
    )

    print(f"[build_dictionary] geometry={args.geometry}  D_gt shape={tuple(D.shape)}")
    _report_cosines(D)

    payload = {"D_gt": D, "geometry": args.geometry, "seed": args.seed,
               "d": args.d, "d_sae": args.d_sae}
    # ``corr_eigvals`` is added only for ``laplacian_*_cofire``;
    # ``corr_factor`` only for ``gaussian_cofire``.
    if corr_eigvals is not None:
        payload["corr_eigvals"] = torch.from_numpy(corr_eigvals).float()
        print(f"[build_dictionary] corr_eigvals: shape={tuple(payload['corr_eigvals'].shape)}  "
              f"λ_min={float(corr_eigvals.min()):.4g}  λ_max={float(corr_eigvals.max()):.4g}")
    if corr_factor is not None:
        payload["corr_factor"] = torch.from_numpy(corr_factor).float()
        payload["cofire_rank"] = int(args.cofire_rank)
        payload["cofire_scale"] = float(args.cofire_scale)
        F = corr_factor
        rows = (F ** 2).sum(axis=1)
        # Mean s²·rows at corr_strength=1; gives the per-latent
        # variance budget the dataset's δ-clamp will see.
        print(f"[build_dictionary] corr_factor: shape={tuple(F.shape)}  "
              f"rank={args.cofire_rank}  scale={args.cofire_scale}  "
              f"E[‖F[i]‖²]={float(rows.mean()):.4f} "
              f"(target≈{args.cofire_rank * args.cofire_scale ** 2:.4f})")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, args.out)
    print(f"[build_dictionary] saved → {args.out}")


if __name__ == "__main__":
    main()
