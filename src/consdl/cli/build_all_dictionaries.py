"""Build all ground-truth dictionaries at the default data location.

Usage: ``python -m consdl.cli.build_all_dictionaries [--data-dir DIR]``

Default output directory: ``<repo>/data/synthetic/dictionaries/``.

Files written:
    D_gt_laplacian_er_seed0.pt       (sparse ER Laplacian embedding)
    D_gt_laplacian_sbm_seed0.pt      (block-structured SBM Laplacian embedding)
    D_gt_gaussian_seed0.pt           (i.i.d. Gaussian columns)
    D_gt_gaussian_cofire_seed0.pt    (gaussian + random low-rank cofire factor)
    D_gt_frame_seed0.pt              (random partial orthonormal frame — upper bound)

Existing files are skipped unless ``--overwrite`` is set.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from consdl.cli.build_dictionary import build_D_T, _report_cosines

GEOMETRIES = ["laplacian_er", "laplacian_sbm", "gaussian", "gaussian_cofire", "frame"]


def default_data_dir() -> Path:
    from consdl.io.paths import REPO_ROOT  # noqa: PLC0415
    return REPO_ROOT / "data" / "synthetic" / "dictionaries"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", type=Path, default=default_data_dir())
    p.add_argument("--d", type=int, default=768)
    p.add_argument("--d-sae", type=int, default=3072)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--geometries", nargs="+", default=GEOMETRIES, choices=GEOMETRIES)
    p.add_argument("--er-avg-degree", type=float, default=10.0)
    p.add_argument("--n-blocks", type=int, default=16)
    p.add_argument("--sbm-p-in", type=float, default=0.20)
    p.add_argument("--sbm-p-out", type=float, default=0.01)
    p.add_argument("--cofire-rank", type=int, default=50)
    p.add_argument("--cofire-scale", type=float, default=0.1)
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    args.data_dir.mkdir(parents=True, exist_ok=True)
    print(f"[build_all_dictionaries] out dir: {args.data_dir}")

    for geom in args.geometries:
        out = args.data_dir / f"D_gt_{geom}_seed{args.seed}.pt"
        if out.exists() and not args.overwrite:
            print(f"  [skip] {out.name} already exists")
            continue
        print(f"\n[build] geometry={geom}")
        D, corr_eigvals, corr_factor = build_D_T(
            geom, d=args.d, d_sae=args.d_sae, seed=args.seed,
            er_avg_degree=args.er_avg_degree, n_blocks=args.n_blocks,
            sbm_p_in=args.sbm_p_in, sbm_p_out=args.sbm_p_out,
            cofire_rank=args.cofire_rank, cofire_scale=args.cofire_scale,
        )
        print(f"  shape={tuple(D.shape)}")
        _report_cosines(D)
        payload = {"D_gt": D, "geometry": geom, "seed": args.seed,
                   "d": args.d, "d_sae": args.d_sae}
        if corr_eigvals is not None:
            payload["corr_eigvals"] = torch.from_numpy(corr_eigvals).float()
        if corr_factor is not None:
            payload["corr_factor"] = torch.from_numpy(corr_factor).float()
            payload["cofire_rank"] = int(args.cofire_rank)
            payload["cofire_scale"] = float(args.cofire_scale)
        torch.save(payload, out)
        print(f"  saved → {out}")


if __name__ == "__main__":
    main()
