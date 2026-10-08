"""Build everything the viewers read for one or more backbones.

Reads a checkpoint tree in the release layout
(``<root>/<modality>/<backbone>_layer<L>/<arch>/x<E>/<level>/seed<S>/``, as
``consdl.cli.sweep`` writes under ``var/checkpoints`` or
``consdl.io.hub.download`` fetches) and writes
``var/results/viewers/<modality>/<backbone>_layer<L>/``:

  kmeans     seed-0 spherical k-means per arch × cell (the colour anchors)
  figures    cluster-size curves per arch
  views      per arch: seed views (one per cell) and the cross-cell view
  archviews  per cell: the cross-architecture view
  tops       top-100 inputs per latent (GPU): ImageNet val images for vision
             (plus per-image footprints), FineWeb token contexts for LMs

``kmeans``/``views``/``archviews`` run on CPU (UMAP, Hungarian); ``tops``
wants a GPU. Stages skip outputs that exist unless ``--force``.

Usage::

    # CLIP and Pythia, x16 only, every arch and level, 3 seeds
    python -m consdl.cli.viewer_precompute --root var/release \\
        --backbones clip-vit-b-16_layer10 pythia-160m_layer8 --expansions 16 --seeds 3

    # one arch at a time (e.g. a job array), then the cross-arch views
    python -m consdl.cli.viewer_precompute --root var/release --backbones clip-vit-b-16 \\
        --archs topk --stages kmeans figures views tops
    python -m consdl.cli.viewer_precompute --root var/release --backbones clip-vit-b-16 \\
        --stages archviews
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

STAGES = ("kmeans", "figures", "views", "archviews", "tops")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", type=Path, required=True,
                   help="checkpoint tree: the release root, a modality dir or one backbone dir")
    p.add_argument("--backbones", nargs="+", default=None, metavar="NAME",
                   help="backbone dir names or short names (default: every one under --root)")
    p.add_argument("--archs", nargs="+", default=None)
    p.add_argument("--expansions", nargs="+", type=int, default=None)
    p.add_argument("--levels", nargs="+", default=None, help="e.g. l0_32 l0_64 l0_128")
    p.add_argument("--seeds", type=int, default=None, help="use the first N seeds of each cell")
    p.add_argument("--stages", nargs="+", default=list(STAGES), choices=STAGES)
    p.add_argument("--out-root", type=Path, default=None,
                   help="default: <repo>/var/results/viewers")
    p.add_argument("--m", type=int, default=48, help="atoms per cluster of the colour anchors")
    p.add_argument("--restarts", type=int, default=3, help="k-means restarts")
    p.add_argument("--images-per-class", type=int, default=None,
                   help="vision top-input pool: N ImageNet val images per class (default: all 50)")
    p.add_argument("--lm-seqs", type=int, default=2048,
                   help="LM top-input pool: packed FineWeb sequences of 1024 tokens")
    p.add_argument("--batch-size", type=int, default=64, help="images per backbone batch")
    p.add_argument("--no-footprints", action="store_true",
                   help="vision: skip the per-image footprints (fp cos in the view)")
    p.add_argument("--gpu", type=int, default=None, help="CUDA device index")
    p.add_argument("--force", action="store_true", help="recompute existing outputs")
    return p.parse_args(argv)


def main(argv=None) -> None:
    a = parse_args(argv)
    if a.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch  # noqa: PLC0415

    from consdl.viewers import layout as lay  # noqa: PLC0415
    from consdl.viewers import precompute as pre  # noqa: PLC0415
    from consdl.viewers import tops  # noqa: PLC0415

    out_root = a.out_root or lay.DEFAULT_OUT_ROOT
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    bbs = lay.find_backbone_dirs(a.root, a.backbones)
    if not bbs:
        raise SystemExit(f"no <backbone>_layer<L> dir under {a.root} matches {a.backbones}")
    for bb_dir in bbs:
        t0 = time.time()
        bb = lay.backbone_spec(bb_dir.name)
        cells = lay.discover(bb_dir, archs=a.archs, expansions=a.expansions, levels=a.levels,
                             n_seeds=a.seeds)
        if not cells:
            print(f"[skip] {bb_dir}: no cells match", flush=True)
            continue
        root = lay.out_dir(bb, out_root)
        pre.write_backbone_json(bb, root)
        n = sum(len(c) for c in cells.values())
        print(f"== {bb.name} ({bb.modality}): {len(cells)} archs, {n} cells -> {root}", flush=True)
        if "kmeans" in a.stages:
            pre.run_kmeans(cells, root, m=a.m, restarts=a.restarts, device=dev, force=a.force)
        if "figures" in a.stages:
            pre.run_figures(cells, root, m=a.m)
        if "views" in a.stages:
            pre.run_views(cells, root, bb, m=a.m)
        if "archviews" in a.stages:
            pre.run_archviews(cells, root, bb, m=a.m)
        if "tops" in a.stages:
            if dev == "cpu":
                print("[warn] tops on CPU: this is slow; use a GPU node", flush=True)
            if bb.modality == "vision":
                tops.vision_tops(cells, root, bb, per_class=a.images_per_class,
                                 batch_size=a.batch_size, device=dev,
                                 footprints=not a.no_footprints, force=a.force)
            else:
                tops.lm_tops(cells, root, bb, n_seqs=a.lm_seqs, device=dev, force=a.force)
        print(f"== {bb.name} done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
