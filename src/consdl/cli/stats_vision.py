"""Compute per-checkpoint SAE stats (mean_act, sparsity, top_idx, ...) for a vision SAE.

Wraps ``SAEInterpreter.compute_stats`` and writes the aggregated stats to
``<step_dir>/stats.<dataset>.split_<split>.seed_no/``. Raw activations are
not saved.

Usage::

    python -m consdl.cli.stats_vision --gpu 0 --ckpt_path <run_dir> \
        --dataset imagenet --dataset_split val
"""

from __future__ import annotations

import argparse
import os
import pathlib

import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Compute SAE statistics on a trained checkpoint")
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to a run dir (with 'backbone' + 'final') or a step dir.")
    p.add_argument("--dataset", type=str, default="imagenet")
    p.add_argument("--dataset_split", type=str, default="val", choices=["train", "val"])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--stats_top_k", type=int, default=100,
                   help="Number of top-activating samples to track per latent.")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=None,
                   help="If set, subsample this many images evenly across the dataset.")
    p.add_argument("--checkpoint_step", type=str, default="final",
                   help="'final', a numeric step, or 'all' to process every step dir.")
    p.add_argument("--progress_file", type=str, default=None)
    p.add_argument("--sparsity-thresholds", type=float, nargs="+",
                   default=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                   help="Thresholds for L0 reporting in metrics.json. Each yields "
                        "'l0_per_token_<thr>' and 'l0_per_latent_<thr>' entries.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if top_act.pt already exists at the target "
                        "stats dir (useful after metric-schema changes).")
    return p.parse_args()


args = parse_args()

from dotenv import load_dotenv

from consdl.io.paths import REPO_ROOT  # noqa: E402
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import datasets
import fastsae
from transformers import AutoProcessor

from consdl.data.vision_datasets import DATASET_MAP, format_dataset, select_subset


def expand_ckpt_path(p: str, step: str = "final") -> list[pathlib.Path]:
    """Expand a root path to a list of checkpoint step dirs.

    If ``p`` already has a sibling ``backbone/``, treat it as a step dir.
    Otherwise, recurse and match step names. ``step="all"`` returns every
    numeric step dir + ``final``.
    """
    root = pathlib.Path(p)
    if (root.parent / "backbone").is_dir() and root.is_dir():
        return [root]

    def _is_step_dir(d: pathlib.Path) -> bool:
        return d.is_dir() and (d.parent / "backbone").is_dir() and any(d.glob("*.json"))

    if step == "all":
        candidates = sorted(
            d for d in root.rglob("*")
            if _is_step_dir(d) and (d.name == "final" or d.name.isdigit())
        )
    else:
        candidates = sorted(d for d in root.rglob(step) if _is_step_dir(d))
    return candidates or [root]


def main() -> None:
    dataset_short = args.dataset
    dataset_name = DATASET_MAP[dataset_short]
    dataset_split = args.dataset_split

    for ckpt_path in expand_ckpt_path(args.ckpt_path, step=args.checkpoint_step):
        if (ckpt_path.parent / "backbone").is_dir():
            final_dir = ckpt_path
            backbone_dir = ckpt_path.parent / "backbone"
        else:
            final_dir = ckpt_path / "final"
            backbone_dir = ckpt_path / "backbone"

        from consdl.io.paths import stats_write_dir  # noqa: PLC0415
        subset_tag = f".{args.max_samples // 1_000_000}m" if args.max_samples else ""
        # The subset tag is part of the dataset name so a subsampled pass
        # does not overwrite the full-dataset stats.
        stats_path = stats_write_dir(final_dir, f"{dataset_short}{subset_tag}",
                                     dataset_split)
        stats_path.mkdir(parents=True, exist_ok=True)
        if (stats_path / "top_act.pt").exists():
            if args.overwrite:
                print(f"[overwrite] Recomputing stats at {stats_path}.")
            else:
                print(f"Stats already exist at {stats_path}. Skipping (pass --overwrite to force).")
                continue

        # ── Model ─────────────────────────────────────────────────────────
        # ``fastsae.models.SAE.load`` only knows the built-in SAE types; this
        # loader also dispatches custom ones such as fista_frozen_dec.
        from consdl.io.checkpoint import load_sae as _load_sae  # noqa: PLC0415
        backbone = fastsae.models.Backbone.load(backbone_dir)
        sae = _load_sae(final_dir)
        # SAEWrapper asserts ``backbone.device == sae.device``, and some
        # checkpoints (e.g. FISTA with frozen decoder) load on CPU, so move
        # the SAE to the backbone's device first.
        if hasattr(backbone, "device") and getattr(sae, "device", None) != backbone.device:
            sae = sae.to(backbone.device)
            if hasattr(sae, "device"):
                sae.device = backbone.device
        wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae)
        backbone_hf_name = wrapper.backbone.config["model_hf_name"]

        # ── Dataset ───────────────────────────────────────────────────────
        auto_processor = AutoProcessor.from_pretrained(backbone_hf_name)
        dataset_instance = datasets.load_dataset(path=dataset_name, split=dataset_split)
        dataset_instance = format_dataset(dataset_name, dataset_instance)
        if args.max_samples:
            dataset_instance = select_subset(dataset_instance, dataset_short, dataset_split, args.max_samples)

        has_labels = "label" in dataset_instance.features
        labels_np = np.array(dataset_instance["label"]) if has_labels else None

        dataloader = fastsae.dataset.flexible.create_dataloader(
            dataset=dataset_instance,
            processor=auto_processor,
            return_pixel_values_only=False,
            return_label=False,
            return_labels_np=False,
            batch_size=args.batch_size,
            shuffle=False,
            collate="auto",
            num_workers=args.num_workers,
            persistent_workers=args.num_workers > 0,
        )

        # ── Compute + save ────────────────────────────────────────────────
        from consdl.stats import SAEInterpreter
        _thrs = list(args.sparsity_thresholds or [])
        intrp = SAEInterpreter(
            wrapped=wrapper,
            backbone_processor=auto_processor,
            stats_dataloader=dataloader,
            stats_labels_np=labels_np,
            stats_top_k_max_samples=args.stats_top_k,
            l0_thresholds=_thrs,      # feeds l0_per_token_<thr>
            _progress_file=args.progress_file or f"/tmp/compute_stats_gpu{args.gpu}.progress",
        )

        try:
            intrp.compute_stats(compute_cls_wise_top_idx=has_labels)
        finally:
            # Drop the raw activation buffers; only aggregated stats are saved.
            intrp.stats.pop("all_act", None)
            intrp.stats.pop("valid_latent_indices", None)
            intrp.save_stats(stats_path)
        print(f"Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
