"""Train and evaluate a vision or language-model SAE sweep.

    python -m consdl.cli.sweep vision --backbone clip-vit-b-16 --layer 10 \\
        --archs consdl topk --levels l0_32 l0_64 \\
        --values consdl=1.2,1.0 topk=32,64 --expansions 16 --seeds 0 1 2 --gpus 0 1
    python -m consdl.cli.sweep lm --backbone pythia-160m --hook-layer 8 ...

A cell is one (architecture, expansion, sparsity level) setting; every
cell is trained for each seed, then evaluated across seeds:

    <out>/<backbone>_layer<L>/<arch>/x<E>/<level>/seed<S>/final/   training (+ stats)
    consdl.cli.dict_metrics --tree <out>            DCS, #dead, #dup
    consdl.cli.code_metrics --tree <out>            MCC, R²

Results land under ``var/results/`` mirroring ``var/checkpoints/``.
Training jobs are spread over ``--gpus``, one at a time per GPU.
"""
from __future__ import annotations

import argparse
import itertools
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

from consdl.io.paths import REPO_ROOT as REPO  # noqa: E402

#: Architecture → (fixed flags, sparsity flag). ``sc``, ``refit`` and
#: ``consdl`` are the ablation rows: FISTA sparse coding, + refitting,
#: + co-active similarity regularization.
RECIPES: dict[str, tuple[list[str], str]] = {
    "relu": (["--architecture", "relu"], "--l1-coeff"),
    "topk": (["--architecture", "topk"], "--k-per-token"),
    "batchtopk": (["--architecture", "batchtopk"], "--k-per-token"),
    "matryoshka": (["--architecture", "matryoshka"], "--k-per-token"),
    "mp": (["--architecture", "mp"], "--k-per-token"),
    "isae": (["--architecture", "isae"], "--k-per-token"),
    "sc": (["--architecture", "fista"], "--l1-coeff"),
    "refit": (["--architecture", "fista", "--refit"], "--l1-coeff"),
    "consdl": (["--architecture", "fista", "--refit"], "--l1-coeff"),
}


def parse_args(modality: str, defaults: dict) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog=f"python -m consdl.cli.sweep {modality}",
        description=f"Train and evaluate a {modality} SAE sweep.",
        epilog="Arguments after `--` are passed to every training run.")
    p.add_argument("--archs", nargs="+", required=True, choices=sorted(RECIPES))
    p.add_argument("--levels", nargs="+", required=True, metavar="NAME",
                   help="Sparsity-level names, used as the cell directory "
                        "(e.g. l0_32 l0_64 l0_128).")
    p.add_argument("--values", nargs="+", required=True, metavar="ARCH=V1,V2",
                   help="Sparsity knob per architecture, one value per level in "
                        "--levels order: L1 coefficient (relu), k (top-k family, "
                        "mp, isae) or λ (sc / refit / consdl).")
    p.add_argument("--expansions", nargs="+", type=int, default=[4, 8, 16])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    p.add_argument("--gpus", nargs="+", type=int, default=[0])
    p.add_argument("--out", type=Path,
                   default=REPO / "var" / "checkpoints" / modality)
    p.add_argument("--refit-iters", type=int, default=10, help="T (refit / consdl)")
    p.add_argument("--co-active-sim-coefficient", type=float, default=5e-3,
                   help="γ (consdl)")
    p.add_argument("--co-active-sim-barrier-margin", type=float, default=0.5)
    p.add_argument("--skip-train", action="store_true")
    p.add_argument("--skip-eval", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    for k, v in defaults.items():
        p.add_argument(f"--{k.replace('_', '-')}", default=v, type=type(v))
    argv = sys.argv[2:]
    extra: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, extra = argv[:i], argv[i + 1:]
    args = p.parse_args(argv)
    args.extra = extra
    values = {}
    for item in args.values:
        arch, _, vs = item.partition("=")
        values[arch] = vs.split(",")
    missing = [a for a in args.archs if a not in values]
    if missing:
        p.error(f"--values missing for {missing}")
    for arch in args.archs:
        if len(values[arch]) != len(args.levels):
            p.error(f"--values {arch}: {len(values[arch])} value(s) for "
                    f"{len(args.levels)} level(s)")
    args.value_map = values
    return args


def cells(args, segment: str) -> list[tuple[Path, list[str]]]:
    """``(seed_dir, training flags)`` for every run of the sweep."""
    runs = []
    for arch in args.archs:
        fixed, knob_flag = RECIPES[arch]
        flags = list(fixed)
        if arch in ("refit", "consdl"):
            flags += ["--refit-iters", str(args.refit_iters)]
        if arch == "consdl":
            flags += ["--co-active-sim-coefficient", str(args.co_active_sim_coefficient),
                      "--co-active-sim-refresh-steps", "1",
                      "--co-active-sim-barrier-margin", str(args.co_active_sim_barrier_margin)]
        levels = list(zip(args.levels, args.value_map[arch]))
        for e, (level, v), s in itertools.product(args.expansions, levels, args.seeds):
            seed_dir = args.out / segment / arch / f"x{e}" / level / f"seed{s}"
            runs.append((seed_dir, flags + [knob_flag, v, "--expansion-factor", str(e),
                                            "--seed", str(s)]))
    return runs


def _run(cmd: list[str], dry_run: bool) -> int:
    print("$", " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.call(cmd, cwd=REPO)


def train_all(args, train_module: str, base_flags: list[str],
              segment: str) -> list[Path]:
    """Train every run not yet complete; returns the failed seed dirs."""
    jobs: queue.Queue = queue.Queue()
    for seed_dir, flags in cells(args, segment):
        if (seed_dir / "final" / ".training_complete").exists():
            print(f"[skip] {seed_dir} (complete)")
            continue
        jobs.put((seed_dir, flags))
    failed: list[Path] = []

    def worker(gpu: int):
        while True:
            try:
                seed_dir, flags = jobs.get_nowait()
            except queue.Empty:
                return
            cmd = [sys.executable, "-u", "-m", train_module, "--gpu", str(gpu),
                   *base_flags, *flags, "--experiment-dir", str(seed_dir), *args.extra]
            if _run(cmd, args.dry_run) != 0:
                failed.append(seed_dir)

    threads = [threading.Thread(target=worker, args=(g,)) for g in args.gpus]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return failed


def evaluate(args, modality: str) -> None:
    py = [sys.executable, "-u", "-m"]
    _run(py + ["consdl.cli.dict_metrics", "--tree", str(args.out),
               "--repo-root", str(REPO)], args.dry_run)
    _run(py + ["consdl.cli.code_metrics", "--tree", str(args.out),
               "--modality", modality, "--gpu", str(args.gpus[0])], args.dry_run)


#: Per modality: training module, its fixed flags, the
#: ``<backbone>_layer<L>`` directory the metric passes read the backbone
#: and layer back from, and the defaults of the modality-specific flags.
MODALITIES = {
    "vision": dict(
        train_module="consdl.cli.train_vision",
        base_flags=lambda a: ["--backbone", a.backbone, "--layer", str(a.layer),
                              "--dataset", a.dataset,
                              "--total-samples", str(a.total_samples)],
        segment=lambda a: f"{a.backbone}_layer{a.layer}",
        # ImageNet train, one epoch.
        defaults=dict(backbone="clip-vit-b-16", layer=10, dataset="imagenet",
                      total_samples=1_281_167),
    ),
    "lm": dict(
        train_module="consdl.cli.train_lm",
        base_flags=lambda a: ["--backbone", a.backbone,
                              "--hook-layer", str(a.hook_layer),
                              "--total-samples", str(a.total_samples)],
        segment=lambda a: f"{a.backbone}_layer{a.hook_layer}",
        defaults=dict(backbone="pythia-160m", hook_layer=8,
                      total_samples=500_000_000),
    ),
}


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in MODALITIES:
        sys.exit(f"usage: python -m consdl.cli.sweep {{{','.join(MODALITIES)}}} ...")
    modality = sys.argv[1]
    m = MODALITIES[modality]
    args = parse_args(modality, m["defaults"])
    os.environ.setdefault("PYTHONPATH", str(REPO / "src"))
    failed: list[Path] = []
    if not args.skip_train:
        failed = train_all(args, m["train_module"], m["base_flags"](args),
                           m["segment"](args))
    if not args.skip_eval:
        evaluate(args, modality)
    if failed:
        print(f"[fail] {len(failed)} run(s):", *failed, sep="\n  ")
        sys.exit(1)


if __name__ == "__main__":
    main()
