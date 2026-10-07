"""Run ``consdl.cli.stats_vision`` for every checkpoint under one or more root dirs.

Discovers ``<run_id>/final/`` (or intermediate) checkpoint directories and
runs one job per checkpoint across the available GPUs, optionally with
several concurrent jobs per GPU.

Usage
-----
    python -m consdl.cli.stats_vision_all \\
        --root-dirs checkpoints/.../layer23/batchtopk_x32 \\
        --dataset imagenet --dataset-split val

    # Two concurrent jobs per GPU, explicit GPU list
    python -m consdl.cli.stats_vision_all \\
        --root-dirs checkpoints/... --gpus 0 1 --jobs-per-gpu 2

    # Process every intermediate step
    python -m consdl.cli.stats_vision_all \\
        --root-dirs checkpoints/... --checkpoint-steps all
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from consdl.io.discovery import discover_checkpoints
from consdl.utils.gpu import (
    FREE_MEMORY_MB_DEFAULT, FREE_UTIL_PCT_DEFAULT,
    Job, poll_loop, print_gpu_table, print_summary, query_gpus, select_gpus,
)
import consdl.utils.gpu as _gpu_mod


_GPU_PLACEHOLDER = "GPU_PLACEHOLDER"


@dataclass
class JobSpec:
    ckpt_path: Path
    dataset: str
    dataset_split: str

    @property
    def label(self) -> str:
        return f"{self.ckpt_path.parent.name}/{self.dataset}_{self.dataset_split}"


_job_counter = 0


def _update_pbar_text(job: Job) -> None:
    """Parse the ``cur total`` progress file written by stats_vision."""
    if job.pbar is None or job.progress_file is None:
        return
    try:
        parts = job.progress_file.read_text().split()
        cur, total = int(parts[0]), int(parts[1])
        if total > 0 and job.pbar.total != total:
            job.pbar.total = total
            job.pbar.refresh()
        if cur > job.pbar.n:
            job.pbar.update(cur - job.pbar.n)
    except Exception:
        pass


def _close_pbar_and_cleanup(job: Job, success: bool) -> None:
    if job.pbar is not None:
        status = "done" if success else "FAILED"
        job.pbar.set_postfix_str(f"{status}  elapsed={job.elapsed}")
        job.pbar.close()
    if not success and job.log_file is not None:
        # Print the log path and its last lines on failure.
        tqdm.write(f"  [FAIL-LOG] {job.log_file}")
        try:
            tail = job.log_file.read_text(errors="replace").splitlines()[-30:]
            if tail:
                tqdm.write("  " + "-" * 60)
                for line in tail:
                    tqdm.write(f"  | {line}")
                tqdm.write("  " + "-" * 60)
        except Exception as e:
            tqdm.write(f"  [FAIL-LOG] (could not read log: {e})")
    try:
        if job.progress_file is not None:
            job.progress_file.unlink(missing_ok=True)
    except Exception:
        pass


def _build_job(spec: JobSpec, extra_args: list[str], log_dir: Path) -> Job:
    global _job_counter
    job_id = _job_counter
    _job_counter += 1
    safe_label = spec.label.replace("/", "_")
    log_file = log_dir / f"{safe_label}_job{job_id}.log"
    progress_file = Path(f"/tmp/compute_stats_job{job_id}.progress")
    cmd = [
        sys.executable, "-u", "-m", "consdl.cli.stats_vision",
        "--gpu", _GPU_PLACEHOLDER,
        "--ckpt_path", str(spec.ckpt_path),
        "--dataset", spec.dataset,
        "--dataset_split", spec.dataset_split,
        "--progress_file", str(progress_file),
    ] + extra_args
    job = Job(
        label=f"gpu? {spec.label}",
        gpu=-1, cmd=cmd,
        log_file=log_file, progress_file=progress_file,
    )
    job.update_pbar = lambda: _update_pbar_text(job)  # type: ignore[assignment]
    job.close_pbar = lambda success: _close_pbar_and_cleanup(job, success)  # type: ignore[assignment]
    job._spec = spec  # type: ignore[attr-defined]
    return job


def _launch_with_gpu_and_env(job: Job) -> Job:
    """Replace placeholder, export CUDA_VISIBLE_DEVICES, launch."""
    job.cmd = [str(job.gpu) if v == _GPU_PLACEHOLDER else v for v in job.cmd]
    job.label = f"gpu{job.gpu} {job._spec.label}"  # type: ignore[attr-defined]
    if job.log_file is not None:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(job.log_file, "w")
    else:
        stdout = subprocess.DEVNULL
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(job.gpu)}
    job.proc = subprocess.Popen(job.cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env)
    job.pbar = tqdm(total=None, desc=f"gpu{job.gpu} {job._spec.label}",  # type: ignore[attr-defined]
                    unit="samples", unit_scale=True, leave=True, dynamic_ncols=True)
    job.started_at = datetime.now()
    tqdm.write(f"  [LAUNCH] {job._spec.label}  gpu={job.gpu}")  # type: ignore[attr-defined]
    return job


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--root-dirs", nargs="+", required=True)
    parser.add_argument("--dataset", type=str, default="imagenet")
    parser.add_argument("--dataset-split", type=str, default="val", choices=["train", "val"])
    parser.add_argument("--gpus", type=int, nargs="+", default=None)
    parser.add_argument("--max-gpus", type=int, default=None)
    parser.add_argument("--jobs-per-gpu", type=int, default=1)
    parser.add_argument("--free-mem-mb", type=int, default=FREE_MEMORY_MB_DEFAULT)
    parser.add_argument("--free-util-pct", type=int, default=FREE_UTIL_PCT_DEFAULT)
    parser.add_argument("--poll-interval", type=int, default=30)
    parser.add_argument("--log-dir", type=str,
                        default=str(Path.cwd() / "var" / "logs" / "compute_stats"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--stats-top-k", type=int, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--sparsity-thresholds", type=float, nargs="+", default=None,
                        help="Thresholds forwarded to compute_stats for l0_<thr> entries.")
    parser.add_argument("--checkpoint-steps", "--checkpoint-step",
                        dest="checkpoint_steps", nargs="+", default=["final"])
    parser.add_argument("--overwrite", action="store_true",
                        help="Recompute even where stats already exist (forwarded to "
                             "compute_stats; useful after metric-schema changes).")
    parser.add_argument("--dry-run", action="store_true")
    args, extra_args = parser.parse_known_args()
    if extra_args and extra_args[0] == "--":
        extra_args = extra_args[1:]

    if args.batch_size is not None:
        extra_args += ["--batch_size", str(args.batch_size)]
    if args.stats_top_k is not None:
        extra_args += ["--stats_top_k", str(args.stats_top_k)]
    if args.max_samples is not None:
        extra_args += ["--max_samples", str(args.max_samples)]
    if args.sparsity_thresholds is not None:
        extra_args += ["--sparsity-thresholds", *[str(t) for t in args.sparsity_thresholds]]
    if args.overwrite:
        extra_args += ["--overwrite"]

    all_ckpts: list[Path] = []
    for root_str in args.root_dirs:
        root = Path(root_str)
        if not root.exists():
            print(f"WARNING: root dir not found: {root}", file=sys.stderr)
            continue
        found = discover_checkpoints(root, steps=args.checkpoint_steps)
        if not found:
            print(f"WARNING: no checkpoints found under {root}", file=sys.stderr)
        all_ckpts.extend(found)

    if not all_ckpts:
        print("ERROR: no checkpoints discovered.", file=sys.stderr)
        sys.exit(1)

    def _is_complete(ckpt: Path) -> bool:
        from consdl.io.paths import stats_dir  # noqa: PLC0415
        d = stats_dir(ckpt, args.dataset, args.dataset_split)
        return d.is_dir() and (d / "top_act.pt").exists()

    if args.overwrite:
        pending = list(all_ckpts)
        skipped = 0
        print(f"[overwrite] recomputing all {len(pending)} checkpoint(s).")
    else:
        pending = [c for c in all_ckpts if not _is_complete(c)]
        skipped = len(all_ckpts) - len(pending)
        if skipped:
            print(f"\nSkipping {skipped} checkpoint(s) with existing stats ({len(pending)} remaining)")
    if not pending:
        print("Nothing to do.")
        return

    job_specs = [JobSpec(ckpt_path=c, dataset=args.dataset, dataset_split=args.dataset_split) for c in pending]

    all_gpus = query_gpus()
    selected = select_gpus(all_gpus, explicit=args.gpus,
                           free_mem_mb=args.free_mem_mb, free_util_pct=args.free_util_pct,
                           max_gpus=args.max_gpus)
    print_gpu_table(all_gpus, selected)
    print(f"Checkpoints : {len(all_ckpts)}  (pending {len(pending)})")
    print(f"Dataset     : {args.dataset} / {args.dataset_split}")
    print(f"GPUs        : {selected}   jobs/gpu={args.jobs_per_gpu}")
    print(f"Extra args  : {extra_args or '(none)'}")
    print(f"Log dir     : {args.log_dir}")

    if args.dry_run:
        free = [g for g in selected for _ in range(args.jobs_per_gpu)]
        running_sim, queue_sim, order = [], list(job_specs), []
        while queue_sim or running_sim:
            while queue_sim and free:
                s = queue_sim.pop(0); g = free.pop(0)
                running_sim.append((s, g)); order.append((s, g))
            if running_sim:
                s, g = running_sim.pop(0); free.append(g)
        for s, g in order:
            print(f"  gpu={g}  {s.label}")
        return

    log_dir = Path(args.log_dir); log_dir.mkdir(parents=True, exist_ok=True)
    queue: list[Job] = [_build_job(s, extra_args, log_dir) for s in job_specs]

    _gpu_mod.launch = _launch_with_gpu_and_env  # custom env-aware launch
    free_gpus = [g for g in selected for _ in range(args.jobs_per_gpu)]
    done = poll_loop(queue, free_gpus, poll_interval=args.poll_interval, stagger_delay=0)
    print_summary(done)


if __name__ == "__main__":
    main()
