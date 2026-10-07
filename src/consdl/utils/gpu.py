"""GPU discovery and a simple local job scheduler.

``query_gpus`` / ``select_gpus`` find usable GPUs (including MIG
instances); ``Job`` and ``poll_loop`` run subprocess jobs one per GPU with
progress bars.
"""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm


FREE_MEMORY_MB_DEFAULT = 10_000
FREE_UTIL_PCT_DEFAULT = 30


@dataclass
class GPUInfo:
    index: int
    name: str
    mem_used_mb: int
    mem_free_mb: int
    mem_total_mb: int
    util_pct: int

    def is_free(self, free_mem_mb: int = FREE_MEMORY_MB_DEFAULT,
                free_util_pct: int = FREE_UTIL_PCT_DEFAULT) -> bool:
        return self.mem_free_mb >= free_mem_mb and self.util_pct < free_util_pct


def _query_mig_instances() -> list[GPUInfo]:
    """MIG compute instances visible to this process, or ``[]`` if none.

    ``nvidia-smi --query-gpu`` enumerates physical GPUs only and so
    undercounts on a MIG allocation. ``nvidia-smi -L`` lists MIG
    instances, indented under their parent::

        GPU 0: NVIDIA A100-PCIE-40GB (UUID: GPU-bc69…)
          MIG 3g.20gb     Device  0: (UUID: MIG-7078…)

    The indices returned here are positional, which is what CUDA uses:
    with ``CUDA_VISIBLE_DEVICES`` set to the allocated MIG UUIDs, torch
    numbers them 0..N-1 in that order.

    Per-instance memory and utilisation are not available, so the fields
    are filled such that a MIG instance always reads as free; the
    allocation has already decided which instances may be used.
    """
    try:
        out = subprocess.run(
            ["nvidia-smi", "-L"], capture_output=True, text=True, check=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return []
    mig: list[GPUInfo] = []
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped.startswith("MIG "):
            continue
        # "MIG 3g.20gb     Device  0: (UUID: MIG-7078…)"
        profile = stripped.split()[1] if len(stripped.split()) > 1 else "mig"
        mig.append(GPUInfo(
            index=len(mig), name=f"MIG {profile}",
            mem_used_mb=0, mem_free_mb=FREE_MEMORY_MB_DEFAULT,
            mem_total_mb=FREE_MEMORY_MB_DEFAULT, util_pct=0,
        ))
    # ``nvidia-smi -L`` may list slices not allocated to this job; inside
    # a SLURM allocation, trust its GPU count.
    alloc = os.environ.get("SLURM_GPUS_ON_NODE")
    if alloc and alloc.isdigit() and 0 < int(alloc) < len(mig):
        mig = mig[: int(alloc)]
    return mig


def query_gpus() -> list[GPUInfo]:
    """Query all GPUs via nvidia-smi.

    MIG instances take precedence when present: they, not their parent
    devices, are what a job on a MIG node can actually run on.
    """
    mig = _query_mig_instances()
    if mig:
        return mig
    result = subprocess.run(
        ["nvidia-smi",
         "--query-gpu=index,name,memory.used,memory.free,memory.total,utilization.gpu",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    gpus: list[GPUInfo] = []
    for line in result.stdout.strip().splitlines():
        idx, name, used, free, total, util = [x.strip() for x in line.split(",")]
        # Utilisation may be reported as "[N/A]" (e.g. MIG); treat it as 0
        # and rely on the memory fields for ``is_free``.
        util_pct = int(util) if util.lstrip("-").isdigit() else 0
        gpus.append(GPUInfo(
            index=int(idx), name=name,
            mem_used_mb=int(used), mem_free_mb=int(free),
            mem_total_mb=int(total), util_pct=util_pct,
        ))
    return gpus


def select_gpus(
    gpus: list[GPUInfo] | None = None,
    explicit: list[int] | None = None,
    free_mem_mb: int = FREE_MEMORY_MB_DEFAULT,
    free_util_pct: int = FREE_UTIL_PCT_DEFAULT,
    max_gpus: int | None = None,
) -> list[int]:
    """Return selected GPU indices (explicit list, or auto-detect free GPUs)."""
    if gpus is None:
        gpus = query_gpus()
    if explicit is not None:
        known = {g.index for g in gpus}
        missing = [i for i in explicit if i not in known]
        if missing:
            raise RuntimeError(f"GPUs not found: {missing}")
        selected = list(explicit)
    else:
        selected = [g.index for g in gpus if g.is_free(free_mem_mb, free_util_pct)]
        if not selected:
            raise RuntimeError("No free GPUs detected. Use --gpus to specify manually.")
    if max_gpus is not None:
        selected = selected[:max_gpus]
    return selected


def print_gpu_table(gpus: list[GPUInfo], selected: list[int]) -> None:
    print("\nGPU status:")
    print(f"  {'GPU':<4}  {'Name':<24}  {'Used':>7}  {'Free':>7}  "
          f"{'Total':>7}  {'Util':>5}  {'Status'}")
    print(f"  {'-'*4}  {'-'*24}  {'-'*7}  {'-'*7}  {'-'*7}  {'-'*5}  {'-'*10}")
    for g in gpus:
        status = "SELECTED" if g.index in selected else ("free" if g.is_free() else "busy")
        print(f"  {g.index:<4}  {g.name:<24}  {g.mem_used_mb:>6}M  {g.mem_free_mb:>6}M  "
              f"{g.mem_total_mb:>6}M  {g.util_pct:>4}%  {status}")
    print()


@dataclass
class Job:
    label: str
    gpu: int
    cmd: list[str] = field(default_factory=list, repr=False)
    log_file: Path | None = None
    progress_file: Path | None = None
    proc: Optional[subprocess.Popen] = field(default=None, repr=False)
    pbar: Optional[tqdm] = field(default=None, repr=False)
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    returncode: Optional[int] = None

    @property
    def elapsed(self) -> str:
        if self.started_at is None:
            return "—"
        end = self.finished_at or datetime.now()
        s = int((end - self.started_at).total_seconds())
        return f"{s // 3600:02d}:{(s % 3600) // 60:02d}:{s % 60:02d}"

    def update_pbar(self) -> None:
        if self.pbar is None or self.progress_file is None:
            return
        try:
            data = json.loads(self.progress_file.read_text())
            n, total = data.get("n", 0), data.get("total", 0)
            if total and self.pbar.total != total:
                self.pbar.total = total
                self.pbar.refresh()
            if n > self.pbar.n:
                self.pbar.update(n - self.pbar.n)
        except Exception:
            pass

    def close_pbar(self, success: bool) -> None:
        if self.pbar is None:
            return
        status = "done" if success else "FAILED"
        self.pbar.set_postfix_str(f"{status}  elapsed={self.elapsed}")
        self.pbar.close()


GPU_PLACEHOLDER = "GPU_PLACEHOLDER"


def launch(job: Job) -> Job:
    """Start the subprocess and attach a tqdm progress bar.

    Each bar is pinned to the job's GPU index (``position``) so concurrent
    jobs render as stacked bars rather than overwriting one line.
    """
    try:
        idx = job.cmd.index(GPU_PLACEHOLDER)
        job.cmd[idx] = str(job.gpu)
    except ValueError:
        pass
    if job.log_file is not None:
        job.log_file.parent.mkdir(parents=True, exist_ok=True)
        stdout = open(job.log_file, "w")
    else:
        stdout = subprocess.DEVNULL
    job.proc = subprocess.Popen(job.cmd, stdout=stdout, stderr=subprocess.STDOUT)
    job.pbar = tqdm(
        total=None,
        desc=f"{job.label} (gpu {job.gpu})",
        unit="samples", unit_scale=True, leave=True,
        position=int(job.gpu) if job.gpu is not None and job.gpu >= 0 else None,
        dynamic_ncols=True,
    )
    job.started_at = datetime.now()
    tqdm.write(f"  [LAUNCH] {job.label}  gpu={job.gpu}")
    return job


def poll_loop(
    queue: list[Job],
    free_gpus: list[int],
    *,
    poll_interval: int = 5,
    stagger_delay: int = 0,
    dry_run: bool = False,
    all_enqueued: threading.Event | None = None,
) -> list[Job]:
    """Run queued jobs on free GPUs until all have finished.

    Pass ``all_enqueued`` (a ``threading.Event``) when jobs are added to
    *queue* from a background thread after this call starts.  The loop will
    not exit until the event is set AND both *queue* and *running* are empty.
    """
    running: list[Job] = []
    done: list[Job] = []

    def _fill() -> None:
        while queue and free_gpus:
            job = queue.pop(0)
            job.gpu = free_gpus.pop(0)
            if dry_run:
                tqdm.write(f"  [DRY] {job.label}  gpu={job.gpu}")
                done.append(job)
            else:
                launch(job)
                running.append(job)
                if queue and free_gpus and stagger_delay > 0:
                    tqdm.write(f"  [STAGGER] waiting {stagger_delay}s...")
                    time.sleep(stagger_delay)

    _fill()
    if dry_run:
        print("\n[DRY RUN] No jobs launched.")
        return done

    def _more_coming() -> bool:
        return all_enqueued is not None and not all_enqueued.is_set()

    while running or queue or _more_coming():
        time.sleep(poll_interval)
        for job in running:
            job.update_pbar()
        still_running: list[Job] = []
        for job in running:
            rc = job.proc.poll()  # type: ignore[union-attr]
            if rc is not None:
                job.returncode = rc
                job.finished_at = datetime.now()
                job.update_pbar()
                job.close_pbar(success=(rc == 0))
                done.append(job)
                free_gpus.append(job.gpu)
                flag = "OK" if rc == 0 else f"FAILED(rc={rc})"
                tqdm.write(f"  [DONE] {job.label}  gpu={job.gpu}  elapsed={job.elapsed}  {flag}")
            else:
                still_running.append(job)
        running = still_running
        if queue and free_gpus:
            _fill()
    return done


def print_summary(done: list[Job]) -> None:
    import sys
    print("\n" + "=" * 60)
    failed = [j for j in done if j.returncode not in (None, 0)]
    for j in sorted(done, key=lambda x: x.label):
        rc = j.returncode
        status = "OK" if rc == 0 else (f"FAILED(rc={rc})" if rc is not None else "DRY")
        print(f"  {j.label:45s}  gpu={j.gpu}  elapsed={j.elapsed}  {status}")
    if failed:
        print(f"\n{len(failed)} job(s) FAILED.")
        sys.exit(1)
    else:
        print(f"\nAll {len(done)} jobs succeeded.")


def attach_fail_log_tail(job: "Job", n_lines: int = 30) -> None:
    """Wrap ``job.close_pbar`` so a FAILED worker dumps the tail of its
    log to the orchestrator's stdout.

    Call right after building a :class:`Job` (before enqueueing) to print
    the last ``n_lines`` of its log on failure. No-op when the job has no
    ``log_file``.
    """
    _orig = job.close_pbar

    def _close_with_tail(success: bool, _job=job, _orig=_orig):
        _orig(success)
        if success or _job.log_file is None:
            return
        try:
            from tqdm import tqdm as _t  # noqa: PLC0415

            _t.write(f"  [FAIL-LOG] {_job.log_file}")
            tail = _job.log_file.read_text(errors="replace").splitlines()[-n_lines:]
            if tail:
                _t.write("  " + "-" * 60)
                for line in tail:
                    _t.write(f"  | {line}")
                _t.write("  " + "-" * 60)
        except Exception as e:  # noqa: BLE001
            print(f"  [FAIL-LOG] (could not read log: {e})")

    job.close_pbar = _close_with_tail  # type: ignore[assignment]
