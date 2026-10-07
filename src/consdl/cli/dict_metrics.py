"""Write ``dict_metrics_step_<step>.json`` for every tag under a tree.

Dictionary metrics (DCS, #dead, #dup, …) need only the decoder matrices
and the per-latent firing rates from the post-training stats: no
activation pool, SAE forward pass or backbone. The same runner therefore
serves vision and LM, on CPU.

Usage
-----
    python -m consdl.cli.dict_metrics \\
        --tree var/checkpoints/vision --min-seeds 2

    # checkpoints outside this repo, results written into this repo
    python -m consdl.cli.dict_metrics \\
        --tree /path/to/checkpoints --repo-root "$(pwd)"

    # one tag
    python -m consdl.cli.dict_metrics \\
        --tag var/checkpoints/vision/clip-vit-b-16_layer10/topk/x16/l0_32
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
from pathlib import Path

from consdl.metrics.dict_metrics import REPORT_BAND, write_tag
from consdl.io.paths import auto_repo_root, find_tags, to_results_path


def _tags(tree: Path, step: str, min_seeds: int) -> list[Path]:
    """Cell dirs under ``tree`` with at least ``min_seeds`` trained seeds."""
    return find_tags(tree, step, min_seeds)


def _pin_blas() -> None:
    """One BLAS thread per worker. See :func:`_one`."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _one(args) -> tuple[str, str]:
    """One tag, in its own process. Returns ``(status, message)``.

    Parallelism is per tag rather than per threshold, so no large shared
    array (e.g. the Gram matrix) has to be copied into workers.

    BLAS is pinned to one thread per worker: the time is spent in the
    single-threaded ``connected_components`` and
    ``linear_sum_assignment``, so extra BLAS threads would only
    oversubscribe the CPUs.
    """
    tag, out, step = args
    try:
        return ("ok", str(tag)) if write_tag(tag, out, step) else (
            "fail", f"no loadable decoder: {tag}")
    except Exception as e:  # noqa: BLE001
        return ("fail", f"{tag}: {type(e).__name__}: {e}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, default=None)
    # Repeatable, so a set of tags can be processed in one invocation
    # sharing one worker pool.
    ap.add_argument("--tag", type=Path, action="append", default=None,
                    metavar="TAG_DIR",
                    help="A single tag directory (the parent of the "
                         "seed dirs). Repeat to pass several; mutually "
                         "exclusive with --tree.")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Repo whose var/results/ the metric jsons are "
                         "written to. Default: inferred per tag. Pass it "
                         "when --tree points at checkpoints outside this "
                         "repo.")
    ap.add_argument("--step", default="final")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--backfill-stats", action="store_true",
                    help="Do NOT recompute. Open each existing "
                         "dict_metrics file and add only the scalars "
                         "that come from compute_stats (l0, recon_r2, "
                         "l1_per_token, n_dense). Seconds for the whole "
                         "tree; a "
                         "full pass would redo the Gram and the "
                         "Hungarian for three JSON reads.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="Worker processes over TAGS (0 = one per CPU). "
                         "The work is single-threaded per tag — scipy's "
                         "connected_components and linear_sum_assignment "
                         "dominate and neither releases to BLAS — so this "
                         "is the only lever that shortens the wall clock. "
                         "Peak memory is ~3 GB per x16 worker (two Grams "
                         "and a similarity matrix at 12288**2 float32).")
    a = ap.parse_args(argv)

    if not a.tree and not a.tag:
        ap.error("one of --tree / --tag is required")
    tags = list(a.tag) if a.tag else _tags(a.tree, a.step, a.min_seeds)
    print(f"[dict-metrics] {len(tags)} tag(s), band={list(REPORT_BAND)}",
          flush=True)

    jobs: list = []
    skipped = 0
    for t in tags:
        # ``--repo-root`` pins where results land. Without it the root is
        # inferred from the tag (``auto_repo_root`` walks up to the first
        # ancestor holding ``checkpoints/``), which is wrong for
        # checkpoints that live outside this repo.
        repo = a.repo_root or auto_repo_root(t)
        out = to_results_path(t, repo) / f"dict_metrics_step_{a.step}.json"
        if a.skip_existing and out.is_file():
            skipped += 1
            continue
        if a.backfill_stats and not out.is_file():
            continue
        jobs.append((t, out, a.step))

    n_proc = max(1, int(a.jobs or (os.cpu_count() or 4)))
    n_proc = min(n_proc, len(jobs)) or 1
    print(f"[dict-metrics] {len(jobs)} to compute, {skipped} skipped, "
          f"{n_proc} process(es)", flush=True)

    if a.backfill_stats:
        from consdl.metrics.dict_metrics import (  # noqa: PLC0415
            backfill_stats,
        )
        ok = skip = bad = 0
        for t, out, step in jobs:
            r = backfill_stats(t, out, step)
            if r == "ok":
                ok += 1
            elif r == "skip":
                skip += 1
            else:
                bad += 1
                print(f"  [skip] {t}: {r}", flush=True)
        print(f"[dict-metrics] backfill: ok={ok} unchanged={skip} "
              f"absent={bad}", flush=True)
        return 0

    ok = failed = 0
    if n_proc == 1:
        results = (_one(j) for j in jobs)
    else:
        # ``spawn``: BLAS thread pools do not survive fork cleanly, so
        # each worker starts fresh and pins its own BLAS threads.
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(n_proc, initializer=_pin_blas)
        results = pool.imap_unordered(_one, jobs, chunksize=1)
    for i, (status, msg) in enumerate(results, 1):
        if status == "ok":
            ok += 1
        else:
            failed += 1
            print(f"  [fail] {msg}", flush=True)
        if i % 20 == 0:
            print(f"[dict-metrics] {i}/{len(jobs)}  ok={ok} fail={failed}",
                  flush=True)
    if n_proc > 1:
        pool.close()
        pool.join()

    print(f"[dict-metrics] done: ok={ok} skip={skipped} fail={failed} "
          f"of {len(tags)}", flush=True)
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
