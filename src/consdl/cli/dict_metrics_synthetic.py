"""Dictionary metrics for synthetic runs, including ground-truth recovery.

Same as ``consdl.cli.dict_metrics``, plus the ground-truth block: each
tag's ground-truth dictionary is resolved from its synthetic config
(:func:`consdl.data.synthetic.config.load_or_infer_config`) and passed
to ``consdl.metrics.dict_metrics.write_tag``.

Usage
-----
    python -m consdl.cli.dict_metrics_synthetic --tree checkpoints/synthetic
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np


from consdl.metrics.dict_metrics import (  # noqa: E402
    REPORT_BAND,
    backfill_stats,
    write_tag,
)
from consdl.io.paths import auto_repo_root, to_results_path  # noqa: E402


def _gt_for(tag_dir: Path):
    """Unit-norm ``(d_sae_gt, d)`` ground truth for a tag, or ``None``.

    Resolved via ``load_or_infer_config``, whose ``gt_dict`` field is
    the dictionary the run was trained on.

    Returns ``None`` (with a printed reason) rather than raising, so the
    caller can still report the tag's cross-seed metrics.
    """
    import torch  # noqa: PLC0415

    from consdl.data.synthetic.config import load_or_infer_config  # noqa: PLC0415

    try:
        # Read from the seed dir, not the step dir:
        # ``infer_config_from_tag`` expects ``run_dir.parent`` to be the tag.
        cfg = load_or_infer_config(tag_dir / "seed0")
        blob = torch.load(cfg["gt_dict"], map_location="cpu")
        D = blob["D_gt"].float().numpy().T          # (d_sae_gt, d)
        D /= np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
        return D.astype(np.float32)
    except Exception as e:  # noqa: BLE001
        print(f"  [ground truth] {tag_dir.name}: {type(e).__name__}: {e}",
              flush=True)
        return None


def _tags(tree: Path, step: str, min_seeds: int) -> list[Path]:
    from consdl.io.paths import find_tags  # noqa: PLC0415
    return find_tags(tree, step, min_seeds)


def _pin_blas() -> None:
    """One BLAS thread per worker — see :func:`_run_parallel`."""
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS"):
        os.environ[v] = "1"


def _one(args) -> tuple[str, str]:
    """One tag, in its own process. Returns ``(status, message)``."""
    t, out, step, no_gt = args
    try:
        D_gt = None if no_gt else _gt_for(t)
        if D_gt is None and not no_gt:
            return ("no_gt", str(t))
        return ("ok", str(t)) if write_tag(t, out, step, gt=D_gt) else (
            "fail", f"no loadable decoder: {t}")
    except Exception as e:  # noqa: BLE001
        return ("fail", f"{t}: {type(e).__name__}: {e}")


def _run_parallel(tags, a) -> int:
    """Tag-level worker pool.

    BLAS is pinned to one thread per worker: the time is spent in
    single-threaded scipy routines, not the matmul, so extra BLAS threads
    would only oversubscribe the CPUs.
    """
    import multiprocessing as mp  # noqa: PLC0415

    jobs = []
    skipped = 0
    for t in tags:
        repo = a.repo_root or auto_repo_root(t)
        out = to_results_path(t, repo) / f"dict_metrics_step_{a.step}.json"
        if a.skip_existing and out.is_file():
            skipped += 1
            continue
        jobs.append((t, out, a.step, a.no_gt))
    n_proc = max(1, int(a.jobs or (os.cpu_count() or 4)))
    n_proc = min(n_proc, len(jobs)) or 1
    print(f"[dict-synth] {len(jobs)} to compute, {skipped} skipped, "
          f"{n_proc} process(es)", flush=True)
    ok = failed = no_gt = 0
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_proc, initializer=_pin_blas) as pool:
        for i, (status, msg) in enumerate(
                pool.imap_unordered(_one, jobs), 1):
            if status == "ok":
                ok += 1
            elif status == "no_gt":
                no_gt += 1
            else:
                failed += 1
                print(f"  [{status}] {msg}", flush=True)
            if i % 5 == 0 or i == len(jobs):
                print(f"  [{i}/{len(jobs)}] ok={ok} fail={failed}",
                      flush=True)
    print(f"[dict-synth] ok={ok} skipped={skipped} failed={failed} "
          f"no_gt={no_gt}", flush=True)
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, default=None)
    ap.add_argument("--tag", type=Path, default=None)
    ap.add_argument("--tags-file", type=Path, default=None,
                    help="One tag dir per line; a targeted alternative "
                         "to --tree that avoids walking the whole tree "
                         "when only a known subset needs processing. "
                         "Blank lines and ``#`` comments are skipped.")
    ap.add_argument("--repo-root", type=Path, default=None,
                    help="Repo whose var/results/ the metric jsons are "
                         "written to. Default: inferred per tag. Pass it "
                         "when --tree points at checkpoints outside this "
                         "repo.")
    ap.add_argument("--step", default="final")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--backfill-stats", action="store_true",
                    help="Do NOT recompute. Add only the scalars that "
                         "come from compute_stats (l0, recon_r2, "
                         "l1_per_token) to each existing dict_metrics "
                         "file. Seconds for the tree; a full pass would "
                         "redo the Gram and the Hungarian for three "
                         "JSON reads.")
    ap.add_argument("--no-gt", action="store_true",
                    help="Cross-seed block only. The ground truth load is the "
                         "one step that can fail on a relocated tree, and "
                         "this keeps the rest reachable when it does.")
    ap.add_argument("--jobs", type=int, default=0,
                    help="Worker processes over TAGS (0 = one per CPU, "
                         "1 = serial). The work is single-threaded per "
                         "tag, so parallelism comes from running tags "
                         "concurrently. Peak memory is ~3 GB per x16 "
                         "worker.")
    a = ap.parse_args(argv)
    if not a.tree and not a.tag and not a.tags_file:
        ap.error("one of --tree / --tag / --tags-file is required")

    if a.tags_file:
        tags = [Path(ln.strip()) for ln in
                a.tags_file.read_text().splitlines()
                if ln.strip() and not ln.startswith("#")]
        missing = [t for t in tags if not t.is_dir()]
        if missing:
            print(f"[dict-synth] {len(missing)} listed tag(s) are not "
                  f"directories; skipping them:", flush=True)
            for t in missing[:10]:
                print(f"    {t}", flush=True)
            tags = [t for t in tags if t.is_dir()]
    elif a.tag:
        tags = [a.tag]
    else:
        tags = _tags(a.tree, a.step, a.min_seeds)
    print(f"[dict-synth] {len(tags)} tag(s), band={list(REPORT_BAND)}",
          flush=True)

    ok = skipped = failed = no_gt = 0

    # Parallel over tags: the work inside a tag is single-threaded
    # (scipy's connected_components and linear_sum_assignment), so
    # tag-level workers are what shortens the wall clock.
    if a.jobs != 1 and not a.backfill_stats and len(tags) > 1:
        return _run_parallel(tags, a)

    for i, t in enumerate(tags, 1):
        # ``--repo-root`` pins where results land. Without it the root is
        # inferred from the tag (``auto_repo_root`` walks up to the first
        # ancestor holding ``checkpoints/``), which is wrong for
        # checkpoints that live outside this repo.
        repo = a.repo_root or auto_repo_root(t)
        out = to_results_path(t, repo) / f"dict_metrics_step_{a.step}.json"
        if a.skip_existing and out.is_file():
            skipped += 1
            continue
        if a.backfill_stats:
            if not out.is_file():
                continue
            r = backfill_stats(t, out, a.step)
            if r == "ok":
                ok += 1
            elif r == "skip":
                skipped += 1
            else:
                failed += 1
                print(f"  [skip] {t}: {r}", flush=True)
            continue
        D_gt = None if a.no_gt else _gt_for(t)
        no_gt += int(D_gt is None and not a.no_gt)
        try:
            if write_tag(t, out, a.step, gt=D_gt):
                ok += 1
            else:
                failed += 1
                print(f"  [fail] no loadable decoder: {t}", flush=True)
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  [fail] {t}: {type(e).__name__}: {e}", flush=True)
        if i % 20 == 0:
            print(f"[dict-synth] {i}/{len(tags)}  ok={ok} skip={skipped} "
                  f"fail={failed} no-ground truth={no_gt}", flush=True)
    print(f"[dict-synth] done: ok={ok} skip={skipped} fail={failed} "
          f"no-ground truth={no_gt} of {len(tags)}", flush=True)
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
