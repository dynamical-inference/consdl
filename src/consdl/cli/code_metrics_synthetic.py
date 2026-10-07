"""Code metrics for synthetic runs, including ground-truth recovery.

Writes ``code_metrics_step_<step>.json`` (MCC, input R², cross-seed and
against the ground truth) for every tag. Unlike real data, the activation
pool is regenerated from the ground-truth dictionary and the recorded
``pool_seed`` (the held-out pool, see
:mod:`consdl.data.synthetic.held_out`), and the ground-truth codes are
known exactly: ``ds[i]`` is ``z = D_gt · a`` and ``ds._sample_code(i)`` is
``a``. These are passed to ``consdl.metrics.code_metrics.write_tag``,
which computes the metrics.

Usage
-----
    python -m consdl.cli.code_metrics_synthetic --tree checkpoints/synthetic --gpu 0
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


from consdl.metrics.activations import seed_encode     # noqa: E402
from consdl.metrics.code_metrics import write_tag        # noqa: E402
from consdl.io.checkpoint import load_decoder               # noqa: E402
from consdl.io.paths import auto_repo_root, to_results_path  # noqa: E402


def _cfg_of(tag_dir: Path) -> dict | None:
    """The cell's synthetic config, or ``None`` with a printed reason.

    The config is read from the seed dir, not the step dir: inference
    from the path expects ``run_dir.parent`` to be the tag.
    """
    from consdl.data.synthetic.config import load_or_infer_config  # noqa: PLC0415

    try:
        return load_or_infer_config(tag_dir / "seed0")
    except Exception as e:  # noqa: BLE001
        print(f"  [cfg] {tag_dir.name}: {type(e).__name__}: {e}", flush=True)
        return None


def _pool_key(cfg: dict) -> tuple:
    """Cells sharing this key share a held-out pool.

    The pool depends only on the ground truth and the sampling spec, so
    every architecture at one setting is scored on the same samples and
    the pool is built once per key.
    """
    return (str(cfg["gt_dict"]),
            cfg["distribution"], int(cfg["k_active"]),
            float(cfg.get("lognormal_mu", 0.0)),
            float(cfg.get("lognormal_sigma", 0.25)),
            int(cfg.get("pool_size", 1_000_000)),
            int(cfg["pool_seed"]),
            int(cfg.get("stats_pool_seed",
                        int(cfg["pool_seed"]) ^ 0xDEADBEEF)))


def _pool(cfg: dict, n_samples: int):
    """``(X, Z_gt, D_gt)`` — the held-out inputs, ground-truth codes, ground-truth D.

    The SAME held-out set every other synthetic metric is measured on
    (``held_out.held_out_dataset``).
    """
    from consdl.data.synthetic.held_out import gt_codes, gt_decoder, held_out_dataset  # noqa: PLC0415

    ds, _seed = held_out_dataset(cfg)
    n = min(int(n_samples), int(ds.spec.pool_size))
    X = np.stack([np.asarray(ds[i]) for i in range(n)]).astype(np.float32)
    Z_gt = gt_codes(ds, n).astype(np.float32)
    D_gt = gt_decoder(cfg).astype(np.float32)
    return X, Z_gt, D_gt


def _tags(tree: Path, step: str, min_seeds: int) -> list[Path]:
    from consdl.io.paths import find_tags  # noqa: PLC0415
    return find_tags(tree, step, min_seeds)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, default=None)
    ap.add_argument("--tag", type=Path, default=None)
    ap.add_argument("--tag-list", type=Path, default=None,
                    help="File of tag dirs, one per line; only these tags "
                         "are measured.")
    ap.add_argument("--step", default="final")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-samples", type=int, default=10_000)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--no-gt", action="store_true",
                    help="Cross-seed block only. The ground truth rebuild is the "
                         "one step that can fail on a relocated tree, and "
                         "this keeps the rest reachable when it does.")
    a = ap.parse_args(argv)
    if not (a.tree or a.tag or a.tag_list):
        ap.error("one of --tree / --tag / --tag-list is required")

    if a.tag_list:
        tags = [Path(ln.strip()) for ln in a.tag_list.read_text().splitlines()
                if ln.strip()]
    elif a.tag:
        tags = [a.tag]
    else:
        tags = _tags(a.tree, a.step, a.min_seeds)

    # Group by pool identity so the pool is regenerated once per group.
    groups: dict = {}
    for t in tags:
        cfg = _cfg_of(t)
        if cfg is None:
            continue
        try:
            groups.setdefault(_pool_key(cfg), (cfg, []))[1].append(t)
        except Exception as e:  # noqa: BLE001
            print(f"  [key] {t.name}: {type(e).__name__}: {e}", flush=True)
    print(f"[code-synth] {len(tags)} tag(s) in {len(groups)} pool group(s)",
          flush=True)

    device = f"cuda:{a.gpu}"
    ok = skipped = failed = 0
    for gi, (cfg, group) in enumerate(groups.values(), 1):
        repo = auto_repo_root(group[0])
        todo = [t for t in group
                if not (a.skip_existing
                        and (to_results_path(t, repo)
                             / f"code_metrics_step_{a.step}.json").is_file())]
        skipped += len(group) - len(todo)
        if not todo:
            continue
        try:
            X, Z_gt, D_gt = _pool(cfg, a.n_samples)
        except Exception as e:  # noqa: BLE001
            failed += len(todo)
            print(f"  [fail] pool for group {gi}: {type(e).__name__}: {e}",
                  flush=True)
            continue
        gt = None if a.no_gt else (D_gt, Z_gt, X)
        print(f"########## group {gi}/{len(groups)}: {len(todo)} tag(s), "
              f"pool {X.shape}", flush=True)

        def _encode(sd):
            return seed_encode(X, sd, device=device, batch_size=a.batch_size)

        for t in todo:
            out = to_results_path(t, repo) / f"code_metrics_step_{a.step}.json"
            try:
                done = write_tag(
                    t, out, z_pool=X, encode=_encode,
                    load_decoder_fn=load_decoder, step=a.step,
                    gt=gt)
                if done:
                    ok += 1
                else:
                    failed += 1
                    print(f"  [fail] nothing encodable: {t}", flush=True)
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"  [fail] {t}: {type(e).__name__}: {e}", flush=True)
    print(f"[code-synth] done: ok={ok} skip={skipped} fail={failed} "
          f"of {len(tags)}", flush=True)
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
