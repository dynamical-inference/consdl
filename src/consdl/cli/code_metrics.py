"""Write ``code_metrics_step_<step>.json`` for every tag under a tree.

Code metrics (MCC, R², …) come from running each seed's SAE over an
activation pool; dictionary metrics are computed separately by
``consdl.cli.dict_metrics``. Pass ``--full-pool`` for the pool used in the
paper (see the flag's comment below).

The activation pool depends only on (backbone, layer, data, size), so it
is built once per (backbone, layer) and shared by every seed of every
tag; ``--activation-cache`` (on by default) also keeps it on disk.

Liveness comes from the post-training stats over the full split, not from
this pool: on a small pool a latent that never fires is not necessarily
dead.

MP-SAE's sparsity is set by its inference round count, not its weights,
so ``--mp-k 32 64 128`` evaluates every MP cell once per round count
into ``code_metrics_step_<step>_mp_k<N>.json``.

Usage
-----
    python -m consdl.cli.code_metrics \\
        --tree var/checkpoints/vision --modality vision --gpu 0 --full-pool

    python -m consdl.cli.code_metrics \\
        --tag var/checkpoints/lm/pythia-160m_layer8/topk/x16/l0_128 \\
        --modality lm --gpu 0 --full-pool

    # MP cells at fixed inference round counts
    python -m consdl.cli.code_metrics \\
        --tree var/checkpoints/vision --modality vision --gpu 0 --mp-k 32 64 128
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

import json
from consdl.metrics.activations import (
    encode_batch_with,
    lm_pool,
    load_seed_sae,
    resolve_lm_backbone_layer,
    seed_encode,
    vision_pool,
    vision_stratified_pool,
)
from consdl.metrics.code_metrics import compute_tag_stream, write_tag
from consdl.metrics.dict_metrics import REPORT_BAND
from consdl.metrics.latent_consistency import resolve_backbone_layer
from consdl.io.checkpoint import load_decoder
from consdl.io.paths import auto_repo_root, find_tags, to_results_path


def _pool_name(modality: str, bb: str, layer: int, a) -> str:
    """File name of the shared activation cache for one backbone+layer.

    Keyed on everything the pool depends on, so a cached pool is only
    reused for the same (backbone, layer, data, size).
    """
    if modality == "vision":
        return (f"{bb}_layer{layer}_{a.dataset}_{a.split}"
                f"_n{a.n_tokens}.npy")
    ds_key = str(a.hf_dataset).replace("/", "_")
    return (f"lm_{bb}_layer{layer}_{ds_key}_{a.hf_dataset_split}"
            f"_ctx{a.context_length}_n{a.n_tokens}.npy")


def _pool_for(modality: str, bb: str, layer: int, a) -> "object":
    """Build (or reload) the activation pool for one backbone+layer."""
    cache = (Path(a.activation_cache_dir) / _pool_name(modality, bb, layer, a)
             if a.activation_cache else None)
    if modality == "vision":
        return vision_pool(
            bb, layer, dataset=a.dataset, split=a.split,
            n_tokens=a.n_tokens, batch_size=a.batch_size,
            cache_dir=a.imagenet_cache_dir, device=f"cuda:{a.gpu}",
            act_cache=cache)
    return lm_pool(
        bb, layer, hf_dataset=a.hf_dataset, hf_dataset_config=a.hf_dataset_config,
        hf_dataset_split=a.hf_dataset_split,
        hf_dataset_text_field=a.hf_dataset_text_field,
        context_length=a.context_length, n_tokens=a.n_tokens,
        batch_seqs=a.batch_seqs, device=f"cuda:{a.gpu}", act_cache=cache)


#: LM full pool: the fp16 activation memmap the post-training stats pass
#: (``consdl.cli.stats_lm``, run by ``train_lm``) writes for each backbone
#: -- 20,000 FineWeb sequences of 1,024 tokens through layer 8.
LM_FULL_POOL = {
    "gpt2-small": "var/cache/lm_stats_acts/openai-community_gpt2_"
                  "transformer.h.8_HuggingFaceFW_fineweb_train_ctx1024_n20000.f16.mm",
    "pythia-160m": "var/cache/lm_stats_acts/EleutherAI_pythia-160m-deduped_"
                   "gpt_neox.layers.8_HuggingFaceFW_fineweb_train_ctx1024_n20000.f16.mm",
}


#: LM MCC pool size, in tokens: MCC has converged to within ~5e-4 of its
#: 2M-token value here. Vision uses the class-stratified pool
#: (``activations.vision_stratified_pool``) instead, sized for class
#: coverage.
LM_POOL_TOKENS = 600_000


#: The in-memory ``--n-tokens`` default. A json with any other
#: ``n_tokens`` is treated as measured on a full pool.
LEGACY_POOL_TOKENS = 20_000


def _on_new_pool(path: Path) -> bool:
    """Whether a code_metrics json was already measured on a full pool,
    so parallel ``--full-pool`` shards never redo a tag."""
    try:
        n = int(json.loads(path.read_text()).get("n_tokens") or 0)
    except (OSError, ValueError, TypeError):
        return False
    return n > 0 and n != LEGACY_POOL_TOKENS


def _chunks(z: np.ndarray, step: int, limit: int = 0):
    """``z`` in row slices of ``step``, capped at ``limit`` rows (0 = all)."""
    n = z.shape[0] if not limit else min(int(limit), z.shape[0])
    for i in range(0, n, step):
        yield np.asarray(z[i:min(i + step, n)], dtype=np.float32)


def lm_full_pool(bb: str, repo: Path, d_in: int = 768) -> np.ndarray:
    """The LM stats memmap as ``(n_tokens, d_in)`` (fp16, not loaded)."""
    p = repo / LM_FULL_POOL[bb]
    if not p.is_file():
        raise FileNotFoundError(
            f"no LM activation pool for {bb}: {p} -- run "
            f"``python -m consdl.cli.stats_lm`` on one of its checkpoints "
            f"(train_lm does this after training)")
    mm = np.memmap(p, dtype=np.float16, mode="r")
    n = mm.shape[0] // d_in
    return mm[: n * d_in].reshape(n, d_in)


def _pool_limit(a) -> int:
    """Tokens to stream, resolving the ``-1`` sentinel per modality.

    The two modalities size their pool by DIFFERENT criteria, so one
    number cannot be the default for both: vision uses the stratified
    pool whole (coverage), LM caps at :data:`LM_POOL_TOKENS` (convergence). ``-1`` asks for whichever
    applies; an explicit value always wins, and ``0`` means no cap.
    """
    lim = int(getattr(a, "full_pool_limit", -1))
    if lim >= 0:
        return lim
    return LM_POOL_TOKENS if a.modality == "lm" else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tree", type=Path, default=None)
    # Repeatable: the activation pool is shared across the tags of one
    # invocation, so several tags cost one backbone pass.
    ap.add_argument("--tag", type=Path, action="append", default=None,
                    metavar="TAG_DIR",
                    help="A single tag directory (the parent of the "
                         "seed dirs). Repeat to pass several; mutually "
                         "exclusive with --tree.")
    ap.add_argument("--modality", choices=("vision", "lm"), required=True)
    ap.add_argument("--step", default="final")
    ap.add_argument("--min-seeds", type=int, default=2)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--n-tokens", type=int, default=20_000)
    # The paper's pool. ``--n-tokens`` sizes an in-memory pool, since
    # ``compute_tag`` materialises ``(n_tokens, d_sae)`` per seed.
    # ``--full-pool`` instead streams a larger pool through
    # ``compute_tag_stream``, whose peak memory follows the batch: for
    # vision the class-stratified ImageNet-val pool (``--per-class``
    # images of every class), for LM the stats-pass activation memmap
    # capped at LM_POOL_TOKENS. The paper's MCC / input R² use this.
    ap.add_argument("--full-pool", action="store_true",
                    help="use the paper's pool (vision: --per-class images "
                         "of every ImageNet-val class; LM: 600k tokens of "
                         "the stats-pass memmap) instead of --n-tokens")
    ap.add_argument("--full-pool-limit", type=int, default=-1,
                    help="cap the streamed tokens. -1 (the DEFAULT) "
                         "means LM_POOL_TOKENS for LM and no cap for "
                         "vision, which stratifies instead; 0 forces no "
                         "cap on either.")
    ap.add_argument("--per-class", type=int, default=3,
                    help="vision --full-pool: images of every ImageNet-val "
                         "class (3 = 3,000 images, 591k CLIP / 771k DINOv2 "
                         "tokens, all 1000 classes present)")
    ap.add_argument("--stream-batch", type=int, default=64 * 197,
                    help="tokens per streamed batch with --full-pool")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--mp-k", type=int, nargs="+", default=None, metavar="K",
                    help="Evaluate MP cells (arch dir ``mp``) at each fixed "
                         "inference round count K, one json per K. Without "
                         "it MP cells use their checkpoint's own inference "
                         "settings.")
    ap.add_argument("--skip-existing", action="store_true")
    ap.add_argument("--out-root", type=Path, default=None,
                    help="write the json under this root (same relative "
                         "path as results/) instead of results/, e.g. to keep "
                         "an alternative run without overwriting the default "
                         "json")
    ap.add_argument("--recompute-pool", action="store_true",
                    help="with --full-pool, recompute tags whose json is "
                         "ALREADY on a full pool; by default they are "
                         "skipped so parallel shards never redo a tag")
    ap.add_argument("--threads", type=int, default=4,
                    help="Tags in flight per (backbone, layer) group. "
                         "Threads, not processes: a tag holds ~1 GB of "
                         "activations per seed and the heavy kernels "
                         "release the GIL. The GPU encode is locked, so "
                         "this overlaps the CPU half only.")
    ap.add_argument("--activation-cache", action=argparse.BooleanOptionalAction,
                    default=True)
    ap.add_argument("--activation-cache-dir", type=Path, default=None)
    # vision
    ap.add_argument("--dataset", default="imagenet")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imagenet-cache-dir", default=None)
    # lm
    ap.add_argument("--hf-dataset", default="HuggingFaceFW/fineweb")
    ap.add_argument("--hf-dataset-config", default=None)
    ap.add_argument("--hf-dataset-split", default="train")
    ap.add_argument("--hf-dataset-text-field", default="text")
    # LM only. Must match the context length the cached pools were built
    # with (it is part of the cache file name).
    ap.add_argument("--context-length", type=int, default=1024)
    ap.add_argument("--batch-seqs", type=int, default=16)
    a = ap.parse_args(argv)

    if not a.tree and not a.tag:
        ap.error("one of --tree / --tag is required")
    tags = list(a.tag) if a.tag else find_tags(a.tree, a.step, a.min_seeds)
    # Resolve the repo from the working directory, not from a tag path,
    # so results always land in this repo's results tree.
    repo = auto_repo_root(Path.cwd())

    def _res_dir(t):
        """Where this tag's json goes: the mirrored ``results/`` dir, or
        the same relative path under ``--out-root``.

        ``--out-root`` lets a second pool size be recorded for a cell
        without overwriting its json in ``results/``.
        """
        base = to_results_path(t, repo)
        if not a.out_root:
            return base
        try:
            rel = base.relative_to(repo / "var" / "results")
        except ValueError:
            rel = base.relative_to(base.anchor)
        return Path(a.out_root) / rel
    def _out(t, k):
        """The json for tag ``t`` at MP round count ``k`` (None: no override)."""
        suffix = "" if k is None else f"_mp_k{k}"
        return _res_dir(t) / f"code_metrics_step_{a.step}{suffix}.json"

    def _mp(k):
        return (None if k is None
                else {"inference_mode": "fixed", "inference_iters": int(k)})
    if a.activation_cache_dir is None:
        a.activation_cache_dir = repo / "var" / "cache" / "activations"
    print(f"[code-metrics] {len(tags)} tag(s), band={list(REPORT_BAND)}",
          flush=True)

    # Group by (backbone, layer) so the backbone forward is paid once per
    # group instead of once per tag; the pool is dropped when its group
    # ends. Vision and LM backbones live in different registries, hence
    # the per-modality resolver.
    _resolve = (resolve_lm_backbone_layer if a.modality == "lm"
                else resolve_backbone_layer)
    groups: dict = {}
    for t in tags:
        try:
            bb, layer = _resolve(t)
        except Exception:  # noqa: BLE001
            print(f"  [skip] no backbone/layer: {t}", flush=True)
            continue
        groups.setdefault((bb, int(layer)), []).append(t)

    ok = skipped = failed = 0
    for (bb, layer), group in sorted(groups.items()):
        print(f"########## {bb} layer{layer}: {len(group)} tag(s)", flush=True)
        # One job per (tag, MP round count); ``None`` = the checkpoint's
        # own inference settings, which is every non-MP tag.
        jobs = [(t, k) for t in group
                for k in ((a.mp_k or [None]) if "mp" in t.parts else [None])]
        todo = [(t, k) for t, k in jobs
                if not ((a.full_pool and not a.recompute_pool
                         and _on_new_pool(_out(t, k)))
                        or (a.skip_existing and _out(t, k).is_file()))]
        skipped += len(jobs) - len(todo)
        if not todo:
            continue
        try:
            z_pool = z_full = None
            if not a.full_pool:
                z_pool = _pool_for(a.modality, bb, layer, a)
            elif a.modality == "vision":
                z_full = vision_stratified_pool(
                    bb, layer, per_class=a.per_class, dataset=a.dataset,
                    split=a.split, cache_dir=a.imagenet_cache_dir,
                    device=f"cuda:{a.gpu}",
                    act_cache=(Path(a.activation_cache_dir)
                               / f"{bb}_layer{layer}_{a.dataset}_{a.split}"
                                 f"_perclass{a.per_class}.npy"))
            else:
                z_full = lm_full_pool(bb, repo)
        except Exception as e:  # noqa: BLE001
            failed += len(todo)
            print(f"  [fail] pool for {bb} layer{layer}: "
                  f"{type(e).__name__}: {e}", flush=True)
            continue
        # Threads over tags, not processes: a tag holds ~1 GB of
        # activations per seed, too much to pickle into a worker, and the
        # dominant scipy kernels (linear_sum_assignment,
        # connected_components) release the GIL.
        #
        # GPU work is serialised by a lock (overlapping it gains nothing
        # and can OOM); only the CPU half overlaps.
        import threading  # noqa: PLC0415
        from concurrent.futures import ThreadPoolExecutor  # noqa: PLC0415

        gpu_lock = threading.Lock()

        def _one_stream(job):
            """One tag, over the streamed full pool."""
            t, k = job
            out = _out(t, k)
            # Re-checked here: another shard may have finished this tag
            # since ``todo`` was built.
            if a.full_pool and not a.recompute_pool and _on_new_pool(out):
                return (t, "skip-new-pool")
            def _open(sd):
                with gpu_lock:
                    return load_seed_sae(sd, device=f"cuda:{a.gpu}",
                                         mp_override=_mp(k))

            def _enc_batch(sae, zb):
                with gpu_lock:
                    return encode_batch_with(sae, zb, device=f"cuda:{a.gpu}",
                                             batch_size=a.batch_size)
            try:
                payload = compute_tag_stream(
                    t,
                    batches=_chunks(z_full, a.stream_batch,
                                    limit=_pool_limit(a)),
                    encode_batch=_enc_batch, open_seed=_open,
                    load_decoder_fn=load_decoder, step=a.step,
                    alive_dataset=("imagenet" if a.modality == "vision"
                                   else "fineweb"),
                    alive_split=("val" if a.modality == "vision"
                                 else "train"))
                if payload is None:
                    return (t, "nothing encodable")
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(payload, indent=1))
                # The caller counts only the literal "ok" as success.
                print(f"    [{t.name}] {payload['n_tokens']:,} tokens",
                      flush=True)
                return (t, "ok")
            except Exception as e:  # noqa: BLE001
                return (t, f"{type(e).__name__}: {e}")

        def _one(job):
            t, k = job
            out = _out(t, k)

            def _encode(sd):
                with gpu_lock:
                    return seed_encode(z_pool, sd, device=f"cuda:{a.gpu}",
                                       batch_size=a.batch_size,
                                       mp_override=_mp(k))
            try:
                done = write_tag(
                    t, out, z_pool=z_pool, encode=_encode,
                    load_decoder_fn=load_decoder, step=a.step)
                return (t, "ok" if done else "nothing encodable")
            except Exception as e:  # noqa: BLE001
                return (t, f"{type(e).__name__}: {e}")

        n_thread = max(1, min(int(a.threads or 4), len(todo)))
        print(f"  {len(todo)} tag(s) on {n_thread} thread(s)", flush=True)
        with ThreadPoolExecutor(n_thread) as pool:
            _fn = _one_stream if a.full_pool else _one
            for i, (t, r) in enumerate(pool.map(_fn, todo), 1):
                if r == "ok":
                    ok += 1
                elif r == "skip-new-pool":
                    skipped += 1
                    print(f"  [skip] {t.name}: already on a full pool",
                          flush=True)
                else:
                    failed += 1
                    print(f"  [fail] {t}: {r}", flush=True)
                if i % 20 == 0:
                    print(f"  {i}/{len(todo)}  ok={ok} fail={failed}",
                          flush=True)
    print(f"[code-metrics] done: ok={ok} skip={skipped} fail={failed}",
          flush=True)
    return 1 if failed and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
