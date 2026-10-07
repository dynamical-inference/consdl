"""Compute per-checkpoint LM-SAE stats (top-K activating tokens + per-token magnitude).

LM counterpart of :mod:`consdl.cli.stats_vision`. Streams a text corpus
through ``LMBackbone`` and the trained SAE and writes:

  * ``top_idx.pt``    — ``(d_sae, K)`` packed-sequence indices of the
                        top-K activating tokens per latent.
  * ``top_pos.pt``    — ``(d_sae, K)`` token position **within** each
                        of those packed sequences.
  * ``top_act.pt``    — ``(d_sae, K)`` activation magnitudes.
  * ``mean_act_per_token.pt`` — ``(d_sae,)`` whole-corpus ``E[|a|]``.
  * ``max_act_per_token.pt``  — ``(d_sae,)`` running max.
  * ``sparsity_per_token.pt`` — ``(d_sae,)`` fraction of tokens where
                                ``|a| > eps`` (``ZERO_EPS``).
  * ``l0_per_token.json``     — scalar L0 / L1 family at every
                                ``--sparsity-thresholds`` value.
  * ``input_ids.pt``  — ``(N_samples, context_length)`` token IDs of the
                        same packed corpus, so any ``(top_idx, top_pos)``
                        span can be decoded without re-streaming.
  * ``meta.json``    — corpus identity, tokenizer name and stream seed.

Unlike the vision path (``SAEInterpreter.compute_stats``, whose ``top_idx``
is per image), top-K here is tracked per token: which token of which packed
sequence fires.

Usage::

    python -m consdl.cli.stats_lm --gpu 0 --ckpt_path <seed_dir>
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import re

import numpy as np


ZERO_EPS = 1e-6


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--ckpt_path", type=str, required=True,
                   help="Path to a seed dir (with sibling 'backbone' + "
                        "'final') or directly to its step dir.")
    p.add_argument("--num-workers", type=int, default=8,
                   help="Dataloader workers; 0 assembles batches inline "
                        "on the main process.")
    p.add_argument("--activation-cache-dir", type=str, default=None,
                   help="Cache the backbone activation pool as a float16 "
                        "memmap keyed on (backbone, layer, corpus, split, "
                        "context, num_samples). The pool does not depend on "
                        "the SAE, so without this every cell in a sweep "
                        "recomputes the identical tensor. ~31 GB at "
                        "20k x 1024 x 768.")
    p.add_argument("--token-cache-tokens", type=int, default=0,
                   help="Read the stats corpus from a pre-tokenised uint16 "
                        "memmap instead of streaming. Streaming re-derives "
                        "the corpus per cell from a live HF shard shuffle, "
                        "so a shard refresh mid-sweep would silently give "
                        "later cells different text; the memmap pins it. "
                        "Use the same value the sweep trained with so the "
                        "stats corpus is the training corpus. 0 = stream.")
    p.add_argument("--token-cache-dir", type=str, default="var/cache/lm_tokens")
    p.add_argument("--dataset", type=str, default="fineweb",
                   help="Short tag written into the stats-dir name "
                        "(matches the trainer's --dataset).")
    p.add_argument("--dataset-split", type=str, default="train",
                   choices=["train", "val"],
                   help="HF split used as the stats corpus.  Default "
                        "``train`` because FineWeb (and most LM corpora) "
                        "ship only the train split; ``val`` against such "
                        "a dataset raises ``Bad split: val``.  Stats "
                        "files encode this in their dir name "
                        "(stats.<dataset>.split_<split>.seed_no/).")
    p.add_argument("--hf-dataset", type=str, default="HuggingFaceFW/fineweb",
                   help="HuggingFace dataset id to stream.  Pair with "
                        "the corresponding ``--dataset`` short name so "
                        "the stats dir matches what the viewer expects.")
    p.add_argument("--hf-dataset-config", type=str, default=None,
                   help="Optional HF dataset config (e.g. ``sample-10BT``).")
    p.add_argument("--hf-dataset-text-field", type=str, default="text")
    p.add_argument("--context-length", type=int, default=1024,
                   help="Tokens per packed sequence.  Must match the "
                        "trainer's --context-length for the activations "
                        "to be on the manifold the SAE saw.")
    p.add_argument("--num-samples", type=int, default=20_000,
                   help="How many packed sequences to stream.  "
                        "20k x 1024 = ~20.5M token positions, enough for "
                        "the tail of a 12k-latent dictionary.")
    p.add_argument("--batch-seqs", type=int, default=32,
                   help="Sequences per forward pass.  32 x 1024 = 32,768 "
                        "tokens/batch — fits a ~12k-latent SAE on an "
                        "80 GB GPU with headroom.")
    p.add_argument("--stats-top-k", type=int, default=100,
                   help="K per latent in the top-K accumulator.  100 "
                        "matches the vision default.")
    p.add_argument("--stream-seed", type=int, default=0,
                   help="Shuffle seed for the streaming HF iterator.  "
                        "Saved in meta.json so the viewer reproducibly "
                        "matches activations back to tokens.")
    p.add_argument("--sparsity-thresholds", type=float, nargs="+",
                   default=[0.001, 0.01, 0.05, 0.1, 0.5, 1.0],
                   help="Thresholds for L0/L1 per-token reporting.")
    p.add_argument("--checkpoint-step", type=str, default="final",
                   help="'final', a numeric step, or 'all'.")
    p.add_argument("--overwrite", action="store_true",
                   help="Recompute even if top_act.pt already exists.")
    p.add_argument("--progress-file", type=str, default=None)
    return p.parse_args()


args = parse_args()


# Load .env and pin the GPU before importing torch / transformers so
# CUDA_VISIBLE_DEVICES takes effect.
from dotenv import load_dotenv  # noqa: E402

from consdl.io.paths import REPO_ROOT  # noqa: E402
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ── Heavy imports (deferred until after GPU pinning) ────────────────────


import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# Silence the HF datasets "Resolving data files" progress bar.
try:
    import datasets as _hf_datasets  # noqa: E402
    _hf_datasets.disable_progress_bar()
    _hf_datasets.logging.set_verbosity_error()
except Exception:  # noqa: BLE001
    pass

from consdl.io.checkpoint import load_sae  # noqa: E402
from consdl.data.lm_backbones import LM_BACKBONE_MAP  # noqa: E402
from consdl.cli.train_lm import (  # noqa: E402
    LMBackbone,
    build_lm_dataloader,
)


# ── Path helpers ────────────────────────────────────────────────────────


def expand_ckpt_path(p: str, step: str = "final") -> list[pathlib.Path]:
    """Expand a root path to a list of step dirs.

    If the given path has a sibling ``backbone/``, treat it as a step dir;
    otherwise recurse looking for step dirs (named ``final`` or a digit).
    """
    root = pathlib.Path(p)
    if (root.parent / "backbone").is_dir() and root.is_dir():
        return [root]

    def _is_step_dir(d: pathlib.Path) -> bool:
        return (d.is_dir() and (d.parent / "backbone").is_dir()
                and any(d.glob("*.json")))

    if step == "all":
        candidates = sorted(
            d for d in root.rglob("*")
            if _is_step_dir(d) and (d.name == "final" or d.name.isdigit())
        )
    else:
        candidates = sorted(d for d in root.rglob(step) if _is_step_dir(d))
    return candidates or [root]


def _resolve_lm_backbone_layer(seed_dir: pathlib.Path) -> tuple[str, int]:
    """Parse ``(backbone_short, layer)`` from an LM seed path.

    Convention: ``…/<dataset>/<arch>/x<E>/<backbone>_layer<L>/<tag>/seed<N>/``.
    """
    for part in reversed(seed_dir.parts):
        if "_layer" not in part:
            continue
        head, _, tail = part.rpartition("_layer")
        if head in LM_BACKBONE_MAP and tail.isdigit():
            return head, int(tail)
    raise ValueError(
        f"Could not infer (backbone, layer) from path {seed_dir} — "
        f"expected a '<backbone>_layer<L>' segment with "
        f"backbone in {sorted(LM_BACKBONE_MAP.keys())}."
    )


# ── Streaming top-K accumulator ─────────────────────────────────────────


def _accumulate(
    backbone: LMBackbone,
    sae,
    loader,
    *,
    act_cache_path: pathlib.Path | None = None,
    d_sae: int,
    context_length: int,
    num_samples: int,
    K: int,
    sparsity_thresholds: list[float],
    device: torch.device,
    progress_file: str | None,
) -> dict:
    """Stream packed-sequence batches through backbone+SAE and accumulate
    all stats in one pass.

    Returns a dict of CPU tensors / scalars ready to save.
    """
    # Running top-K per latent — stored as (d_sae, K).  ``top_act`` is
    # initialised at -inf so the first batch's real activations always
    # win the topk.
    top_act = torch.full(
        (d_sae, K), float("-inf"), dtype=torch.float32, device=device,
    )
    top_sample = torch.full(
        (d_sae, K), -1, dtype=torch.int64, device=device,
    )
    top_pos = torch.full(
        (d_sae, K), -1, dtype=torch.int64, device=device,
    )

    # Reconstruction accumulators: float64 running sums rather than a mean
    # of per-batch means, since batch sizes are not all equal.
    #
    # Sums are kept per dimension so that SST is
    #
    #     sum_n ||z_n - zbar||^2  =  sum_d sum_n (z_nd - zbar_d)^2
    #
    # with ``zbar`` the mean activation vector (the paper's definition,
    # sklearn's ``multioutput="variance_weighted"``). Centring about a single
    # scalar mean instead would inflate SST by the spread of the
    # per-dimension means and overstate R2.
    sse: "float | None" = 0.0
    z_sum: "torch.Tensor | None" = None    # (d_in,)  running sum
    z_sqsum: "torch.Tensor | None" = None  # (d_in,)  running sum of squares
    z_rows = 0                             # tokens, NOT elements
    z_elems = 0                            # tokens * d_in, for l2_loss

    # Per-token magnitude / sparsity accumulators.
    abs_sum = torch.zeros(d_sae, dtype=torch.float64, device=device)
    max_act = torch.zeros(d_sae, dtype=torch.float32, device=device)
    eps_active = torch.zeros(d_sae, dtype=torch.int64, device=device)
    thr_active: dict[float, int] = {float(t): 0 for t in sparsity_thresholds}
    total_positions = 0
    total_eps_nonzero = 0
    # Per-token active-count histogram: index i = # tokens with exactly
    # i active latents. Same format as the vision stats.
    active_count_hist = torch.zeros(d_sae + 1, dtype=torch.int64, device=device)

    # Raw token IDs of every packed sequence, to reconstruct the text span
    # at any (sample_idx, pos).
    id_chunks: list[torch.Tensor] = []

    sample_offset = 0
    seen_samples = 0
    seen_batches = 0
    last_progress_report = 0

    sae.eval()
    backbone.model.eval()


    # Activation cache. The pool depends only on
    # (backbone, hook, corpus, split, context, num_samples), not on the SAE,
    # so one pool serves every SAE in a sweep. It is read only if its
    # ``.done`` sidecar exists, so a half-written cache is recomputed. The
    # write side is opened lazily because d_in is only known after the first
    # batch.
    _acache_read = _acache_write = None
    _acache_pos = 0
    _acache_rows = int(num_samples) * int(context_length)
    if act_cache_path is not None:
        _done = act_cache_path.with_suffix(act_cache_path.suffix + ".done")
        if _done.exists() and act_cache_path.exists():
            _bytes = act_cache_path.stat().st_size
            _d_in = _bytes // (_acache_rows * 2)
            try:
                _valid = int(_done.read_text().strip())
            except (ValueError, OSError):
                _valid = 0
            if _d_in > 0 and _bytes == _acache_rows * _d_in * 2 and _valid > 0:
                _full = np.memmap(act_cache_path, dtype=np.float16,
                                  mode="r", shape=(_acache_rows, _d_in))
                _acache_read = _full[:_valid]
                print(f"[stats] activation cache HIT "
                      f"({_valid:,}x{_d_in}) {act_cache_path}")
            else:
                print(f"[stats] activation cache size mismatch — recomputing")

    with torch.no_grad():
        for batch in loader:
            if seen_samples >= num_samples:
                break
            ids = batch["input_ids"]  # (B, T) CPU
            B, T = int(ids.shape[0]), int(ids.shape[1])
            if T != context_length:
                raise RuntimeError(
                    f"context-length mismatch: batch yielded T={T} but "
                    f"--context-length={context_length}.  The trainer "
                    f"and stats run must use the same value."
                )
            id_chunks.append(ids.cpu())

            # Forward + encode. ``forward_to_z`` flattens to (B*T, d_in);
            # ``sae.encode`` returns (B*T, d_sae). Cached activations are
            # stored as float16, which is enough for statistics.
            _n = int(ids.shape[0]) * int(ids.shape[1])
            if _acache_read is not None and _acache_pos + _n <= _acache_read.shape[0]:
                z = torch.from_numpy(
                    np.ascontiguousarray(_acache_read[_acache_pos:_acache_pos + _n]),
                ).to(device=device, dtype=torch.float32)
                _acache_pos += _n
            else:
                z = backbone.forward_to_z(batch).to(device)
                if act_cache_path is not None and _acache_read is None:
                    if _acache_write is None:
                        act_cache_path.parent.mkdir(parents=True, exist_ok=True)
                        _acache_write = np.memmap(
                            act_cache_path, dtype=np.float16, mode="w+",
                            shape=(_acache_rows, int(z.shape[-1])))
                        print(f"[stats] building activation cache "
                              f"({_acache_rows * int(z.shape[-1]) * 2 / 1e9:.1f} GB)"
                              f" {act_cache_path}")
                    if _acache_pos + _n <= _acache_write.shape[0]:
                        _acache_write[_acache_pos:_acache_pos + _n] = (
                            z.detach().to(torch.float16).cpu().numpy())
                        _acache_pos += _n
            enc_out = sae.encode(z)
            a = enc_out[0] if isinstance(enc_out, tuple) else enc_out

            # Reconstruction over the same corpus as the activation stats,
            # with the same estimator as vision: R2 = 1 - SSE/SST, SST taken
            # about the mean of z.
            try:
                _zh = sae.decode(a if not isinstance(enc_out, tuple)
                                 else enc_out[0])
                _d = (z - _zh).to(torch.float64)
                sse += float((_d * _d).sum().item())
                _zd = z.to(torch.float64).reshape(-1, z.shape[-1])
                if z_sum is None:
                    z_sum = torch.zeros(_zd.shape[1], dtype=torch.float64,
                                        device=_zd.device)
                    z_sqsum = torch.zeros_like(z_sum)
                z_sum += _zd.sum(dim=0)
                z_sqsum += (_zd * _zd).sum(dim=0)
                z_rows += int(_zd.shape[0])
                z_elems += int(_zd.numel())
            except Exception as _e:  # noqa: BLE001
                if sse is not None:
                    print(f"  [stats] decode unavailable, R2 skipped: "
                          f"{type(_e).__name__}: {_e}")
                sse = None

            a = a.detach().abs().float()  # (B*T, d_sae)

            # Per-token magnitude / sparsity stats.
            abs_sum.add_(a.sum(dim=0).to(torch.float64))
            batch_max = a.max(dim=0).values
            torch.maximum(max_act, batch_max, out=max_act)
            eps_mask = a > ZERO_EPS
            eps_active.add_(eps_mask.sum(dim=0).to(torch.int64))
            total_eps_nonzero += int(eps_mask.sum().item())
            # Per-token active-count histogram accumulator.
            per_token_l0 = eps_mask.sum(dim=-1).to(torch.int64)   # (B*T,)
            active_count_hist.add_(
                torch.bincount(per_token_l0, minlength=d_sae + 1),
            )
            for t in sparsity_thresholds:
                thr_active[float(t)] += int((a >= float(t)).sum().item())
            total_positions += int(a.shape[0])

            # ── Top-K update (vectorised across latents) ─────────────
            # Combined matrix per latent: (B*T + K, d_sae).
            #   - new candidates: a               (B*T, d_sae)
            #   - existing top  : top_act         (d_sae, K) → transpose
            # Indices into combined: rows [0, B*T) = new candidates;
            #                       rows [B*T, B*T+K) = existing entries.
            combined = torch.cat([a, top_act.transpose(0, 1)], dim=0)  # (B*T+K, d_sae)
            top_vals, top_local = torch.topk(combined, K, dim=0)
            # top_vals / top_local: (K, d_sae)
            came_from_new = top_local < (B * T)
            # New-candidate position within batch → which (b, t).
            new_local = top_local.clamp_max(B * T - 1)  # only used where came_from_new
            new_b = (new_local // T).to(torch.int64)    # (K, d_sae)
            new_t = (new_local %  T).to(torch.int64)    # (K, d_sae)
            new_sample_idx = sample_offset + new_b      # (K, d_sae)

            # Old-entry index within current top buffer.
            old_local = (top_local - (B * T)).clamp_min(0)  # (K, d_sae)
            # gather along K: top_sample[d, old_local[k,d]]
            old_sample = torch.gather(
                top_sample.transpose(0, 1), 0, old_local,
            )  # (K, d_sae)
            old_pos = torch.gather(
                top_pos.transpose(0, 1), 0, old_local,
            )  # (K, d_sae)

            new_sample_chosen = torch.where(
                came_from_new, new_sample_idx, old_sample,
            )
            new_pos_chosen = torch.where(
                came_from_new, new_t, old_pos,
            )

            top_act = top_vals.transpose(0, 1).contiguous()
            top_sample = new_sample_chosen.transpose(0, 1).contiguous()
            top_pos = new_pos_chosen.transpose(0, 1).contiguous()

            sample_offset += B
            seen_samples += B
            seen_batches += 1

            # Periodic progress for the launcher's per-GPU progress file.
            if (progress_file
                    and (seen_samples - last_progress_report) >= 512):
                last_progress_report = seen_samples
                try:
                    pathlib.Path(progress_file).write_text(
                        f"{seen_samples}/{num_samples}\n"
                    )
                except Exception:  # noqa: BLE001
                    pass

            if seen_batches % 50 == 0:
                print(f"  [stats] {seen_samples:,}/{num_samples:,} samples "
                      f"({seen_batches} batches)", flush=True)

    if seen_samples == 0:
        raise RuntimeError("loader produced zero batches — corpus empty?")

    # Concatenate saved IDs + truncate to the actual sample count.
    all_ids = torch.cat(id_chunks, dim=0)[:seen_samples].contiguous()

    # Sort top-K descending per latent for downstream readability.
    sort_vals, sort_idx = torch.sort(top_act, dim=1, descending=True)
    top_act_sorted = sort_vals
    top_sample_sorted = torch.gather(top_sample, 1, sort_idx)
    top_pos_sorted = torch.gather(top_pos, 1, sort_idx)

    n = float(total_positions)
    # Truncate the trailing all-zero tail from the L0 histogram — no
    # token has more than ``d_sae`` actives, and typically many fewer.
    ach = active_count_hist.cpu()
    last_nz = int(torch.nonzero(ach).max()) + 1 if ach.any() else 1
    ach = ach[:last_nz]
    out = {
        "top_idx": top_sample_sorted.cpu(),                      # (d_sae, K)
        "top_pos": top_pos_sorted.cpu(),                         # (d_sae, K)
        "top_act": top_act_sorted.cpu(),                         # (d_sae, K)
        "mean_act_per_token": (abs_sum / n).float().cpu(),       # (d_sae,)
        "max_act_per_token": max_act.cpu(),                       # (d_sae,)
        "sparsity_per_token": (eps_active.double() / n).float().cpu(),
        # Alias for ``sparsity_per_token``, which is per latent despite its
        # name: entry k is the fraction of token positions where latent k
        # fired, and mean(firing_rate_per_latent) * d_sae == l0_per_token.
        "firing_rate_per_latent": (eps_active.double() / n).float().cpu(),
        # Same file name as the vision stats.
        "active_count_hist_per_token": ach,                        # (K,)
        "input_ids": all_ids,                                     # (N, T)
        "scalars": {
            "n_samples": int(seen_samples),
            "n_positions": int(total_positions),
            "l0_per_token_0": (total_eps_nonzero / n),
            **{
                f"l0_per_token_{float(t)}": (thr_active[float(t)] / n)
                for t in sparsity_thresholds
            },
            "l1_per_token": float(abs_sum.sum().item() / n),
            # SST about the mean activation vector; see the accumulator
            # comment above.
            **({} if sse is None or not z_elems or z_sum is None else {
                "r2_score": float(1.0 - sse / max(float(
                    (z_sqsum - z_sum * z_sum / max(z_rows, 1)).sum().item()
                ), 1e-30)),
                "l2_loss": float(sse / z_elems),
            }),
        },
    }
    if _acache_write is not None:
        # Mark complete only here, so an interrupted write is recomputed.
        _acache_write.flush()
        # Record the row count actually written: with drop_last the pool can
        # be shorter than num_samples*context_length, and the unwritten tail
        # rows are zeros.
        act_cache_path.with_suffix(
            act_cache_path.suffix + ".done").write_text(f"{_acache_pos}\n")
        print(f"[stats] activation cache written ({_acache_pos:,} rows)")
    return out


def _save_stats(out_dir: pathlib.Path, out: dict, meta: dict) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in ("top_idx", "top_pos", "top_act",
                "mean_act_per_token", "max_act_per_token",
                "sparsity_per_token", "firing_rate_per_latent",
                "active_count_hist_per_token",
                "input_ids"):
        torch.save(out[key], out_dir / f"{key}.pt")
    (out_dir / "l0_per_token.json").write_text(
        json.dumps(out["scalars"], indent=2)
    )
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"[stats] saved to {out_dir}")
    print(f"  d_sae × K  : {tuple(out['top_act'].shape)}")
    print(f"  samples    : {out['scalars']['n_samples']:,}")
    print(f"  positions  : {out['scalars']['n_positions']:,}")
    print(f"  l1_per_tok : {out['scalars']['l1_per_token']:.4f}")
    print(f"  l0_per_tok : {out['scalars']['l0_per_token_0']:.4f}  "
          f"(|a|>{ZERO_EPS})")


def main() -> None:
    for ckpt_path in expand_ckpt_path(args.ckpt_path, step=args.checkpoint_step):
        # Resolve final_dir + seed_dir.  ``ckpt_path`` is either the seed
        # dir itself (in which case final_dir = ckpt_path / "final") or
        # the step dir directly.
        if (ckpt_path.parent / "backbone").is_dir():
            final_dir = ckpt_path
            seed_dir = ckpt_path.parent
        else:
            final_dir = ckpt_path / "final"
            seed_dir = ckpt_path

        from consdl.io.paths import stats_write_dir  # noqa: PLC0415
        stats_dir = stats_write_dir(final_dir, args.dataset,
                                    args.dataset_split)
        if (stats_dir / "top_act.pt").exists() and not args.overwrite:
            print(f"[skip] stats already present at {stats_dir}  "
                  f"(pass --overwrite to recompute)")
            continue

        backbone_short, layer = _resolve_lm_backbone_layer(seed_dir)
        cfg = LM_BACKBONE_MAP[backbone_short]
        hf_name = cfg["hf_name"]
        hook_layer_path = cfg["hook_layer_fmt"].format(int(layer))

        print(f"[lm-stats gpu={args.gpu}] {seed_dir}")
        print(f"  backbone   : {backbone_short}  ({hf_name})")
        print(f"  hook       : {hook_layer_path}")
        print(f"  corpus     : {args.hf_dataset}"
              f"{f' ({args.hf_dataset_config})' if args.hf_dataset_config else ''}"
              f"  split={args.dataset_split}")
        print(f"  ctx × N    : {args.context_length} × {args.num_samples:,}  "
              f"(~{args.context_length * args.num_samples:,} tokens)")
        print(f"  top-K      : {args.stats_top_k}")

        device = torch.device("cuda")
        backbone = LMBackbone(
            model_hf_name=hf_name,
            hook_layer_path=hook_layer_path,
            device="cuda",
        )

        sae = load_sae(final_dir)
        sae = sae.to(device)
        if hasattr(sae, "device"):
            sae.device = device
        d_sae = int(sae.d_sae)
        print(f"  d_sae      : {d_sae}")

        tokenizer = AutoTokenizer.from_pretrained(hf_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        loader = build_lm_dataloader(
            hf_dataset=args.hf_dataset,
            hf_config=args.hf_dataset_config,
            split=args.dataset_split,
            text_field=args.hf_dataset_text_field,
            tokenizer=tokenizer,
            context_length=int(args.context_length),
            batch_seqs=int(args.batch_seqs),
            seed=int(args.stream_seed),
            token_cache_tokens=int(args.token_cache_tokens),
            token_cache_dir=str(args.token_cache_dir),
            num_workers=int(args.num_workers),
        )

        # Cache key: everything the pool depends on. The SAE is deliberately
        # excluded so the pool is shared across SAEs.
        _acp = None
        if args.activation_cache_dir:
            _key = (f"{getattr(backbone, 'model_hf_name', 'lm')}"
                    f"_{getattr(backbone, 'hook_target_name', 'hook')}"
                    f"_{args.hf_dataset}_{args.dataset_split}"
                    f"_ctx{int(args.context_length)}_n{int(args.num_samples)}")
            _acp = pathlib.Path(args.activation_cache_dir) / (
                re.sub(r"[^A-Za-z0-9._-]", "_", _key) + ".f16.mm")
        out = _accumulate(
            backbone, sae, loader,
            act_cache_path=_acp,
            d_sae=d_sae,
            context_length=int(args.context_length),
            num_samples=int(args.num_samples),
            K=int(args.stats_top_k),
            sparsity_thresholds=list(args.sparsity_thresholds or []),
            device=device,
            progress_file=(
                args.progress_file
                or f"/tmp/compute_stats_lm_gpu{args.gpu}.progress"
            ),
        )
        meta = {
            "dataset": args.dataset,
            "split": args.dataset_split,
            "hf_dataset": args.hf_dataset,
            "hf_dataset_config": args.hf_dataset_config,
            "hf_dataset_text_field": args.hf_dataset_text_field,
            "context_length": int(args.context_length),
            "num_samples_requested": int(args.num_samples),
            "num_samples_actual": int(out["scalars"]["n_samples"]),
            "batch_seqs": int(args.batch_seqs),
            "stats_top_k": int(args.stats_top_k),
            "stream_seed": int(args.stream_seed),
            "sparsity_thresholds": list(args.sparsity_thresholds or []),
            "model_hf_name": hf_name,
            "hook_target_name": hook_layer_path,
            "tokenizer_hf_name": hf_name,
            "d_sae": d_sae,
            "d_in": int(backbone.hook_target_dim),
        }
        _save_stats(stats_dir, out, meta)

        # Free the backbone weights before the next step dir.
        del backbone, sae
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
