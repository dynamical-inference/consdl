"""Train an SAE on a single LM residual-stream layer.

LM counterpart of :mod:`consdl.cli.train_vision`. Same CLI and
architecture dispatch, different data pipeline:

  * **dataset** — HuggingFace text dataset streamed (default
    ``HuggingFaceFW/fineweb``) + ``AutoTokenizer`` + sequence packing
    into ``--context-length`` chunks. Only the shards actually read are
    downloaded. ``--token-cache-tokens`` instead pins a fixed token set
    in a local memmap.
  * **backbone** — :class:`LMBackbone` wraps
    ``AutoModelForCausalLM.from_pretrained(<hf_name>)`` with a forward
    hook on the transformer-block submodule at layer ``--hook-layer``.
    Hook output is the **resid_post** residual stream:
    ``(B, T, d_in)`` → reshaped to ``(B·T, d_in)``.

Everything below the activation tensor is shared with the vision
trainer: the SAE families (relu / topk / batchtopk / matryoshka /
mp / fista / isae) and the trainer classes.

Supported LM backbones are listed in :data:`LM_BACKBONE_MAP`
(``consdl.data.lm_backbones``). After training, statistics are computed
with :mod:`consdl.cli.stats_lm` unless ``--no-stats`` is given.

Usage::

    python -m consdl.cli.train_lm --gpu 0 --backbone pythia-160m \
        --architecture topk --seed 0
"""

from __future__ import annotations

import argparse
import sys


# ── LM backbone table ──────────────────────────────────────────────
from consdl.data.lm_backbones import LM_BACKBONE_MAP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    # ── Hardware / output ──────────────────────────────────────────
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--experiment-dir", type=str, default=None)
    p.add_argument("--progress-file", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--data-seed", type=int, default=None,
                   help="Seed for the DATA stream only (shard shuffle / token-cache "
                        "order). Defaults to --seed, which couples data order to "
                        "the training seed. Set it FIXED across training seeds to mirror the "
                        "iSAE reference, which pins seed.data=42 and varies only "
                        "seed.train -- otherwise each seed trains on DIFFERENT "
                        "documents, which depresses any cross-seed metric.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--total-checkpoints", type=int, default=3)
    p.add_argument("--log-interval", type=int, default=1000)

    # ── Backbone + hook ────────────────────────────────────────────
    p.add_argument("--backbone", type=str, default="pythia-160m",
                   choices=sorted(LM_BACKBONE_MAP.keys()))
    p.add_argument("--hook-layer", type=int, default=None,
                   help="Transformer-block index. Default = the "
                        "backbone's canonical layer (LM_BACKBONE_MAP).")

    # ── Dataset / tokenisation ─────────────────────────────────────
    p.add_argument("--dataset", type=str, default="fineweb",
                   help="Logical dataset short-name for the checkpoint tree.")
    p.add_argument("--hf-dataset", type=str, default="HuggingFaceFW/fineweb",
                   help="HuggingFace dataset id to stream "
                        "(the dictionary_learning reference default).")
    p.add_argument("--hf-dataset-config", type=str, default=None,
                   help="Optional HF dataset config (e.g. ``sample-10BT`` "
                        "for FineWeb's smaller subset).")
    p.add_argument("--hf-dataset-split", type=str, default="train")
    p.add_argument("--hf-dataset-text-field", type=str, default="text")
    p.add_argument("--context-length", type=int, default=128,
                   help="Tokens per sequence (the dictionary_learning reference "
                        "uses 1024).")
    p.add_argument("--token-cache-tokens", type=int, default=0,
                   help="Pre-tokenise this many tokens into a uint16 memmap "
                        "and train from it instead of the live stream. "
                        "Streaming + ds.shuffle(seed) changes WHICH documents "
                        "a seed reads, so seeds differ in data as well as in "
                        "init; with the cache the token set is pinned and seed "
                        "only permutes sequence order. 500M = 1.0 GB. "
                        "0 (default) keeps the streaming path.")
    p.add_argument("--token-cache-dir", type=str, default="var/cache/lm_tokens")
    p.add_argument("--bf16", action="store_true", default=False,
                   help="bfloat16 autocast over the whole training step — "
                        "backbone forward, SAE forward, loss and backward. "
                        "Parameters and optimizer state stay fp32: autocast "
                        "casts the matmuls, not the master weights, which is "
                        "what keeps Adam's updates from being quantised "
                        "away by bf16's 7-bit mantissa.")
    p.add_argument("--normalize-activations", action="store_true", default=False,
                   help="Divide each per-token activation by its L2 norm before "
                        "feeding the SAE. Not implemented: passing it raises an error.")

    # ── Training shape ─────────────────────────────────────────────
    p.add_argument("--total-samples", type=int, default=10_000_000,
                   help="Token budget (the dictionary_learning reference uses 500_000_000).")
    p.add_argument("--batch-size", type=int, default=4096,
                   help="SAE batch size, in tokens.")
    p.add_argument("--learning-rate", "--lr", dest="learning_rate",
                   type=float, default=5e-5)
    p.add_argument("--lr-warmup-steps", type=int, default=1_000)
    p.add_argument("--sparsity-warmup-steps", type=int, default=5_000)
    p.add_argument("--decay-start-fraction", type=float, default=0.8,
                   help="LR decay start as a fraction of total steps. 0.8 is "
                        "the dictionary_learning reference default.")

    # ── SAE family + per-arch knobs (same as train_vision) ─────────
    p.add_argument("--architecture", type=str, default="topk",
                   choices=["relu", "topk", "batchtopk", "matryoshka",
                            "mp", "fista", "isae", "isae_me"])
    p.add_argument("--l1-coeff", type=float, default=0.02)

    # ── iSAE / iSAE-ME (AbsTopK + aRIP regularization) ────────────────
    # Port of third_party/isaes (rsae.RIPTopK). ``isae`` = single-step
    # AbsTopK + aRIP; ``isae_me`` = same with a T-step iterative encoder.
    p.add_argument("--rip-weight", type=float, default=0.1,
                   help="aRIP regularization weight (paper: 0.1).")
    p.add_argument("--rip-multiplier", type=int, default=2,
                   help="k' = k * multiplier for the aRIP Gram submatrices.")
    p.add_argument("--rip-weighted", action="store_true", default=False,
                   help="Use real code values (True) vs random-Gaussian "
                        "substitutes on the same support (False, paper default).")
    p.add_argument("--isae-l1", type=float, default=0.01,
                   help="L1 coefficient for iSAE codes (fixed scalar, paper 0.01).")
    p.add_argument("--num-encode-steps", type=int, default=1,
                   help="Iterative-encoder steps (isae=1, isae_me=5).")
    p.add_argument("--initial-step-size", type=float, default=0.9,
                   help="Initial learned step size for the iterative encoder.")
    p.add_argument("--no-abstopk", dest="use_abstopk", action="store_false",
                   default=True,
                   help="Disable magnitude-based TopK (default on for iSAE).")
    p.add_argument("--isae-input-norm", action="store_true", default=False,
                   help="Train/encode iSAE on per-sample mean-centred, unit-L2 "
                        "inputs -- the reference convention (see IsaeSAE."
                        "input_norm). OFF by default: every existing iSAE "
                        "checkpoint was trained on raw activations.")

    p.add_argument("--expansion-factor", type=int, default=8)
    p.add_argument("--d-sae", type=int, default=None)
    p.add_argument("--k-per-token", type=int, default=80)

    # MP
    p.add_argument("--mp-iters", type=int, default=0)
    p.add_argument("--mp-inference-mode", choices=["fixed", "adaptive"],
                   default="fixed")
    p.add_argument("--mp-inference-iters", type=int, default=0)
    p.add_argument("--mp-residual-thr", type=float, default=0.05)
    p.add_argument("--mp-max-iters", type=int, default=0)
    p.add_argument("--mp-sparsity-cap", type=float, default=0.1)
    p.add_argument("--mp-patience", type=int, default=2)
    p.add_argument("--mp-nonneg", dest="mp_nonneg", action="store_true",
                   default=True)
    p.add_argument("--no-mp-nonneg", dest="mp_nonneg", action="store_false")
    p.add_argument("--mp-dropout", type=float, default=0.0)
    p.add_argument("--mp-revive-coef", type=float, default=1e-2)

    # FISTA-family
    p.add_argument("--fista-iters", type=int, default=50)
    p.add_argument("--fista-step", type=float, default=0.0)
    p.add_argument("--fista-nonneg", dest="fista_nonneg",
                   action="store_true", default=True)
    p.add_argument("--no-fista-nonneg", dest="fista_nonneg",
                   action="store_false")
    # FISTA speedup options (see train_vision for descriptions).
    p.add_argument("--fista-r2-tol", type=float, default=1e-4)
    p.add_argument("--fista-check-every", type=int, default=5)
    p.add_argument("--fista-use-bf16", action="store_true", default=False)
    p.add_argument("--fista-step-cache-every", type=int, default=0)
    p.add_argument("--fista-compile", action="store_true", default=False)
    # ── L0 early abort (relu / FISTA; top-k pins L0 at k) ──
    p.add_argument(
        "--l0-abort-min", type=float, default=0.0,
        help="Abort the run when L0 (active latents per token, the "
        "``metrics/l0`` panel) falls below this. 0 disables the abort. "
        "A lambda off the usable range does not crash — it trains to "
        "the end of the budget and yields a cell nobody can use, so "
        "the cheapest failure is an early one.",
    )
    p.add_argument(
        "--l0-abort-max", type=float, default=0.0,
        help="Abort when L0 rises above this. 0 disables. Both bounds "
        "must be set for the abort to arm.",
    )
    p.add_argument(
        "--l0-abort-warmup-frac", type=float, default=0.10,
        help="Fraction of the sample budget to run before the abort can "
        "fire — early L0 is the dictionary settling, not the verdict.",
    )
    p.add_argument(
        "--l0-abort-patience", type=int, default=5,
        help="Consecutive out-of-band log intervals required. One tick "
        "outside the band is noise; five in a row is the cell.",
    )
    # Refitting: OLS refit on the FISTA support to strip the ℓ1
    # shrinkage bias off the coefficient values. See FistaSAE.
    p.add_argument("--refit", action="store_true", default=False)
    p.add_argument("--refit-iters", type=int, default=30)
    p.add_argument("--n-outer", type=int, default=20)
    p.add_argument("--samples-per-outer", type=int, default=200_000)
    p.add_argument("--dl-ridge", type=float, default=1e-6)
    p.add_argument("--early-stop-tol", type=float, default=1e-4)

    # Co-active similarity regularization (γ > 0 enables it).
    p.add_argument("--co-active-sim-coefficient", type=float, default=0.0)
    p.add_argument("--co-active-sim-r", type=int, default=32)
    p.add_argument("--co-active-sim-refresh-steps", type=int, default=100)
    p.add_argument("--co-active-sim-barrier-delta", type=float, default=1e-4)
    p.add_argument("--co-active-sim-barrier-margin", type=float, default=0.0,
                   help="Cosine below which a co-active pair is left alone. 0 (default) applies the barrier to every co-active pair, whose only minimum is full orthogonality; a positive margin restricts the penalty to pairs whose cosine exceeds it.")
    p.add_argument("--co-active-sim-activation-eps", type=float, default=1e-6)
    p.add_argument("--co-active-sim-cache-block-size", type=int, default=512)

    # Ghost-grad / aux losses
    p.add_argument("--no-ghost-grads", action="store_true")
    # saprmarks-style dead-mask-gated AuxK loss for TopK SAEs (Gao et al.
    # 2024 §A.2); default 1/32.
    p.add_argument("--topk-aux-coefficient", type=float, default=1.0 / 32.0)
    p.add_argument("--topk-aux-dead-threshold", type=int, default=0,
                   help="Tokens-since-last-fire before a latent is "
                        "considered dead and routed through the AuxK "
                        "loss. ``0`` ⇒ auto = 2%% of --total-samples "
                        "(saprmarks' 10M-on-500M ratio).")
    p.add_argument("--topk-aux-k-aux", type=int, default=0,
                   help="``k_aux`` for the AuxK loss; 0 ⇒ d_in // 2 "
                        "(Gao Appendix B.1 heuristic).")
    # iSAE only. The reference declares a latent dead after zero firings across the
    # last ``activation_window_batches`` batches (config default 8 -> ~32k tokens at
    # batch 4096), so AuxK revives it almost immediately. A token-count threshold
    # in the millions is far more lenient, and death is absorbing under TopK: an
    # unselected latent gets no recon and no L1 gradient.
    # Ref: third_party/isaes/rsae/rip_topk.py:317-335, configs/model/base.yaml.
    p.add_argument("--isae-lr-decay-start-frac", type=float, default=0.8,
                   help="iSAE only. Fraction of total steps at which the LR begins a linear "
                        "decay to zero (reference: 0.8). 0 disables decay, leaving a "
                        "constant-with-warmup schedule.")
    p.add_argument("--isae-dead-window-batches", type=int, default=8,
                   help="iSAE dead-latent window in BATCHES; the AuxK dead threshold "
                        "becomes this x --batch-size tokens (reference default: 8). "
                        "Set 0 to fall back to --topk-aux-dead-threshold instead.")
    # ── saprmarks-style neuron resampling (relu only, LM only) ────
    p.add_argument("--use-resampling", action="store_true",
                   help="Enable saprmarks-style periodic dead-latent "
                        "resampling for relu SAEs.  LM-only — vision "
                        "(CLIP patch tokens) already converges at low "
                        "dead-rate without it.  Active only when "
                        "architecture=relu.")
    p.add_argument("--resample-steps", type=int, default=0,
                   help="Resampling period in training steps. "
                        "``0`` ⇒ auto-scale to ``total_steps // 8`` "
                        "(min 2000) so the resampler fires ~8 times "
                        "regardless of training budget.  "
                        "A fixed period such as saprmarks' 25000 never "
                        "fires when training is shorter than that, so "
                        "auto is the safe default.")
    p.add_argument("--resample-buffer-size", type=int, default=8_192,
                   help="Size of the recent-activations buffer used "
                        "by the loss-weighted multinomial sampler. "
                        "Larger = more diverse candidates but more "
                        "CPU memory.")
    p.add_argument("--resample-tail-skip-frac", type=float, default=0.25,
                   help="Fraction of training at the tail where the "
                        "resampler stops firing (default 0.25 ⇒ no "
                        "resamples in the last 25%% of steps).  Set to "
                        "0.0 to disable the cutoff.  Reason: revived "
                        "latents need gradient runway to settle into "
                        "useful directions; at LR-floor (cosine tail) "
                        "they enter eval as quasi-random firing rows "
                        "in W_dec and crash recon_r2.")
    p.add_argument("--no-stats", action="store_true",
                   help="Skip post-train stats (LM-side stats not yet "
                        "implemented; emits the .training_complete sentinel "
                        "either way).")

    return p.parse_args()


# Parse argv only when this module is run directly: module-level setup
# (``CUDA_VISIBLE_DEVICES``, progress-file tqdm patch) needs ``args``
# before torch is imported. When the module is imported for
# ``LMBackbone`` / ``build_lm_dataloader`` (e.g. by ``consdl.cli.stats_lm``),
# the importer's argv is not ours and the setup is skipped.
import sys as _sys

try:
    # ``python -m consdl.cli.train_lm`` sets argv[0] to this file's path.
    import os.path as _osp
    _argv_is_ours = (
        _osp.realpath(_sys.argv[0]) == _osp.realpath(__file__)
        or _sys.argv[0] == "consdl.cli.train_lm"
    )
except Exception:  # noqa: BLE001
    _argv_is_ours = False

if _argv_is_ours:
    args = parse_args()
    # ``--normalize-activations`` is not implemented; fail loudly rather than
    # silently ignore it.
    if getattr(args, "normalize_activations", False):
        raise SystemExit(
            "--normalize-activations is not implemented. Remove the flag."
        )
else:
    args = None  # type: ignore[assignment]


# ── Setup (after argparse so CUDA_VISIBLE_DEVICES is set first) ────
import json
import math
import os
os.environ.setdefault("WANDB_DIR", "var/logs/wandb")  # keep run dirs out of the repo root
import pathlib
from datetime import datetime

from dotenv import load_dotenv

from consdl.io.paths import REPO_ROOT  # noqa: E402
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
# Guard module-level setup behind ``args`` presence so importers that
# don't run main() never trigger CUDA / progress-file side-effects.
if args is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Progress reporting — same tqdm monkey-patch as train_vision.
_PROGRESS_FILE = args.progress_file if args is not None else None
if _PROGRESS_FILE:
    import tqdm as _tqdm_module
    _OrigTqdm = _tqdm_module.tqdm
    _last = {"n": 0}

    class _ProgressTqdm(_OrigTqdm):
        def update(self, n=1):
            r = super().update(n)
            _last["n"] += 1
            if _last["n"] >= 50:
                _last["n"] = 0
                try:
                    with open(_PROGRESS_FILE, "w") as f:
                        json.dump({"n": self.n or 0, "total": self.total or 0}, f)
                except Exception:
                    pass
            return r

    _tqdm_module.tqdm = _ProgressTqdm

import datasets
import fastsae
from consdl.training.build import build_sae, build_trainer
import numpy as np
import shortuuid
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


# Rebind already-imported fastsae tqdm references to the patched class
# (fastsae sub-modules bind ``tqdm`` at first import, which may precede
# the patch above).
if _PROGRESS_FILE:
    import fastsae.training.trainer as _fst_trainer
    _fst_trainer.tqdm = _ProgressTqdm


# ── LM backbone wrapper ────────────────────────────────────────────
#
# Mirrors ``fastsae.models.Backbone``'s interface (``forward_to_z``,
# ``hook_target_dim``, ``eval``) so ``SAEWrapper`` and the trainers work
# unchanged. Instead of an image batch it takes ``{"input_ids": (B, T)}``;
# the hook captures the layer's resid_post output ``(B, T, d_in)``, and
# ``forward_to_z`` flattens that to ``(B·T, d_in)``.


class _EarlyStopForward(Exception):
    """Internal control-flow exception to abort forward after capturing the
    hook output (same role as in fastsae's ``Backbone``)."""


class LMBackbone(nn.Module):
    """Causal-LM backbone with a forward hook on a transformer-block.

    Wraps ``AutoModelForCausalLM`` and registers a forward hook on the
    submodule at ``backbone.<hook_layer_fmt.format(layer)>`` — for
    Pythia that's ``gpt_neox.layers.<L>``, for GPT-2 that's
    ``transformer.h.<L>``. The submodule's output is the residual-stream
    state immediately after the block (``resid_post``), ``(B, T, d_in)``.

    ``forward_to_z(batch)`` is the unified accessor:
      * ``batch`` may already be a precomputed activation dict
        ``{"latent": (B·T, d_in)}`` — returned as-is (no LM forward).
      * Otherwise the dict carries ``{"input_ids": (B, T),
        "attention_mask": (B, T)}`` — runs the LM, returns the captured
        residual stream flattened to ``(B·T, d_in)``.
    """

    def __init__(self, model_hf_name: str, hook_layer_path: str,
                 device: str = "cuda", dtype: torch.dtype | None = None):
        super().__init__()
        # Identifying info for ``save()``, which writes metadata instead of
        # the HF weights.
        self.model_hf_name = model_hf_name
        self.hook_target_name = hook_layer_path
        self.model = AutoModelForCausalLM.from_pretrained(model_hf_name)
        # The hook path is the full submodule chain from the
        # ``AutoModelForCausalLM`` root (the base-transformer attribute name
        # differs per family: gpt_neox / transformer / model).
        target = self.model
        for part in hook_layer_path.split("."):
            target = getattr(target, part)
        self._hook_target = target
        self._cache: torch.Tensor | None = None

        def _capture(_mod, _inp, out):
            # Block submodules return either a Tensor or a tuple
            # ``(hidden, ...)``; the first element is the residual stream.
            self._cache = out[0] if isinstance(out, tuple) else out
            # Abort the rest of the forward: later blocks and the LM head
            # cannot affect the captured activation, so computing them is
            # wasted work.
            raise _EarlyStopForward

        self._hook_handle = self._hook_target.register_forward_hook(_capture)
        self.device = device
        self.model = self.model.to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self._dtype = dtype or next(self.model.parameters()).dtype
        # ``d_in`` — the residual-stream dimension, from the model config.
        cfg = self.model.config
        d_in = (getattr(cfg, "hidden_size", None)
                or getattr(cfg, "n_embd", None)
                or getattr(cfg, "d_model", None))
        if d_in is None:
            raise ValueError(f"Could not infer d_in from {cfg!r}")
        self._hook_target_dim = int(d_in)

    @property
    def hook_target_dim(self) -> int:
        return self._hook_target_dim

    @property
    def dtype(self) -> torch.dtype:
        # ``SAEWrapper`` asserts ``backbone.dtype == sae.dtype``.
        return self._dtype

    def save(self, path) -> None:
        # Called by ``SAETrainer.save_backbone``. The weights are in the HF
        # cache, so only a small JSON record of the backbone is written.
        import json
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_hf_name": self.model_hf_name,
            "hook_target_name": self.hook_target_name,
            "d_in": self._hook_target_dim,
            "dtype": str(self._dtype),
        }
        (path / "lm_backbone.json").write_text(json.dumps(meta, indent=2))

    @torch.no_grad()
    def forward_to_z(self, batch) -> torch.Tensor:
        """Return the per-token residual-stream activations ``(B·T, d_in)``."""
        if isinstance(batch, dict) and "latent" in batch:
            # Precomputed activations (same protocol as the vision Backbone).
            z = batch["latent"]
            if z.ndim == 3:
                z = z.reshape(-1, z.shape[-1])
            return z.to(self.device)
        input_ids = batch["input_ids"].to(self.device)
        attention_mask = batch.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        try:
            self.model(input_ids=input_ids, attention_mask=attention_mask,
                       use_cache=False)
        except _EarlyStopForward:
            pass          # expected: the hook fired and stopped the forward
        z = self._cache  # (B, T, d_in)
        self._cache = None
        if z is None:
            raise RuntimeError("LMBackbone hook did not fire — check hook layer path.")
        # Left in the backbone's output dtype (bf16 under --bf16 autocast),
        # so the SAE step also runs in bf16.
        return z.reshape(-1, z.shape[-1])


# ── Streaming text → sequence-packed token DataLoader ──────────────


def build_lm_dataloader(
    hf_dataset: str,
    hf_config: str | None,
    split: str,
    text_field: str,
    tokenizer: AutoTokenizer,
    context_length: int,
    batch_seqs: int,
    seed: int | None,
    token_cache_tokens: int = 0,
    token_cache_dir: str = "var/cache/lm_tokens",
    skip_seqs: int = 0,
    num_workers: int = 0,
) -> torch.utils.data.DataLoader:
    """Stream ``hf_dataset`` → tokenise → sequence-pack into
    ``(B, context_length)`` token batches.

    Documents are concatenated with the tokenizer's EOS between them and
    sliced into fixed-length chunks (no padding, no document-boundary
    handling beyond EOS), as in ``dictionary_learning``. Streaming means
    only the shards actually read are downloaded.

    When ``token_cache_tokens > 0`` the stream is instead pinned once into
    a uint16 token memmap (see :mod:`consdl.data.token_cache`) and served
    map-style, so ``seed`` only permutes the order of a fixed token set
    (same documents across seeds).
    """
    if int(token_cache_tokens) > 0:
        from consdl.data import token_cache as _tc
        bin_path = _tc.build(
            token_cache_dir, hf_dataset=hf_dataset, hf_config=hf_config,
            split=split, text_field=text_field, tokenizer=tokenizer,
            context_length=int(context_length), n_tokens=int(token_cache_tokens),
        )
        return _tc.build_dataloader(
            bin_path, context_length=int(context_length),
            batch_seqs=int(batch_seqs), seed=seed,
            num_workers=int(num_workers), skip_seqs=int(skip_seqs),
        )

    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer lacks an EOS token — sequence packing needs one.")

    # Pre-tokenised memmap path. With streaming, ``ds.shuffle(seed=...)``
    # changes which documents a seed reads, so seeds differ in data as well
    # as in init — a confound for cross-seed consistency. With the memmap
    # the token set is pinned and ``seed`` only permutes sequence order.
    if token_cache_tokens:
        from consdl.data import token_cache  # noqa: PLC0415

        bin_path = token_cache.build(
            token_cache_dir, hf_dataset=hf_dataset, hf_config=hf_config,
            split=split, text_field=text_field, tokenizer=tokenizer,
            context_length=context_length, n_tokens=int(token_cache_tokens),
        )
        return token_cache.build_dataloader(
            bin_path, context_length=context_length, batch_seqs=batch_seqs,
            seed=seed, num_workers=int(num_workers), skip_seqs=skip_seqs,
        )
    if skip_seqs:
        # Only the memmap path can seek. The streaming loader restarts at
        # the top of the shard shuffle, so a resumed run retrains on tokens
        # it has already seen.
        print(f"[resume] WARNING: streaming loader cannot skip "
              f"{skip_seqs:,} sequences — the first tokens will be seen "
              f"twice. Use --token-cache-tokens for an exact resume.")

    ds = datasets.load_dataset(
        hf_dataset,
        name=hf_config,
        split=split,
        streaming=True,
    )
    if seed is not None:
        ds = ds.shuffle(seed=int(seed), buffer_size=10_000)

    eos_id = int(tokenizer.eos_token_id)
    ctx = int(context_length)

    def _packed_iter():
        buf: list[int] = []
        for row in ds:
            text = row.get(text_field)
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False).input_ids
            buf.extend(ids)
            buf.append(eos_id)
            while len(buf) >= ctx:
                yield {"input_ids": torch.tensor(buf[:ctx], dtype=torch.long)}
                buf = buf[ctx:]

    class _IterableTokenDataset(torch.utils.data.IterableDataset):
        def __iter__(self):
            return _packed_iter()

    def _collate(rows):
        ids = torch.stack([r["input_ids"] for r in rows], dim=0)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    return torch.utils.data.DataLoader(
        _IterableTokenDataset(),
        batch_size=int(batch_seqs),
        collate_fn=_collate,
        num_workers=0,  # iterable + tokenizer ⇒ workers don't compose cleanly
    )


# ── Main ───────────────────────────────────────────────────────────


def main() -> None:
    backbone_cfg = LM_BACKBONE_MAP[args.backbone]
    hf_name = backbone_cfg["hf_name"]
    hook_layer = (int(args.hook_layer) if args.hook_layer is not None
                  else int(backbone_cfg["default_layer"]))
    hook_layer_path = backbone_cfg["hook_layer_fmt"].format(hook_layer)

    device = "cuda"
    dataset_short = args.dataset

    # ── Knobs derived from CLI ─────────────────────────────────────
    d_sae_explicit = int(args.d_sae) if args.d_sae is not None else None
    d_sae_factor = args.expansion_factor
    architecture = args.architecture
    if architecture in ("isae", "isae_me"):
        # Reference iSAE normalizes L1 per active feature: sum(|a|)/(batch·k)
        # (rip_topk.py::compute_l1_loss). Our trainer's sparsity_loss is
        # sum(|a|)/token with no ÷k, so pass isae_l1/k to match. Without it
        # the L1 term is k× too strong and collapses all codes to zero.
        sparsity_coefficient = float(args.isae_l1) / float(args.k_per_token)
    elif architecture == "relu":
        sparsity_coefficient = args.l1_coeff
    else:
        sparsity_coefficient = 0.0
    learning_rate = args.learning_rate
    batch_seqs = max(1, int(args.batch_size) // int(args.context_length))
    sae_batch_tokens = batch_seqs * int(args.context_length)
    total_samples = int(args.total_samples)
    total_checkpoints = int(args.total_checkpoints)
    # FISTA-SAE and MP-SAE have no learned encoder, so ghost grads are off.
    use_ghost_grads = (
        (not args.no_ghost_grads)
        and architecture not in ("fista", "mp")
    )
    sae_b_dec_init_method = "geometric_median"

    # Default on; set WANDB_ENABLED=false in .env to disable.
    wandb_enabled = os.getenv("WANDB_ENABLED", "true").lower() != "false"
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "consdl")
    wandb_entity = os.getenv("WANDB_ENTITY", "")

    print(f"[lm gpu={args.gpu}] Starting SAE training (LM backbone)")
    print(f"  backbone        : {args.backbone}  ({hf_name})")
    print(f"  hook layer      : {hook_layer}  → {hook_layer_path}")
    print(f"  hf-dataset      : {args.hf_dataset}"
          f"{f' ({args.hf_dataset_config})' if args.hf_dataset_config else ''}")
    print(f"  context length  : {args.context_length}")
    print(f"  total tokens    : {total_samples:,}")
    print(f"  batch tokens    : {sae_batch_tokens}  ({batch_seqs} seqs × {args.context_length})")
    print(f"  architecture    : {architecture}")
    print(f"  expansion       : x{d_sae_factor}")
    if architecture in ("topk", "batchtopk"):
        print(f"  k_per_token     : {args.k_per_token}")
    if architecture == "relu":
        print(f"  l1_coeff        : {sparsity_coefficient}")
    print(f"  lr              : {learning_rate}  (warmup {args.lr_warmup_steps})")
    print(f"  precision       : {'bf16 autocast (fwd+bwd)' if args.bf16 else 'fp32'}")
    print(f"  seed            : {args.seed if args.seed is not None else 'none'}")

    # Seed
    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Data seed can be decoupled from the training seed (default: --seed
    # drives both). Pinning it, as the iSAE reference does (seed.data=42),
    # gives every training seed the same documents.
    data_seed = args.data_seed if args.data_seed is not None else args.seed
    print(f"  data seed       : {data_seed if data_seed is not None else 'none'}"
          f"{' (decoupled from --seed)' if args.data_seed is not None else ''}")

    # ── Backbone + tokenizer ──────────────────────────────────────
    print(f"[lm] loading backbone…")
    # Not cast to bf16 even under --bf16: SAEWrapper asserts
    # ``backbone.dtype == sae.dtype``, and the backbone forward runs inside
    # the same autocast region as the SAE step anyway.
    backbone = LMBackbone(model_hf_name=hf_name, hook_layer_path=hook_layer_path,
                          device=device)
    d_in = backbone.hook_target_dim
    if d_in != int(backbone_cfg["d_in"]):
        print(f"  [warn] LM_BACKBONE_MAP d_in={backbone_cfg['d_in']} but "
              f"model.config gives {d_in}; using {d_in}.")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Resolve construction-time factor if --d-sae is set.
    if d_sae_explicit is not None:
        d_sae_factor = max(1, math.ceil(d_sae_explicit / d_in))
        print(f"  d_sae           : {d_sae_explicit} (exact; constructed at "
              f"factor={d_sae_factor} → {d_in * d_sae_factor}, then resized)")
    else:
        print(f"  d_sae           : {d_in * d_sae_factor} = {d_in} × {d_sae_factor}")

    # ── SAE construction ───────────────────────────────────────────
    sae_instance = build_sae(
        args, architecture, d_in=d_in, d_sae_factor=d_sae_factor, device=device,
        fista_lam=float(args.l1_coeff), mp_default_iters=int(args.k_per_token),
        mp_cap_d_sae=(d_sae_explicit if d_sae_explicit is not None
                      else int(d_in) * int(d_sae_factor)),
        d_sae=d_sae_explicit)

    wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae_instance,
                                         _caching=False)

    # ── DataLoader ─────────────────────────────────────────────────
    # On resume, start the permutation partway in so the loader does not
    # replay tokens already trained on (the budget is one pass). The resume
    # state is read here because the trainer is built after the loader.
    _skip_seqs = 0
    if args.experiment_dir is not None and os.getenv("SAE_RESUME", "1") != "0":
        from consdl.training.resume import resume_state
        _rs = resume_state(args.experiment_dir)
        if _rs is not None:
            _seen = int(_rs[1].get("n_training_samples", 0))
            _skip_seqs = _seen // int(args.context_length)
            print(f"[resume] {_seen:,} tokens already seen → dataloader skips "
                  f"{_skip_seqs:,} sequences of the seed-{args.seed} order")
    dataloader = build_lm_dataloader(
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_dataset_config,
        split=args.hf_dataset_split,
        text_field=args.hf_dataset_text_field,
        tokenizer=tokenizer,
        context_length=args.context_length,
        batch_seqs=batch_seqs,
        seed=data_seed,
        token_cache_tokens=int(args.token_cache_tokens),
        token_cache_dir=str(args.token_cache_dir),
        num_workers=int(args.num_workers),
        skip_seqs=_skip_seqs,
    )

    # ── Checkpoint path (same layout as train_vision) ──────────────
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shortuuid.uuid()[:6]}"
    config_tag = f"{architecture}_x{d_sae_factor}"
    if args.experiment_dir is not None:
        ckpt_path = pathlib.Path(args.experiment_dir)
        if not ckpt_path.is_absolute():
            ckpt_path = pathlib.Path.cwd() / args.experiment_dir
    else:
        ckpt_path = pathlib.Path(
            os.getenv("CHECKPOINT_PATH", str(pathlib.Path.cwd() / "var" / "checkpoints")),
            dataset_short, architecture, f"x{d_sae_factor}",
            f"{args.backbone}_layer{hook_layer}", run_id,
        )
    # With --experiment-dir, the wandb run name is derived from the
    # checkpoint dir, so it maps 1:1 onto it.
    if args.experiment_dir is not None:
        _exp = pathlib.Path(args.experiment_dir)
        wandb_run_id = (
            f"x{d_sae_factor}_{_exp.parent.name}_{_exp.name}"
            f"_{run_id.split('_')[-1]}"
        )
    else:
        wandb_run_id = f"{args.backbone}_L{hook_layer}_{config_tag}_{run_id}"

    # ── Optimizer + scheduler + train ──────────────────────────────
    optimizer = torch.optim.Adam(wrapper.sae.parameters(), lr=learning_rate)
    if architecture in ("isae", "isae_me") and float(args.isae_lr_decay_start_frac) > 0:
        # The reference iSAE uses a 3-phase schedule
        # (rsae/trainers/base_trainer.py::_create_lr_scheduler): linear warmup ->
        # constant -> linear decay to zero over the final (1 - decay_start_fraction)
        # of training. fastsae's "linearwarmupdecay" decays from warmup onward,
        # so the reference lambda is reproduced here.
        _warm = max(1, int(args.lr_warmup_steps))
        _total = max(1, int(total_samples // sae_batch_tokens))
        _decay_start = int(float(args.isae_lr_decay_start_frac) * _total)

        def _isae_lr_lambda(step: int) -> float:
            if step < _warm:
                return (step + 1) / _warm
            if step < _decay_start or _decay_start >= _total:
                return 1.0
            return max(0.0, (_total - step) / max(1, _total - _decay_start))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _isae_lr_lambda)
        print(f"  lr schedule     : 3-phase warmup({_warm}) -> constant -> "
              f"linear decay from step {_decay_start}/{_total} (reference convention)")
    else:
        scheduler = fastsae.training.util.get_scheduler(
            "constantwithwarmup", optimizer=optimizer,
            warm_up_steps=int(args.lr_warmup_steps),
        )
    wrapper.backbone.eval()
    wrapper.sae.train()

    # Set on the base class, not passed as a kwarg: ``autocast_dtype`` is a
    # class attribute, not a constructor field. All trainers descend from
    # SAETrainer, so this covers every architecture.
    if args.bf16:
        from consdl.training.trainer import (  # noqa: PLC0415
            SAETrainer as _BaseTrainer,
        )
        _BaseTrainer.autocast_dtype = torch.bfloat16

    train_kwargs = dict(
        dataloader=dataloader, optimizer=optimizer, scheduler=scheduler,
        total_n_training_samples=total_samples,
        sparsity_coefficient=sparsity_coefficient,
        use_ghost_grads=use_ghost_grads,
        sae_b_dec_init_method=sae_b_dec_init_method,
        checkpoint_path=str(ckpt_path),
        total_n_checkpoints=total_checkpoints,
        log_to_wandb=wandb_enabled, wandb_run_id=wandb_run_id,
        wandb_project=wandb_project, wandb_entity=wandb_entity,
        wandb_log_frequency=max(args.log_interval // sae_batch_tokens, 1),
    )

    # In-repo trainers throughout: fastsae's ``use_ghost_grads`` is
    # declared but never read, so ``--no-ghost-grads`` only takes effect
    # through them. ``total_samples`` counts tokens here.
    isae_dead_thr = None
    if architecture in ("isae", "isae_me"):
        # Reference-equivalent dead window (see --isae-dead-window-batches).
        win = int(args.isae_dead_window_batches)
        isae_dead_thr = (win * int(args.batch_size) if win > 0
                         else int(args.topk_aux_dead_threshold))
        print(f"  dead threshold  : {isae_dead_thr} tokens")
    resampling = None
    if architecture == "relu" and bool(args.use_resampling):
        # ReLU with saprmarks-style neuron resampling, scaled to fire ~8
        # times over the run.
        n_steps = max(1, int(total_samples // sae_batch_tokens))
        resample_steps = (int(args.resample_steps) if int(args.resample_steps) > 0
                          else max(2000, n_steps // 8))
        print(f"[resample] resample_steps={resample_steps} "
              f"(n_steps={n_steps}; expected events ≈ {n_steps // resample_steps})")
        resampling = dict(resample_steps=resample_steps,
                          resample_buffer_size=int(args.resample_buffer_size),
                          resample_tail_skip_frac=float(args.resample_tail_skip_frac))
    trainer = build_trainer(
        args, architecture, wrapper=wrapper, train_kwargs=train_kwargs,
        aux_off=bool(args.no_ghost_grads), dead_threshold_total=total_samples,
        isae_dead_threshold=isae_dead_thr, resampling=resampling)
    if architecture in ("relu", "fista") and resampling is None:
        # Only the λ-controlled families can drift out of their L0 band.
        trainer.set_l0_abort(
            minimum=float(args.l0_abort_min),
            maximum=float(args.l0_abort_max),
            warmup_frac=float(args.l0_abort_warmup_frac),
            patience=int(args.l0_abort_patience),
        )
    from consdl.training.trainer import L0AbortError  # noqa: PLC0415
    try:
        trainer.train()
    except L0AbortError as exc:
        # ``final/.l0_aborted`` records the abort in place of
        # ``.training_complete``.
        print(f"[l0-abort] {exc}", flush=True)
        try:
            _ab = pathlib.Path(ckpt_path) / "final" / ".l0_aborted"
            _ab.parent.mkdir(parents=True, exist_ok=True)
            _ab.write_text(json.dumps({
                "reason": str(exc),
                "aborted_at": datetime.now().isoformat(timespec="seconds"),
                "l0_abort_min": float(args.l0_abort_min),
                "l0_abort_max": float(args.l0_abort_max),
            }))
        except Exception as _e:          # never mask the abort itself
            print(f"[l0-abort] (marker not written: {_e})", flush=True)
        sys.exit(3)

    _complete_path = pathlib.Path(ckpt_path) / "final" / ".training_complete"
    _complete_path.parent.mkdir(parents=True, exist_ok=True)
    _complete_path.write_text(json.dumps({
        "total_samples": total_samples,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }))
    print(f"[lm gpu={args.gpu}] Training complete. Checkpoint: {ckpt_path}")

    if args.no_stats:
        print(f"[lm gpu={args.gpu}] Skipping stats (--no-stats).")
        return
    # Post-train stats via ``consdl.cli.stats_lm`` (the vision
    # ``SAEInterpreter`` assumes image batches). Run as a subprocess: the
    # module has its own CLI and device setup, and a failure there must not
    # affect the saved checkpoints.
    import subprocess  # noqa: PLC0415
    _stats_cmd = [
        sys.executable, "-u", "-m", "consdl.cli.stats_lm",
        "--gpu", str(int(args.gpu)),
        "--ckpt_path", str(ckpt_path),
        "--dataset", str(args.dataset),
        "--dataset-split", str(args.hf_dataset_split),
        "--hf-dataset", str(args.hf_dataset),
        "--hf-dataset-text-field", str(args.hf_dataset_text_field),
        "--context-length", str(int(args.context_length)),
        # Same corpus as training; streaming would re-derive it from a live
        # shard shuffle.
        "--token-cache-tokens", str(int(args.token_cache_tokens)),
        "--token-cache-dir", str(args.token_cache_dir),
        # Shared activation pool, so runs on the same backbone/corpus do not
        # repeat the backbone forward.
        "--activation-cache-dir", "var/cache/lm_stats_acts",
    ]
    if args.hf_dataset_config:
        _stats_cmd += ["--hf-dataset-config", str(args.hf_dataset_config)]
    print(f"[lm gpu={args.gpu}] computing stats: {' '.join(_stats_cmd)}")
    _rc = subprocess.call(_stats_cmd)
    if _rc != 0:
        print(f"[lm gpu={args.gpu}] stats FAILED (rc={_rc}) — checkpoints "
              f"are intact; rerun with `python -m consdl.cli.stats_lm "
              f"--gpu {args.gpu} --ckpt_path {ckpt_path}`")


if __name__ == "__main__":
    # ``args`` is parsed at import time (the ``_argv_is_ours`` block) so
    # module-level setup runs with the right values.
    assert args is not None, "main() invoked but args weren't parsed."
    main()
