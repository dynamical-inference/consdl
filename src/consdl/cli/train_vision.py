"""Train an SAE on one layer of a vision backbone (CLIP / DINOv2).

Streams ImageNet images through the backbone, hooks the chosen layer,
and trains one of the paper's SAE architectures on the patch tokens
(ReLU / TopK / BatchTopK / Matryoshka / MP-SAE / iSAE / ConSDL), with
seeded training, a multi-checkpoint schedule and optional wandb logging.
Statistics on the stats dataset are computed after training unless
``--no-stats`` is given.

Usage::

    python -m consdl.cli.train_vision --gpu 0 --layer 23 \
        --backbone clip-vit-l-14 --architecture topk --seed 0
"""

from __future__ import annotations

import argparse
import os
os.environ.setdefault("WANDB_DIR", "var/logs/wandb")  # keep run dirs out of the repo root
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--layer", type=int, required=True)
    p.add_argument("--backbone", type=str, default="clip-vit-l-14",
                   choices=["clip-vit-l-14", "clip-vit-b-16",
                            "dinov2-b", "dinov2-l",
                            "siglip-b", "siglip-l"])
    p.add_argument("--dataset", type=str, default="imagenet")
    p.add_argument("--stats-dataset", type=str, default="imagenet",
                   help="Dataset used to compute post-training stats. "
                        "Defaults to imagenet, independent of ``--dataset``, "
                        "so metrics are comparable across training "
                        "datasets.")
    p.add_argument("--total-samples", type=int, default=5_000_000)
    # ── Per-seed data subset ─────────────────────────────────────────
    # Two slicing modes; ``none`` is the default (full data).
    #   * ``class_bucket`` — restrict to one disjoint class range
    #     ``[bucket_idx * (n_classes / n_buckets),
    #        (bucket_idx + 1) * (n_classes / n_buckets))``.  Class-
    #     disjoint across buckets.
    #   * ``index_residue`` — keep rows where ``row_index % n_buckets
    #     == bucket_idx``.  Class-balanced, sample-disjoint.
    # Applied AFTER ``format_dataset`` and BEFORE ``dataset_fraction`` /
    # the ``shuffle`` re-order.
    p.add_argument(
        "--data-subset-mode",
        choices=["none", "class_bucket", "index_residue"],
        default="none",
        help="Per-seed data subset mode for the consistency-under-"
        "partial-coverage experiment.  ``none`` = full data (default).",
    )
    p.add_argument(
        "--data-subset-bucket-idx", type=int, default=0,
        help="Which bucket / residue to keep (0-indexed).  Ignored when "
        "--data-subset-mode is ``none``.",
    )
    p.add_argument(
        "--data-subset-n-buckets", type=int, default=4,
        help="Total number of buckets / modulus.  Ignored when "
        "--data-subset-mode is ``none``.  Default 4 (25%% slices).",
    )
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--l1-coeff", type=float, default=0.00008,
                   help="L1 sparsity coefficient (relu).  For "
                        "--architecture fista this value is reused as "
                        "the FISTA L1 penalty λ (fista_lam) — the FISTA "
                        "prox produces the sparsity, so there is no "
                        "external L1 loss term.")
    p.add_argument("--lr", type=float, default=0.0004)
    p.add_argument("--expansion-factor", type=int, default=32)
    p.add_argument("--d-sae", type=int, default=None,
                   help="Optional exact dictionary width. When set, "
                        "overrides --expansion-factor: the SAE is "
                        "constructed at the smallest integer factor "
                        "that gives ≥ this many latents, then resized "
                        "in place to exactly --d-sae (kaiming-reseeded "
                        "with --seed). Lets you target widths that "
                        "aren't integer multiples of d_in (e.g. 2000 "
                        "with d_in=768).")
    p.add_argument("--architecture", type=str, default="relu",
                   choices=["relu", "topk", "batchtopk", "matryoshka",
                            "mp", "fista", "isae", "isae_me"])
    p.add_argument("--k-per-token", type=int, default=64,
                   help="Active latents per token; batchtopk budget = k_per_token * seq_len.")
    # ── MP-SAE (--architecture mp) ──────────────────────────────────
    # Greedy matching-pursuit encoder + learned decoder.  Sparsity is
    # structural (one atom per round), so there's no L1 penalty.  All
    # ignored for non-mp arches.
    p.add_argument("--mp-iters", type=int, default=0,
                   help="MP-SAE: matching-pursuit rounds AT TRAIN TIME.  "
                        "0 → fall back to --k-per-token.")
    p.add_argument("--mp-inference-mode", choices=["fixed", "adaptive"],
                   default="fixed",
                   help="MP-SAE: eval-time round-count rule.  ``fixed`` runs "
                        "exactly --mp-inference-iters rounds; ``adaptive`` "
                        "unrolls until the relative residual drops below "
                        "--mp-residual-thr (or support stops growing for "
                        "--mp-patience rounds), capped by --mp-max-iters.")
    p.add_argument("--mp-inference-iters", type=int, default=0,
                   help="MP-SAE: round count for --mp-inference-mode=fixed.  "
                        "0 → mirror --mp-iters.")
    p.add_argument("--mp-residual-thr", type=float, default=0.05,
                   help="MP-SAE: adaptive-mode relative-residual stop "
                        "threshold.  Default 0.05.")
    p.add_argument("--mp-max-iters", type=int, default=0,
                   help="MP-SAE: adaptive-mode safety cap.  0 → fall back to "
                        "--mp-sparsity-cap (when > 0) else d_sae.")
    p.add_argument("--mp-sparsity-cap", type=float, default=0.1,
                   help="MP-SAE: per-token sparsity cap for adaptive mode "
                        "(max_iters = ceil(cap × d_sae)).  Default 0.1; "
                        "0 disables.")
    p.add_argument("--mp-patience", type=int, default=2,
                   help="MP-SAE: adaptive-mode support-stabilisation patience.")
    p.add_argument("--mp-nonneg", dest="mp_nonneg", action="store_true",
                   default=True,
                   help="MP-SAE: restrict atom selection to non-negative "
                        "correlations.  Default on.")
    p.add_argument("--no-mp-nonneg", dest="mp_nonneg", action="store_false",
                   help="MP-SAE: allow signed atom selection.")
    p.add_argument("--mp-dropout", type=float, default=0.0,
                   help="MP-SAE: Bernoulli atom dropout fraction during "
                        "training (disabled at eval).  Default 0.")
    p.add_argument("--mp-revive-coef", type=float, default=1e-2,
                   help="MP-SAE: dead-atom revival aux-loss coefficient "
                        "(0 disables; off entirely under --no-ghost-grads).")
    # ── FISTA-SAE (--architecture fista) ────────────────────────────
    # FISTA-SAE: cold-start FISTA sparse-coding encoder + jointly
    # trained decoder (online dictionary learning).  λ is taken from
    # ``--l1-coeff``; the knobs below are the FISTA solver settings.
    # All ignored for non-fista arches.
    p.add_argument("--fista-iters", type=int, default=50,
                   help="FISTA-SAE: inner FISTA iteration count T.")
    p.add_argument("--fista-power-iters", type=int, default=32,
                   help="FISTA-SAE: number of power-iteration steps "
                        "for the σ_max estimate (η = 1/σ_max²).  "
                        "Default 32.  Lower values (e.g. 8) leave "
                        "σ_max noisy across seeds, which hurts "
                        "cross-seed consistency.")
    p.add_argument("--fista-step", type=float, default=0.0,
                   help="FISTA-SAE: proximal-gradient step η.  0 "
                        "(default) auto-resolves 1/σ_max(D)² every "
                        "forward via power iteration.")
    p.add_argument("--fista-r2-tol", type=float, default=1e-4,
                   help="FISTA-SAE: early-stop tolerance on per-batch "
                        "Input R² improvement.  Loop breaks once "
                        "ΔR² = (SSE_prev − SSE_curr) / SST < this.  "
                        "0 disables the check (run exactly T iters).  "
                        "Default 1e-4.  Raise "
                        "to 5e-4 for ~10-30%% wall-clock at full-sweep "
                        "scale.")
    p.add_argument("--fista-check-every", type=int, default=5,
                   help="FISTA-SAE: how often (in FISTA iters) to "
                        "check the early-stop criterion — one extra "
                        "``h @ W_dec`` matmul per check, so larger "
                        "amortises better.  Default 5.")
    p.add_argument("--fista-use-bf16", action="store_true", default=False,
                   help="FISTA-SAE: run the encode loop in bfloat16.  "
                        "Roughly 1.8-2x faster on Ampere/Ada GPUs.  "
                        "The encode is no-grad and matmul-bound so "
                        "bf16's ~1e-3 noise is well below normal λ-prox "
                        "thresholds.  Output is cast back to fp32 "
                        "before return so backward stays fp32.  Default "
                        "off (exact fp32 encode).")
    p.add_argument("--fista-step-cache-every", type=int, default=0,
                   help="FISTA-SAE: cache + warm-start the power-"
                        "iteration estimate of σ_max(W_dec).  0 "
                        "(default) recomputes from a random vector "
                        "every forward.  >0 reuses the cached "
                        "dominant singular vector and refreshes σ_max "
                        "every N forwards in ~4 warm-started iters "
                        "instead of the cold-start 32.  Safe because "
                        "W_dec changes slowly between steps.")
    p.add_argument("--fista-compile", action="store_true", default=False,
                   help="FISTA-SAE: torch.compile the inner FISTA iter "
                        "body (mode='reduce-overhead' + CUDA Graphs). "
                        "Fuses the per-iter elementwise launches into "
                        "one Triton kernel.  Profile shows ~40%% of "
                        "encode GPU time is small elementwise kernels; "
                        "expect ~1.4-1.7x speedup on top of bf16.  "
                        "First step pays ~30s of compile + autotune "
                        "overhead.  Requires fixed input shape — "
                        "drop_last on the dataloader or pad partial "
                        "batches.  Default off (eager).")
    p.add_argument("--fista-nonneg", dest="fista_nonneg",
                   action="store_true", default=True,
                   help="FISTA-SAE: constrain codes a ≥ 0 "
                        "(prox = ReLU(z − η·λ)).  Default on.")
    p.add_argument("--no-fista-nonneg", dest="fista_nonneg",
                   action="store_false",
                   help="FISTA-SAE: allow signed codes (soft-threshold "
                        "prox) instead of the non-negative ReLU prox.")
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
    p.add_argument("--refit", action="store_true", default=False,
                   help="FISTA-SAE: after the FISTA solve, refit "
                        "coefficients on the discovered support via "
                        "projected GD (OLS, no L1). Removes the ℓ1 "
                        "shrinkage bias. Fast batched implementation.")
    p.add_argument("--refit-iters", type=int, default=30,
                   help="FISTA-SAE: refit budget (projected "
                        "GD steps). 15 usually converges within 1e-4 "
                        "relative residual on the support submatrix.")
    # ── iSAE / iSAE-ME (--architecture isae / isae_me) ────────────────
    # Port of third_party/isaes (rsae.RIPTopK), "Toward Identifiable
    # Sparse Autoencoders".  ``isae`` = AbsTopK + aRIP with a single-step
    # encoder; ``isae_me`` = the same model with a T-step iterative encoder
    # (T=5).  The classes live in ``consdl.models.isae``; these flags match
    # those of ``consdl.cli.train_lm``.
    p.add_argument("--rip-weight", type=float, default=0.1,
                   help="iSAE: aRIP regularization weight (paper: 0.1).")
    p.add_argument("--rip-multiplier", type=int, default=2,
                   help="iSAE: k' = k * multiplier for the aRIP Gram "
                        "submatrices (reference default 2).")
    p.add_argument("--rip-weighted", action="store_true", default=False,
                   help="iSAE: use real code values in the aRIP loss "
                        "instead of random-Gaussian substitutes on the "
                        "same support (reference default: off).")
    p.add_argument("--isae-l1", type=float, default=0.01,
                   help="iSAE: fixed-scalar L1 coefficient on the codes "
                        "(paper 0.01).  Divided by --k-per-token before it "
                        "reaches the trainer — see the sparsity_coefficient "
                        "note below.")
    p.add_argument("--num-encode-steps", type=int, default=1,
                   help="iSAE iterative-encoder steps (isae=1, isae_me=5).")
    p.add_argument("--initial-step-size", type=float, default=0.9,
                   help="iSAE-ME: initial learned step size (step = I - F, "
                        "so 0.9 ⇒ F = 0.1·I).")
    p.add_argument("--no-abstopk", dest="use_abstopk", action="store_false",
                   default=True,
                   help="iSAE: disable magnitude-based (sign-preserving) "
                        "TopK, falling back to ReLU+TopK.  On by default.")
    p.add_argument("--isae-input-norm", action="store_true", default=False,
                   help="iSAE: train/encode on per-sample mean-centred, "
                        "unit-L2 inputs -- the reference convention (see "
                        "IsaeSAE.input_norm).  Off by default: inputs are "
                        "raw activations.")
    p.add_argument("--isae-lr-decay-start-frac", type=float, default=0.8,
                   help="iSAE only.  Fraction of total steps at which the LR "
                        "begins a linear decay to zero (reference: 0.8, "
                        "rsae/trainers/base_trainer.py::_create_lr_scheduler). "
                        "0 disables decay, leaving a "
                        "constant-with-warmup schedule.  Ending un-annealed "
                        "is bad for cross-seed decoder agreement (DCS): "
                        "seeds are still taking full-size steps when "
                        "training stops.")
    p.add_argument("--isae-dead-window-batches", type=int, default=8,
                   help="iSAE dead-latent window in BATCHES; the AuxK dead "
                        "threshold becomes this × (batch_size × seq_len) "
                        "tokens (reference default: 8 batches, "
                        "configs/model/base.yaml::activation_window_batches). "
                        "Set 0 to fall back to --topk-aux-dead-threshold. "
                        "A multi-million-token threshold is far more lenient, "
                        "so AuxK rarely engages and latents killed early by "
                        "the L1 stay dead forever — death is absorbing under "
                        "TopK, since an unselected latent gets neither a "
                        "recon nor an L1 gradient.")

    # Co-active similarity regularization (ConSDL), applied by
    # SAETrainer when ``--co-active-sim-coefficient > 0`` (relu +
    # fista only); the other knobs are no-ops at the default (coef=0).
    p.add_argument("--co-active-sim-coefficient", type=float, default=0.0,
                   help="γ ≥ 0. Weight of the co-active similarity term. "
                        "0 (default) disables it.")
    p.add_argument("--co-active-sim-r", type=int, default=32,
                   help="Global top-r "
                        "candidate-neighbor cache size per atom. Default 32.")
    p.add_argument("--co-active-sim-refresh-steps", type=int, default=1,
                   help="Optimizer-step "
                        "cadence for refreshing the neighbor cache. Default "
                        "1 (refresh every step — always-exact neighbors; "
                        "the cost is small).")
    p.add_argument("--co-active-sim-barrier-delta", type=float, default=1e-4,
                   help="δ in the "
                        "-log1p(-cos²/(1+δ)) barrier. Default 1e-4.")
    p.add_argument("--co-active-sim-barrier-margin", type=float, default=0.0,
                   help="Cosine below which a co-active pair is left alone. 0 (default) applies the barrier to every co-active pair, whose only minimum is full orthogonality; a positive margin restricts the penalty to pairs whose cosine exceeds it.")
    p.add_argument("--co-active-sim-activation-eps", type=float, default=1e-6,
                   help="|a| threshold for "
                        "an atom to count as active (anchor-eligible). "
                        "Matches the repo's ZERO_EPS convention.")
    p.add_argument("--co-active-sim-cache-block-size", type=int, default=512,
                   help="Row-block size "
                        "for the neighbor-cache refresh matmul. Default 512.")
    p.add_argument("--log-interval", type=int, default=1000,
                   help="Wandb log cadence in *samples*. Translated to "
                        "fastsae's batch-cadence ``wandb_log_frequency`` "
                        "as ``max(--log-interval // --batch-size, 1)``.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-stats", action="store_true")
    p.add_argument("--no-ghost-grads", action="store_true",
                   help="Disable all auxiliary losses: fastsae's ghost-grad "
                        "dead-latent revival aux (relu etc.) "
                        "AND the topk overcomplete aux (when "
                        "architecture=topk). Maps to ``use_ghost_grads=False`` "
                        "for non-topk and ``aux_loss_coefficient=0`` for topk.")
    p.add_argument("--topk-aux-coefficient", type=float, default=1.0 / 32.0,
                   help="Active only when --architecture=topk and "
                        "ghost-grads are not disabled. Coefficient on "
                        "TopKAuxLossTrainer's saprmarks-style dead-mask-"
                        "gated AuxK loss (Gao et al. 2024 §A.2). Default "
                        "1/32, matching saprmarks/dictionary_learning.")
    p.add_argument("--topk-aux-dead-threshold", type=int, default=0,
                   help="Samples-since-last-fire before a latent is "
                        "considered dead and routed through the AuxK "
                        "loss. ``0`` ⇒ auto = 2%% of --total-samples "
                        "(saprmarks' 10M-on-500M ratio), so the threshold "
                        "scales with the training budget.")
    p.add_argument("--topk-aux-k-aux", type=int, default=0,
                   help="``k_aux`` for the AuxK loss; 0 ⇒ d_in // 2 "
                        "(Gao Appendix B.1 heuristic).")
    p.add_argument("--progress-file", type=str, default=None)
    p.add_argument("--wandb-project", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument(
        "--imagenet-cache-dir", type=str,
        default=os.environ.get(
            "IMAGENET_CACHE_DIR"),  # None -> the Hugging Face default cache
    )
    p.add_argument("--total-checkpoints", type=int, default=5)
    p.add_argument("--experiment-dir", type=str, default=None)
    p.add_argument("--dataset-order", type=str, default="shuffle",
                   choices=["shuffle", "ascending", "descending"])
    p.add_argument("--dataset-fraction", type=float, default=1.0,
                   help="Train on only this fraction of the (train-split) "
                        "dataset — a fixed seed-0 shuffled slice (default "
                        "1.0 = full).  imagenet is class-ordered, so the "
                        "slice is taken after a fixed shuffle to keep all "
                        "classes represented.  The stats / eval split is "
                        "unaffected.")
    p.add_argument("--cache-activations", action="store_true",
                   help="Disk-cache the backbone layer activations to an "
                        "explicit, reproducible cache file so repeated runs "
                        "on the same (backbone, layer, dataset, fraction) "
                        "reload it instead of re-running the backbone.  "
                        "Auto-disabled when the estimate exceeds 100 GB.")
    p.add_argument("--activation-cache-dir", type=str, default=None,
                   help="Directory for the --cache-activations cache files. "
                        "Default <cwd>/cache/train_activations.")
    p.add_argument("--sparsity-thresholds", type=float, nargs="+",
                   default=[],
                   help="Extra thresholds for `l0_per_token_<thr>` / `firing_rate_per_latent_<thr>`. Empty by default: the eps-based `l0_per_token_0` (|a| > zero_eps) is always written and is the L0 used downstream; each extra threshold adds a (d_sae,) tensor per run. Pass values explicitly for a threshold sweep.")
    return p.parse_args()


args = parse_args()

import json
import math
import os
import pathlib
from datetime import datetime

from dotenv import load_dotenv

from consdl.io.paths import REPO_ROOT  # noqa: E402
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ── Progress reporting: monkey-patch tqdm before fastsae import ─────────────
# Replace ``tqdm.tqdm`` with a subclass that, every 50 ``update()`` calls,
# writes ``{n, total}`` to ``--progress-file``.  fastsae creates a single
# "Training SAE" pbar during training, so no per-bar filtering is needed.
_PROGRESS_FILE = args.progress_file
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
from transformers import AutoProcessor

from consdl.data.vision_backbones import BACKBONE_MAP
from consdl.data.vision_datasets import (
    DATASET_MAP, apply_data_subset, format_dataset,
)

# Rebind the ``from tqdm import tqdm`` names in fastsae sub-modules that
# were imported (transitively, via ``consdl``) before the patch above and
# so still point at the original class.
if _PROGRESS_FILE:
    import fastsae.training.trainer as _fst_trainer
    import fastsae.interpret.compute_sae_stats as _fst_stats
    import fastsae.interpret.sae_interpreter as _fst_interp
    _fst_trainer.tqdm = _ProgressTqdm
    _fst_stats.tqdm = _ProgressTqdm
    _fst_interp.tqdm = _ProgressTqdm

# Size cap for the ``--cache-activations`` disk cache. Disabled (inf):
# caching is opt-in, so a silent cap would only hide that decision. A full
# ImageNet-train cache is large (1.28M imgs x 197 tokens x 768 dims x 4B
# = 775 GB).
_ACTIVATION_CACHE_CAP_GB = float("inf")


def _resume_skip_samples(experiment_dir) -> int:
    """Samples already trained, for slicing them off the dataloader.

    0 when there is nothing to resume, when no ``--experiment-dir`` is
    given, or when ``SAE_RESUME=0`` opts out. Any failure to read the state
    returns 0 rather than raising.
    """
    import os  # noqa: PLC0415
    if experiment_dir is None or os.getenv("SAE_RESUME", "1") == "0":
        return 0
    try:
        from consdl.training.resume import resume_state  # noqa: PLC0415
        rs = resume_state(experiment_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  resume: skip-count unavailable ({type(e).__name__}: {e})")
        return 0
    if rs is None:
        return 0
    return int(rs[1].get("n_training_samples", 0))


def _cached_latents_dataloader(dataloader, backbone, cache_file, map_batch_size):
    """Explicit, reproducible backbone-activation cache.

    Returns a dataloader that yields **precomputed layer activations**
    (a ``latent`` column) instead of images — the trainer's
    ``Backbone.forward_to_z`` consumes a ``latent`` key directly, so no
    backbone forward runs during training.

    This does **not** use fastsae's ``SAEWrapper._caching``, which lets HF
    name the ``.map`` cache by hashing the backbone closure; that hash is
    unstable across processes, so every job would recompute.  Here the
    cache is keyed on an explicit file path (``cache_file``): the first run
    writes it, later runs reload it.
    """
    import functools  # noqa: PLC0415

    import torch  # noqa: PLC0415
    from fastsae.models.wrapper import clone_dataloader  # noqa: PLC0415
    from filelock import FileLock  # noqa: PLC0415

    def _encode(bk, batch):  # batched .map fn → one {'latent': …} column
        with torch.no_grad():
            return {"latent": bk.forward_to_z(batch)}

    cache_file = pathlib.Path(cache_file)
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    img_dataset = dataloader.dataset
    # Concurrent jobs may all start before the cache exists.  A
    # cross-process ``FileLock`` makes the first job build it while the
    # rest wait and then reload it via ``load_from_cache_file``.
    with FileLock(str(cache_file) + ".lock"):
        print(f"[cache] {'reload' if cache_file.exists() else 'build'}: "
              f"{cache_file}")
        latents = img_dataset.map(
            functools.partial(_encode, backbone),
            batched=True,
            keep_in_memory=False,
            batch_size=int(map_batch_size),
            writer_batch_size=int(map_batch_size),
            remove_columns=img_dataset.column_names,
            cache_file_name=str(cache_file),     # explicit → reused if present
            load_from_cache_file=True,
            desc=f"activation cache → {cache_file.name}",
        )
    latents.reset_format()
    latents = latents.with_format("torch")
    return clone_dataloader(dataloader, latents)


def main() -> None:
    dataset_short = args.dataset
    dataset_name = DATASET_MAP[dataset_short]
    stats_dataset_short = args.stats_dataset
    stats_dataset_name = DATASET_MAP[stats_dataset_short]
    dataset_split = "train"
    stats_dataset_split = "val"
    collate = "auto"
    device = "cuda"

    backbone_cfg = BACKBONE_MAP[args.backbone]
    backbone_name = backbone_cfg["hf_name"]
    # Hook-layer path is backbone-specific: CLIP uses
    # ``vision_model.encoder.layers.<i>``; DINOv2 uses ``encoder.layer.<i>``.
    backbone_layer = backbone_cfg["hook_layer_fmt"].format(args.layer)

    # ``--d-sae N`` overrides the expansion factor; see below.
    d_sae_explicit = int(args.d_sae) if args.d_sae is not None else None
    d_sae_factor = args.expansion_factor
    architecture = args.architecture
    if architecture in ("isae", "isae_me"):
        # Reference iSAE normalizes L1 per active feature: sum(|a|)/(batch·k)
        # (third_party/isaes/rsae/rip_topk.py::compute_l1_loss). Our trainer's
        # sparsity_loss is sum(|a|)/token with no ÷k, so pass isae_l1/k to
        # match. Without it the L1 term is k× too strong and collapses all
        # codes to zero.
        sparsity_coefficient = float(args.isae_l1) / float(args.k_per_token)
    elif architecture == "relu":
        sparsity_coefficient = args.l1_coeff
    else:
        sparsity_coefficient = 0.0
    learning_rate = args.lr
    scheduler_name = "constantwithwarmup"
    warm_up_steps = 500
    batch_size = args.batch_size
    total_samples = args.total_samples
    total_checkpoints = args.total_checkpoints
    # FISTA-SAE and MP-SAE have no learned encoder, so the ghost-grad
    # dead-latent revival aux (which acts through the encoder) is
    # meaningless for them.
    use_ghost_grads = (
        (not args.no_ghost_grads)
        and architecture not in ("fista", "mp")
    )
    sae_b_dec_init_method = "geometric_median"
    stats_top_k_max_samples = 100

    # Default on; set WANDB_ENABLED=false in .env to disable.
    wandb_enabled = os.getenv("WANDB_ENABLED", "true").lower() != "false"
    wandb_project = args.wandb_project or os.getenv("WANDB_PROJECT", "consdl")
    wandb_entity = os.getenv("WANDB_ENTITY", "")

    print(f"[layer={args.layer} gpu={args.gpu}] Starting SAE training")
    print(f"  backbone       : {args.backbone}  ({backbone_name})")
    print(f"  backbone_layer : {backbone_layer}")
    print(f"  dataset        : {dataset_name}")
    print(f"  total_samples  : {total_samples:,}")
    print(f"  batch_size     : {batch_size}")
    print(f"  architecture   : {architecture}")
    print(f"  l1_coeff       : {sparsity_coefficient}")
    print(f"  expansion      : {d_sae_factor}x")
    if architecture == "fista":
        bits = (
            f"T={args.fista_iters}  "
            f"step={'auto' if float(args.fista_step) <= 0 else args.fista_step}  "
            f"λ={args.l1_coeff}  nonneg={'on' if args.fista_nonneg else 'off'}"
        )
        print(f"  {architecture:18s}: {bits}")
    if architecture in ("isae", "isae_me"):
        print(f"  {architecture:18s}: k={args.k_per_token}  "
              f"abstopk={'on' if args.use_abstopk else 'off'}  "
              f"rip={args.rip_weight} (k'={args.rip_multiplier}×k, "
              f"weighted={'on' if args.rip_weighted else 'off'})  "
              f"l1={args.isae_l1} (→ {sparsity_coefficient:.3e} after ÷k)  "
              f"T={args.num_encode_steps if architecture == 'isae_me' else 1}")
    print(f"  seed           : {args.seed if args.seed is not None else 'none'}")

    # Seed
    if args.seed is not None:
        import random
        random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)

    # Model
    backbone = fastsae.models.Backbone(
        model_hf_name=backbone_name, hook_target_name=backbone_layer, device=device,
    )
    d_in = backbone.hook_target_dim
    seq_len = (backbone.image_size // backbone.patch_size) ** 2 + 1

    # If --d-sae was passed, construct at the smallest integer factor
    # that yields ≥ d_sae_explicit latents; the SAE is then resized to the
    # exact width.
    if d_sae_explicit is not None:
        d_sae_factor = max(1, math.ceil(d_sae_explicit / d_in))
        print(f"  d_sae          : {d_sae_explicit} (exact; "
              f"constructed at factor={d_sae_factor} → "
              f"{d_in * d_sae_factor}, then resized down)")
    else:
        print(f"  d_sae          : {d_in * d_sae_factor} "
              f"(d_in × factor = {d_in} × {d_sae_factor})")

    sae_instance = build_sae(
        args, architecture, d_in=d_in, d_sae_factor=d_sae_factor, device=device,
        fista_lam=float(args.l1_coeff), mp_default_iters=int(args.k_per_token),
        mp_cap_d_sae=(d_sae_explicit if d_sae_explicit is not None
                      else int(d_in) * int(d_sae_factor)),
        d_sae=d_sae_explicit)

    wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae_instance, _caching=False)

    # Dataloader
    auto_processor = AutoProcessor.from_pretrained(backbone_name)

    def processor(img):
        return auto_processor(images=img, text="a photo", return_tensors="pt")

    _imagenet_hf = DATASET_MAP["imagenet"]
    train_cache_dir = args.imagenet_cache_dir if dataset_name == _imagenet_hf else None
    ds = None  # populated by either the cache-only branch or the HF load

    # ── Activation-cache short-circuit ───────────────────────────────
    # The cache file name is fixed by (backbone, layer, dataset,
    # dataset_fraction).  If it already exists, skip the HF dataset load
    # entirely: the cache holds post-backbone features, so the raw images
    # are not needed.
    use_activation_cache = False   # a built cache exists -> read it
    build_activation_cache = False  # no cache yet -> build it this run
    activation_cache_file = None
    if args.cache_activations:
        cache_root = pathlib.Path(
            args.activation_cache_dir
            or (REPO_ROOT / "var" / "cache" / "train_activations")
        )
        activation_cache_file = cache_root / (
            f"{args.backbone}_layer{args.layer}_{dataset_short}"
            f"_frac{args.dataset_fraction:g}.arrow"
        )
        if activation_cache_file.exists():
            print(f"[cache] reuse existing activation cache → "
                  f"{activation_cache_file}  ({activation_cache_file.stat().st_size / 1e9:.1f} GB)")
            use_activation_cache = True
            ds = None        # signals the cache-only path below
        else:
            print(f"[cache] activation cache MISSING → will build at "
                  f"{activation_cache_file}")
            # Separate flag from ``use_activation_cache`` (which means
            # "read an existing cache"): the normal image pipeline loads,
            # then its dataloader is wrapped in ``_cached_latents_dataloader``.
            build_activation_cache = True

    if ds is None and use_activation_cache:
        # Cache-only path: load the Arrow file directly, skip HF dataset.
        import datasets as _hfds  # noqa: PLC0415
        latents = _hfds.Dataset.from_file(str(activation_cache_file))
        latents.reset_format()
        latents = latents.with_format("torch")
        from torch.utils.data import DataLoader  # noqa: PLC0415
        dataloader = DataLoader(
            latents,
            batch_size=batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=lambda rows: {"latent": torch.stack([r["latent"] for r in rows])},
        )
        if _resume_skip_samples(args.experiment_dir):
            # The cache path builds its own loader from the Arrow file and
            # is not sliced, so a resume here replays the front of the
            # epoch. Flag it instead of reporting a clean resume.
            print("  resume: WARNING activation-cache loader is not "
                  "skip-aware — already-seen samples will repeat")
        # Skip the HF dataset / processor / dataset-fraction code below.
        ds = latents              # downstream uses ``len(ds)`` for logging only
    else:
        ds = datasets.load_dataset(path=dataset_name, split=dataset_split, cache_dir=train_cache_dir)
        ds = format_dataset(dataset_name, ds)
        # Apply the per-seed slice before ``dataset_fraction`` and the
        # ``shuffle`` re-order, so both operate on the filtered rows.
        if args.data_subset_mode and args.data_subset_mode != "none":
            _before = len(ds)
            ds = apply_data_subset(
                ds,
                mode=args.data_subset_mode,
                bucket_idx=int(args.data_subset_bucket_idx),
                n_buckets=int(args.data_subset_n_buckets),
            )
            print(
                f"  data-subset: mode={args.data_subset_mode} "
                f"bucket={args.data_subset_bucket_idx}/{args.data_subset_n_buckets} "
                f"→ {len(ds):,} / {_before:,} rows kept"
            )
        if float(args.dataset_fraction) < 1.0:
            # Deterministic class-mixed subset.  imagenet's train split is
            # ordered by class, so a raw head-slice would keep only the
            # first ~N classes — shuffle with a fixed seed first, then
            # slice, so the subset is reproducible *and* spans all classes.
            _n_keep = max(1, int(float(args.dataset_fraction) * len(ds)))
            ds = ds.shuffle(seed=0).select(range(_n_keep))
            print(f"  dataset-fraction: {args.dataset_fraction}  "
                  f"→ {_n_keep:,} train samples (seed-0 slice)")
        if args.dataset_order == "shuffle":
            if args.seed is not None:
                ds = ds.shuffle(seed=args.seed)
                # Resume: drop the rows this run already trained on. The
                # budget is a single epoch, so a restarted loader would
                # replay the front of this permutation and never reach the
                # tail. The order is a pure function of the seed, so the
                # slice lines up exactly.
                _skip = _resume_skip_samples(args.experiment_dir)
                if 0 < _skip < len(ds):
                    print(f"  resume: skipping {_skip:,} already-seen samples "
                          f"of the seed-{args.seed} order "
                          f"({len(ds) - _skip:,} left)")
                    ds = ds.select(range(_skip, len(ds)))
                elif _skip >= len(ds):
                    print(f"  resume: {_skip:,} seen >= {len(ds):,} rows — "
                          f"serving the full epoch again")
        else:
            if _resume_skip_samples(args.experiment_dir):
                print(f"  resume: WARNING dataset-order="
                      f"{args.dataset_order!r} is not skip-aware — "
                      f"already-seen samples will repeat")
            datasets.disable_caching()
            ds = ds.sort("label", reverse=(args.dataset_order == "descending"))
            datasets.enable_caching()

        # ── Activation cache size-cap + cold build ──────────────────
        if build_activation_cache:
            est_gb = len(ds) * int(seq_len) * int(d_in) * 4 / 1e9
            if est_gb > _ACTIVATION_CACHE_CAP_GB:
                print(f"[cache] estimated activation cache ≈ {est_gb:.0f} GB > "
                      f"{_ACTIVATION_CACHE_CAP_GB:.0f} GB cap — caching DISABLED")
                if activation_cache_file.exists():
                    activation_cache_file.unlink()
                activation_cache_file = None
                build_activation_cache = False
            else:
                print(f"[cache] activation cache ON — est ≈ {est_gb:.1f} GB → "
                      f"{activation_cache_file}")

        dataloader = fastsae.dataset.flexible.create_dataloader(
            dataset=ds, processor=processor,
            return_pixel_values_only=True, return_labels_np=False,
            batch_size=batch_size, shuffle=True, collate=collate,
            num_workers=args.num_workers,
        )
        # Cold build: the backbone forward runs once and writes the
        # cache; later runs hit the short-circuit above.
        if build_activation_cache:
            dataloader = _cached_latents_dataloader(
                dataloader, backbone, activation_cache_file, batch_size,
            )

    # Checkpoint path: ``--experiment-dir`` if given, otherwise
    #     <dataset>/<arch>/x<E>/<backbone>_layer<L>/<run_id>
    # under $CHECKPOINT_PATH.
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{shortuuid.uuid()[:6]}"
    config_tag = f"{architecture}_x{d_sae_factor}"
    if str(architecture) == "fista":
        # --l1-coeff doubles as the FISTA λ.
        config_tag += f"_lam{float(args.l1_coeff):g}"
    if float(getattr(args, "co_active_sim_coefficient", 0.0)) > 0.0:
        config_tag += f"_co_active_sim{float(args.co_active_sim_coefficient):g}"
        config_tag += f"_rf{int(args.co_active_sim_refresh_steps)}"
    if args.experiment_dir is not None:
        ckpt_path = pathlib.Path(args.experiment_dir)
        if not ckpt_path.is_absolute():
            ckpt_path = pathlib.Path.cwd() / args.experiment_dir
    else:
        ckpt_path = pathlib.Path(
            os.getenv("CHECKPOINT_PATH", str(pathlib.Path.cwd() / "var" / "checkpoints")),
            dataset_short, architecture, f"x{d_sae_factor}",
            f"{args.backbone}_layer{args.layer}", run_id,
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
        wandb_run_id = f"{args.backbone}_L{args.layer}_{config_tag}_{run_id}"

    # Train
    optimizer = torch.optim.Adam(wrapper.sae.parameters(), lr=learning_rate)
    if (architecture in ("isae", "isae_me")
            and float(args.isae_lr_decay_start_frac) > 0):
        # The reference iSAE uses a 3-phase schedule
        # (rsae/trainers/base_trainer.py::_create_lr_scheduler): linear warmup
        # → constant → linear decay to zero over the final
        # (1 - decay_start_fraction) of training. fastsae's
        # "linearwarmupdecay" decays from warmup onward, so the reference
        # lambda is reproduced here.
        _warm = max(1, int(warm_up_steps))
        _total = max(1, int(total_samples // batch_size))
        _decay_start = int(float(args.isae_lr_decay_start_frac) * _total)

        def _isae_lr_lambda(step: int) -> float:
            if step < _warm:
                return (step + 1) / _warm
            if step < _decay_start or _decay_start >= _total:
                return 1.0
            return max(0.0, (_total - step) / max(1, _total - _decay_start))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _isae_lr_lambda)
        print(f"  lr schedule    : 3-phase warmup({_warm}) → constant → "
              f"linear decay from step {_decay_start}/{_total} "
              f"(reference convention)")
    else:
        scheduler = fastsae.training.util.get_scheduler(scheduler_name, optimizer=optimizer, warm_up_steps=warm_up_steps)
    wrapper.backbone.eval()
    wrapper.sae.train()

    # ``--log-interval`` is in samples; ``wandb_log_frequency`` is in
    # batches.
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
        wandb_log_frequency=max(args.log_interval // args.batch_size, 1),
    )
    # In-repo trainers throughout: fastsae's ``use_ghost_grads`` is
    # declared but never read, so ``--no-ghost-grads`` only takes effect
    # through them. ``total_samples`` counts images while the dead-latent
    # counters count tokens, hence ``* seq_len``.
    isae_dead_thr = None
    if architecture in ("isae", "isae_me"):
        # Reference-equivalent dead window: ``activation_window_batches``
        # counts optimizer steps, one step = batch_size x seq_len tokens.
        win = int(args.isae_dead_window_batches)
        isae_dead_thr = (win * int(batch_size) * int(seq_len) if win > 0
                         else int(args.topk_aux_dead_threshold))
        print(f"  dead threshold : {isae_dead_thr:,} tokens")
    trainer = build_trainer(
        args, architecture, wrapper=wrapper, train_kwargs=train_kwargs,
        aux_off=bool(args.no_ghost_grads),
        dead_threshold_total=total_samples * int(seq_len),
        isae_dead_threshold=isae_dead_thr)
    if architecture in ("relu", "fista"):
        # Only the λ-controlled families can drift out of their L0 band;
        # top-k pins L0 at k by construction.
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
        # Exit non-zero without post-training stats. ``final/.l0_aborted``
        # records the abort in place of ``.training_complete``.
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
    # Completion sentinel. fastsae writes `final/state_dict.pt` in a
    # try/finally, so it exists even if training was killed; this marker is
    # written only after train() returns cleanly.
    _complete_path = pathlib.Path(ckpt_path) / "final" / ".training_complete"
    _complete_path.parent.mkdir(parents=True, exist_ok=True)
    _complete_path.write_text(json.dumps({
        "total_samples": total_samples,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }))
    print(f"[layer={args.layer} gpu={args.gpu}] Training complete. Checkpoint: {ckpt_path}")

    # Post-train stats
    if args.no_stats:
        print(f"[layer={args.layer} gpu={args.gpu}] Skipping stats (--no-stats).")
        return

    print(f"[layer={args.layer} gpu={args.gpu}] Computing stats on {stats_dataset_short}...")
    from consdl.io.paths import stats_write_dir  # noqa: PLC0415
    stats_path = stats_write_dir(
        pathlib.Path(ckpt_path, "final"),
        stats_dataset_short, stats_dataset_split,
    )
    stats_path.mkdir(parents=True, exist_ok=True)
    if (stats_path / "metrics.json").exists():
        print(f"[layer={args.layer} gpu={args.gpu}] Stats already exist. Skipping.")
        return

    stats_cache_dir = args.imagenet_cache_dir if stats_dataset_name == _imagenet_hf else None
    stats_ds = datasets.load_dataset(path=stats_dataset_name, split=stats_dataset_split, cache_dir=stats_cache_dir)
    stats_ds = format_dataset(stats_dataset_name, stats_ds)
    has_labels = "label" in stats_ds.features
    labels_np = np.array(stats_ds["label"]) if has_labels else None

    dataloader_stats = fastsae.dataset.flexible.create_dataloader(
        dataset=stats_ds, processor=auto_processor,
        return_pixel_values_only=True, return_labels_np=False,
        batch_size=batch_size, shuffle=False, collate=collate,
        num_workers=args.num_workers,
    )
    # Don't call ``wrapper.eval()``: ``nn.Module.eval`` calls
    # ``self.train(False)``, but ``SAEWrapper.train`` is overridden as the
    # training entry point. Set the two child modules directly instead.
    wrapper.backbone.eval()
    wrapper.sae.eval()
    from consdl.stats import SAEInterpreter
    _thrs = list(args.sparsity_thresholds or [])
    intrp = SAEInterpreter(
        wrapped=wrapper, backbone_processor=auto_processor,
        stats_dataloader=dataloader_stats, stats_labels_np=labels_np,
        stats_top_k_max_samples=stats_top_k_max_samples,
        l0_thresholds=_thrs,     # feeds l0_per_token_<thr>
    )
    try:
        intrp.compute_stats(compute_cls_wise_top_idx=has_labels)
    finally:
        # Drop raw activation buffers; only aggregated stats are saved.
        intrp.stats.pop("all_act", None)
        intrp.stats.pop("valid_latent_indices", None)
        intrp.save_stats(stats_path)
    print(f"[layer={args.layer} gpu={args.gpu}] Stats saved to {stats_path}")


if __name__ == "__main__":
    main()
