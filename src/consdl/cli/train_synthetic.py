"""Train one SAE on a synthetic ``z = D_gt a`` stream.

Wraps the SAE with :class:`IdentityBackbone`, so the optimiser, scheduler
and dead-latent handling are the same as for real-data backbones; only the
training target differs. Writes ``synthetic_config.json`` next to the
checkpoint and, unless ``--no-stats``, latent stats on a held-out pool.

Usage
-----
    python -m consdl.cli.train_synthetic \\
        --gpu 0 --seed 0 \\
        --gt-dict data/synthetic/dictionaries/D_gt_laplacian_er_seed0.pt \\
        --distribution uniform --k-active 90 \\
        --experiment-dir checkpoints/<run>/seed0
"""
from __future__ import annotations

import argparse

#: Architectures whose ``k`` is a real hyper-parameter and has to survive
#: into ``synthetic_config.json`` — everything the AuxK trainer covers.
_TOPK_FAMILY = ("topk", "batchtopk", "matryoshka", "isae", "isae_me")

#: Architectures built on :class:`FistaSAE`, i.e. the ones for which the
#: refit is a meaningful setting.
_FISTA_FAMILY = ("fista",)


def _distribution(name: str) -> str:
    """argparse type for ``--distribution``.

    A fixed ``choices=`` list cannot validate the sd family, whose name
    carries its parameters (``sd_r10_f010``). A typo fails at argument
    parsing instead of training on the wrong distribution.
    """
    from consdl.data.synthetic.dataset import is_valid_distribution  # noqa: PLC0415

    if is_valid_distribution(name):
        return name
    raise argparse.ArgumentTypeError(
        f"unknown distribution {name!r}; expected one of "
        f"uniform / lt03 / lt04 / lt05 / lt07 / lt10 / gauss05 / gauss10, "
        f"or an sd name of the form sd_r<R>_f<FFF> (R >= 1 integer, "
        f"FFF = dense-group percent 0-100, e.g. sd_r10_f010)")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gt-dict", required=True, help="Path to a D_gt .pt file.")
    p.add_argument("--distribution", required=True,
                   type=_distribution,
                   help="Per-latent firing-probability distribution p_i. "
                        "Fixed families, or sd_r<R>_f<FFF> for a "
                        "sparse/dense split.")
    p.add_argument("--k-active", type=int, default=90,
                   help="Target E[# active latents] per sample. Default 90.")
    p.add_argument("--lognormal-mu", type=float, default=0.0,
                   help="μ for the activation-magnitude law.  Interpreted "
                        "per-distribution (see --magnitude-distribution).  "
                        "Default 0.")
    p.add_argument("--lognormal-sigma", type=float, default=0.25,
                   help="σ for the activation-magnitude law.  Default 0.25 "
                        "(Elhage lognormal).  Ignored when "
                        "--magnitude-distribution=exponential.")
    p.add_argument("--magnitude-distribution", default="lognormal",
                   choices=["lognormal", "exponential", "uniform", "constant"],
                   help="Distribution for c_i (active-latent coefficient). "
                        "``lognormal`` is the default.  ``exponential`` "
                        "uses scale=exp(μ) (σ ignored).  ``uniform`` uses "
                        "Uniform(max(0, exp(μ)−σ), exp(μ)+σ).  ``constant`` "
                        "sets c_i = exp(μ) deterministically (default μ=0 "
                        "→ value 1); σ ignored.")
    p.add_argument("--sampling", default="bernoulli",
                   choices=["bernoulli", "fixed_k", "correlated"],
                   help="How to draw the active-latent set per sample. "
                        "``bernoulli`` (default): independent "
                        "Bernoulli(p_i); # active is Poisson-Binomial around "
                        "k_active.  ``fixed_k``: weighted sampling without "
                        "replacement of exactly k_active latents per sample, "
                        "with weights p_i/Σp_i; the tag gets an ``_fk`` suffix. "
                        "``correlated``: Gaussian copula derived from the "
                        "ground truth's graph-Laplacian eigenpairs (ground truth must be "
                        "built with a ``_cofire`` geometry); preserves marginal "
                        "firing rates while injecting graph-neighbour co-firing.")
    p.add_argument("--corr-rank", type=int, default=None,
                   help="(correlated sampling) rank of the copula's low-rank "
                        "factor F.  Default = d (use all stored eigenpairs).")
    p.add_argument("--corr-strength", type=float, default=1.0,
                   help="(correlated sampling) co-firing strength in [0, 1].  "
                        "0 ≡ bernoulli; 1 = full Laplacian-derived correlation.  "
                        "Marginal firing rates are preserved at every strength.")
    p.add_argument("--corr-tau", type=float, default=0.01,
                   help="(correlated sampling) ridge added to λ in the GMRF "
                        "precision Θ = L + τI.")
    p.add_argument("--pool-size", type=int, default=1_000_000)
    p.add_argument("--pool-seed", type=int, default=0,
                   help="Seed for the 1M-sample pool (shared across SAE seeds).")
    p.add_argument("--total-samples", type=int, default=5_000_000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--l1-coeff", type=float, default=8e-5)
    p.add_argument("--expansion-factor", type=int, default=4)
    p.add_argument("--d-sae", type=int, default=None,
                   help="Optional exact dictionary width. When set, "
                        "overrides --expansion-factor: the SAE is "
                        "constructed at the smallest integer factor "
                        "that gives >= this many latents, then resized "
                        "in place to exactly --d-sae (kaiming-reseeded "
                        "with --seed). Lets you target widths that "
                        "aren't integer multiples of d_in (e.g. 200 "
                        "with d_in=9).")
    p.add_argument("--architecture", default="relu",
                   choices=["relu", "topk", "batchtopk", "matryoshka",
                            "mp", "fista", "isae", "isae_me"])
    # ── iSAE / iSAE-ME (--architecture isae | isae_me) ──────────────────
    # Same flags and defaults as the vision and LM trainers, so the
    # synthetic iSAE is the published baseline rather than a re-tuned one.
    p.add_argument("--rip-weight", type=float, default=0.1,
                   help="iSAE: aRIP regularization weight (paper: 0.1).")
    p.add_argument("--rip-multiplier", type=int, default=2,
                   help="iSAE: k' = k * multiplier for the aRIP Gram "
                        "submatrices.")
    p.add_argument("--rip-weighted", action="store_true", default=False,
                   help="iSAE: use real code values in the aRIP loss "
                        "instead of N(0,1) on the same support.")
    p.add_argument("--isae-l1", type=float, default=0.01,
                   help="iSAE: fixed-scalar L1 on the codes. Divided by k "
                        "before it reaches the trainer — see the "
                        "sparsity_coefficient comment below.")
    p.add_argument("--num-encode-steps", type=int, default=1,
                   help="iSAE iterative-encoder steps (isae=1, isae_me=5).")
    p.add_argument("--initial-step-size", type=float, default=0.9,
                   help="iSAE-ME: initial learned step size.")
    p.add_argument("--no-abstopk", dest="use_abstopk", action="store_false",
                   default=True,
                   help="iSAE: disable magnitude (sign-preserving) top-k.")
    p.add_argument("--isae-input-norm", action="store_true", default=False,
                   help="iSAE: unit-norm the input before encoding.")
    # FISTA-SAE (``--architecture=fista``): FISTA sparse-coding encoder
    # plus a trained decoder. The code is solved under no_grad each step;
    # the reconstruction MSE trains the decoder.
    p.add_argument("--fista-iters", type=int, default=100,
                   help="FISTA inner-iteration count per encode "
                        "(--architecture=fista).  Default 100.")
    p.add_argument("--fista-step", type=float, default=0.0,
                   help="FISTA proximal-gradient step η.  ``0`` (default) "
                        "auto-resolves 1/σ_max(D)² per forward via power "
                        "iteration; pass a positive value to pin it "
                        "(faster — skips the per-step estimate).")
    p.add_argument("--fista-lam", type=float, default=1e-3,
                   help="FISTA L1 penalty λ.  Sparsity is enforced by the "
                        "prox, so there is no separate --l1-coeff term.  "
                        "Default 1e-3.")
    p.add_argument("--fista-nonneg", action="store_true", default=True,
                   help="Constrain the FISTA code a ≥ 0 (prox = ReLU(z−η·λ)). "
                        "Default True — synthetic ground-truth dictionaries emit "
                        "non-negative codes.")
    p.add_argument("--no-fista-nonneg", dest="fista_nonneg",
                   action="store_false",
                   help="Allow signed codes (signed soft-threshold prox).")
    # Refit: same meaning as ``--refit{,-iters}`` on the LM and vision
    # trainers.
    p.add_argument("--refit", action="store_true", default=False,
                   help="Refit the coefficients on the support FISTA "
                        "found, by projected GD with no L1, so the code "
                        "the decoder sees keeps FISTA's support choice "
                        "but loses the ℓ1 shrinkage bias on its values.")
    p.add_argument("--refit-iters", type=int, default=30,
                   help="Projected-GD budget for the refit. "
                        "15 usually converges to within 1e-4 relative "
                        "residual.")
    p.add_argument("--k-per-token", type=int, default=90,
                   help="Used by topk / batchtopk AT TRAIN TIME (hard "
                        "per-token budget); ignored for relu / mp.  "
                        "Defaults to --k-active.")
    p.add_argument("--k-per-token-inference", type=int, default=0,
                   help="Per-token budget for topk AT EVAL TIME.  "
                        "``0`` → mirror --k-per-token (train / eval "
                        "use the same budget).  Pass a different value to study "
                        "how a small-budget-trained encoder "
                        "generalises to a larger eval budget.")
    # AuxK (Gao et al. 2024, App. B.1), same defaults as the LM / vision
    # trainers. The dead threshold is counted in samples; ``0`` resolves to
    # ``--total-samples``.
    p.add_argument("--topk-aux-coefficient", type=float, default=1.0 / 32.0,
                   help="Active only when --architecture=topk and ghost-grads "
                        "is ON.  Weight on the dead-latent residual "
                        "reconstruction aux term (Gao AuxK).  Default "
                        "1/32 — matches LM / vision.")
    p.add_argument("--topk-aux-dead-threshold", type=int, default=0,
                   help="Samples-since-last-fire before a latent is "
                        "considered dead and routed through the AuxK "
                        "loss.  ``0`` ⇒ auto = ``--total-samples`` "
                        "(mirrors LM / vision, where threshold == "
                        "training budget by default).")
    p.add_argument("--topk-aux-k-aux", type=int, default=0,
                   help="``k_aux`` for the AuxK loss; 0 ⇒ d_in // 2 "
                        "(Gao Appendix B.1 heuristic, matches LM / vision).")
    # MP-specific knobs (active when --architecture=mp).
    p.add_argument("--mp-iters", type=int, default=0,
                   help="Matching-pursuit rounds AT TRAIN TIME.  "
                        "0 → fall back to --k-active.")
    p.add_argument("--mp-inference-mode",
                   choices=["fixed", "adaptive"], default="fixed",
                   help="Eval-time round-count rule.  ``fixed``: run "
                        "exactly ``--mp-inference-iters`` rounds.  "
                        "``adaptive``: unroll until the relative "
                        "residual norm drops below "
                        "``--mp-residual-thr`` or the support stops "
                        "growing for ``--mp-patience`` rounds, "
                        "capped by ``--mp-max-iters``.  Adaptive "
                        "mode lets the model pick its own per-input "
                        "budget — no fixed sparsity hyperparameter "
                        "to tune at eval time.")
    p.add_argument("--mp-inference-iters", type=int, default=0,
                   help="Round count for ``--mp-inference-mode=fixed``.  "
                        "``0`` → mirror ``--mp-iters`` (train and "
                        "eval use the same budget).")
    p.add_argument("--mp-residual-thr", type=float, default=0.05,
                   help="Adaptive-mode threshold on relative residual "
                        "norm — stop when ``max ‖res‖ / ‖z‖ < thr``.  "
                        "Default 0.05 (5%% of input norm).")
    p.add_argument("--mp-max-iters", type=int, default=0,
                   help="Safety cap for adaptive mode.  ``0`` → fall "
                        "back to ``--mp-sparsity-cap`` (when > 0) or "
                        "to ``d_sae`` (the worst-case useful budget).")
    p.add_argument("--mp-sparsity-cap", type=float, default=0.1,
                   help="Per-token sparsity cap for "
                        "``--mp-inference-mode=adaptive``: "
                        "``max_iters = ceil(cap × d_sae)`` (each MP "
                        "round picks at most one new atom).  Default "
                        "``0.1`` (10%% of ``d_sae``); pass ``0`` to "
                        "disable.  ``--mp-max-iters`` (when > 0) wins "
                        "over this so explicit budgets are still "
                        "honoured.  Applied via the shared "
                        "``mp_inference`` helper.")
    p.add_argument("--mp-patience", type=int, default=2,
                   help="Adaptive-mode support-stabilisation patience.")
    p.add_argument("--mp-nonneg", action="store_true", default=True,
                   help="Restrict atom selection to non-negative correlations "
                        "(matches non-negative GT codes).")
    p.add_argument("--no-mp-nonneg", dest="mp_nonneg", action="store_false",
                   help="Allow signed atom selection.")
    p.add_argument("--mp-dropout", type=float, default=0.0,
                   help="Bernoulli atom dropout fraction during training "
                        "(disabled at eval).  Default 0.")
    p.add_argument("--mp-revive-coef", type=float, default=1e-2,
                   help="Coefficient for the dead-atom revival aux loss "
                        "(0 disables).  Active only when --architecture=mp "
                        "and ghost-grads is ON.  Default 1e-2.")
    # Co-active similarity regularization (ConSDL).
    p.add_argument("--co-active-sim-coefficient", type=float, default=0.0,
                   help="γ ≥ 0.  ``0`` (default) disables the co-active "
                        "similarity term.")
    # ── co-active similarity knobs ──
    p.add_argument("--co-active-sim-r", type=int, default=32,
                   help="Candidate pool size per decoder atom (top-r "
                        "positive-cosine neighbors, cached). Default 32, "
                        "which covers ~90%% (vision) / ~99%% (LM) of "
                        "co-active neighbors; r=4 covers ~53%%/82%%.")
    p.add_argument("--co-active-sim-refresh-steps", type=int, default=100,
                   help="Steps between neighbor-cache refreshes. Default "
                        "100 (refreshing every step roughly doubles run "
                        "time without a measurable change in the "
                        "consistency metrics). Lower to 1 for "
                        "always-exact nearest co-active neighbors when "
                        "the exact cost is affordable; raise for "
                        "faster/looser cells.")
    p.add_argument("--co-active-sim-barrier-delta", type=float, default=1e-4,
                   help="Stability constant in the co-active similarity barrier "
                        "-log((1-s^2+delta)/(1+delta)). Bounds the "
                        "barrier's max gradient (~2/delta at s=1). "
                        "Must NOT be implemented via "
                        "clamp_min/clamp — both zero the gradient exactly "
                        "on near-exact duplicates.")
    p.add_argument("--co-active-sim-barrier-margin", type=float, default=0.0,
                   help="Cosine below which a co-active pair is left "
                        "alone. 0 (default) applies the barrier to every "
                        "co-active pair; a positive margin restricts the "
                        "penalty to pairs whose cosine exceeds it.")
    p.add_argument("--co-active-sim-activation-eps", type=float, default=1e-6,
                   help="Activation threshold defining the anchor/co-active "
                        "set (same default as the ZERO_EPS convention used "
                        "by the stats pass).")
    p.add_argument("--co-active-sim-cache-block-size", type=int, default=512,
                   help="Row-block size for the periodic neighbor-cache "
                        "refresh's Gram computation, and for the per-step "
                        "candidate-cosine table (also blockwise, same "
                        "rationale).")
    p.add_argument("--total-checkpoints", type=int, default=10)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--log-interval", type=int, default=5000)
    # Disables the arch's dead-latent revival aux loss: ghost grads
    # (relu / fista), AuxK (top-k family) or revival aux (mp).
    p.add_argument("--no-aux-loss", action="store_true",
                   help="Disable the arch's revival aux loss.")
    p.add_argument("--detect-anomaly", action="store_true",
                   help="Debug: wrap training in "
                        "torch.autograd.set_detect_anomaly + register NaN "
                        "forward hooks on the SAE, to pin the exact op that "
                        "emits the first non-finite value.  Slows training; "
                        "use with a small --total-samples.")
    p.add_argument("--experiment-dir", required=True)
    p.add_argument("--progress-file", default=None)
    p.add_argument("--wandb-project", default=None)
    # Post-training stats pass.
    p.add_argument("--no-stats", action="store_true",
                   help="Skip the in-process stats pass after training.")
    p.add_argument("--stats-top-k", type=int, default=100,
                   help="Top-K activating samples tracked per latent.")
    p.add_argument("--stats-max-samples", type=int, default=100_000,
                   help="Cap the stats pass at this many synthetic samples "
                        "(independently drawn from the held-out pool below).")
    p.add_argument("--stats-pool-seed", type=int, default=0,
                   help="pool_seed for the HELD-OUT stats pool. "
                        "0 (default) → derive as --pool-seed XOR 0xDEADBEEF so "
                        "the stats pass uses samples disjoint from training.")
    p.add_argument("--sparsity-thresholds", type=float, nargs="+",
                   default=[],
                   help="Extra thresholds for `l0_per_token_<thr>` / `firing_rate_per_latent_<thr>`. Empty by default: the eps-based `l0_per_token_0` (|a| > zero_eps) is always written and is the L0 used downstream; each extra threshold adds a (d_sae,) tensor per run. Pass values explicitly for a threshold sweep.")
    return p.parse_args()


args = parse_args()

import os
import pathlib
import json
import random
from datetime import datetime

from dotenv import load_dotenv

from consdl.io.paths import REPO_ROOT  # noqa: E402
load_dotenv(dotenv_path=REPO_ROOT / ".env", override=True)
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

# Mirror tqdm progress to ``--progress-file`` as JSON.
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

import numpy as np
import torch
import fastsae
from consdl.training.build import build_sae, build_trainer
from torch.utils.data import DataLoader

from consdl.data.synthetic.dataset import SyntheticSparseDataset, SyntheticSpec, collate_z, collate_z_seq
from consdl.data.synthetic.backbone import IdentityBackbone


def main() -> None:
    device = "cuda"
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)

    blob = torch.load(args.gt_dict, map_location="cpu")
    D_gt: torch.Tensor = blob["D_gt"]
    d, d_sae_gt = D_gt.shape
    # Cofire payloads; at most one is present:
    #   ``corr_eigvals`` → laplacian_*_cofire (graph spectrum)
    #   ``corr_factor``  → gaussian_cofire   (random low-rank F)
    # Both are ``None`` for every other geometry.
    corr_eigvals = blob.get("corr_eigvals")
    corr_factor = blob.get("corr_factor")
    cofire_tag = (
        "eigvals" if corr_eigvals is not None
        else "factor" if corr_factor is not None
        else "no"
    )
    print(f"[ground truth] {args.gt_dict}  geometry={blob.get('geometry')}  "
          f"D_gt={tuple(D_gt.shape)}  cofire={cofire_tag}")

    # Dataset + dataloader.
    spec = SyntheticSpec(
        distribution=args.distribution,
        k_active=args.k_active,
        lognormal_mu=args.lognormal_mu,
        lognormal_sigma=args.lognormal_sigma,
        pool_size=args.pool_size,
        pool_seed=args.pool_seed,
        magnitude_distribution=args.magnitude_distribution,
        sampling=args.sampling,
        corr_rank=args.corr_rank,
        corr_strength=args.corr_strength,
        corr_tau=args.corr_tau,
    )
    ds = SyntheticSparseDataset(D_gt, spec, corr_eigvals=corr_eigvals,
                                corr_factor=corr_factor)
    print(f"[pool] dist={spec.distribution}  k={spec.k_active}  "
          f"mag={spec.magnitude_distribution}  "
          f"sampling={spec.sampling}  "
          f"μ={spec.lognormal_mu:+.2f}  σ={spec.lognormal_sigma:.2f}  "
          f"Σp={ds._p_bern.sum():.3g}  "
          f"max_p={ds._p_bern.max():.3g}  min_p={ds._p_bern.min():.3g}")
    loader = DataLoader(
        ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_z,
        pin_memory=True, drop_last=True,
        generator=torch.Generator().manual_seed(args.seed),
    )

    # When ``--d-sae N`` is set, override --expansion-factor with the
    # smallest integer factor yielding >= N latents; exact width is
    # enforced via resize_sae_in_place right after SAE construction.
    d_sae_explicit = int(args.d_sae) if args.d_sae is not None else None
    if d_sae_explicit is not None:
        import math as _math
        args.expansion_factor = max(1, _math.ceil(d_sae_explicit / int(d)))
        print(f"[d_sae] exact={d_sae_explicit}  "
              f"construct at factor={args.expansion_factor} "
              f"({int(d)} × {args.expansion_factor} = "
              f"{int(d) * args.expansion_factor}), then resize down.")

    # SAE. MP's training round count defaults to the planted k.
    sae = build_sae(
        args, args.architecture, d_in=int(d), d_sae_factor=int(args.expansion_factor),
        device=device, fista_lam=float(args.fista_lam),
        mp_default_iters=int(args.k_active),
        mp_cap_d_sae=int(d) * int(args.expansion_factor),
        d_sae=d_sae_explicit,
        extra={"topk": {"top_k_inference": int(args.k_per_token_inference)}})

    # SAEWrapper does a strict `backbone.device == sae.device` equality check;
    # torch.device("cuda") != torch.device("cuda", 0), so mirror sae.device exactly.
    backbone = IdentityBackbone(d_in=d, device=sae.device, dtype=sae.dtype).to(sae.device)
    backbone.device = sae.device
    wrapper = fastsae.models.SAEWrapper(backbone=backbone, sae=sae, _caching=False)

    # Optim / scheduler.
    optimizer = torch.optim.Adam(wrapper.sae.parameters(), lr=args.lr)
    scheduler = fastsae.training.util.get_scheduler(
        "constantwithwarmup", optimizer=optimizer, warm_up_steps=500,
    )
    wrapper.backbone.eval(); wrapper.sae.train()

    ckpt = pathlib.Path(args.experiment_dir)
    ckpt.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # wandb rejects run names over 128 characters, so clamp the descriptive
    # part and keep the timestamp tail for uniqueness.
    _wandb_name = (
        f"synth_{blob.get('geometry','?')}_{args.architecture}_{args.distribution}"
        f"_x{int(args.expansion_factor)}"
        + (f"_lam{float(args.fista_lam):g}"
           if str(args.architecture).startswith("fista")
           else "")
        # Refit runs use ``--architecture fista``; tag them so the name
        # differs from plain FISTA at the same λ.
        + ("_refit" if bool(getattr(args, "refit", False)) else "")
        + (
            f"_co_active_sim{float(args.co_active_sim_coefficient):g}"
            f"_rf{int(args.co_active_sim_refresh_steps)}"
            if float(getattr(args, "co_active_sim_coefficient", 0.0)) > 0.0 else ""
        )
        + f"_u{args.lognormal_mu:+.2f}_s{args.lognormal_sigma:.2f}"
        f"_k{args.k_active}_seed{args.seed}"
    )
    _tail = f"_{run_id}"  # 16-char timestamp, kept for uniqueness
    if len(_wandb_name) + len(_tail) > 128:
        _wandb_name = _wandb_name[: 128 - len(_tail)]
    _wandb_name += _tail

    train_kwargs = dict(
        dataloader=loader, optimizer=optimizer, scheduler=scheduler,
        total_n_training_samples=args.total_samples,
        # The L1 penalty applies to relu; topk / mp enforce sparsity
        # structurally. iSAE's reference normalises its L1 per active
        # feature (sum|a| / (batch·k)) while this trainer's sparsity loss
        # does not divide by k, so the coefficient is divided here;
        # otherwise the term is k× too strong and collapses the encoder.
        sparsity_coefficient=(
            args.l1_coeff if args.architecture == "relu"
            else float(args.isae_l1) / float(args.k_per_token)
            if args.architecture in ("isae", "isae_me")
            else 0.0),
        use_ghost_grads=not args.no_aux_loss,
        sae_b_dec_init_method="geometric_median",
        checkpoint_path=str(ckpt),
        total_n_checkpoints=args.total_checkpoints,
        # Default ON — set WANDB_ENABLED=false in .env to suppress.
        log_to_wandb=os.getenv("WANDB_ENABLED", "true").lower() != "false",
        wandb_run_id=_wandb_name,
        wandb_project=args.wandb_project or os.getenv("WANDB_PROJECT", "synthetic"),
        wandb_entity=os.getenv("WANDB_ENTITY", ""),
        # Counted in batches; convert from the sample-based --log-interval.
        wandb_log_frequency=max(args.log_interval // args.batch_size, 1),
    )
    # fastsae's ``use_ghost_grads`` is never read; ``--no-aux-loss`` takes
    # effect through the in-repo trainers.
    trainer = build_trainer(
        args, args.architecture, wrapper=wrapper, train_kwargs=train_kwargs,
        aux_off=bool(args.no_aux_loss), dead_threshold_total=args.total_samples)

    if getattr(args, "detect_anomaly", False):
        # Locate the first non-finite value. ``encode`` / ``decode`` are
        # wrapped because the trainer calls them directly, bypassing
        # ``forward`` hooks. Each step checks parameters and outputs with
        # ``torch.isfinite`` (an overflow shows up as Inf before NaN). The
        # weight abs-max is printed every 20 steps to expose a gradual
        # blow-up.
        _anom = {"hit": False, "step": 0}
        _sae = wrapper.sae

        def _first_bad_param():
            for pn, p in _sae.named_parameters():
                if not torch.isfinite(p).all():
                    return pn, float(p.detach().abs().max())
            return None

        def _wrap(fn, fname):
            def _checked(*a, **k):
                _anom["step"] += (1 if fname == "encode" else 0)
                if not _anom["hit"]:
                    bad = _first_bad_param()
                    if bad is not None:
                        _anom["hit"] = True
                        print(f"[detect-anomaly] step {_anom['step']}: "
                              f"non-finite PARAMETER '{bad[0]}' at entry of "
                              f"{fname} (abs-max {bad[1]:.3e}) — the optimizer "
                              f"step before this produced it", flush=True)
                out = fn(*a, **k)
                if not _anom["hit"]:
                    outs = out if isinstance(out, (tuple, list)) else (out,)
                    for i, o in enumerate(outs):
                        if isinstance(o, torch.Tensor) and o.numel() \
                                and not torch.isfinite(o).all():
                            _anom["hit"] = True
                            fin = o.detach()[torch.isfinite(o.detach())]
                            fmax = float(fin.abs().max()) if fin.numel() else float("nan")
                            print(f"[detect-anomaly] step {_anom['step']}: "
                                  f"non-finite OUTPUT[{i}] of {fname}() with "
                                  f"all parameters still finite — finite-part "
                                  f"abs-max {fmax:.3e}", flush=True)
                if fname == "encode" and _anom["step"] % 20 == 0 \
                        and not _anom["hit"]:
                    we = float(_sae.W_enc.detach().abs().max())
                    wd = float(_sae.W_dec.detach().abs().max())
                    bd = float(_sae.b_dec.detach().abs().max())
                    print(f"[detect-anomaly] step {_anom['step']:5d}  "
                          f"|W_enc|max={we:.3e}  |W_dec|max={wd:.3e}  "
                          f"|b_dec|max={bd:.3e}", flush=True)
                return out
            return _checked

        _sae.encode = _wrap(_sae.encode, "encode")
        _sae.decode = _wrap(_sae.decode, "decode")
        print("[detect-anomaly] enabled — encode/decode finiteness scan + "
              "weight-norm trajectory, autograd anomaly mode on", flush=True)
        with torch.autograd.set_detect_anomaly(True):
            trainer.train()
    else:
        trainer.train()

    # Completion sentinel: final/state_dict.pt is written in a try/finally,
    # so only this marker (written after train() returns) shows the run was
    # not killed mid-loop.
    _complete_path = ckpt / "final" / ".training_complete"
    _complete_path.parent.mkdir(parents=True, exist_ok=True)
    _complete_path.write_text(json.dumps({
        "total_samples": args.total_samples,
        "finished_at": datetime.now().isoformat(timespec="seconds"),
    }))

    # Record the synthetic data recipe beside the checkpoint so evaluation
    # can rebuild the exact pool the SAE saw without parsing the tag.
    cfg_out = ckpt / "synthetic_config.json"
    cfg_out.write_text(json.dumps({
        "gt_dict": str(pathlib.Path(args.gt_dict).resolve()),
        "geometry": blob.get("geometry"),
        "distribution": args.distribution,
        "k_active": args.k_active,
        "lognormal_mu": args.lognormal_mu,
        "lognormal_sigma": args.lognormal_sigma,
        "magnitude_distribution": args.magnitude_distribution,
        "sampling": args.sampling,
        "pool_size": args.pool_size,
        "pool_seed": args.pool_seed,
        "batch_size": args.batch_size,
        "d": int(d),
        "seed": args.seed,
        "architecture": args.architecture,
        # k is needed to rebuild any top-k-family encoder.
        "top_k": (args.k_per_token
                   if args.architecture in _TOPK_FAMILY else None),
        "top_k_inference": (int(getattr(sae, "top_k_inference", 0)) or
                             int(getattr(sae, "top_k", 0))
                             if args.architecture in _TOPK_FAMILY else None),
        "matryoshka_group_sizes": (list(getattr(sae, "group_sizes", []))
                                    if args.architecture == "matryoshka"
                                    else None),
        "refit": (bool(args.refit)
                               if args.architecture in _FISTA_FAMILY else None),
        "refit_iters": (int(args.refit_iters)
                                     if args.architecture in _FISTA_FAMILY
                                     and args.refit else None),
        "mp_iters": (int(getattr(sae, "k_iters", 0))
                     if args.architecture == "mp" else None),
        "mp_inference_mode": (str(getattr(sae, "inference_mode", "fixed"))
                               if args.architecture == "mp" else None),
        "mp_inference_iters": (int(getattr(sae, "inference_iters", 0))
                                if args.architecture == "mp" else None),
        "mp_residual_thr": (float(getattr(sae, "residual_thr", 0.0))
                             if args.architecture == "mp" else None),
        "mp_max_iters": (int(getattr(sae, "max_iters", 0))
                          if args.architecture == "mp" else None),
        "mp_sparsity_cap": (float(args.mp_sparsity_cap)
                             if args.architecture == "mp" else None),
        "d_sae": (int(getattr(sae, "d_sae", 0))
                   if args.architecture == "mp" else None),
        "co_active_sim_coefficient": float(args.co_active_sim_coefficient),
        "co_active_sim_r": (int(args.co_active_sim_r)
                            if float(args.co_active_sim_coefficient) > 0.0 else None),
        "co_active_sim_refresh_steps": (int(args.co_active_sim_refresh_steps)
                            if float(args.co_active_sim_coefficient) > 0.0 else None),
        "co_active_sim_barrier_delta": (float(args.co_active_sim_barrier_delta)
                            if float(args.co_active_sim_coefficient) > 0.0 else None),
        "co_active_sim_activation_eps": (float(args.co_active_sim_activation_eps)
                            if float(args.co_active_sim_coefficient) > 0.0 else None),
        "co_active_sim_barrier_margin": (float(args.co_active_sim_barrier_margin)
                            if float(args.co_active_sim_coefficient) > 0.0 else None),
        "mp_patience": (int(getattr(sae, "patience", 0))
                         if args.architecture == "mp" else None),
        "mp_nonneg": (bool(args.mp_nonneg)
                      if args.architecture == "mp" else None),
        "mp_dropout": (float(args.mp_dropout)
                       if args.architecture == "mp" else None),
        "stats_pool_seed": (args.stats_pool_seed if args.stats_pool_seed != 0
                            else (args.pool_seed ^ 0xDEADBEEF)),
    }, indent=2))

    # ── Post-train stats on a HELD-OUT synthetic pool ──────────────────
    # Same spec, different pool_seed: the stats samples are drawn
    # independently of the training pool.
    if args.no_stats:
        print(f"[done] {ckpt} (skipped stats per --no-stats)")
        return

    from dataclasses import replace as _replace
    from torch.utils.data import DataLoader as _DataLoader, Subset as _Subset
    from consdl.stats import SAEInterpreter

    final_dir = ckpt / "final"
    stats_path = final_dir / "stats.synthetic.split_pool.seed_no"
    if (stats_path / "top_act.pt").exists():
        print(f"[stats] skip — already present at {stats_path}")
    else:
        stats_path.mkdir(parents=True, exist_ok=True)
        wrapper.backbone.eval(); wrapper.sae.eval()
        # Reload the saved final weights so the stats match the checkpoint
        # on disk (the in-memory copy can differ after decoder-norm ops).
        # ``load_sae`` dispatches on ``class_type`` in the saved config.
        from consdl.io.checkpoint import load_sae
        sae_final = load_sae(final_dir)
        # The SAE loads on CPU and SAEWrapper asserts
        # backbone.device == sae.device, so move it first.
        sae_final = sae_final.to(backbone.device)
        sae_final.device = backbone.device
        wrapper_final = fastsae.models.SAEWrapper(
            backbone=backbone, sae=sae_final, _caching=False,
        )
        wrapper_final.backbone.eval(); wrapper_final.sae.eval()

        # The XOR derivation is deterministic, so the held-out pool can be
        # rebuilt later from the config.
        stats_pool_seed = args.stats_pool_seed if args.stats_pool_seed != 0 \
            else (args.pool_seed ^ 0xDEADBEEF)
        stats_spec = _replace(spec, pool_seed=stats_pool_seed,
                              pool_size=args.stats_max_samples
                              or spec.pool_size)
        stats_ds_full = SyntheticSparseDataset(D_gt, stats_spec, corr_eigvals=corr_eigvals,
                                               corr_factor=corr_factor)
        print(f"[stats] held-out pool_seed={stats_pool_seed}  "
              f"(training pool_seed={args.pool_seed})")

        stats_ds = stats_ds_full
        if args.stats_max_samples is not None and args.stats_max_samples < len(stats_ds_full):
            stats_ds = _Subset(stats_ds_full, list(range(int(args.stats_max_samples))))
        stats_loader = _DataLoader(
            stats_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers, collate_fn=collate_z_seq,
            pin_memory=True, drop_last=False,
            persistent_workers=args.num_workers > 0,
        )
        _thrs = list(args.sparsity_thresholds or [])
        # The synthetic pool has no labels; pass zeros so the top-label
        # lookup works. The resulting label entropy is degenerate and unused.
        import numpy as _np
        _labels = _np.zeros(len(stats_ds), dtype=_np.int64)
        intrp = SAEInterpreter(
            wrapped=wrapper_final,
            backbone_processor=None,
            stats_dataloader=stats_loader,
            stats_labels_np=_labels,
            stats_top_k_max_samples=args.stats_top_k,
            sparsity_thr=_thrs,
            l0_thresholds=_thrs,
        )
        try:
            intrp.compute_stats(compute_cls_wise_top_idx=False)
        finally:
            intrp.stats.pop("all_act", None)
            intrp.stats.pop("valid_latent_indices", None)
            intrp.save_stats(stats_path)
        print(f"[stats] wrote → {stats_path}")

    print(f"[done] checkpoint tree under {ckpt}")


if __name__ == "__main__":
    main()
