"""SAE and trainer construction shared by the vision, LM and synthetic trainers.

The three ``consdl.cli.train_*`` scripts differ in their data and backbone;
which SAE class an ``--architecture`` builds, which trainer trains it and
how co-active similarity regularization is attached is the same for all
three and lives here.
"""
from __future__ import annotations

from typing import Any

ARCHITECTURES = ("relu", "topk", "batchtopk", "matryoshka", "mp", "fista",
                 "isae", "isae_me")

#: FISTA options passed through when the calling script defines them;
#: the rest keep ``FistaSAE``'s defaults.
_FISTA_OPTIONAL = ("fista_r2_tol", "fista_check_every", "fista_use_bf16",
                   "fista_step_cache_every", "fista_power_iters", "fista_compile")


def build_sae(args, architecture: str, *, d_in: int, d_sae_factor: int,
              device, fista_lam: float, mp_default_iters: int,
              mp_cap_d_sae: int, d_sae: int | None = None,
              extra: dict[str, dict[str, Any]] | None = None):
    """The SAE for ``architecture``, resized to exactly ``d_sae`` if given.

    ``d_sae_factor`` is the construction width (``ceil(d_sae / d_in)`` when
    ``d_sae`` is set). ``fista_lam`` is the FISTA λ (the ConSDL sparsity
    knob), ``mp_default_iters`` MP's training round count when
    ``--mp-iters`` is 0, and ``mp_cap_d_sae`` the width MP's sparsity cap
    is resolved against. ``extra[architecture]`` adds caller-specific
    keyword arguments.
    """
    import fastsae  # noqa: PLC0415

    kw: dict[str, Any] = dict(d_in=d_in, d_sae_factor=d_sae_factor,
                              architecture=architecture, device=device)
    if getattr(args, "seed", None) is not None:
        kw["seed"] = args.seed
    kw.update((extra or {}).get(architecture, {}))

    if architecture == "relu":
        kw.setdefault("activation_name", "relu")
        sae = fastsae.models.VanillaSAE(**kw)
    elif architecture == "topk":
        from consdl.models.topk import TopKSAE  # noqa: PLC0415
        sae = TopKSAE(top_k=args.k_per_token, **kw)
    elif architecture in ("batchtopk", "matryoshka"):
        # ``k`` is the per-token budget in both; BatchTopK spends it as a
        # batch-wide pool, so the same value gives a matched MEAN L0.
        from consdl.models.batchtopk import (  # noqa: PLC0415
            BatchTopKSAE,
            MatryoshkaBatchTopKSAE,
        )
        cls = BatchTopKSAE if architecture == "batchtopk" else MatryoshkaBatchTopKSAE
        sae = cls(top_k=args.k_per_token, **kw)
    elif architecture in ("isae", "isae_me"):
        from consdl.models.isae import IsaeMESAE, IsaeSAE  # noqa: PLC0415
        kw.update(
            top_k=args.k_per_token,
            use_abstopk=bool(args.use_abstopk),
            rip_weight=float(args.rip_weight),
            rip_multiplier=int(args.rip_multiplier),
            rip_weighted=bool(args.rip_weighted),
            l1_weight=float(args.isae_l1),
            initial_step_size=float(args.initial_step_size),
            input_norm=bool(getattr(args, "isae_input_norm", False)),
        )
        if architecture == "isae_me":
            # The ME step-size factor is a dense (d_sae, d_sae) parameter:
            # keep T from the CLI but never below the reference's 5.
            steps = int(args.num_encode_steps)
            sae = IsaeMESAE(num_encode_steps=steps if steps > 1 else 5, **kw)
        else:
            sae = IsaeSAE(num_encode_steps=1, **kw)
    elif architecture == "fista":
        # ConSDL and its ablations: FISTA sparse coding (+ refitting); the
        # co-active similarity term is attached by ``build_trainer``.
        from consdl.models.fista import FistaSAE  # noqa: PLC0415
        kw.update(
            fista_t_iters=int(args.fista_iters),
            fista_step=float(args.fista_step),
            fista_lam=float(fista_lam),
            fista_nonneg=bool(args.fista_nonneg),
            refit=bool(args.refit),
            refit_iters=int(args.refit_iters),
            fista_topk_k=0,
        )
        for name in _FISTA_OPTIONAL:
            if hasattr(args, name):
                kw[name] = type(getattr(args, name))(getattr(args, name))
        sae = FistaSAE(**kw)
    elif architecture == "mp":
        # Greedy matching-pursuit encoder + learned decoder. ``max_iters``
        # (adaptive-mode cap) is resolved against the final width.
        from consdl.models.mp import MPSAE, resolve_max_iters  # noqa: PLC0415
        max_iters = resolve_max_iters(d_sae=int(mp_cap_d_sae),
                                      explicit=int(args.mp_max_iters),
                                      sparsity_cap=float(args.mp_sparsity_cap))
        kw.update(
            k_iters=int(args.mp_iters) if int(args.mp_iters) > 0 else int(mp_default_iters),
            mp_nonneg=bool(args.mp_nonneg),
            mp_dropout=float(args.mp_dropout),
            inference_mode=str(args.mp_inference_mode),
            inference_iters=int(args.mp_inference_iters),
            residual_thr=float(args.mp_residual_thr),
            max_iters=max_iters if max_iters is not None else 0,
            patience=int(args.mp_patience),
        )
        sae = MPSAE(**kw)
    else:
        raise SystemExit(f"unknown --architecture {architecture!r}; "
                         f"expected one of {', '.join(ARCHITECTURES)}")

    if d_sae is not None and int(d_sae) != int(sae.d_sae):
        # Constructed one factor wide, shrunk to the exact width; re-seeded
        # so the init is fully determined by --seed.
        from consdl.models._resize import resize_sae_in_place  # noqa: PLC0415
        resize_sae_in_place(sae, int(d_sae), seed=getattr(args, "seed", None))
        assert int(sae.d_sae) == int(d_sae)
    return sae


def co_active_sim_kwargs(args) -> dict:
    """``SAETrainer.set_co_active_sim`` arguments from the CLI flags."""
    return dict(
        coefficient=float(args.co_active_sim_coefficient),
        r=int(args.co_active_sim_r),
        refresh_steps=int(args.co_active_sim_refresh_steps),
        barrier_delta=float(args.co_active_sim_barrier_delta),
        barrier_margin=float(args.co_active_sim_barrier_margin),
        activation_eps=float(args.co_active_sim_activation_eps),
        cache_block_size=int(args.co_active_sim_cache_block_size),
    )


def build_trainer(args, architecture: str, *, wrapper, train_kwargs: dict,
                  aux_off: bool, dead_threshold_total: int,
                  isae_dead_threshold: int | None = None,
                  resampling: dict | None = None):
    """The trainer for ``architecture``.

    ``aux_off`` disables the auxiliary dead-latent term; the top-k family's
    dead threshold is resolved against ``dead_threshold_total`` (the
    budget in the unit the trainer counts, tokens). ``isae_dead_threshold``
    overrides iSAE's threshold. ``resampling`` (LM ReLU only) are the
    ``ResamplingSAETrainer`` arguments. ReLU and FISTA get co-active
    similarity regularization (a no-op at γ = 0).
    """
    if float(args.co_active_sim_coefficient) > 0.0 and architecture not in ("relu", "fista"):
        raise SystemExit("--co-active-sim-coefficient applies to "
                         "--architecture relu / fista only")
    train_kwargs = dict(train_kwargs)

    if architecture in ("topk", "batchtopk", "matryoshka", "isae", "isae_me"):
        from consdl.models.topk import resolve_dead_feature_threshold  # noqa: PLC0415
        train_kwargs.pop("use_ghost_grads", None)  # the subclass forces it off
        aux = dict(
            aux_loss_coefficient=0.0 if aux_off else args.topk_aux_coefficient,
            top_k_aux=(int(args.topk_aux_k_aux) or None),
        )
        if architecture in ("isae", "isae_me"):
            from consdl.models.isae import IsaeAuxRipTrainer  # noqa: PLC0415
            thr = (isae_dead_threshold if isae_dead_threshold is not None
                   else resolve_dead_feature_threshold(
                       args.topk_aux_dead_threshold, total_samples=dead_threshold_total))
            return IsaeAuxRipTrainer(wrapped=wrapper, dead_feature_threshold=thr,
                                     **aux, **train_kwargs)
        from consdl.models.topk import TopKAuxLossTrainer  # noqa: PLC0415
        cls = TopKAuxLossTrainer
        if architecture in ("batchtopk", "matryoshka"):
            from consdl.models.batchtopk import (  # noqa: PLC0415
                BatchTopKAuxLossTrainer,
                MatryoshkaAuxLossTrainer,
            )
            cls = (BatchTopKAuxLossTrainer if architecture == "batchtopk"
                   else MatryoshkaAuxLossTrainer)
            if architecture == "matryoshka":
                print(f"  matryoshka      : groups={wrapper.sae.group_sizes}")
        return cls(wrapped=wrapper,
                   dead_feature_threshold=resolve_dead_feature_threshold(
                       args.topk_aux_dead_threshold, total_samples=dead_threshold_total),
                   **aux, **train_kwargs)

    if architecture == "mp":
        from consdl.models.mp import MPRevivalLossTrainer  # noqa: PLC0415
        train_kwargs.pop("use_ghost_grads", None)
        return MPRevivalLossTrainer(
            wrapped=wrapper, revive_coef=0.0 if aux_off else float(args.mp_revive_coef),
            **train_kwargs)

    if resampling is not None:
        from consdl.models.resampling import ResamplingSAETrainer  # noqa: PLC0415
        trainer = ResamplingSAETrainer(wrapped=wrapper, **resampling, **train_kwargs)
    else:
        from consdl.training.trainer import SAETrainer  # noqa: PLC0415
        trainer = SAETrainer(wrapped=wrapper, **train_kwargs)
    return trainer.set_co_active_sim(**co_active_sim_kwargs(args))
