"""``FistaSAE``: FISTA sparse-coding encoder with a trained decoder.

This is the model behind ConSDL: online dictionary learning with a FISTA
inference step.

  1. Encode: cold-start FISTA solves, per input ``x``, the LASSO
     ::
         a* = argmin_a  ½‖(x − b_dec) − a·W_dec‖²₂ + λ‖a‖₁     (a ≥ 0)

     under ``torch.no_grad``, so ``a*`` is a detached constant.
     Optionally the coefficients are then refit on the selected support
     (``refit=True``) to remove the L1 shrinkage bias.
  2. Decode: ``x̂ = a*·W_dec + b_dec`` keeps ``W_dec`` / ``b_dec`` in the
     autograd graph, so the reconstruction loss trains only the decoder
     (alternating dictionary update with the code held fixed).

Sparsity comes from FISTA's L1 prox, so :meth:`aux_losses` is empty.
Co-active similarity regularization is applied by the trainer
(:mod:`consdl.training.co_active_sim`).

FISTA iterate (Nesterov-accelerated proximal gradient), with
``r = x − b_dec`` and ``D = W_decᵀ``::

    y_k     = a_k + ((t_k − 1)/t_{k+1})·(a_k − a_{k−1})
    a_{k+1} = prox_{η·λ}( y_k − η·(D·Dᵀ·y_k − Dᵀ·r) )

``prox`` is ``ReLU(z − η·λ)`` when ``fista_nonneg``, else the signed
soft-threshold.

``fista_step > 0`` fixes η; ``0`` (default) sets ``η = 1/σ_max(D)²``
every forward via power iteration, since the decoder changes every step.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch
from config_dataclass import config_field, torch_dataclass
from fastsae.models.sae import VanillaSAE
from torch import Tensor


def _fused_momentum(h: Tensor, h_prev: Tensor, mom: Tensor) -> Tensor:
    """``q = h + mom*(h - h_prev)`` — 2 reads, 1 write when fused."""
    return h + mom * (h - h_prev)


def _fused_grad_prox(q: Tensor, g: Tensor, b_drive: Tensor,
                     eta: Tensor, thr: Tensor) -> Tensor:
    """``clamp(q - eta*g + b_drive - thr, 0)`` — 3 reads, 1 write."""
    return torch.clamp(q - eta * g + b_drive - thr, min=0.0)


#: Compiled ``(momentum, grad+prox)`` pair, or ``None`` where Dynamo
#: cannot run. Built once, on first use, and never rebuilt.
_FUSED: "tuple | None | bool" = False        # False = not tried yet


def _fused_kernels():
    """The two fused elementwise kernels, or ``None``.

    The FISTA loop's elementwise ops are memory-bandwidth-bound; fusing
    them cuts the passes over the coefficient tensor roughly from 19 to 7.
    The result is numerically the same as the eager path, only faster.

    Returns ``None`` when ``torch.compile`` is unavailable (e.g. Python
    3.12 with torch < 2.4); the eager body is used instead.

    The kernels take only coefficient tensors and 0-dim tensor scalars,
    never the decoder, so changing weights do not trigger recompilation.
    Scalars are tensors rather than floats because Dynamo specializes on
    float values and ``mom`` changes every iteration.
    """
    global _FUSED
    if _FUSED is not False:
        return _FUSED
    # ``CONSDL_NO_FUSION=1`` forces the eager body, e.g. for FLOP counting
    # (``FlopCounterMode`` does not work inside a compiled region).
    if os.environ.get("CONSDL_NO_FUSION", "").strip() not in ("", "0"):
        _FUSED = None
        return _FUSED
    try:
        _FUSED = (torch.compile(_fused_momentum, dynamic=False),
                  torch.compile(_fused_grad_prox, dynamic=False))
    except Exception:                                   # noqa: BLE001
        _FUSED = None
    return _FUSED


@torch_dataclass
class FistaSAE(VanillaSAE):
    """SAE with a FISTA sparse-coding encoder and a trained decoder.

    Subclasses :class:`VanillaSAE` to inherit its parameters, save/load
    and unit-norm decoder projection. ``W_enc`` / ``b_enc`` are unused
    (the encoder is cold-start FISTA) but kept in the state dict.

    Attributes
    ----------
    fista_t_iters : int
        FISTA inner-iteration count.
    fista_step : float
        Proximal-gradient step η.  ``> 0`` pins a fixed value; ``0``
        (default) auto-resolves ``1/σ_max(D)²`` per forward.
    fista_lam : float
        L1 penalty coefficient λ.
    fista_nonneg : bool
        Constrain ``a ≥ 0`` (``prox = ReLU(z − η·λ)``).  Default True.
    fista_power_iters : int
        Power-iteration count for the auto step-size estimate. Decoders
        with a small spectral gap need many iterations; 32 keeps the
        relative η error under ~1.5%.
    fista_r2_tol : float
        Early-stop tolerance on the per-batch Input R² improvement.
        Iteration breaks early once
        ``ΔR² = (SSE_prev − SSE_curr) / SST`` drops below this
        threshold. ``0`` (or negative) disables the check, making
        ``fista_t_iters`` a fixed iteration count.
    fista_check_every : int
        How often (in FISTA iterations) to check the early-stop
        criterion. Each check costs one extra ``h @ W_dec`` matmul.
    fista_use_bf16 : bool
        Unused; kept so older config files load.
    fista_step_cache_every : int
        ``0`` (default) re-estimates σ_max(W_dec) from a random vector
        every forward. ``> 0`` caches the dominant singular vector and
        refreshes σ_max only every that many forwards, with a 4-step
        warm-started power iteration.

        Caution: power iteration approaches σ_max from below, so
        ``η = 1/σ̂²`` overshoots ``1/σ_max²``, and a stale cache can be
        further off once ``W_dec`` has moved. Only the spectral-norm
        vector is warm-started; every FISTA solve still starts at 0.
    fista_compile : bool
        Unused; kept so older config files load.
    """

    architecture: str = config_field(default="fista")
    fista_t_iters: int = config_field(default=50)
    fista_step: float = config_field(default=0.0)   # 0 ⇒ auto per-forward
    fista_lam: float = config_field(default=1e-1)
    fista_nonneg: bool = config_field(default=True)
    fista_power_iters: int = config_field(default=32)
    fista_r2_tol: float = config_field(default=1e-4)
    fista_check_every: int = config_field(default=5)
    # Unused; kept so older config files load.
    fista_use_bf16: bool = config_field(default=False)
    fista_step_cache_every: int = config_field(default=0)
    # Unused; kept so older config files load.
    fista_compile: bool = config_field(default=False)
    # Unused; kept so older config files load.
    fista_topk_k: int = config_field(default=0)
    # Coefficient refitting: after the FISTA solve, refit the coefficients
    # on the selected support by least squares to remove the L1 shrinkage
    # bias (see ``_refit``).
    refit: bool = config_field(default=False)
    refit_iters: int = config_field(default=30)
    refit_thresh: float = config_field(default=1e-8)

    def __lazy_post_init__(self):
        super().__lazy_post_init__()
        if int(self.fista_t_iters) <= 0:
            raise ValueError(
                f"fista_t_iters must be > 0; got {self.fista_t_iters!r}"
            )
        if float(self.fista_step) < 0:
            raise ValueError(
                f"fista_step must be >= 0 (0 = auto); got {self.fista_step!r}"
            )
        if float(self.fista_lam) < 0:
            raise ValueError(
                f"fista_lam must be >= 0; got {self.fista_lam!r}"
            )
        if int(self.fista_power_iters) <= 0:
            raise ValueError(
                f"fista_power_iters must be > 0; got {self.fista_power_iters!r}"
            )
        if int(self.fista_check_every) <= 0:
            raise ValueError(
                f"fista_check_every must be > 0; got {self.fista_check_every!r}"
            )

    # ── step size ─────────────────────────────────────────────────────

    @torch.no_grad()
    def _resolve_step(self) -> float:
        """FISTA step η.

        ``fista_step > 0`` ⇒ that fixed value.  Otherwise auto:
        ``η = 1/σ_max(D)²`` (the convergence bound η ≤ 1/L with
        ``L = σ_max(D)²``).  σ_max is estimated by power iteration on
        ``W_dec``.

        With ``fista_step_cache_every > 0`` the estimate is cached; see
        the class docstring for why that path is off by default.
        """
        eta = float(self.fista_step)
        if eta > 0.0:
            return eta
        W = self.W_dec                                  # (d_sae, d_in)
        cache_every = int(self.fista_step_cache_every)

        if cache_every <= 0:
            # Cold-start from a random vector every call.
            v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
            v = v / (v.norm() + 1e-12)
            for _ in range(int(self.fista_power_iters)):
                u = W @ v
                u = u / (u.norm() + 1e-12)
                v = W.T @ u
                v = v / (v.norm() + 1e-12)
            sigma_max = float((W @ v).norm())
            return 1.0 / max(sigma_max ** 2, 1e-30)

        # Cached path. State is created lazily so the state dict is
        # unchanged when caching is off.
        if not hasattr(self, "_eta_dom_v") or self._eta_dom_v is None:
            v = torch.randn(W.shape[1], device=W.device, dtype=W.dtype)
            v = v / (v.norm() + 1e-12)
            self._eta_dom_v = v
            self._eta_cached_sigma_max = None
            self._eta_call_counter = 0
        # Reuse cached σ_max for ``cache_every - 1`` consecutive calls.
        if (
            self._eta_cached_sigma_max is not None
            and (self._eta_call_counter % cache_every) != 0
        ):
            self._eta_call_counter += 1
            return 1.0 / max(self._eta_cached_sigma_max ** 2, 1e-30)
        # Refresh, warm-starting from the cached ``v``; W_dec changes
        # slowly, so a few iterations suffice. The first refresh uses the
        # full ``fista_power_iters``.
        v = self._eta_dom_v.to(device=W.device, dtype=W.dtype)
        is_cold = (
            self._eta_cached_sigma_max is None
            or self._eta_call_counter == 0
        )
        n_iters = int(self.fista_power_iters) if is_cold else 4
        for _ in range(n_iters):
            u = W @ v
            u = u / (u.norm() + 1e-12)
            v = W.T @ u
            v = v / (v.norm() + 1e-12)
        self._eta_dom_v = v
        sigma_max = float((W @ v).norm())
        self._eta_cached_sigma_max = sigma_max
        self._eta_call_counter += 1
        return 1.0 / max(sigma_max ** 2, 1e-30)

    # ── post-prox hook ────────────────────────────────────────────────

    def _project_after_prox(self, h_new: Tensor) -> Tensor:
        """Optional projection applied to ``h_new`` after the proximal
        soft-threshold inside each FISTA iter.

        Identity here (plain λ-FISTA). Subclasses can override it to add
        a projection without copying the loop body.
        """
        return h_new

    # ── FISTA core ────────────────────────────────────────────────────

    def _fista_iterate(
        self,
        r: Tensor,
        W: Tensor,
        *,
        eta: float,
        lam: float,
        max_iters: int,
        nonneg: bool,
        r2_tol: float = 0.0,
        check_every: int = 1,
    ) -> Tensor:
        """One cold-start (``a₀ = 0``) Nesterov-accelerated proximal
        (FISTA) solve of ``argmin_a 0.5·||r − a·W||² + lam·|a|₁`` (with
        an optional non-negativity prox), for an arbitrary decoder-row
        matrix ``W`` (``(n, d_in)``) and target residual ``r``
        (``(..., d_in)``). Returns the code ``(..., n)``.

        ``self._project_after_prox`` is applied every iteration.
        ``r2_tol > 0`` early-stops once the per-batch Input-R² improvement
        drops below the tolerance (checked every ``check_every``
        iterations); ``r2_tol = 0`` runs exactly ``max_iters``. The caller
        must be under ``torch.no_grad``.
        """
        thresh = eta * float(lam)
        # Driver term η·Dᵀ·r with D = Wᵀ ⇒ r @ W.T, precomputed once.
        b_drive = eta * (r @ W.T)                       # (..., n)
        n = W.shape[0]
        h = torch.zeros(*r.shape[:-1], n, device=r.device, dtype=r.dtype)
        h_prev = h
        t = 1.0

        early_stop_enabled = float(r2_tol) > 0.0
        if early_stop_enabled:
            r_mean = r.mean(dim=-2, keepdim=True)
            sst = ((r - r_mean) ** 2).sum()
            prev_sse = None

        # Number of iterations actually run (``max_iters`` is only a cap
        # when early stopping is on).
        self._fista_iters_run = 0
        # Fused kernels are used only for the non-negative prox; the signed
        # soft-threshold uses the eager body.
        fused = _fused_kernels() if nonneg else None
        if fused is not None:
            # Build the 0-dim scalar tensors once: each ``as_tensor`` is a
            # host-to-device copy. ``mom`` changes per iteration and is
            # written into a reused buffer with ``fill_``.
            _eta_t = torch.as_tensor(eta, device=h.device, dtype=h.dtype)
            _thr_t = torch.as_tensor(thresh, device=h.device, dtype=h.dtype)
            _mom_t = torch.zeros((), device=h.device, dtype=h.dtype)

        for it in range(int(max_iters)):
            self._fista_iters_run = it + 1
            # ``t`` must advance every iteration, including the first.
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            if it == 0:
                # With h = h_prev = 0, q = 0 exactly, so the gradient step
                # reduces to ``b_drive``; skip the two matmuls.
                grad_step = b_drive
            elif fused is not None:
                # Fused: momentum in one kernel, grad+prox in another.
                _mom_t.fill_((t - 1.0) / t_next)
                q = fused[0](h, h_prev, _mom_t)
                h_new = fused[1](q, (q @ W) @ W.T, b_drive, _eta_t, _thr_t)
                grad_step = None                        # nothing to prox
            else:
                mom = (t - 1.0) / t_next
                q = h + mom * (h - h_prev)
                # ``q @ W`` is kept dense: momentum makes ``q`` far less
                # sparse than the final code, and sparse kernels are much
                # slower than a dense GEMM at these densities.
                q_dt = q @ W                            # (..., d_in)
                grad_step = q - eta * (q_dt @ W.T) + b_drive
            if grad_step is None:                       # fused path
                pass                                    # h_new already set
            elif nonneg:
                h_new = torch.clamp(grad_step - thresh, min=0.0)
            else:
                h_new = torch.sign(grad_step) * torch.clamp(
                    grad_step.abs() - thresh, min=0.0,
                )
            h_new = self._project_after_prox(h_new)
            h_prev = h
            h = h_new
            t = t_next

            # Skip the check on the last iteration, where it cannot
            # change the result.
            if (early_stop_enabled and ((it + 1) % check_every == 0)
                    and it + 1 < int(max_iters)):
                y_hat = h @ W                           # (..., d_in)
                cur_sse = ((r - y_hat) ** 2).sum()
                if prev_sse is not None:
                    delta_r2 = (prev_sse - cur_sse) / torch.clamp(sst, min=1e-30)
                    if float(delta_r2) < r2_tol:
                        break
                prev_sse = cur_sse
        return h

    # ── encode (cold-start FISTA, detached) ───────────────────────────

    def encode(self, x: Tensor) -> Tensor:
        """Cold-start FISTA sparse code for ``x``.

        Runs under ``torch.no_grad``: the code is a detached constant, so
        the reconstruction loss trains the decoder only through
        :meth:`decode`. When ``refit`` is set, the coefficients are refit
        on the FISTA support (:meth:`_refit`).

        When ``fista_r2_tol > 0`` the loop stops once
        ``(SSE_prev − SSE_curr) / SST < fista_r2_tol``, so
        ``fista_t_iters`` is a maximum; ``fista_r2_tol = 0`` runs exactly
        ``fista_t_iters`` iterations.
        """
        with torch.no_grad():
            eta = self._resolve_step()
            r = x - self.b_dec                          # (..., d_in)
            z = self._fista_iterate(
                r, self.W_dec,
                eta=eta,
                lam=float(self.fista_lam),
                max_iters=int(self.fista_t_iters),
                nonneg=bool(self.fista_nonneg),
                r2_tol=float(self.fista_r2_tol),
                check_every=int(self.fista_check_every),
            )
            if bool(self.refit):
                z = self._refit(z, r, eta=eta)
            return z

    @torch.no_grad()
    def _refit(self, z: Tensor, r: Tensor, *, eta: float) -> Tensor:
        """Coefficient refitting on the support selected by FISTA.

        FISTA's L1 penalty shrinks every active coefficient towards zero.
        Refitting solves the unpenalized least-squares problem
        ``min ½||r − Dz||²`` restricted to the FISTA support S (latents
        outside S stay zero), which removes that bias.

        The solve is projected gradient descent from the FISTA output:
        ``z ← z − η Dᵀ(Dz − r)``, then re-apply the support mask (and the
        non-negativity clamp when ``fista_nonneg``). This is fully batched,
        reuses the FISTA step size ``η = 1/σ_max²``, and avoids a per-sample
        ``(D_Sᵀ D_S)⁻¹``. ``refit_iters`` sets the number of steps.

        The matmuls are dense over all of ``d_sae`` on purpose: gathering
        per-sample support rows of ``W`` is bandwidth-bound and slower in
        practice despite far fewer FLOPs.
        """
        thresh = float(self.refit_thresh)
        # Support mask from the FISTA output, as float for cheap masking.
        mask = (z > thresh).to(z.dtype)
        z_p = z * mask
        W = self.W_dec                                 # (d_sae, d_in)
        n_iters = int(self.refit_iters)
        nonneg = bool(self.fista_nonneg)
        for _ in range(n_iters):
            recon = z_p @ W                            # (..., d_in)
            e = recon - r                              # (..., d_in)
            grad = e @ W.T                             # (..., d_sae)
            z_p = z_p - eta * grad
            z_p = z_p * mask
            if nonneg:
                z_p = z_p.clamp(min=0.0)
        return z_p

    @torch.no_grad()
    def encode_with_diagnostics(
        self, x: Tensor, n_tail: int = 3,
    ) -> Tuple[Tensor, Dict[str, Any]]:
        """Sibling of :meth:`encode` that returns per-iteration FISTA
        diagnostics in addition to the code.

        For debugging; not used in training or inference. The arithmetic
        mirrors the FISTA loop of :meth:`encode` (without refitting).
        Reports per iteration:

          * ``iter``               — iteration index (0-based)
          * ``thresh``             — soft-threshold ``η · λ``
          * ``grad_step_median``   — ``median |grad_step|``; the typical
            magnitude of the value being thresholded
          * ``ratio``              — ``thresh / grad_step_median``;
            an atom survives roughly when ``grad_step[i] > thresh``,
            so the ratio governs survival rate
          * ``l0_per_token``       — final L0 of ``h`` at this iter
          * ``l1_per_token``       — final L1 of ``h`` at this iter

        Plus batch-level scalars:

          * ``eta``                — resolved FISTA step
          * ``r_norm_mean``        — mean ``‖x − b_dec‖₂`` (input scale)

        ``n_tail`` keeps only the last N iterations' stats. Early stopping
        is not applied, so all ``fista_t_iters`` iterations run.
        """
        eta = self._resolve_step()
        r = x - self.b_dec                              # (..., d_in)
        b_drive = eta * (r @ self.W_dec.T)              # (..., d_sae)
        thresh = eta * float(self.fista_lam)
        r_norm_mean = float(r.norm(dim=-1).mean())

        h = torch.zeros(
            *x.shape[:-1], self.d_sae, device=x.device, dtype=x.dtype,
        )
        h_prev = h
        t = 1.0
        max_iters = int(self.fista_t_iters)
        tail: list[dict] = []

        for it in range(max_iters):
            t_next = 0.5 * (1.0 + (1.0 + 4.0 * t * t) ** 0.5)
            mom = (t - 1.0) / t_next
            q = h + mom * (h - h_prev)
            q_dt = q @ self.W_dec
            grad_step = q - eta * (q_dt @ self.W_dec.T) + b_drive
            if self.fista_nonneg:
                h_new = torch.clamp(grad_step - thresh, min=0.0)
            else:
                h_new = torch.sign(grad_step) * torch.clamp(
                    grad_step.abs() - thresh, min=0.0,
                )
            h_new = self._project_after_prox(h_new)
            h_prev = h
            h = h_new
            t = t_next

            gs_med = float(grad_step.abs().median())
            l0 = float((h > 0).float().sum(dim=-1).mean())
            l1 = float(h.abs().sum(dim=-1).mean())
            tail.append(dict(
                iter=int(it),
                thresh=float(thresh),
                grad_step_median=gs_med,
                ratio=float(thresh) / max(gs_med, 1e-30),
                l0_per_token=l0,
                l1_per_token=l1,
            ))
            if len(tail) > int(n_tail):
                tail.pop(0)

        return h, dict(
            eta=float(eta),
            thresh=float(thresh),
            fista_lam=float(self.fista_lam),
            r_norm_mean=r_norm_mean,
            n_iters_run=int(max_iters),
            tail=tail,
        )

    def forward(
        self,
        x: Tensor,
        steer_cfg: Optional[Dict[str, Any]] = None,
        return_act: bool = True,
        return_pre_act: bool = False,
    ) -> Tuple[Tensor, Tensor] | Tensor:
        a = self.encode(x)                              # detached code
        if steer_cfg is not None:
            a = self.steer(a, steer_cfg)
        # Decode in-graph so the reconstruction loss trains the decoder.
        z_hat = self.decode(a)
        if return_pre_act:
            # FISTA has no separate pre-activation; return ``a`` twice.
            return (z_hat, a, a)
        return (z_hat, a) if return_act else z_hat

    def aux_losses(self, a: Tensor) -> Dict[str, Tensor]:
        # Sparsity comes from FISTA's L1 prox; no extra penalty term.
        return {}

