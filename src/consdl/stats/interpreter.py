"""fastsae ``SAEInterpreter`` subclass that computes per-token firing
statistics during the stats pass.

A forward hook on ``wrapped.sae`` accumulates, in the same pass fastsae
already runs, the per-token L0/L1 scalars (``l0_per_token_<thr>``,
``l1_per_token``), the per-latent firing rate over all token positions
(``firing_rate_per_latent``, used for #dead), and a pooled reconstruction
R^2. fastsae's per-latent ``l0_<thr>`` scalars are removed; the per-token
form is the sparsity measure used throughout.

``l0_per_token_0`` counts ``|a| > ZERO_EPS`` (default ``1e-6``) rather
than ``a != 0`` so rounding noise does not count as active.
"""

from __future__ import annotations

from typing import Iterable

import torch
from fastsae.interpret.sae_interpreter import SAEInterpreter as _FastSAEInterpreter


ZERO_EPS = 1e-6

# Firing-rate gates for the summary scalars; identical to the defaults of
# ``consdl.metrics.quality.quality_scalars``.
DEAD_THRESHOLD = 1e-6
DENSE_THRESHOLD = 0.2
LIVE_THRESHOLD = 0.1


class SAEInterpreter(_FastSAEInterpreter):
    """``SAEInterpreter`` that writes the per-token L0/L1 family.

    Parameters
    ----------
    l0_thresholds : iterable of float, optional
        Thresholds for ``l0_per_token_<thr>``. ``l0_per_token_0`` (counted
        with ``|a| > zero_eps``) is always written.
    zero_eps : float, default ``ZERO_EPS`` (1e-6)
        Treat activations with ``|a| <= zero_eps`` as zero.

    Any ``sparsity_thr`` keyword is passed through to fastsae; the
    resulting ``l0_<thr>`` scalars are removed by ``compute_stats``.
    """

    def __init__(
        self,
        *args,
        l0_thresholds: Iterable[float] | None = None,
        zero_eps: float = ZERO_EPS,
        **kwargs,
    ):
        self._zero_eps: float = float(zero_eps)
        # Callers may pass ``_progress_file`` for progress tracking;
        # fastsae's config dataclass rejects unknown kwargs, so pop it.
        self._progress_file = kwargs.pop("_progress_file", None)
        super().__init__(*args, **kwargs)
        self._l0_per_token_thr: list[float] = sorted(
            set(float(t) for t in (l0_thresholds or []))
        )

    def compute_stats(self, *args, **kwargs):  # type: ignore[override]
        counts_thr: dict[float, int] = {t: 0 for t in self._l0_per_token_thr}
        counts = {"eps_nonzero": 0, "total": 0}
        eps = self._zero_eps
        # Per-latent accumulators over all (batch, seq) positions, sized on
        # the first hook call: fire counts at ``eps`` and each threshold,
        # sum of |a| and max |a|. ``sparsity_per_token[k] = fires(k) /
        # positions`` is the per-latent firing rate over token positions,
        # unlike fastsae's ``sparsity.pt``, which uses a per-image
        # convention.
        per_lat: dict = {
            "eps": None, "thr": {}, "positions": 0,
            "abs_sum": None, "max_act": None,
        }
        # Histogram of per-token active counts ``(|a| > eps).sum(-1)``:
        # index = active count, entry = number of tokens. Grows on demand.
        ac_hist: dict = {"hist": None, "max_seen": -1, "total": 0}

        # Reconstruction R^2 as defined in the paper: one ratio over the
        # whole pass, with the total sum of squares centred per dimension
        # (sum_n ||x_n - xbar||^2). fastsae's ``r2_score`` instead centres
        # on a scalar mean and averages per-batch ratios, so it is
        # overwritten below. ``l2_loss`` is left as fastsae computes it.
        recon: dict = {"sse": 0.0, "sum": None, "sq": None, "rows": 0}

        def _hook(_module, _inputs, output):
            # SAE forward returns (z_hat, a); we want the sparse activations a.
            if isinstance(output, tuple) and len(output) >= 2:
                a = output[1]
            else:
                a = output
            if isinstance(output, tuple) and _inputs:
                try:
                    _z = _inputs[0]
                    _zh = output[0]
                    if getattr(_z, "shape", None) == getattr(_zh, "shape", None):
                        zf = _z.detach().reshape(-1, _z.shape[-1]).double()
                        hf = _zh.detach().reshape(-1, _zh.shape[-1]).double()
                        df = zf - hf
                        recon["sse"] += float((df * df).sum().item())
                        if recon["sum"] is None:
                            recon["sum"] = torch.zeros(zf.shape[1],
                                                       dtype=torch.float64)
                            recon["sq"] = torch.zeros_like(recon["sum"])
                        recon["sum"].add_(zf.sum(dim=0).cpu())
                        recon["sq"].add_((zf * zf).sum(dim=0).cpu())
                        recon["rows"] += int(zf.shape[0])
                except Exception:  # noqa: BLE001
                    # Without a comparable z_hat, keep fastsae's value.
                    pass
            a_abs = a.detach().abs()
            counts["total"] += a_abs.numel()
            counts["eps_nonzero"] += int((a_abs > eps).sum().item())
            for t in self._l0_per_token_thr:
                counts_thr[t] += int((a_abs >= t).sum().item())

            # Per-latent: flatten every leading dim (batch, seq, ...) so the
            # last axis is d_sae, then sum over all positions.
            flat = a_abs.reshape(-1, a_abs.shape[-1])
            if per_lat["eps"] is None:
                d_sae = flat.shape[1]
                per_lat["eps"] = torch.zeros(d_sae, dtype=torch.int64)
                per_lat["abs_sum"] = torch.zeros(d_sae, dtype=torch.float64)
                per_lat["max_act"] = torch.zeros(d_sae, dtype=torch.float32)
                for t in self._l0_per_token_thr:
                    per_lat["thr"][t] = torch.zeros(d_sae, dtype=torch.int64)
            per_lat["eps"].add_((flat > eps).sum(dim=0).to(torch.int64).cpu())
            per_lat["abs_sum"].add_(flat.sum(dim=0).to(torch.float64).cpu())
            chunk_max = flat.max(dim=0).values.to(torch.float32).cpu()
            torch.maximum(per_lat["max_act"], chunk_max, out=per_lat["max_act"])
            for t in self._l0_per_token_thr:
                per_lat["thr"][t].add_((flat >= t).sum(dim=0).to(torch.int64).cpu())
            per_lat["positions"] += flat.shape[0]

            # Per-token active-count histogram.
            per_tok = (flat > eps).sum(dim=-1).to(torch.int64).cpu()
            if per_tok.numel() > 0:
                local_max = int(per_tok.max().item())
                if local_max > ac_hist["max_seen"]:
                    new_len = local_max + 1
                    new_buf = torch.zeros(new_len, dtype=torch.int64)
                    if ac_hist["hist"] is not None:
                        new_buf[: ac_hist["hist"].numel()] = ac_hist["hist"]
                    ac_hist["hist"] = new_buf
                    ac_hist["max_seen"] = local_max
                ac_hist["hist"].scatter_add_(
                    0, per_tok, torch.ones_like(per_tok),
                )
                ac_hist["total"] += int(per_tok.numel())

        handle = self.wrapped.sae.register_forward_hook(_hook)
        try:
            super().compute_stats(*args, **kwargs)
        finally:
            handle.remove()

        # Pooled, per-dimension-centred R^2 (see above).
        if recon["rows"] > 0 and recon["sum"] is not None:
            _sst = float((recon["sq"]
                          - recon["sum"] * recon["sum"] / recon["rows"]
                          ).sum().item())
            self.stats["r2_score"] = float(
                1.0 - recon["sse"] / max(_sst, 1e-30))

        # Write l0_per_token_<thr>. The _0 slot uses `|a| > eps`.
        if counts["total"] > 0:
            self.stats["l0_per_token_0"] = counts["eps_nonzero"] / counts["total"]
            for t in self._l0_per_token_thr:
                self.stats[f"l0_per_token_{t}"] = counts_thr[t] / counts["total"]

        # Per-latent tensors; fastsae saves each as ``<key>.pt``.
        if per_lat["eps"] is not None and per_lat["positions"] > 0:
            n = float(per_lat["positions"])
            self.stats["sparsity_per_token"] = per_lat["eps"].float() / n
            for t in self._l0_per_token_thr:
                self.stats[f"sparsity_per_token_{t}"] = per_lat["thr"][t].float() / n

            # ``firing_rate_per_latent`` is the same tensor under a clearer
            # name: entry k is the fraction of token positions at which
            # latent k fired. Its mean equals ``l0_per_token_0`` (a fraction
            # of d_sae); times d_sae it is the per-token L0 count.
            self.stats["firing_rate_per_latent"] = self.stats["sparsity_per_token"]
            for t in self._l0_per_token_thr:
                self.stats[f"firing_rate_per_latent_{t}"] = (
                    self.stats[f"sparsity_per_token_{t}"]
                )

            # Firing-rate summary scalars, defined as in
            # ``consdl.metrics.quality.quality_scalars``:
            #   dead_rate        = frac of latents firing on <= 1e-6 of tokens
            #   dense_rate_0.2   = frac of latents firing on >= 20% of tokens
            #   live_latent_frac = frac with max|a| over the pass > 0.1
            _firing = self.stats["sparsity_per_token"]
            self.stats["dead_rate"] = float((_firing <= DEAD_THRESHOLD).float().mean())
            self.stats["dense_rate_0.2"] = float((_firing >= DENSE_THRESHOLD).float().mean())
            self.stats["live_latent_frac"] = float(
                (per_lat["max_act"] > LIVE_THRESHOLD).float().mean()
            )

            # Per-latent E[|a|] over all positions (zeros included).
            self.stats["mean_act_per_token"] = (
                per_lat["abs_sum"] / per_lat["positions"]
            ).float()
            # Per-latent max |a| over all positions.
            self.stats["max_act_per_token"] = per_lat["max_act"].clone()

            # Scalar L1 per token: E_token[sum_k |a_k|].
            self.stats["l1_per_token"] = float(
                per_lat["abs_sum"].sum().item() / per_lat["positions"]
            )

        # Active-count histogram, as raw counts and as proportions.
        if ac_hist["hist"] is not None and ac_hist["total"] > 0:
            hist = ac_hist["hist"].clone()
            self.stats["active_count_hist_per_token"] = hist
            self.stats["active_count_proportion_per_token"] = (
                hist.to(torch.float64) / float(ac_hist["total"])
            ).float()

        # Drop fastsae's per-latent ``l0_<thr>`` scalars, keeping
        # ``l0_per_token_<thr>``.
        for k in [
            key for key in self.stats
            if key.startswith("l0_") and not key.startswith("l0_per_")
        ]:
            del self.stats[k]
