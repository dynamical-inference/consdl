"""Per-seed reconstruction and sparsity scalars, shared by the vision, LM
and synthetic stats passes.

``ScalarAccumulator``   used by a stats pass. Streaming float64 running
                        sums rather than a mean of per-batch means, so a
                        short last batch is weighted correctly.
``canonical_scalars``   used by readers. Normalises a stats file into the
                        schema below; vision/synthetic files store L0 as a
                        fraction of ``d_sae``, LM files as a count.

Canonical schema:

    l0_abs          mean active latents per token, an absolute COUNT
    l0_frac         the same as a fraction of d_sae
    l1_per_token    mean sum |a| per token
    r2_score        1 - SSE/SST, SST about the MEAN of z, so predicting
                    the mean scores 0 rather than something flattering
    l2_loss         SSE / n_elements
    dead_rate       fraction of latents never firing
    n_positions     how many token positions the estimate is over

Everything else the source file carried is passed through untouched.

MP has no single L0 determined by its weights: its sparsity is the number
of inference rounds. Callers handle it explicitly.
"""
from __future__ import annotations


class ScalarAccumulator:
    """Streaming reconstruction + sparsity scalars over ``(z, a, z_hat)``.

    ``update_recon`` may be called with ``z_hat=None``: a pass that does
    not decode still gets the sparsity scalars, and ``r2_score`` is
    omitted rather than guessed.
    """

    def __init__(self, d_sae: int, thresholds=()):
        self.d_sae = int(d_sae)
        self.thresholds = tuple(float(t) for t in thresholds)
        self.sse = 0.0
        self.z_sum = 0.0
        self.z_sqsum = 0.0
        self.z_elems = 0
        self.have_recon = True
        self.abs_sum = 0.0
        self.n_positions = 0
        self.n_nonzero = 0
        self.thr_active = {t: 0 for t in self.thresholds}

    def update_recon(self, z, z_hat) -> None:
        """Accumulate SSE and the moments of ``z``. No-op once failed."""
        if not self.have_recon:
            return
        if z_hat is None:
            self.have_recon = False
            return
        import torch  # noqa: PLC0415

        with torch.no_grad():
            d = (z - z_hat).to(torch.float64)
            zd = z.to(torch.float64)
            self.sse += float((d * d).sum().item())
            self.z_sum += float(zd.sum().item())
            self.z_sqsum += float((zd * zd).sum().item())
            self.z_elems += int(zd.numel())

    def update_sparsity(self, a_abs, n_positions: int, eps: float = 0.0) -> None:
        """``a_abs`` is ``|a|`` as ``(n_positions, d_sae)``."""
        import torch  # noqa: PLC0415

        with torch.no_grad():
            self.abs_sum += float(a_abs.sum().item())
            self.n_positions += int(n_positions)
            self.n_nonzero += int((a_abs > eps).sum().item())
            for t in self.thresholds:
                self.thr_active[t] += int((a_abs > t).sum().item())

    def finalize(self, firing_rate=None, dead_threshold: float = 1e-6) -> dict:
        """The canonical dict. ``firing_rate`` is a ``(d_sae,)`` vector."""
        n = max(self.n_positions, 1)
        out: dict = {
            "n_positions": int(self.n_positions),
            "l0_abs": self.n_nonzero / n,
            "l0_frac": (self.n_nonzero / n) / max(self.d_sae, 1),
            "l1_per_token": self.abs_sum / n,
        }
        for t in self.thresholds:
            out[f"l0_per_token_{t}"] = self.thr_active[t] / n
        if self.have_recon and self.z_elems:
            sst = max(self.z_sqsum - self.z_sum * self.z_sum / self.z_elems,
                      1e-30)
            out["r2_score"] = 1.0 - self.sse / sst
            out["l2_loss"] = self.sse / self.z_elems
        if firing_rate is not None:
            import numpy as _np  # noqa: PLC0415

            fr = _np.asarray(firing_rate, dtype=_np.float64).reshape(-1)
            out["dead_rate"] = float((fr <= dead_threshold).mean())
        return out


def canonical_scalars(raw: dict, d_sae: int | None = None) -> dict:
    """Normalise a stats file of EITHER layout into the schema above.

    The only ambiguous field is the sparsity: vision and synthetic store
    ``l0_per_token_0`` as a fraction of ``d_sae``, LM as an absolute count.
    A value below 1 is read as a fraction.
    """
    if not isinstance(raw, dict) or not raw:
        return {}
    out = dict(raw)
    v = raw.get("l0_abs", raw.get("l0_per_token_0"))
    if v is not None:
        v = float(v)
        if "l0_abs" not in out:
            out["l0_abs"] = v * float(d_sae) if (v < 1.0 and d_sae) else v
        if "l0_frac" not in out and d_sae:
            out["l0_frac"] = out["l0_abs"] / float(d_sae)
    return out
