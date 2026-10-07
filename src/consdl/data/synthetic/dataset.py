"""Synthetic sparse dataset: a fixed pool of samples ``z = D_gt @ a``.

Each sample is generated on demand from an RNG seeded by
``(pool_seed, sample_idx)``, so the pool needs no disk storage and is
identical across epochs and workers.

``pool_seed`` is independent of the SAE seed: all SAEs for one
``(decoder, distribution, k_active)`` configuration see the same samples
and differ only in initialisation and shuffle order, so cross-seed
disagreement is attributable to the optimiser, not the data.

Firing-probability distributions (which latents are active; the
magnitudes are set separately by ``magnitude_distribution``):

    uniform  : every latent equally likely.
    lt03     : Zipf exponent 0.3  (mild tail, top-5% holds ~12% mass)
    lt04     : Zipf exponent 0.4
    lt05     : Zipf exponent 0.5  (moderate, ~20%)
    lt07     : Zipf exponent 0.7  (heavy, ~35%)
    lt10     : Zipf exponent 1.0  (classic Zipf, ~59%)
    gauss05  : p_i ~ Normal(k/d_sae, 0.5·k/d_sae), clipped ≥0, Σ=k
    gauss10  : p_i ~ Normal(k/d_sae, 1.0·k/d_sae), clipped ≥0, Σ=k
    sd_r<R>_f<FFF> : dense group (fraction FFF/100) firing R× as often
                     as the sparse rest, Σ=k
"""

from __future__ import annotations

from dataclasses import dataclass

import re

import numpy as np
import torch
from torch.utils.data import Dataset

DISTRIBUTION_EXPONENTS: dict[str, float] = {
    "uniform": 0.0,
    "lt03": 0.3,
    "lt04": 0.4,
    "lt05": 0.5,
    "lt07": 0.7,
    "lt10": 1.0,
}

# Gaussian-prior p_i modes. ``sigma`` here is a *relative* spread: the
# per-latent firing probability is drawn from ``Normal(k/d_sae, σ·k/d_sae)``,
# clipped non-negative, then rescaled so Σ p_i = k (matching the Zipf
# modes). μ is not exposed — any non-default μ would be cancelled by
# the Σ=k rescaling.
GAUSSIAN_SIGMAS: dict[str, float] = {
    "gauss05": 0.5,
    "gauss10": 1.0,
}


#: ``sd_r<R>_f<FFF>`` — sparse/dense split, R = density ratio, FFF =
#: dense-group percentage, zero-padded to three digits so paths sort.
#:
#: The parameters are part of the name so they appear in the run tag and
#: runs with different r / f never share a directory.
_SD_RE = re.compile(r"^sd_r(\d+)_f(\d{1,3})$")


def parse_sd(name: str) -> tuple[int, float] | None:
    """``"sd_r10_f010" -> (10, 0.10)``; ``None`` if not an sd name.

    Raises on a well-formed name with out-of-range numbers (e.g.
    ``sd_r0_f10``) instead of returning ``None``, so a typo cannot fall
    through to another distribution.
    """
    m = _SD_RE.match(name)
    if m is None:
        return None
    r, f_pct = int(m.group(1)), int(m.group(2))
    if r < 1 or not (0 <= f_pct <= 100):
        raise ValueError(f"sd name out of range: {name}")
    return r, f_pct / 100.0


def is_valid_distribution(name: str) -> bool:
    """Whether ``name`` is a known firing-probability distribution.

    Used for argparse validation, since the parameterised sd family cannot
    be listed as fixed ``choices``.
    """
    if name in DISTRIBUTION_EXPONENTS or name in GAUSSIAN_SIGMAS:
        return True
    try:
        return parse_sd(name) is not None
    except ValueError:
        return False


MAGNITUDE_DISTRIBUTIONS = (
    "lognormal", "exponential", "uniform", "constant", "energy_softmax",
)


@dataclass
class SyntheticSpec:
    """Controls how sparse codes are drawn for the synthetic pool.

    Following Elhage et al.:

        x = Σ_i b_i · c_i · f_i,
            b_i ~ Bernoulli(p_i),
            c_i ~ MagnitudeDistribution(μ, σ),

    where the ``p_i`` distribution is set by ``distribution`` and the
    ``c_i`` draw is set by ``magnitude_distribution``:

    - ``p_i`` options:
        ``"uniform"``              — constant p_i = k_active / d_sae.
        ``"lt03/lt04/lt05/lt07/lt10"`` — Zipf-weighted p_i, Σ p_i = k_active.
        ``"gauss05/gauss10"``      — per-latent Normal prior, Σ p_i = k_active.
        ``"sd_r<R>_f<FFF>"``       — sparse/dense split, Σ p_i = k_active.

    - ``magnitude_distribution`` options (all non-negative):
        ``"lognormal"`` — LogNormal(μ, σ).  Default; heavy right tail.
        ``"exponential"`` — Exponential(scale=exp(μ)).  σ ignored.
        ``"uniform"`` — Uniform(max(0, exp(μ) − σ), exp(μ) + σ).
        ``"constant"`` — deterministic ``exp(μ)`` (default μ=0 → value 1).
                         σ ignored.
        ``"energy_softmax"`` — see ``energy_tau`` below.
    """

    distribution: str             # "uniform" | "lt03" | "lt05" | "lt07" | "lt10" | ...
    k_active: int = 90            # target E[# active]
    lognormal_mu: float = 0.0
    lognormal_sigma: float = 0.25
    pool_size: int = 1_000_000
    pool_seed: int = 0
    magnitude_distribution: str = "lognormal"
    # How to draw the active-latent mask from ``_p_bern``:
    #   "bernoulli"  — independent Bernoulli(p_i) per latent
    #                   (# active is Poisson-Binomial around k).
    #   "fixed_k"    — sample exactly ``k_active`` latents per sample
    #                   via weighted sampling without replacement with
    #                   weights ``p_i / Σp_i``.  # active is
    #                   deterministically ``k_active``.
    #   "correlated" — Gaussian copula whose covariance is derived from
    #                   the ground truth's graph Laplacian (see fields
    #                   ``corr_*`` below).  Marginal firing rates equal
    #                   the bernoulli mode's; only the joint structure
    #                   (co-firing) differs.  Requires the ground-truth .pt
    #                   to have been built with a ``_cofire`` geometry
    #                   so its ``corr_eigvals`` is available.
    sampling: str = "bernoulli"

    # ── Correlated-sampling (Gaussian copula) parameters ─────────────
    #
    # Only consulted when ``sampling == "correlated"``. The defaults give
    # full rank and full strength.
    #
    # ``corr_rank``     — rank ``r`` of the low-rank correlation factor
    #                     ``F`` (shape ``(d_sae, r)``).  ``None`` → use
    #                     all ``d`` available eigenpairs.  ``min(r, d)``
    #                     is taken when set.
    # ``corr_strength`` — co-firing strength ``s`` ∈ [0, 1].  0 makes the
    #                     joint distribution statistically identical to
    #                     bernoulli (independent firings); 1 gives the
    #                     full Laplacian-derived correlation.  Marginal
    #                     firing rates are preserved at every ``s``.
    # ``corr_tau``      — small ridge added to ``λ`` for invertibility
    #                     of the GMRF precision ``Θ = L + τI``.  Default
    #                     0.01 is enough to bound the largest σ.
    corr_rank: int | None = None
    corr_strength: float = 1.0
    corr_tau: float = 0.01

    # ── Energy-softmax magnitude-family parameters ───────────────────
    #
    # Only consulted when ``magnitude_distribution == "energy_softmax"``.
    # Decouples the per-sample TOTAL activation energy from the
    # WITHIN-sample allocation, giving finer control over the peakiness
    # of |a_i| across active latents than a plain LogNormal(μ, σ) does.
    #
    # Draw:
    #   g_ni ~ N(0, 1)                       (independent auxiliary)
    #   q_ni = softmax(g_ni / τ)             over i ∈ S_n
    #   a_ni = R_n · sqrt(q_ni)              so Σ_i a_ni² = R_n²
    #
    # ``R_n`` is sampled from LogNormal(``lognormal_mu``, ``lognormal_sigma``²)
    # so ``lognormal_mu`` / ``lognormal_sigma`` retain their moment-
    # controlling roles at the SAMPLE level (as opposed to per-atom).
    # ``energy_tau`` is the softmax temperature:
    #   τ → 0    : one latent takes all the energy (extreme peakiness)
    #   τ ≈ 0.5  : heavy-tailed within-sample allocation (few dominate)
    #   τ → ∞    : uniform allocation across active latents (≈ constant)
    energy_tau: float = 1.0


class SyntheticSparseDataset(Dataset):
    """Deterministic pool of ``pool_size`` samples.

    Each ``__getitem__(idx)`` draws a sparse code using an RNG seeded by
    ``(pool_seed, idx)`` so the pool is reproducible and identical across
    training epochs and workers.
    """

    def __init__(
        self, D_gt: torch.Tensor, spec: SyntheticSpec,
        corr_eigvals: np.ndarray | torch.Tensor | None = None,
        corr_factor: np.ndarray | torch.Tensor | None = None,
    ):
        """Construct from ground-truth dictionary + sample-distribution spec.

        ``corr_eigvals`` / ``corr_factor`` are optional and only consulted
        when ``spec.sampling == "correlated"``.  Exactly one of them
        should be set:

        * ``corr_eigvals`` — low-frequency normalised-Laplacian
          eigenvalues, used by ``laplacian_*_cofire`` ground-truth dictionaries.  The
          dataset builds ``F = V · √(1/(λ+τ))`` from D_gt's columns and
          this spectrum, so co-firing follows graph geometry.
        * ``corr_factor`` — pre-built low-rank factor of shape
          ``(d_sae, rank)``, used by ``gaussian_cofire`` ground-truth dictionaries
          (sae_lens-style random ``F ~ N(0, scale²·I)``).  The dataset
          uses it directly, so co-firing follows arbitrary random
          directions with no dictionary tie-in.

        Pass ``blob.get(...)`` straight from the ground-truth .pt blob; at
        most one is present. Both are ignored for other sampling modes.
        """
        assert D_gt.ndim == 2, f"D_gt must be (d, d_sae), got {D_gt.shape}"
        self.D_gt = D_gt.float().contiguous()
        self.d, self.d_sae = D_gt.shape
        self.spec = spec

        if not is_valid_distribution(spec.distribution):
            raise ValueError(
                f"unknown distribution '{spec.distribution}'. valid: "
                f"{sorted(DISTRIBUTION_EXPONENTS) + sorted(GAUSSIAN_SIGMAS)} "
                f"or sd_r<R>_f<FFF>"
            )
        if spec.k_active <= 0:
            raise ValueError("k_active must be > 0.")
        k = float(spec.k_active)

        sd = parse_sd(spec.distribution)
        if sd is not None:
            # Two-population firing: a dense group of fraction ``f`` that
            # fires ``r`` times as often as the sparse rest.
            #
            # Sigma p_i = k is one equation in two unknowns, so ``r``
            # supplies the missing constraint:
            #
            #     n_d*(r*p_s) + (d_sae - n_d)*p_s = k
            #     p_s = k / (n_d*r + d_sae - n_d)
            #
            # Total activity is held at k and the contrast at r, so
            # varying f changes only the composition. (Fixing p_dense
            # absolutely would exceed the budget k once f is more than a
            # few percent.)
            r, f = sd
            n_d = int(round(f * self.d_sae))
            p_s = k / (n_d * r + (self.d_sae - n_d))
            p = np.full(self.d_sae, p_s, dtype=np.float64)
            # Shuffle which latents are dense (as the Zipf branch shuffles
            # rank -> latent) so index-correlated structure in the ground
            # truth does not alias onto the split. The XOR constant differs
            # from the Zipf branch's so the permutations differ.
            perm_rng = np.random.default_rng(spec.pool_seed ^ 0x5D_5D_5D)
            dense_idx = perm_rng.permutation(self.d_sae)[:n_d]
            p[dense_idx] = r * p_s
            # Raise rather than clip: clipping would break Sigma p_i = k,
            # the property this family guarantees. p_d > 1 needs
            # r > d_sae/k.
            if p.max() > 1.0:
                raise ValueError(
                    f"{spec.distribution}: p_dense={p.max():.4f} > 1 "
                    f"(d_sae={self.d_sae}, k={k:g}). Reduce r or raise f; "
                    f"r must stay below d_sae/k = {self.d_sae / k:.0f}.")
            self._p_bern = p
        elif spec.distribution in GAUSSIAN_SIGMAS:
            # Gaussian-prior p_i: Normal(mean, σ·mean) with mean = k/d_sae,
            # clipped non-negative, rescaled so Σ p_i = k, then clipped to
            # [0, 1] for Bernoulli safety.  Seeded off pool_seed so the
            # per-latent firing-rate structure is reproducible.
            sigma = GAUSSIAN_SIGMAS[spec.distribution]
            mean = k / self.d_sae
            rng = np.random.default_rng(spec.pool_seed ^ 0xA17_7A17)
            p = rng.normal(loc=mean, scale=sigma * mean, size=self.d_sae)
            np.clip(p, 0.0, None, out=p)
            s = p.sum()
            if s > 0:
                p *= (k / s)
            np.clip(p, 0.0, 1.0, out=p)
            self._p_bern = p
        else:
            exponent = DISTRIBUTION_EXPONENTS[spec.distribution]
            if exponent == 0.0:
                # Constant-Probability Model (Elhage): p_i = k / d_sae.
                self._p_bern = np.full(self.d_sae, k / self.d_sae, dtype=np.float64)
            else:
                # Variable-Probability Model: p_i ∝ Zipf(exponent), Σ p_i = k.
                ranks = np.arange(1, self.d_sae + 1, dtype=np.float64)
                w = 1.0 / (ranks ** exponent)
                w /= w.sum()
                # Randomise rank→latent mapping per pool_seed so "popular"
                # latents aren't always the low indices.
                perm_rng = np.random.default_rng(spec.pool_seed ^ 0xA17_7A17)
                p = np.empty(self.d_sae, dtype=np.float64)
                p[perm_rng.permutation(self.d_sae)] = w * k
                np.clip(p, 0.0, 1.0, out=p)
                self._p_bern = p

        # ── Correlated (Gaussian-copula) sampling setup ────────────────
        #
        # Only built for ``sampling="correlated"``. The factors are stored
        # on the dataset so ``_sample_code`` and
        # ``compute_pool_latent_stats_fast`` share them.
        self._F: np.ndarray | None = None
        self._sqrt_delta: np.ndarray | None = None
        self._threshold: np.ndarray | None = None
        sampling = getattr(spec, "sampling", "bernoulli")
        if sampling == "correlated":
            from scipy.stats import norm  # noqa: PLC0415

            if corr_factor is not None:
                # ── Random low-rank F (sae_lens style) ─────────────────
                # ``corr_factor`` is F itself, saved at dictionary-build time.
                if isinstance(corr_factor, torch.Tensor):
                    F = corr_factor.detach().cpu().numpy().astype(np.float64)
                else:
                    F = np.asarray(corr_factor, dtype=np.float64)
                if F.ndim != 2 or F.shape[0] != self.d_sae:
                    raise ValueError(
                        f"corr_factor must be shape (d_sae={self.d_sae}, rank); "
                        f"got {F.shape}"
                    )
                # ``corr_rank`` truncates to fewer columns; None keeps
                # the full saved rank.
                if spec.corr_rank is not None:
                    r = max(1, min(int(spec.corr_rank), int(F.shape[1])))
                    F = F[:, :r]
            elif corr_eigvals is not None:
                # ── laplacian-Gaussian-copula path ─────────────────────
                if isinstance(corr_eigvals, torch.Tensor):
                    lam = corr_eigvals.detach().cpu().numpy().astype(np.float64)
                else:
                    lam = np.asarray(corr_eigvals, dtype=np.float64)
                if lam.ndim != 1 or lam.shape[0] < 1:
                    raise ValueError(
                        f"corr_eigvals must be a 1-D array, got shape {lam.shape}"
                    )
                # The Laplacian eigenvectors are D_gt's rows: each column
                # of ``D_gt`` ``(d, d_sae)`` is a latent's spectral
                # coordinates. Column normalisation breaks exact
                # orthonormality, but the embedding is still a
                # low-frequency Laplacian projection.
                V_full = self.D_gt.detach().cpu().numpy().T  # (d_sae, d)
                d_available = min(int(V_full.shape[1]), int(lam.shape[0]))
                r_req = (
                    d_available if spec.corr_rank is None
                    else min(int(spec.corr_rank), d_available)
                )
                r = max(1, int(r_req))
                V_r = V_full[:, :r].astype(np.float64)
                lam_r = lam[:r]
                # GMRF spectrum: σ_j = 1 / (λ_j + τ).  Smallest λ → largest
                # σ → strongest low-frequency mode → graph neighbours
                # share large weight on the same low-freq eigenvectors and
                # therefore co-fire.
                tau = float(spec.corr_tau)
                sigma_r = 1.0 / (lam_r + tau)
                F = V_r * np.sqrt(sigma_r)[None, :]
            else:
                raise ValueError(
                    "sampling='correlated' requires the ground truth to carry "
                    "either `corr_eigvals` (laplacian_*_cofire geometries) "
                    "or `corr_factor` (gaussian_cofire geometry), but "
                    "neither was provided.  Rebuild the ground truth with a "
                    "`_cofire` geometry and pass blob['corr_eigvals'] or "
                    "blob['corr_factor'] into SyntheticSparseDataset."
                )
            # Marginal-preserving normalisation: with ε ~ N(0, I_r) the
            # per-latent variance of ``s · F·ε`` is ``s² · Σ_k F[i,k]²``.
            # Adding a residual ``√(1 − s²·rows) · η`` (η ~ N(0,1)) keeps
            # Var(g_i) = 1, so the marginal firing rate is p_i for any s.
            # The residual variance is clamped at 0.01 where
            # ``s²·rows[i]`` is too large; those latents' marginals drift
            # slightly and the count is printed.
            rows = (F ** 2).sum(axis=1)
            s = float(spec.corr_strength)
            delta_raw = 1.0 - (s ** 2) * rows
            n_clamped = int((delta_raw < 0.01).sum())
            if n_clamped > 0:
                print(
                    f"[SyntheticSparseDataset] correlated sampling: "
                    f"{n_clamped}/{self.d_sae} latents needed δ-clamp "
                    f"(s²·rows > 0.99) — their marginals will drift slightly"
                )
            delta = np.clip(delta_raw, 0.01, None)
            sqrt_delta = np.sqrt(delta)
            # Per-latent threshold: Φ⁻¹(1 - p_i) so {g > θ_i} fires with
            # prob p_i when Var(g_i) = 1.
            p_clamped = np.clip(self._p_bern, 1e-6, 1.0 - 1e-6)
            threshold = norm.ppf(1.0 - p_clamped)
            # float32 for memory and to match torch defaults.
            self._F = F.astype(np.float32)
            self._sqrt_delta = sqrt_delta.astype(np.float32)
            self._threshold = threshold.astype(np.float32)

    def __len__(self) -> int:
        return self.spec.pool_size

    def _sample_code(self, idx: int) -> np.ndarray:
        rng = np.random.default_rng((self.spec.pool_seed, int(idx)))
        a = np.zeros(self.d_sae, dtype=np.float32)
        # Active-mask draw — dispatch on ``sampling``:
        #   bernoulli:  per-latent independent Bernoulli(p_i)
        #   fixed_k:    weighted sampling without replacement of exactly
        #               k latents (weights = p_i normalised).
        #   correlated: Gaussian copula whose covariance comes from the
        #               ground truth's graph Laplacian (see __init__).
        #               Marginals match bernoulli's p_i; joint structure
        #               concentrates co-firings on graph neighbours.
        sampling = getattr(self.spec, "sampling", "bernoulli")
        if sampling == "fixed_k":
            total = float(self._p_bern.sum())
            if total <= 0:
                active_mask = np.zeros(self.d_sae, dtype=bool)
            else:
                p_norm = self._p_bern / total
                k = min(int(self.spec.k_active), self.d_sae)
                chosen = rng.choice(self.d_sae, size=k, replace=False, p=p_norm)
                active_mask = np.zeros(self.d_sae, dtype=bool)
                active_mask[chosen] = True
        elif sampling == "correlated":
            F = self._F
            r = int(F.shape[1])
            s = float(self.spec.corr_strength)
            eps = rng.standard_normal(r).astype(np.float32)         # (r,)
            eta = rng.standard_normal(self.d_sae).astype(np.float32)  # (d_sae,)
            g = s * (F @ eps) + self._sqrt_delta * eta
            active_mask = g > self._threshold
            # The shared factor can occasionally push every latent below
            # threshold. An all-zero code is a degenerate target, so turn
            # on the latent with the largest g; every sample then has at
            # least one active latent.
            if not active_mask.any():
                active_mask[int(np.argmax(g))] = True
        else:
            active_mask = rng.random(self.d_sae) < self._p_bern
        n_active = int(active_mask.sum())
        if n_active == 0:
            return a
        mu = float(self.spec.lognormal_mu)
        sigma = float(self.spec.lognormal_sigma)
        mag = self.spec.magnitude_distribution
        if mag == "lognormal":
            vals = rng.lognormal(mu, sigma, size=n_active)
        elif mag == "exponential":
            # Scale = exp(μ) so mean = exp(μ).  σ is ignored (one-param family).
            vals = rng.exponential(scale=float(np.exp(mu)), size=n_active)
        elif mag == "uniform":
            # Center at exp(μ) with half-width σ; clip to [0, ∞) so the
            # support stays non-negative when σ ≥ exp(μ).
            center = float(np.exp(mu))
            low = max(0.0, center - sigma)
            high = center + sigma
            if high <= low:
                high = low + 1e-9
            vals = rng.uniform(low=low, high=high, size=n_active)
        elif mag == "constant":
            # Deterministic ``exp(μ)`` on every active latent; σ ignored.
            vals = np.full(n_active, float(np.exp(mu)), dtype=np.float64)
        elif mag == "energy_softmax":
            # Sample-level energy R_n ~ LogNormal(μ, σ²), split across
            # active latents by q_ni = softmax(g_ni / τ), g_ni ~ N(0, 1);
            # a_ni = R_n · √q_ni so ‖a_n‖₂ = R_n.
            R_n = float(rng.lognormal(mu, sigma))
            tau = max(float(self.spec.energy_tau), 1e-6)
            g = rng.standard_normal(n_active)
            # Log-sum-exp shift for stability at small τ.
            z_log = g / tau
            z_log -= float(np.max(z_log))
            w = np.exp(z_log)
            q = w / max(float(w.sum()), 1e-300)
            vals = R_n * np.sqrt(q)
        else:
            raise ValueError(
                f"unknown magnitude_distribution {mag!r}; "
                f"valid: {MAGNITUDE_DISTRIBUTIONS}"
            )
        a[active_mask] = vals.astype(np.float32)
        return a

    def __getitem__(self, idx: int) -> torch.Tensor:
        a = self._sample_code(idx)  # (d_sae,)
        z = self.D_gt @ torch.from_numpy(a)  # (d,)
        return z

    @property
    def num_rows(self) -> int:
        """HF-dataset-style length, so fastsae's ``get_len_dataset()`` does
        not index ``dataset['image']``."""
        return self.spec.pool_size


def collate_z(batch) -> torch.Tensor:
    """Stack into ``(B, d)`` for training.

    With ``IdentityBackbone`` the batch is the training target directly.
    """
    return torch.stack(batch, dim=0)


def collate_z_seq(batch) -> torch.Tensor:
    """Stack + unsqueeze into ``(B, 1, d)`` for stats-time consumption.

    ``SAEInterpreter.compute_stats`` expects a ``(B, seq, d)`` layout;
    ``seq=1`` lets it run unchanged on synthetic data.
    """
    return torch.stack(batch, dim=0).unsqueeze(1)


# ── Ground-truth pool stats ──────────────────────────────────────────────────

_POOL_STATS_CACHE: dict = {}


def compute_pool_latent_stats(
    ds: SyntheticSparseDataset,
    n_samples: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Loop version — iterates ``_sample_code`` in order, matching the
    DataLoader's RNG.  Whole-data mean convention (zeros included).

    Prefer :func:`compute_pool_latent_stats_fast` unless you specifically
    need the training-order sample sequence.
    """
    spec = ds.spec
    n = min(int(n_samples) if n_samples else spec.pool_size, spec.pool_size)
    # Sampling mode and copula params are part of the cache key.
    key = (
        "loop",
        spec.distribution, int(spec.k_active), int(spec.pool_seed),
        float(spec.lognormal_mu), float(spec.lognormal_sigma),
        str(getattr(spec, "magnitude_distribution", "lognormal")),
        str(getattr(spec, "sampling", "bernoulli")),
        None if getattr(spec, "corr_rank", None) is None else int(spec.corr_rank),
        float(getattr(spec, "corr_strength", 1.0)),
        float(getattr(spec, "corr_tau", 0.01)),
        n,
    )
    if key in _POOL_STATS_CACHE:
        return _POOL_STATS_CACHE[key][:2]

    d_sae = ds.d_sae
    fire = np.zeros(d_sae, dtype=np.int64)
    abs_sum = np.zeros(d_sae, dtype=np.float64)
    for idx in range(n):
        a = ds._sample_code(idx)
        fire += a > 0
        abs_sum += np.abs(a)
    sparsity = (fire / n).astype(np.float32)
    mean_act = (abs_sum / float(n)).astype(np.float32)
    _POOL_STATS_CACHE[key] = (mean_act, sparsity, None)
    return mean_act, sparsity


def compute_pool_latent_stats_fast(
    ds: SyntheticSparseDataset,
    n_samples: int = 100_000,
    chunk_size: int = 10_000,
    progress_cb=None,
    return_max: bool = False,
) -> tuple:
    """Vectorised per-latent (mean_act, sparsity[, max_act]) for the pool.

    Uses a single RNG seeded by ``pool_seed`` (no per-index reseeding) so
    the sample order differs from ``_sample_code``; use this **only** for
    aggregate per-latent statistics.  Whole-data mean convention:
    ``mean_act_k = (Σ_t |a_k,t|) / n``.
    """
    spec = ds.spec
    n = int(n_samples)
    mag = getattr(spec, "magnitude_distribution", "lognormal")
    sampling = getattr(spec, "sampling", "bernoulli")
    # ``corr_*`` only matter for sampling="correlated" but are always
    # part of the key.
    corr_rank_key = (None if getattr(spec, "corr_rank", None) is None
                     else int(spec.corr_rank))
    corr_strength_key = float(getattr(spec, "corr_strength", 1.0))
    corr_tau_key = float(getattr(spec, "corr_tau", 0.01))
    key = (
        spec.distribution,
        int(spec.k_active),
        int(spec.pool_size),
        int(spec.pool_seed),
        float(spec.lognormal_mu),
        float(spec.lognormal_sigma),
        str(mag),
        str(sampling),
        corr_rank_key,
        corr_strength_key,
        corr_tau_key,
        n,
    )
    if key in _POOL_STATS_CACHE:
        if progress_cb is not None:
            progress_cb(n, n)
        cached = _POOL_STATS_CACHE[key]
        return cached if return_max else cached[:2]

    d_sae = ds.d_sae
    rng = np.random.default_rng(spec.pool_seed)
    fire = np.zeros(d_sae, dtype=np.int64)
    abs_sum = np.zeros(d_sae, dtype=np.float64)
    max_act = np.zeros(d_sae, dtype=np.float64)

    p_bern = ds._p_bern
    mu, sigma = float(spec.lognormal_mu), float(spec.lognormal_sigma)

    # ``energy_softmax`` normalises per sample over the active support,
    # which the chunked draw cannot express; use the per-sample loop.
    if mag == "energy_softmax":
        stats = compute_pool_latent_stats(ds, n_samples=n_samples)
        if progress_cb is not None:
            progress_cb(n, n)
        return stats

    # Same magnitude laws as ``_sample_code``.
    def _draw_coeffs(shape: tuple[int, int]) -> np.ndarray:
        if mag == "lognormal":
            return rng.lognormal(mu, sigma, size=shape).astype(np.float32)
        if mag == "exponential":
            return rng.exponential(scale=float(np.exp(mu)),
                                   size=shape).astype(np.float32)
        if mag == "uniform":
            center = float(np.exp(mu))
            low = max(0.0, center - sigma)
            high = center + sigma
            if high <= low:
                high = low + 1e-9
            return rng.uniform(low=low, high=high,
                               size=shape).astype(np.float32)
        if mag == "constant":
            return np.full(shape, float(np.exp(mu)), dtype=np.float32)
        raise ValueError(f"unknown magnitude_distribution {mag!r}")

    # fixed_k: Gumbel-top-k, i.e. top-k of ``log p_i + Gumbel(0, 1)``
    # per row, a vectorised equivalent of weighted sampling without
    # replacement.
    if sampling == "fixed_k":
        total = float(p_bern.sum())
        log_p = (np.log(np.clip(p_bern, 1e-30, None)).astype(np.float32)
                 if total > 0 else np.zeros(d_sae, dtype=np.float32))
        k = min(int(spec.k_active), d_sae)
    # correlated: uses the factors cached by ``__init__``; chunk_size is
    # capped so each (chunk, d_sae) array stays under ``MAX_ELEM``
    # float32s.
    if sampling == "correlated":
        if ds._F is None:
            raise ValueError(
                "compute_pool_latent_stats_fast: sampling='correlated' "
                "but the dataset has no F factor — was the ground truth "
                "built with a `_cofire` geometry?"
            )
        F_corr = ds._F
        thr_corr = ds._threshold
        sd_corr = ds._sqrt_delta
        s_corr = float(spec.corr_strength)
        MAX_ELEM = 2e8  # ≈ 800 MB at float32; reduce chunk to fit
        if chunk_size * d_sae > MAX_ELEM:
            chunk_size = max(1, int(MAX_ELEM // max(d_sae, 1)))
            print(f"[compute_pool_latent_stats_fast] correlated: "
                  f"chunk_size shrunk to {chunk_size} for d_sae={d_sae}")
        r_corr = int(F_corr.shape[1])

    for start in range(0, n, chunk_size):
        bs = min(chunk_size, n - start)
        if sampling == "fixed_k":
            # Gumbel-max weighted sampling without replacement.
            u = rng.random((bs, d_sae)).astype(np.float32)
            gumbel = -np.log(-np.log(np.clip(u, 1e-30, 1.0)))
            scores = log_p[None, :] + gumbel                  # (bs, d_sae)
            top_idx = np.argpartition(-scores, kth=k - 1, axis=1)[:, :k]
            active = np.zeros((bs, d_sae), dtype=bool)
            rows = np.arange(bs)[:, None]
            active[rows, top_idx] = True
        elif sampling == "correlated":
            # Vectorised version of ``_sample_code``'s correlated branch.
            eps = rng.standard_normal((bs, r_corr)).astype(np.float32)
            eta = rng.standard_normal((bs, d_sae)).astype(np.float32)
            g = s_corr * (eps @ F_corr.T) + sd_corr[None, :] * eta
            active = g > thr_corr[None, :]
            # As in ``_sample_code``, empty rows fire their highest-g
            # latent.
            _empty = ~active.any(axis=1)
            if _empty.any():
                active[_empty, g[_empty].argmax(axis=1)] = True
        else:
            active = rng.random((bs, d_sae)) < p_bern
        coeffs = _draw_coeffs((bs, d_sae))
        a = coeffs * active
        fire += active.sum(axis=0)
        abs_sum += a.sum(axis=0, dtype=np.float64)
        max_act = np.maximum(max_act, a.max(axis=0))
        if progress_cb is not None:
            progress_cb(min(start + bs, n), n)

    sparsity = (fire / n).astype(np.float32)
    # Whole-data mean: E[|a_k|] over ALL tokens (zeros included).
    mean_act = (abs_sum / float(n)).astype(np.float32)
    max_act_f = max_act.astype(np.float32)
    _POOL_STATS_CACHE[key] = (mean_act, sparsity, max_act_f)
    if return_max:
        return mean_act, sparsity, max_act_f
    return mean_act, sparsity
