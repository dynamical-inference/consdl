"""Load / reconstruct / format ``synthetic_config.json`` for a training run.

Run directory layout::

    checkpoints/<exp>/<arch>/d<D>/x<E>/<geom>/<base_tag>/seed<N>/

where ``<base_tag>`` is::

    <dist>_k<k>_l1_<l1>_<actalias>_s<sss>_mu<mm>[_fk]

Every axis except sampling is always present, so a path describes its
full configuration without reading the config JSON.

Tag-suffix vocabulary:
    <arch>      — relu | topk | fista
    d<D>        — ground-truth feature-dim, e.g. ``d768``
    <E>         — expansion factor (4, 8, 16, …)
    <geom>      — gaussian | laplacian_er | laplacian_sbm | frame
    <dist>      — uniform | lt03 | lt05 | lt07 | lt10 | gauss05 | gauss10
    <k>         — target k_active (E[# active latents] per token)
    <l1>        — sparsity coefficient, e.g. ``2e-05``
    <actalias>  — magnitude-distribution alias: lnorm | exp | uni | con
    <sss>       — σ × 100 as a 3-digit int (``s025`` → σ=0.25)
    <mm>        — µ × 100 as a signed int (``mu0`` → µ=0,
                  ``mu050`` → µ=0.5, ``mu-150`` → µ=-1.5)
    _fk         — only when sampling is ``fixed_k``; bernoulli is implicit

An older tag format (magnitude suffix omitted for the default, or
written as ``_mag_<alias>_s<sss>``) and layouts without the ``<geom>/``
or ``d<D>/`` segment are also parsed; see :func:`parse_tag` and
:func:`infer_config_from_tag`.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

DISTRIBUTIONS = ("uniform", "lt03", "lt04", "lt05", "lt07", "lt10",
                 "gauss05", "gauss10")

# Geometry assumed when a path has no ``<geom>/`` segment.
DEFAULT_GEOMETRY = "gaussian"

# Ground-truth feature dim assumed when a path has no ``d<D>/`` segment.
DEFAULT_D = 768

_X_DIR_RE = re.compile(r"^x\d+$")


def geom_segment(geometry: str) -> str:
    """Path segment for the geometry axis (the geometry name itself)."""
    return geometry


def dim_segment(d: int) -> str:
    """Path segment for the feature-dim axis — always ``d<D>``."""
    return f"d{int(d)}"


def append_geom(root: Path, geometry: str) -> Path:
    """``root / <geom>``."""
    return root / geom_segment(geometry)


def geom_label(geometry: str) -> str:
    """``"<geom>/"`` (with trailing slash) for f-string label building."""
    return f"{geom_segment(geometry)}/"


def geometry_of_tag_dir(tag_dir: Path) -> str:
    """Geometry for a tag dir: the name of its ``<geom>/`` parent.

    A tag directly under ``x<E>/`` (no geometry segment) resolves to
    :data:`DEFAULT_GEOMETRY`.
    """
    parent = tag_dir.parent.name
    return DEFAULT_GEOMETRY if _X_DIR_RE.match(parent) else parent


def split_layout_path(rest: list[str]) -> tuple[str, str, str] | None:
    """Split a ``rest`` parts list into ``(x_seg, geometry, tag)``.

    Accepts both layouts:

    * ``[x<E>, <geom>, <tag>]``;
    * ``[x<E>, <tag>]`` -- geometry is :data:`DEFAULT_GEOMETRY`.

    Returns ``None`` when ``rest`` doesn't match either length, so
    callers can ``continue`` on stray paths.  Use this after stripping
    any leading axis prefixes (``topk/``, ``d<D>/``, ``enc_depth<M>/``)
    so ``rest`` is purely the layout suffix.
    """
    if len(rest) == 2:
        return rest[0], DEFAULT_GEOMETRY, rest[1]
    if len(rest) == 3:
        return rest[0], rest[1], rest[2]
    return None

# Magnitude-distribution aliases used in the ``_<alias>_s<sss>_mu<mm>``
# tag suffix.
MAGNITUDE_TAG_ALIASES = {
    "exponential":    "exp",
    "uniform":        "uni",
    "constant":       "con",
    "lognormal":      "lnorm",
    "energy_softmax": "esm",   # μ/σ = R_n LogNormal moments;
                                # τ = within-sample softmax temperature
                                # (encoded separately as ``_tau<int>``).
}
_MAG_ALIAS_TO_DIST = {v: k for k, v in MAGNITUDE_TAG_ALIASES.items()}

# Default lognormal magnitude parameters.
DEFAULT_LOGNORMAL_MU = 0.0
DEFAULT_LOGNORMAL_SIGMA = 0.25

# Magnitude suffix ``_<alias>_s<sss>_mu<mm>``: ``<sss>`` = round(σ × 100),
# 3-digit zero-padded; ``<mm>`` = round(µ × 100), signed.
_MAG_SUFFIX_RE = (
    r"_(?P<mag>[a-z]+)_s(?P<magsigma>\d{3})_mu(?P<magmu>-?\d+)"
)

# Older magnitude suffix: absent (default lognormal) or
# ``_mag_<alias>_s<sss>[_mu<signed>]``.
_LEGACY_MAG_SUFFIX_RE = (
    r"(?:_mag_(?P<lmag>[a-z]+)_s(?P<lmagsigma>\d{3})"
    r"(?:_mu(?P<lmagmu>-?\d+))?)?"
)

# Sampling suffix. ``_fk`` flags a weighted-without-replacement draw of
# exactly ``k_active`` latents per sample; ``_cor<sss>[_corrank<r>]
# [_cortau<sss>]`` flags Gaussian-copula correlated co-firing (see
# :func:`mag_sampling_suffix`). Bernoulli is implicit (no suffix).
_SAMPLING_SUFFIX_RE = (
    r"(?:(?P<fk>_fk)"
    r"|_cor(?P<corstr>\d{3})"
    r"(?:_corrank(?P<corrank>\d+))?"
    r"(?:_cortau(?P<cortau>\d{3}))?)?"
)

# Current format: magnitude suffix required.
_TAG_RE = re.compile(
    r"^(?P<dist>" + "|".join(DISTRIBUTIONS) + r")"
    r"_k(?P<k>\d+)"
    r"_l1_(?P<l1>[0-9.]+e[+-]?\d+)"
    + _MAG_SUFFIX_RE
    + _SAMPLING_SUFFIX_RE
    + r"$"
)

# Older format: magnitude suffix optional, ``_mag_`` prefix.
_LEGACY_TAG_RE = re.compile(
    r"^(?P<dist>" + "|".join(DISTRIBUTIONS) + r")"
    r"_k(?P<k>\d+)"
    r"_l1_(?P<l1>[0-9.]+e[+-]?\d+)"
    + _LEGACY_MAG_SUFFIX_RE
    + _SAMPLING_SUFFIX_RE
    + r"$"
)

# FISTA-family format: a ``_fista_T<T>_lam<λ>_step<auto|η>`` core in
# place of ``_l1_<l1>``, followed by the same magnitude and sampling
# suffixes. Only :func:`infer_config_from_tag` uses it; :func:`parse_tag`
# deliberately returns ``None`` for these tags.
_FISTA_CORE_RE = (
    r"_fista_T(?P<t>\d+)"
    r"_lam(?P<lam>[0-9.]+e[+-]?\d+)"
    r"_step(?P<step>auto|[0-9.]+e[+-]?\d+)"
)
_FISTA_TAG_RE = re.compile(
    r"^(?P<dist>" + "|".join(DISTRIBUTIONS) + r")"
    r"_k(?P<k>\d+)"
    + _FISTA_CORE_RE
    + _MAG_SUFFIX_RE
    + _SAMPLING_SUFFIX_RE
    + r"$"
)


def mag_sampling_suffix(
    *,
    magnitude_distribution: str = "lognormal",
    magnitude_mu: float = DEFAULT_LOGNORMAL_MU,
    magnitude_sigma: float = DEFAULT_LOGNORMAL_SIGMA,
    sampling: str = "bernoulli",
    corr_rank: int | None = None,
    corr_strength: float = 1.0,
    corr_tau: float = 0.01,
    energy_tau: float = 1.0,
) -> str:
    """The ``_<alias>_s<sss>_mu<mm>[_fk | _cor<sss>[…]]`` tag suffix.

    Bernoulli sampling is implicit (no token); ``_fk`` and ``_cor…`` are
    the two conditional sampling segments. ``energy_softmax`` adds a
    ``_tau<sss>`` block.

    For ``sampling == "correlated"`` the suffix grows by
    ``_cor<strength_int>`` (mandatory) and the optional pieces:

      * ``_corrank<r>``  when ``corr_rank`` is not ``None``,
      * ``_cortau<sss>`` when ``corr_tau`` differs from the 0.01 default.

    so runs at different correlation strengths land in different
    directories.
    """
    alias = MAGNITUDE_TAG_ALIASES.get(magnitude_distribution)
    if alias is None:
        raise ValueError(
            f"unknown magnitude_distribution {magnitude_distribution!r}; "
            f"valid: {sorted(MAGNITUDE_TAG_ALIASES)}"
        )
    sss = int(round(float(magnitude_sigma) * 100))
    mu_int = int(round(float(magnitude_mu) * 100))
    suffix = f"_{alias}_s{sss:03d}_mu{mu_int}"
    # τ × 100, zero-padded to 3 digits (``_tau050`` ≡ τ = 0.5).
    if magnitude_distribution == "energy_softmax":
        tau_int = int(round(float(energy_tau) * 100))
        suffix += f"_tau{tau_int:03d}"
    if sampling == "fixed_k":
        suffix += "_fk"
    elif sampling == "correlated":
        cor_int = int(round(float(corr_strength) * 100))
        suffix += f"_cor{cor_int:03d}"
        if corr_rank is not None:
            suffix += f"_corrank{int(corr_rank)}"
        # tau is encoded only when it differs from the 0.01 default.
        if abs(float(corr_tau) - 0.01) > 1e-9:
            tau_int = int(round(float(corr_tau) * 100))
            suffix += f"_cortau{tau_int:03d}"
    elif sampling != "bernoulli":
        raise ValueError(
            f"unknown sampling {sampling!r}; "
            f"valid: 'bernoulli', 'fixed_k', 'correlated'"
        )
    return suffix


def _l1_tag_str(l1: float) -> str:
    """Shortest scientific-notation string that round-trips to ``l1``.

    Single-significant-figure values keep one digit (``1e-4`` →
    ``1e-04``); others keep the digits they need to stay distinct
    (``1.2e-4`` → ``1.2e-04``), so nearby sweep points never share a tag.
    """
    for prec in range(0, 7):
        s = f"{float(l1):.{prec}e}"
        if float(s) == float(l1):
            return s
    return f"{float(l1):.6e}"


def format_tag(
    distribution: str, k_active: int, l1: float,
    *,
    magnitude_distribution: str = "lognormal",
    magnitude_mu: float = DEFAULT_LOGNORMAL_MU,
    magnitude_sigma: float = DEFAULT_LOGNORMAL_SIGMA,
    sampling: str = "bernoulli",
    corr_rank: int | None = None,
    corr_strength: float = 1.0,
    corr_tau: float = 0.01,
) -> str:
    """Base tag for a run.

    Format: ``<dist>_k<k>_l1_<l1>_<alias>_s<sss>_mu<mm>[_fk]``. Every
    axis is emitted except bernoulli sampling, which is implicit (see
    :func:`mag_sampling_suffix`).

    * ``alias`` ∈ {lnorm, exp, uni, con} (see :data:`MAGNITUDE_TAG_ALIASES`).
    * ``sss`` = round(σ × 100), 3-digit zero-padded — e.g. ``s025`` for σ=0.25.
    * ``mm``  = round(µ × 100), signed — e.g. ``mu0`` for µ=0,
      ``mu050`` for µ=0.5, ``mu-150`` for µ=-1.5.
    * ``l1`` carries the minimal precision needed to round-trip (see
      :func:`_l1_tag_str`) so sub-decade sweep points don't collide.
    """
    base = f"{distribution}_k{int(k_active)}_l1_{_l1_tag_str(l1)}"
    return base + mag_sampling_suffix(
        magnitude_distribution=magnitude_distribution,
        magnitude_mu=magnitude_mu,
        magnitude_sigma=magnitude_sigma,
        sampling=sampling,
        corr_rank=corr_rank,
        corr_strength=corr_strength,
        corr_tau=corr_tau,
    )


def _sampling_from_match(m: "re.Match") -> dict:
    """Sampling fields from a tag match — ``{}`` for implicit bernoulli.

    ``_fk`` → ``fixed_k``; ``_cor…`` → ``correlated`` plus
    ``corr_strength`` and the optional ``corr_rank`` / ``corr_tau``
    (``corr_tau`` defaults to 0.01 when the tag omits it, mirroring
    :func:`mag_sampling_suffix`'s off-default-only convention).
    """
    if m.group("fk"):
        return {"sampling": "fixed_k"}
    if m.group("corstr") is not None:
        out: dict = {
            "sampling": "correlated",
            "corr_strength": int(m.group("corstr")) / 100.0,
        }
        if m.group("corrank") is not None:
            out["corr_rank"] = int(m.group("corrank"))
        out["corr_tau"] = (
            int(m.group("cortau")) / 100.0
            if m.group("cortau") is not None else 0.01
        )
        return out
    return {}


def parse_tag(tag: str) -> dict | None:
    """Parse a base tag into its components (current and older formats).

    Returns a dict with ``distribution``, ``k_active``, ``l1_coeff``;
    plus ``magnitude_distribution`` and ``magnitude_sigma`` (and
    ``magnitude_mu`` when present).  Sampling is included when the tag
    carries a ``_fk`` (fixed_k) or ``_cor…`` (correlated) suffix —
    bernoulli is implicit.

    The current format requires the magnitude suffix; the older format
    (no suffix → default lognormal, or ``_mag_<alias>…``) is also
    accepted. FISTA-family tags return ``None``.
    """
    m = _TAG_RE.match(tag)
    if m is not None:
        mag_alias = m.group("mag")
        mag = _MAG_ALIAS_TO_DIST.get(mag_alias)
        if mag is None:
            return None
        out = {
            "distribution":           m.group("dist"),
            "k_active":               int(m.group("k")),
            "l1_coeff":               float(m.group("l1")),
            "magnitude_distribution": mag,
            "magnitude_sigma":        int(m.group("magsigma")) / 100.0,
            "magnitude_mu":           int(m.group("magmu")) / 100.0,
        }
        out.update(_sampling_from_match(m))
        return out

    # Older format.
    m = _LEGACY_TAG_RE.match(tag)
    if m is None:
        return None
    out = {
        "distribution": m.group("dist"),
        "k_active":     int(m.group("k")),
        "l1_coeff":     float(m.group("l1")),
    }
    mag_alias = m.group("lmag")
    if mag_alias is not None:
        mag = _MAG_ALIAS_TO_DIST.get(mag_alias)
        if mag is None:
            return None
        out["magnitude_distribution"] = mag
        out["magnitude_sigma"] = int(m.group("lmagsigma")) / 100.0
        lmag_mu = m.group("lmagmu")
        if lmag_mu is not None:
            out["magnitude_mu"] = int(lmag_mu) / 100.0
    out.update(_sampling_from_match(m))
    return out


def _parse_fista_tag(tag: str) -> dict | None:
    """Parse a FISTA-family tag.

    Returns the same GT-pool-relevant fields :func:`parse_tag`
    yields (``distribution`` / ``k_active`` / ``magnitude_*`` /
    ``sampling``) so :func:`infer_config_from_tag` can reconstruct the
    held-out pool, plus the FISTA knobs (``fista_lam`` / ``fista_t_iters``
    / ``fista_step``).  ``l1_coeff`` is ``None`` — these arches sweep λ,
    not l1.  Returns ``None`` for non-fista tags.

    Kept separate from :func:`parse_tag` because callers of that
    function expect a numeric ``l1_coeff``.
    """
    m = _FISTA_TAG_RE.match(tag)
    if m is None:
        return None
    mag = _MAG_ALIAS_TO_DIST.get(m.group("mag"))
    if mag is None:
        return None
    step_tok = m.group("step")
    out = {
        "distribution":           m.group("dist"),
        "k_active":               int(m.group("k")),
        "l1_coeff":               None,
        "fista_t_iters":          int(m.group("t")),
        "fista_lam":              float(m.group("lam")),
        "fista_step":             None if step_tok == "auto" else float(step_tok),
        "magnitude_distribution": mag,
        "magnitude_sigma":        int(m.group("magsigma")) / 100.0,
        "magnitude_mu":           int(m.group("magmu")) / 100.0,
    }
    out.update(_sampling_from_match(m))
    return out


def iter_geom_tag_dirs(exp_dir: Path, accepts=None):
    """Yield ``(geometry, tag_dir)`` for every tag directory under ``exp_dir``.

    ``exp_dir`` is the ``x<E>/`` parent. A child of ``x<E>/`` that is a
    tag dir has geometry :data:`DEFAULT_GEOMETRY`; any other child is
    treated as a ``<geom>/`` segment whose children are tag dirs.

    ``accepts`` is a bool predicate over directory names — a child for
    which ``accepts(name)`` is true is a tag dir; otherwise it's a geom.
    The default uses :func:`parse_tag`; pass a custom one for other
    tag formats. Non-directories and non-matching entries are skipped.
    """
    if accepts is None:
        accepts = lambda n: parse_tag(n) is not None  # noqa: E731
    if not exp_dir.is_dir():
        return
    for child in sorted(exp_dir.iterdir()):
        if not child.is_dir():
            continue
        if accepts(child.name):
            yield DEFAULT_GEOMETRY, child
            continue
        for tag_dir in sorted(child.iterdir()):
            if not tag_dir.is_dir():
                continue
            if accepts(tag_dir.name):
                yield child.name, tag_dir


def pool_seed_for(geometry: str, distribution: str, k_active: int,
                  expansion: int) -> int:
    """Deterministic pool_seed — pools differ by (geom, dist, k, exp).

    μ and σ do not enter the hash, so the firing mask is identical across
    magnitude settings and only the magnitude draws differ.
    """
    tag = f"{geometry}_{distribution}_k{int(k_active)}_x{expansion}"
    return int(hashlib.sha1(tag.encode()).hexdigest()[:8], 16)


def _repo_root() -> Path:
    from consdl.io.paths import REPO_ROOT  # noqa: PLC0415
    return REPO_ROOT


def infer_config_from_tag(run_dir: Path) -> dict | None:
    """Reconstruct the synthetic config from the path.

    Accepted layouts:

    * ``d<D>/x<E>/<geom>/<tag>/seed<N>/``;
    * ``x<E>/<geom>/<tag>/seed<N>/`` -- ``d`` is :data:`DEFAULT_D`;
    * ``x<E>/<tag>/seed<N>/`` -- geometry is :data:`DEFAULT_GEOMETRY`.

    ``d`` and the expansion determine which ground-truth dictionary file
    is used.
    """
    tag_dir = run_dir.parent              # .../<tag>/seed<N> → .../<tag>
    parent = tag_dir.parent
    parent_m = re.match(r"^x(\d+)$", parent.name)
    if parent_m is not None:
        # No <geom> segment: default geometry.
        geometry = DEFAULT_GEOMETRY
        exp_dir = parent
        exp_m = parent_m
    else:
        # Explicit <geom> segment between x<E> and the tag; an optional
        # ``d<D>/`` above ``x<E>/`` is handled below.
        geom_dir = parent
        exp_dir = geom_dir.parent
        exp_m = re.match(r"^x(\d+)$", exp_dir.name)
        if exp_m is None:
            return None
        geometry = geom_dir.name
    # Both parsers expose the pool fields read below (distribution,
    # k_active, magnitude_*, sampling), so the held-out pool is the same
    # regardless of architecture.
    parsed = parse_tag(tag_dir.name) or _parse_fista_tag(tag_dir.name)
    if parsed is None:
        return None

    expansion = int(exp_m.group(1))
    seed_m = re.match(r"^seed(\d+)$", run_dir.name)
    seed = int(seed_m.group(1)) if seed_m else 0

    # Optional ``d<D>/`` above ``x<E>/``; 768 when absent.
    d = 768
    d_parent = exp_dir.parent
    if d_parent is not None:
        d_m = re.match(r"^d(\d+)$", d_parent.name)
        if d_m is not None:
            d = int(d_m.group(1))
    d_sae = d * expansion

    pool_seed = pool_seed_for(
        geometry, parsed["distribution"], parsed["k_active"], expansion,
    )
    # Non-default d mixes d into pool_seed so each d gets its own pool.
    # Must match the pool_seed used at training time.
    if d != 768:
        pool_seed = (pool_seed ^ (int(d) * 0x9E3779B1)) & 0xFFFFFFFF
    # Ground-truth filename:
    #   d=768, x=4         → D_gt_<geom>_seed0.pt
    #   d=768, x!=4        → D_gt_<geom>_dsae<d_sae>_seed0.pt
    #   d!=768 (any x)     → D_gt_<geom>_d<d>_dsae<d_sae>_seed0.pt
    if d == 768 and expansion == 4:
        gt_name = f"D_gt_{geometry}_seed0.pt"
    elif d == 768:
        gt_name = f"D_gt_{geometry}_dsae{d_sae}_seed0.pt"
    else:
        gt_name = f"D_gt_{geometry}_d{d}_dsae{d_sae}_seed0.pt"
    gt_dict = (
        _repo_root() / "data" / "synthetic"
        / "dictionaries" / gt_name
    )
    # Magnitude fields come from the tag; a tag without a magnitude
    # suffix means lognormal with the default μ / σ.
    mag = parsed.get("magnitude_distribution", "lognormal")
    mag_sigma = parsed.get("magnitude_sigma")
    mag_mu = parsed.get("magnitude_mu")
    sampling = parsed.get("sampling", "bernoulli")
    out = {
        "gt_dict": str(gt_dict),
        "geometry": geometry,
        "distribution": parsed["distribution"],
        "k_active": parsed["k_active"],
        "lognormal_mu": (float(mag_mu) if mag_mu is not None else DEFAULT_LOGNORMAL_MU),
        "lognormal_sigma": (float(mag_sigma) if mag_sigma is not None else DEFAULT_LOGNORMAL_SIGMA),
        "magnitude_distribution": mag,
        "sampling": sampling,
        "pool_size": 1_000_000,
        "pool_seed": pool_seed,
        "batch_size": 128,
        "d": d,
        "seed": seed,
        "_inferred_from_tag": True,
    }
    return out


def load_or_infer_config(run_dir: Path, *, write_back: bool = True) -> dict:
    cfg_path = run_dir / "synthetic_config.json"
    if cfg_path.exists():
        cached = json.loads(cfg_path.read_text())
        # A config previously inferred from the tag is re-inferred if
        # its ``d`` disagrees with the path.
        if bool(cached.get("_inferred_from_tag")):
            re_inferred = infer_config_from_tag(run_dir)
            if re_inferred is not None and re_inferred.get("d") != cached.get("d"):
                if write_back:
                    cfg_path.write_text(json.dumps(re_inferred, indent=2))
                    print(
                        f"[repair] re-inferred synthetic_config at {cfg_path}: "
                        f"d {cached.get('d')} → {re_inferred.get('d')}"
                    )
                return re_inferred
        return cached
    inferred = infer_config_from_tag(run_dir)
    if inferred is None:
        raise FileNotFoundError(
            f"synthetic_config.json missing at {cfg_path}, and the enclosing "
            f"tag does not match the canonical x<E>/<geom>/<dist>_k<k>_l1_<l1>"
            " layout. Pass --config-override or re-run train_synthetic.py."
        )
    if write_back:
        cfg_path.write_text(json.dumps(inferred, indent=2))
        print(f"[config] reconstructed synthetic_config at {cfg_path}")
    return inferred
