"""Decoder-matching kernels shared by the dictionary and code metrics.

Cross-seed consistency starts from one-to-one Hungarian matching of two
dictionaries' unit-norm decoder atoms on cosine similarity. ``DCS`` is
the mean matched cosine; ``MCC`` correlates the codes of the matched
atoms (``code_metrics``).

SIGN. The match is computed twice: on the signed cosine, the right
reading for architectures with a canonical atom sign (ReLU / prox
codes), and on ``|cos|`` for the AbsTopK family (``isae``), where
``(w, +a)`` and ``(-w, -a)`` are the same solution and about half of the
matched atoms pair anti-parallel.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment


def unit_rows(D: np.ndarray) -> np.ndarray:
    """Row-normalise, leaving all-zero rows at zero instead of NaN."""
    X = np.asarray(D, dtype=np.float32)
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.maximum(n, 1e-12)


def hungarian_match(sim: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One-to-one matching maximising summed similarity:
    ``(row_idx, col_idx, matched similarities)``."""
    r, c = linear_sum_assignment(-sim)
    return r, c, sim[r, c]


def decoder_match(D_a: np.ndarray, D_b: np.ndarray) -> dict:
    """Hungarian match of two decoders (rows unit-norm).

    Returns ``dcs_signed`` / ``dcs_abs`` (mean matched cosine under the
    signed and ``|cos|`` matches), ``frac_sign_flipped`` (share of the
    ``|cos|`` matches that are anti-parallel) and ``cols``, the signed
    match's column permutation used to pair codes for ``MCC``.
    """
    sim = (D_a @ D_b.T).astype(np.float32)
    _r_h, c_h, s_h = hungarian_match(sim)
    r_a, c_a, s_a = hungarian_match(np.abs(sim))
    return {
        "dcs_signed": float(s_h.mean()),
        "dcs_abs": float(s_a.mean()),
        "frac_sign_flipped": float((sim[r_a, c_a] < 0).mean()),
        "cols": c_h,
    }
