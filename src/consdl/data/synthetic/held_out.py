"""The held-out synthetic pool every ground-truth metric is measured on.

Rebuilt from a cell's ``synthetic_config`` so all architectures of one
(geometry, distribution, k) are scored on the SAME samples.
"""
from __future__ import annotations

import numpy as np
import torch

from consdl.data.synthetic.dataset import SyntheticSparseDataset, SyntheticSpec


def gt_decoder(cfg: dict) -> np.ndarray:
    """L2-normalised ground-truth decoder ``(d_sae, d)`` as float32."""
    blob = torch.load(cfg["gt_dict"], map_location="cpu")
    D = blob["D_gt"].float().numpy().T  # (d_sae, d)
    D /= np.clip(np.linalg.norm(D, axis=1, keepdims=True), 1e-12, None)
    return D.astype(np.float32)


def held_out_dataset(cfg: dict) -> tuple[SyntheticSparseDataset, int]:
    """The held-out pool: same ground truth and sampling spec as training,
    drawn with ``stats_pool_seed`` (default ``pool_seed ^ 0xDEADBEEF``).

    ``pool_seed`` is required (recorded by the trainer or reconstructed by
    :func:`consdl.data.synthetic.config.load_or_infer_config`); deriving it
    from anything else would silently score this run on a different sample
    than the other architectures.
    """
    D_gt = torch.load(cfg["gt_dict"], map_location="cpu")["D_gt"]
    if "pool_seed" not in cfg:
        raise KeyError(
            "synthetic_config is missing 'pool_seed'. Load configs through "
            "synthetic_config.load_or_infer_config.")
    stats_pool_seed = int(cfg.get("stats_pool_seed",
                                  int(cfg["pool_seed"]) ^ 0xDEADBEEF))
    spec = SyntheticSpec(
        distribution=cfg["distribution"],
        k_active=int(cfg["k_active"]),
        lognormal_mu=float(cfg.get("lognormal_mu", 0.0)),
        lognormal_sigma=float(cfg.get("lognormal_sigma", 0.25)),
        pool_size=int(cfg.get("pool_size", 1_000_000)),
        pool_seed=stats_pool_seed,
    )
    return SyntheticSparseDataset(D_gt, spec), stats_pool_seed


def gt_codes(ds: SyntheticSparseDataset, n_samples: int) -> np.ndarray:
    """Ground-truth sparse codes ``(n, d_sae)`` of the first ``n`` samples."""
    n = int(n_samples)
    acts = np.zeros((n, ds.d_sae), dtype=np.float32)
    for idx in range(n):
        acts[idx] = ds._sample_code(idx)
    return acts
