"""Co-active similarity regularization.

Implements the ``co_active_sim`` loss that
:meth:`consdl.training.trainer.SAETrainer.set_co_active_sim` adds to the
training objective (``--co-active-sim-coefficient γ``). It takes any
``(W, a)`` pair of decoder rows and codes.

For every active atom instance with at least one co-active candidate
among its cached top-r cosine neighbours, ``s`` is the cosine to its most
similar co-active candidate, and the loss is the mean of
``-log(1 - s^2 / (1 + delta))``. Since each term depends only on the
dictionary pair ``(i, q)``, the sum is computed exactly by grouping
instances into directed edges ``(i, q)`` weighted by their selection
counts, so only the selected edges need a differentiable cosine.

Codes are treated as active when ``> activation_eps`` (FISTA's
non-negative prox gives exact zeros). Computation is in fp32 because the
barrier is sensitive near ``s -> 1``.

Gradients: candidate selection always runs under ``no_grad``; the barrier
follows the caller's grad context, so callers can compute the value for
logging without a gradient.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

# Memory bound (not a hyperparameter): active instances / selected edges
# processed per inner chunk.
CO_ACTIVE_SIM_INSTANCE_CHUNK = 131072

# Default hyperparameters for the co-active similarity barrier.
CO_ACTIVE_SIM_R_DEFAULT = 32
CO_ACTIVE_SIM_REFRESH_STEPS_DEFAULT = 100
CO_ACTIVE_SIM_BARRIER_DELTA_DEFAULT = 1e-4
# Optional cosine margin below which a co-active pair is not penalized;
# 0.0 gives the plain barrier, whose minimum is full orthogonality.
CO_ACTIVE_SIM_BARRIER_MARGIN_DEFAULT = 0.0
CO_ACTIVE_SIM_ACTIVATION_EPS_DEFAULT = 1e-6
CO_ACTIVE_SIM_CACHE_BLOCK_SIZE_DEFAULT = 512


@torch.no_grad()
def _refresh_neighbor_cache(W: torch.Tensor, r: int, cache_block_size: int) -> torch.Tensor:
    """Top-r positive-cosine neighbor cache, one row per atom of ``W``
    (n, d_in). Returns ``neighbor_idx`` (n, r) — LOCAL indices into
    ``W``'s own row order.

    Computed in blocks of anchors so the full ``n x n`` Gram is never
    materialized. Self-similarity is masked at each block row's global
    diagonal position (the block slice is ``(block, n)``, not square).
    """
    W = F.normalize(W.detach().float(), dim=-1)
    n = W.shape[0]
    r = min(r, max(n - 1, 1))
    block = cache_block_size
    blocks = []
    for start in range(0, n, block):
        end = min(start + block, n)
        cosine = (W[start:end] @ W.T).clamp_min(0.0)  # (C, n)
        local_rows = torch.arange(end - start, device=W.device)
        global_cols = torch.arange(start, end, device=W.device)
        cosine[local_rows, global_cols] = float("-inf")  # exclude self
        blocks.append(cosine.topk(r, dim=-1).indices)
    return torch.cat(blocks, dim=0)


def _candidate_cosine_table(
    W_det: torch.Tensor, neighbor_idx: torch.Tensor, cache_block_size: int,
) -> torch.Tensor:
    """``[n, r]`` candidate-cosine table, computed once per step.

    The cache refresh cadence only governs which candidate indices are
    cached; the cosine values are recomputed every step because ``W``
    changes. Blockwise so the ``[n, r, d_in]`` gather is never
    materialized at once.
    """
    n, r = neighbor_idx.shape
    block = cache_block_size
    out = torch.empty((n, r), device=W_det.device, dtype=W_det.dtype)
    for start in range(0, n, block):
        end = min(start + block, n)
        anchor_w = W_det[start:end, None, :]                # (C, 1, d_in)
        candidate_w = W_det[neighbor_idx[start:end]]        # (C, r, d_in)
        out[start:end] = (anchor_w * candidate_w).sum(-1).clamp_min(0.0)
    return out


def co_active_sim_loss(
    W: torch.Tensor,
    a: torch.Tensor,
    cache: dict,
    *,
    step: int,
    r: int,
    refresh_steps: int,
    activation_eps: float,
    barrier_delta: float,
    cache_block_size: int,
    force_refresh: bool = False,
    barrier_margin: float = 0.0,
) -> torch.Tensor:
    """Co-active similarity barrier over atoms ``W`` (n, d_in) with codes
    ``a`` (..., n). Candidates are searched only among the rows of ``W``.

    ``cache`` is a caller-owned dict (``{"neighbor_idx": Tensor|None,
    "last_refresh_step": int}``), mutated in place. The neighbour cache is
    refreshed every ``refresh_steps`` steps, or immediately when
    ``force_refresh`` is set (needed if the rows of ``W`` change meaning).

    Three steps: (1) ``[n, r]`` candidate-cosine table; (2) for each active
    instance, pick its most similar co-active candidate and accumulate
    directed edge counts; (3) recompute the cosine with the grad-tracked
    ``W`` only for the selected edges and take the count-weighted barrier
    mean.
    """
    n = W.shape[0]
    if n < 2:
        return torch.zeros((), device=W.device, dtype=torch.float32)
    if (
        cache.get("neighbor_idx") is None
        or force_refresh
        or step - cache.get("last_refresh_step", -(10 ** 9)) >= refresh_steps
    ):
        cache["neighbor_idx"] = _refresh_neighbor_cache(W, r, cache_block_size)
        cache["last_refresh_step"] = step

    neighbor_idx = cache["neighbor_idx"]
    r_eff = neighbor_idx.shape[1]
    device = W.device
    chunk = CO_ACTIVE_SIM_INSTANCE_CHUNK

    # ── Steps 1-2: selection only, no grad. ──────────────────────────────
    with torch.no_grad():
        W_det = F.normalize(W.detach().float(), dim=-1)
        candidate_cos = _candidate_cosine_table(W_det, neighbor_idx, cache_block_size)  # (n, r)

        active = a.detach().reshape(-1, n) > activation_eps  # (B, n)
        batch_idx, atom_idx = active.nonzero(as_tuple=True)

        edge_counts = torch.zeros(n * r_eff, dtype=torch.int64, device=device)
        n_active = atom_idx.numel()
        for start in range(0, n_active, chunk):
            end = min(start + chunk, n_active)
            b = batch_idx[start:end]
            i = atom_idx[start:end]
            candidates = neighbor_idx[i]                       # (C, r)
            candidate_active = active[b[:, None], candidates]  # (C, r)
            scores = candidate_cos[i]                          # (C, r)
            masked = scores.masked_fill(~candidate_active, float("-inf"))
            best_score, best_q = masked.max(dim=1)
            valid = torch.isfinite(best_score)
            edge_id = i[valid] * r_eff + best_q[valid]
            edge_counts += torch.bincount(edge_id, minlength=n * r_eff)

    # ── Step 3: differentiable barrier over unique selected edges only.
    # Inherits the caller's ambient grad context. ─────────────────────
    selected_edge = edge_counts.nonzero(as_tuple=True)[0]
    if selected_edge.numel() == 0:
        return torch.zeros((), device=device, dtype=torch.float32)

    loss_sum = torch.zeros((), device=device, dtype=torch.float32)
    count_sum = torch.zeros((), device=device, dtype=torch.float32)
    W_norm = F.normalize(W.float(), dim=-1)
    for start in range(0, selected_edge.numel(), chunk):
        end = min(start + chunk, selected_edge.numel())
        edge = selected_edge[start:end]
        counts = edge_counts[edge].to(torch.float32)
        anchor_idx = edge // r_eff
        slot_idx = edge % r_eff
        partner_idx = neighbor_idx[anchor_idx, slot_idx]
        cosine = (W_norm[anchor_idx] * W_norm[partner_idx]).sum(dim=-1).clamp_min(0.0)
        if barrier_margin > 0.0:
            # Rescale so the margin maps to 0 and 1.0 stays 1.0: pairs below
            # the margin cost nothing, while exact duplicates still hit the
            # barrier's pole.
            cosine = ((cosine - barrier_margin)
                      / (1.0 - barrier_margin)).clamp_min(0.0)
            # Average only over edges above the margin, so the mean is not
            # diluted by pairs that contribute nothing.
            active = cosine > 0.0
            counts = counts * active.to(counts.dtype)
        barrier = -torch.log1p(-cosine.square() / (1.0 + barrier_delta))
        loss_sum = loss_sum + (counts * barrier).sum()
        count_sum = count_sum + counts.sum()
    return loss_sum / count_sum.clamp_min(1.0)
