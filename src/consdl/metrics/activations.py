"""Activation pools and per-seed encoding for the code metrics.

The backbone forward is the expensive step and is seed-independent: a
pool of layer activations ``z`` (n_tokens, d_in) is built once per
(backbone, layer) and every seed's SAE encodes the same ``z``. Heavy
imports are local so the module loads without torch / HF on the path.
"""
from __future__ import annotations

import contextlib
import io as _stdio
from pathlib import Path
from typing import Mapping

import numpy as np

from consdl.metrics.quality import quality_scalars

#: Activation pools larger than this (GB) are not cached; recomputing the
#: backbone forward is cheaper. Any stale cache file is removed.
CACHE_CAP_GB = 100.0


def _collect(loader, backbone, d_in: int, n_tokens: int) -> np.ndarray:
    import torch  # noqa: PLC0415

    chunks: list[np.ndarray] = []
    have = 0
    with torch.no_grad():
        for batch in loader:
            z = backbone.forward_to_z(batch).reshape(-1, d_in)
            chunks.append(z.float().cpu().numpy())
            have += z.shape[0]
            if have >= n_tokens:
                break
    if not chunks:
        raise RuntimeError("no activations from the dataset")
    return np.concatenate(chunks, axis=0)[: int(n_tokens)].astype(np.float32)


def _load_cached(act_cache: Path | None) -> np.ndarray | None:
    if act_cache is not None and act_cache.exists():
        print(f"  [act-cache] reload {act_cache}")
        return np.load(act_cache).astype(np.float32)
    return None


def _save_cached(pool: np.ndarray, act_cache: Path | None) -> None:
    if act_cache is None:
        return
    cache_gb = pool.nbytes / 1e9
    if cache_gb > CACHE_CAP_GB:
        print(f"  [act-cache] pool ≈ {cache_gb:.0f} GB > "
              f"{CACHE_CAP_GB:.0f} GB cap — not cached")
        if act_cache.exists():
            act_cache.unlink()
        return
    act_cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(act_cache, pool)
    print(f"  [act-cache] saved {act_cache}  ({cache_gb * 1e3:.0f} MB)")


def vision_pool(backbone_short: str, layer: int, *, dataset: str, split: str,
                n_tokens: int, batch_size: int, cache_dir: str | None,
                device: str, act_cache: Path | None = None) -> np.ndarray:
    """Stream an image pool through a vision backbone → ``z`` (n_tokens, d_in)."""
    pool = _load_cached(act_cache)
    if pool is not None:
        return pool

    import datasets as hfds  # noqa: PLC0415
    import fastsae  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    from consdl.data.vision_backbones import (  # noqa: PLC0415
        BACKBONE_MAP,
        Backbone,
        hook_layer_name,
    )
    from consdl.data.vision_datasets import (  # noqa: PLC0415
        DATASET_MAP,
        format_dataset,
    )

    hf_name = BACKBONE_MAP[backbone_short]["hf_name"]
    hook = hook_layer_name(backbone_short, layer)
    backbone = Backbone(model_hf_name=hf_name, hook_target_name=hook, device=device)
    d_in = int(backbone.hook_target_dim)

    ds_path = DATASET_MAP.get(dataset, dataset)
    ds = hfds.load_dataset(path=ds_path, split=split, cache_dir=cache_dir)
    ds = format_dataset(ds_path, ds)
    auto = AutoProcessor.from_pretrained(hf_name)

    def _processor(img):
        return auto(images=img, text="a photo", return_tensors="pt")

    loader = fastsae.dataset.flexible.create_dataloader(
        dataset=ds, processor=_processor,
        return_pixel_values_only=True, return_labels_np=False,
        batch_size=int(batch_size), shuffle=False,
        collate="auto", num_workers=0,
    )
    pool = _collect(loader, backbone, d_in, n_tokens)
    _save_cached(pool, act_cache)
    return pool


#: The vision MCC pool used in the paper: ``per_class`` images of every
#: ImageNet class from the validation split (3 x 1000 = 3,000 images, i.e.
#: 591k CLIP / 771k DINOv2 patch tokens), drawn with a fixed seed. The
#: pool is class-stratified rather than a random or leading slice because
#: class coverage affects MCC more than pool size does. The HF validation
#: split is stored class-sorted, 50 images per class, so the draw is a
#: pure index computation.
VISION_POOL_PER_CLASS = 3
IMAGES_PER_CLASS = 50


def stratified_rows(n: int, per_class: int, seed: int = 0) -> np.ndarray:
    """Sorted row indices of ``per_class`` images of every class, for a
    split stored class-sorted with :data:`IMAGES_PER_CLASS` per class.

    A fixed random choice within each class, not the first few, so pools
    of different ``per_class`` are independent draws rather than nested.
    """
    rng = np.random.default_rng(seed)
    k = min(int(per_class), IMAGES_PER_CLASS)
    out = np.concatenate([
        c * IMAGES_PER_CLASS + rng.choice(IMAGES_PER_CLASS, k, replace=False)
        for c in range(n // IMAGES_PER_CLASS)])
    out.sort()
    return out


def vision_stratified_pool(backbone_short: str, layer: int, *,
                           per_class: int = VISION_POOL_PER_CLASS,
                           seed: int = 0, dataset: str = "imagenet",
                           split: str = "val", batch_size: int = 64,
                           cache_dir: str | None = None, device: str,
                           act_cache: Path | None = None) -> np.ndarray:
    """``per_class`` images of every class through a vision backbone →
    all their patch tokens ``z`` (n_images * n_tokens, d_in), in image
    order. See :data:`VISION_POOL_PER_CLASS`."""
    pool = _load_cached(act_cache)
    if pool is not None:
        return pool

    import datasets as hfds  # noqa: PLC0415
    import fastsae  # noqa: PLC0415
    from transformers import AutoProcessor  # noqa: PLC0415

    from consdl.data.vision_backbones import (  # noqa: PLC0415
        BACKBONE_MAP,
        Backbone,
        hook_layer_name,
    )
    from consdl.data.vision_datasets import DATASET_MAP, format_dataset  # noqa: PLC0415

    ds_path = DATASET_MAP.get(dataset, dataset)
    ds = hfds.load_dataset(path=ds_path, split=split, cache_dir=cache_dir)
    ds = format_dataset(ds_path, ds)
    labels = np.asarray(ds["label"])
    n_cls = len(labels) // IMAGES_PER_CLASS
    if (len(labels) != n_cls * IMAGES_PER_CLASS or not np.array_equal(
            labels, np.repeat(np.arange(n_cls), IMAGES_PER_CLASS))):
        raise ValueError(f"{ds_path}/{split} is not stored class-sorted with "
                         f"{IMAGES_PER_CLASS} images per class")
    ds = ds.select(stratified_rows(len(labels), per_class, seed).tolist())

    hf_name = BACKBONE_MAP[backbone_short]["hf_name"]
    backbone = Backbone(model_hf_name=hf_name,
                        hook_target_name=hook_layer_name(backbone_short, layer),
                        device=device)
    auto = AutoProcessor.from_pretrained(hf_name)

    def _processor(img):
        return auto(images=img, text="a photo", return_tensors="pt")

    loader = fastsae.dataset.flexible.create_dataloader(
        dataset=ds, processor=_processor,
        return_pixel_values_only=True, return_labels_np=False,
        batch_size=int(batch_size), shuffle=False,
        collate="auto", num_workers=0,
    )
    pool = _collect(loader, backbone, int(backbone.hook_target_dim), 1 << 62)
    _save_cached(pool, act_cache)
    return pool


def lm_pool(backbone_short: str, layer: int, *, hf_dataset: str,
            hf_dataset_config: str | None, hf_dataset_split: str,
            hf_dataset_text_field: str, context_length: int, n_tokens: int,
            batch_seqs: int, device: str,
            act_cache: Path | None = None) -> np.ndarray:
    """Stream a token pool through an LM backbone → ``z`` (n_tokens, d_in)."""
    pool = _load_cached(act_cache)
    if pool is not None:
        return pool

    from transformers import AutoTokenizer  # noqa: PLC0415

    from consdl.data.lm_backbones import LM_BACKBONE_MAP  # noqa: PLC0415
    from consdl.cli.train_lm import (  # noqa: PLC0415
        LMBackbone,
        build_lm_dataloader,
    )

    cfg = LM_BACKBONE_MAP[backbone_short]
    hf_name = cfg["hf_name"]
    backbone = LMBackbone(
        model_hf_name=hf_name,
        hook_layer_path=cfg["hook_layer_fmt"].format(int(layer)),
        device=device,
    )
    d_in = int(backbone.hook_target_dim)

    tokenizer = AutoTokenizer.from_pretrained(hf_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    loader = build_lm_dataloader(
        hf_dataset=hf_dataset, hf_config=hf_dataset_config,
        split=hf_dataset_split, text_field=hf_dataset_text_field,
        tokenizer=tokenizer, context_length=int(context_length),
        batch_seqs=int(batch_seqs), seed=None,
    )
    pool = _collect(loader, backbone, d_in, n_tokens)
    _save_cached(pool, act_cache)
    return pool


def resolve_lm_backbone_layer(cell_dir: Path) -> tuple[str, int] | None:
    """``(backbone_short, layer)`` from the ``<backbone>_layer<L>`` segment
    of an LM checkpoint path, or ``None`` when no segment names a known
    LM backbone."""
    from consdl.data.lm_backbones import LM_BACKBONE_MAP  # noqa: PLC0415

    for part in reversed(Path(cell_dir).parts):
        head, _, tail = part.rpartition("_layer")
        if head in LM_BACKBONE_MAP and tail.isdigit():
            return head, int(tail)
    return None


# ── Per-seed encoding ───────────────────────────────────────────────────────


def load_seed_sae(step_dir, *, device: str,
                  mp_override: Mapping[str, object] | None = None):
    """One seed's SAE, loaded once, on ``device``, in eval mode.

    ``mp_override`` (MP checkpoints only) sets the inference round count:
    MP's sparsity is not in its weights, so the same checkpoint lands on
    L0 32, 64 or 128 depending only on how many rounds it is given.
    """
    from consdl.io.checkpoint import load_sae  # noqa: PLC0415

    try:
        with contextlib.redirect_stdout(_stdio.StringIO()):
            sae = load_sae(step_dir)
    except Exception as e:  # noqa: BLE001
        print(f"    [skip-seed] load_sae failed ({type(e).__name__}: {e})")
        return None
    if mp_override:
        from consdl.models.mp import apply_mp_override  # noqa: PLC0415
        apply_mp_override(sae, mp_override)
    return sae.to(device).eval()


def encode_batch_with(sae, z, *, device: str, batch_size: int):
    """``(acts, z_hat)`` for one batch on an already loaded SAE."""
    import torch  # noqa: PLC0415

    n = z.shape[0]
    acts, zhat = [], []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            zb = torch.from_numpy(np.ascontiguousarray(z[s:s + batch_size])).to(device)
            a = sae.encode(zb)
            zh = sae.decode(a)
            acts.append(a.float().cpu().numpy())
            zhat.append(zh.float().cpu().numpy())
    return (np.concatenate(acts, 0) if acts else np.zeros((0, 0), dtype=np.float32),
            np.concatenate(zhat, 0) if zhat else np.zeros((0, 0), dtype=np.float32))


def seed_quality(z: np.ndarray, acts: np.ndarray, z_hat: np.ndarray,
                 *, live_threshold: float = 0.1) -> dict:
    """Single-model quality scalars from one seed's pass."""
    z64 = z.astype(np.float64)
    sse = float(((z64 - z_hat.astype(np.float64)) ** 2).sum())
    sst = float(((z64 - z64.mean(axis=0)) ** 2).sum())
    return quality_scalars(sse=sse, sst=sst, n_elems=z.size, acts=acts,
                           live_threshold=live_threshold)


def seed_encode(z_pool: np.ndarray, step_dir: Path, *, device: str,
                batch_size: int,
                mp_override: Mapping[str, object] | None = None,
                ) -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Encode the shared ``z`` pool with one seed's SAE.

    Returns ``(acts, z_hat, quality)`` or ``None`` when the checkpoint
    can't be loaded.
    """
    import torch  # noqa: PLC0415

    sae = load_seed_sae(step_dir, device=device, mp_override=mp_override)
    if sae is None:
        return None
    acts, z_hat = encode_batch_with(sae, z_pool, device=device,
                                    batch_size=batch_size)
    del sae
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    acts, z_hat = acts.astype(np.float32), z_hat.astype(np.float32)
    return acts, z_hat, seed_quality(z_pool, acts, z_hat)
