"""Image-dataset loading, the HF short-name registry (``DATASET_MAP``),
image validity caching and per-seed data subsets."""

from __future__ import annotations

import pathlib
from typing import Any

import datasets as hf_datasets
import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

DATASET_MAP = {
    "datacomp": "cornuHGF/datacomp-medium-12m",
    "yfcc": "Leonardo6/yfcc15m",
    "cc": "ooutlierr/cc12m-recaptioned",
    "imagenet": "evanarlian/imagenet_1k_resized_256",
    "food101": "zguo0525/food101",
    "sun397": "dpdl-benchmark/sun397",
    "dtd": "tanganke/dtd",
    "celeba": "flwrlabs/celeba",
    "raf-db": "deanngkl/raf-db-7emotions",
    "stanford_action": "zrchen03/Stanford40_Dataset",
    "caltech101": "dpdl-benchmark/caltech101",
    # ``test`` serves as the evaluation split (see DATASET_META); the
    # smaller ``validation`` split is unused.
    "caltech256": "ilee0022/Caltech-256",
    "pets": "timm/oxford-iiit-pet",
    "cub200": "Donghyun99/CUB-200-2011",
    "cars": "tanganke/stanford_cars",
    "eurosat": "tanganke/eurosat",
    "aircraft": "Donghyun99/FGVC-Aircraft",
    "flowers": "dpdl-benchmark/oxford_flowers102",
    "ucf101": "flwrlabs/ucf101",
}


# ── Per-dataset split layout ──────────────────────────────────────────────
# ``val_split=None`` marks datasets whose only split is ``train``; the
# caller splits those deterministically.
DATASET_META: dict[str, dict] = {
    "imagenet":        {"train_split": "train", "val_split": "val"},
    "sun397":          {"train_split": "train", "val_split": "test"},
    "food101":         {"train_split": "train", "val_split": "test"},
    "dtd":             {"train_split": "train", "val_split": "test"},
    "caltech256":      {"train_split": "train", "val_split": "test"},
    # CelebA is multi-attribute (no single ``label`` column); the caller
    # selects one attribute and renames it to ``label``.
    "celeba":          {"train_split": "train", "val_split": "valid"},
    "raf-db":          {"train_split": "train", "val_split": None},
    "stanford_action": {"train_split": "train", "val_split": None},
}

# Class names for datasets whose HF mirror has no ``ClassLabel`` feature.
# RAF-DB order follows its canonical label indexing (1-7, minus 1).
_HARDCODED_CLASS_NAMES: dict[str, list[str]] = {
    "raf-db": [
        "surprise", "fear", "disgust", "happy",
        "sad", "angry", "neutral",
    ],
}


def _label_feature(dataset_short_name: str):
    builder = hf_datasets.load_dataset_builder(DATASET_MAP[dataset_short_name])
    features = builder.info.features
    if "label" not in features:
        raise ValueError(f"Dataset '{dataset_short_name}' has no 'label' feature.")
    return features["label"]


def get_n_classes(dataset_short_name: str) -> int:
    if dataset_short_name in _HARDCODED_CLASS_NAMES:
        return len(_HARDCODED_CLASS_NAMES[dataset_short_name])
    return _label_feature(dataset_short_name).num_classes


def get_class_names(dataset_short_name: str) -> list[str]:
    if dataset_short_name in _HARDCODED_CLASS_NAMES:
        return list(_HARDCODED_CLASS_NAMES[dataset_short_name])
    return _label_feature(dataset_short_name).names


def format_dataset(dataset_name: str, dataset):
    if dataset_name == "cornuHGF/datacomp-medium-12m":
        dataset = dataset.rename_column("jpg", "image")
    elif dataset_name == "Leonardo6/yfcc15m":
        dataset = dataset.rename_column("image.png", "image")
    elif dataset_name == "ooutlierr/cc12m-recaptioned":
        dataset = dataset.rename_column("image.jpg", "image")
    return dataset


def load_image_split(dataset_short: str = "imagenet", split: str = "val", cache_dir: str | None = None) -> Any:
    """Load a HF image dataset by short name."""
    if dataset_short not in DATASET_MAP:
        raise KeyError(f"unknown dataset {dataset_short!r}; known: {list(DATASET_MAP)}")
    ds = hf_datasets.load_dataset(path=DATASET_MAP[dataset_short], split=split, cache_dir=cache_dir)
    return format_dataset(DATASET_MAP[dataset_short], ds)


_INDICES_CACHE_DIR = pathlib.Path("~/.cache/sae_dataset_indices").expanduser()


def build_valid_indices(dataset, dataset_short_name, split, image_key="image", min_bytes=1024):
    """Scan dataset bytes via pyarrow and cache indices of non-truncated images."""
    cache_path = _INDICES_CACHE_DIR / f"{dataset_short_name}_{split}_valid_indices.npy"
    if cache_path.exists():
        print(f"Loading cached valid indices from {cache_path}")
        return np.load(cache_path)
    print(f"Scanning {len(dataset)} items for valid images (min_bytes={min_bytes})...")
    col = dataset.data.column(image_key)
    first_type = col.chunks[0].type if col.chunks else col.type
    if pa.types.is_struct(first_type):
        bytes_col = pa.chunked_array([c.field("bytes") for c in col.chunks])
    else:
        bytes_col = col
    lengths = pc.fill_null(pc.binary_length(bytes_col), 0)
    lengths_np = lengths.to_numpy(zero_copy_only=False)
    indices = np.where(lengths_np >= min_bytes)[0].astype(np.int32)
    _INDICES_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, indices)
    print(f"Found {len(indices)} valid images. Cached to {cache_path}")
    return indices


def select_subset(dataset, dataset_short_name, split, max_samples, image_key="image", min_bytes=1024):
    """Return a subset of *dataset* with *max_samples* images evenly spanned across valid indices."""
    all_valid = build_valid_indices(dataset, dataset_short_name, split, image_key=image_key, min_bytes=min_bytes)
    if len(all_valid) <= max_samples:
        return dataset.select(all_valid.tolist())
    step = len(all_valid) // max_samples
    return dataset.select(all_valid[::step][:max_samples].tolist())


# ── Per-seed data subsets ─────────────────────────────────────────────────
# Each seed gets a disjoint slice of the training set, to measure
# cross-seed consistency under class or sample disjointness:
#
#   * ``class_bucket(bucket_idx, n_buckets)`` — class-disjoint.  With 1000
#     ImageNet classes + n_buckets=4, bucket 0 keeps labels [0, 250),
#     bucket 1 [250, 500), …  Buckets are mutually exclusive and exhaustive.
#
#   * ``index_residue(residue, modulus)`` — class-balanced sample-disjoint.
#     Keeps rows whose row-index mod ``modulus`` equals ``residue``.  Each
#     residue covers ~1/modulus of the data, all 1000 classes preserved.
#     Index is over the dataset's intrinsic row order; if you ``.shuffle``
#     downstream, do so AFTER applying this filter.


def apply_class_bucket(dataset, bucket_idx: int, n_buckets: int,
                        label_key: str = "label"):
    """Restrict ``dataset`` to rows whose ``label`` falls in bucket
    ``bucket_idx`` of ``n_buckets`` equal-sized class ranges.

    E.g. 1000 classes / 4 buckets gives [0, 250), [250, 500), ...  When
    the class count is not divisible by ``n_buckets``, the last bucket
    absorbs the remainder so the union covers every label.
    """
    if bucket_idx < 0 or bucket_idx >= int(n_buckets):
        raise ValueError(
            f"bucket_idx={bucket_idx} out of range for n_buckets={n_buckets}")
    feat = dataset.features.get(label_key)
    if feat is None:
        raise KeyError(f"{label_key!r} not in dataset.features "
                       f"(have: {list(dataset.features)})")
    n_classes = getattr(feat, "num_classes", None)
    if n_classes is None:
        # No ClassLabel feature: use max(label) + 1.
        n_classes = int(max(dataset[label_key]) + 1)
    step = int(n_classes) // int(n_buckets)
    lo = int(bucket_idx) * step
    hi = (int(bucket_idx) + 1) * step if int(bucket_idx) + 1 < int(n_buckets) else int(n_classes)
    return dataset.filter(
        lambda lbl: lo <= int(lbl) < hi,
        input_columns=[label_key],
    )


def apply_index_residue(dataset, residue: int, modulus: int):
    """Restrict ``dataset`` to rows whose row index mod ``modulus`` equals
    ``residue``.
    """
    if residue < 0 or residue >= int(modulus):
        raise ValueError(
            f"residue={residue} out of range for modulus={modulus}")
    n = len(dataset)
    keep = list(range(int(residue), n, int(modulus)))
    return dataset.select(keep)


def apply_data_subset(dataset, *, mode: str | None,
                       bucket_idx: int, n_buckets: int):
    """Dispatch on ``mode`` ∈ {``None``, ``"class_bucket"``, ``"index_residue"``}.

    No-op when ``mode`` is ``None``, empty or ``"none"``.
    """
    if not mode or mode == "none":
        return dataset
    if mode == "class_bucket":
        return apply_class_bucket(dataset, bucket_idx=bucket_idx,
                                    n_buckets=n_buckets)
    if mode == "index_residue":
        return apply_index_residue(dataset, residue=bucket_idx,
                                     modulus=n_buckets)
    raise ValueError(f"unknown data-subset mode: {mode!r}")
