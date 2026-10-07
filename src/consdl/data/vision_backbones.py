"""Vision backbone registry (``BACKBONE_MAP``) and hook-layer lookup.

Also re-exports fastsae's ``Backbone`` and ``SAEWrapper``.
"""

from __future__ import annotations

import fastsae

Backbone = fastsae.models.Backbone
SAEWrapper = fastsae.models.SAEWrapper

BACKBONE_MAP: dict[str, dict] = {
    # ``hook_layer_fmt`` is a ``str.format`` template for the transformer
    # block path: ``vision_model.encoder.layers.<i>`` for CLIP and SigLIP,
    # ``encoder.layer.<i>`` for DINOv2.
    "clip-vit-l-14": {
        "hf_name": "openai/clip-vit-large-patch14",
        "n_layers": 24,
        "hook_layer_fmt": "vision_model.encoder.layers.{}",
    },
    "clip-vit-b-16": {
        "hf_name": "openai/clip-vit-base-patch16",
        "n_layers": 12,
        "hook_layer_fmt": "vision_model.encoder.layers.{}",
    },
    "dinov2-b": {
        "hf_name": "facebook/dinov2-base",
        "n_layers": 12,
        "hook_layer_fmt": "encoder.layer.{}",
    },
    "dinov2-l": {
        "hf_name": "facebook/dinov2-large",
        "n_layers": 24,
        "hook_layer_fmt": "encoder.layer.{}",
    },
    "siglip-b": {
        "hf_name": "google/siglip-base-patch16-224",
        "n_layers": 12,
        "hook_layer_fmt": "vision_model.encoder.layers.{}",
    },
    "siglip-l": {
        "hf_name": "google/siglip-large-patch16-384",
        "n_layers": 24,
        "hook_layer_fmt": "vision_model.encoder.layers.{}",
    },
}

_HF_NAME_TO_CFG = {v["hf_name"]: v for v in BACKBONE_MAP.values()}


def hook_layer_name(backbone_short_name: str, layer: int) -> str:
    """Return the fastsae ``hook_target_name`` for a given (backbone, layer)."""
    if backbone_short_name not in BACKBONE_MAP:
        raise ValueError(
            f"Unknown backbone '{backbone_short_name}'. Known: {list(BACKBONE_MAP)}"
        )
    fmt = BACKBONE_MAP[backbone_short_name]["hook_layer_fmt"]
    return fmt.format(layer)


def final_encoder_layer(hf_name: str) -> str:
    """Return the hook name of the final vision encoder layer for a given HF name."""
    if hf_name not in _HF_NAME_TO_CFG:
        raise ValueError(
            f"Unknown backbone '{hf_name}'. Known HF names: {list(_HF_NAME_TO_CFG)}"
        )
    cfg = _HF_NAME_TO_CFG[hf_name]
    return cfg["hook_layer_fmt"].format(cfg["n_layers"] - 1)


__all__ = ["Backbone", "SAEWrapper", "BACKBONE_MAP", "hook_layer_name",
           "final_encoder_layer"]
