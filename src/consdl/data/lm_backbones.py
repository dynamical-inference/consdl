"""Registry of LM backbones for SAE training (``LM_BACKBONE_MAP``).

Each entry pins:
  * ``hf_name``         — HF repo id passed to
    ``AutoModelForCausalLM.from_pretrained``.
  * ``n_layers``        — total transformer blocks (informational).
  * ``d_in``            — residual-stream dimension; the SAE's input
    dimension matches this.
  * ``default_layer``   — default hook layer.
  * ``hook_layer_fmt``  — ``str.format`` template for the
    transformer-block submodule path that
    :class:`consdl.cli.train_lm.LMBackbone` resolves
    via ``model.<path>`` to register the forward hook.
"""

from __future__ import annotations


LM_BACKBONE_MAP: dict[str, dict] = {
    "pythia-70m": {
        "hf_name": "EleutherAI/pythia-70m-deduped",
        "n_layers": 6, "d_in": 512, "default_layer": 3,
        "hook_layer_fmt": "gpt_neox.layers.{}",
    },
    "pythia-160m": {
        "hf_name": "EleutherAI/pythia-160m-deduped",
        "n_layers": 12, "d_in": 768, "default_layer": 8,
        "hook_layer_fmt": "gpt_neox.layers.{}",
    },
    # Non-deduped Pythia-160m, the base model of the iSAE paper
    # (arXiv:2605.31245); default_layer=11 is that paper's "layer 12"
    # (0-indexed).
    "pythia-160m-nondedup": {
        "hf_name": "EleutherAI/pythia-160m",
        "n_layers": 12, "d_in": 768, "default_layer": 11,
        "hook_layer_fmt": "gpt_neox.layers.{}",
    },
    "pythia-410m": {
        "hf_name": "EleutherAI/pythia-410m-deduped",
        "n_layers": 24, "d_in": 1024, "default_layer": 12,
        "hook_layer_fmt": "gpt_neox.layers.{}",
    },
    "gpt2-small": {
        "hf_name": "openai-community/gpt2",
        "n_layers": 12, "d_in": 768, "default_layer": 8,
        "hook_layer_fmt": "transformer.h.{}",
    },
    "gemma-2-2b": {
        "hf_name": "google/gemma-2-2b",
        "n_layers": 26, "d_in": 2304, "default_layer": 12,
        "hook_layer_fmt": "model.layers.{}",
    },
}


__all__ = ["LM_BACKBONE_MAP"]
