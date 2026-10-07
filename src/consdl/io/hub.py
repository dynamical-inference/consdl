"""Load the released SAE checkpoints from the Hugging Face Hub.

Every checkpoint of the paper's vision and language tables is one folder

    <modality>/<backbone>_layer<L>/<arch>/x<E>/<level>/seed<S>/

e.g. ``vision/clip-vit-b-16_layer10/consdl/x16/l0_64/seed0``, holding
``config.json``, ``model.safetensors`` and the firing statistics
(``stats.*``) the dictionary metrics read. ``<level>`` is the target L0
(``l0_32`` / ``l0_64`` / ``l0_128``).

    from consdl.io.hub import load_pretrained
    sae = load_pretrained("vision/clip-vit-b-16_layer10/consdl/x16/l0_64/seed0")

To evaluate a whole cell, download it and point the metric CLIs at it:

    path = download("vision/clip-vit-b-16_layer10/consdl/x16/l0_64")
"""
from __future__ import annotations

import os
from pathlib import Path

#: Hub repo holding the checkpoints; ``CONSDL_HF_REPO`` overrides it.
HF_REPO = os.environ.get("CONSDL_HF_REPO", "jjho-choi/consdl")


def download(path_in_repo: str, *, repo_id: str = HF_REPO,
             revision: str | None = None, cache_dir: str | None = None) -> Path:
    """Download one folder (a seed, a cell, or more) and return its local path."""
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    path_in_repo = path_in_repo.strip("/")
    root = snapshot_download(repo_id=repo_id, revision=revision,
                             cache_dir=cache_dir,
                             allow_patterns=[f"{path_in_repo}/**"])
    return Path(root) / path_in_repo


def load_pretrained(path_in_repo: str, *, repo_id: str = HF_REPO,
                    revision: str | None = None, device: str = "cpu"):
    """Download one seed's checkpoint and load it as an SAE module."""
    from consdl.io.checkpoint import load_sae  # noqa: PLC0415

    sae = load_sae(download(path_in_repo, repo_id=repo_id, revision=revision))
    return sae.to(device)
