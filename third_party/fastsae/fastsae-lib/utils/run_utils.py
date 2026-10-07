from dataclasses import fields
from dataclasses import MISSING
import os

from datasets import load_dataset
from fastsae.config.compute_sae_stats import StatsConfig
from fastsae.config.dotdict import DotDict
from fastsae.config.train import TrainConfig
from fastsae.dataset.clip_hf_dataset import HFDataset
from fastsae.models.sae import VanillaSAE
from fastsae.training import SAETrainer
from fastsae.utils.variables import DATASET_NAME_HF
from fastsae.utils.variables import MODEL_NAME_HF
import torch
from transformers import AutoImageProcessor
from transformers import AutoModel
from transformers import CLIPModel
from transformers import CLIPProcessor
import yaml


def load_data(dataset, processor=None, split="train", shuffle=False, shuffle_seed=1):
    if isinstance(dataset, str):
        dataset_name_hf = DATASET_NAME_HF[dataset]
        dataset = load_dataset(dataset_name_hf, cache_dir=os.getenv("HF_HOME"), split=split)
    if shuffle:
        dataset = dataset.shuffle(seed=shuffle_seed)
    clip_hf_dataset = HFDataset(dataset, processor) if processor is not None else dataset

    return clip_hf_dataset


def load_train_config(path: str, device: str) -> TrainConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    configs = TrainConfig(**cfg_dict)
    configs.device = device
    return configs


def load_model_and_processor(cfg):
    if hasattr(cfg, "model_name_hf"):
        model_name_hf = cfg.model_name_hf
    else:
        model_name_hf = MODEL_NAME_HF[cfg.model_type]  # legacy

    if "clip" in cfg.model_type:
        model = CLIPModel.from_pretrained(model_name_hf)
        processor = CLIPProcessor.from_pretrained(model_name_hf)
    elif "dino" in cfg.model_type:
        processor = AutoImageProcessor.from_pretrained(model_name_hf)
        model = AutoModel.from_pretrained(model_name_hf)
    else:
        raise ValueError(f"Model {cfg.model_name} not supported")

    model.eval()
    model.to(cfg.device)

    return model, processor


def get_len_dataset(dataset):
    """Get the length of the dataset."""
    if hasattr(dataset, "num_rows"):
        if dataset.num_rows > 0:
            return dataset.num_rows
    else:
        return len(dataset["image"])


def parse_sae_name(sae_path):
    path_names = sae_path.split("/")
    if "checkpoints_raw" in path_names:
        checkpoints_root_dir = "checkpoints_raw"
        _, run_name, sae_name, train_dataset_name, train_step, f_name = sae_path.split(checkpoints_root_dir)[-1].split("/")
    else:
        checkpoints_root_dir = "checkpoints"
        _, sae_name, train_dataset_name, f_name = sae_path.split(checkpoints_root_dir)[-1].split("/")

    return sae_name


def load_stat_config(path: str) -> StatsConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
    return StatsConfig(**cfg_dict)


def fill_missing_sae_cfg(sae_cfg, **kwargs):
    for f in fields(TrainConfig):
        if f.name not in sae_cfg:
            if f.default is not MISSING:
                sae_cfg[f.name] = f.default
            elif f.name in kwargs:
                sae_cfg[f.name] = kwargs[f.name]
            else:
                if f.name == "d_sae":
                    sae_cfg[f.name] = sae_cfg.d_in * sae_cfg.expansion_factor
                elif f.name == "run_name":
                    sae_cfg[f.name] = sae_cfg.checkpoint_path
                elif f.name == "model_name_hf":
                    sae_cfg[f.name] = MODEL_NAME_HF[sae_cfg.model_type]
                else:
                    raise ValueError(f"Missing required config field: {f.name}")
    return sae_cfg


def load_sae_by_path(sae_path, device):
    assert sae_path.endswith(".pt")

    loaded = torch.load(sae_path, map_location="cpu")

    sae_cfg = DotDict(loaded["config"])
    sae_cfg["device"] = device
    sae_cfg.print_all()

    sae_name = parse_sae_name(sae_path)
    model_type, hook_module_layer = sae_name.split(".")
    kwargs = {"model_type": model_type, "hook_module_layer": hook_module_layer}
    sae_cfg = fill_missing_sae_cfg(sae_cfg, **kwargs)

    if sae_cfg.sae_type == "base":
        sae = VanillaSAE(sae_cfg)

    else:
        raise ValueError(f"Invalid SAE type: {sae_cfg.sae_type}")

    sae.load_state_dict(loaded["state_dict"])
    sae.to(sae_cfg.device).eval()

    return sae


def load_sae_and_trainer(cfg):
    if cfg.sae_type == "base":
        sae = VanillaSAE(cfg)
        trainer = SAETrainer
    else:
        raise ValueError(f"Invalid SAE type: {cfg.sae_type}")

    return sae.train(), trainer
