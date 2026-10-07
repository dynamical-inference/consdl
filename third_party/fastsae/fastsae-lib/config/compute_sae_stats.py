from dataclasses import dataclass


@dataclass
class StatsConfig:
    sae_path: str
    stats_dataset_name: str
    stats_dataset_seed: int
    stats_dataset_split: str
    n_ref: int
    batch_size: int
    sparsity_thr: list[float]
