from dataclasses import dataclass
from dataclasses import field

import torch
import wandb

str_to_dtype = {
    "float32": torch.float32,
    "float": torch.float,  # alias
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.double,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "int16": torch.int16,
    "short": torch.short,
    "int32": torch.int32,
    "int": torch.int,
    "int64": torch.int64,
    "long": torch.long,
    "bool": torch.bool,
}


@dataclass
class TrainConfig:
    """
    Configuration for training a sparse autoencoder on a vision transformer.
    """

    # backbone architecture
    model_type: str
    model_name_hf: str
    # SAE architecture

    d_in: int
    expansion_factor: int
    b_dec_init_method: str = "geometric_median"
    activation: str = "relu"
    sae_type: str = "base"

    # SAE input
    class_token: bool = False  # If you want to train SAE ONLY using CLS token, then True. PatchSAE uses False as default
    hook_module_name: str = "resid"
    hook_module_layer: int = -2

    # training dataset
    training_dataset_name: str = "imagenet"

    # training steps
    total_training_samples: int = 2_000_000

    # training parameters
    l1_coefficient: float = 0.00008
    lr: float = 0.0004
    lr_scheduler_name: str = "constantwithwarmup"  # constant, constantwithwarmup, linearwarmupdecay, cosineannealing, cosineannealingwarmup
    lr_warm_up_steps: int = 500
    batch_size: int = 16

    top_k: int = 5

    # resample protocol
    use_ghost_grads: bool = True
    latent_sampling_window: int = 2000  # May need to change this since by default I will use ghost grads
    latent_sampling_method: str = "anthropic"  # None or Anthropic
    resample_batches: int = 32
    latent_reinit_scale: float = 0.2
    dead_latent_window: int = 1000  # unless this window is larger latent sampling,
    dead_latent_estimation_method: str = "no_fire"
    dead_latent_threshold: float = 1e-8

    # WANDB
    log_to_wandb: bool = True
    wandb_project: str = "patchsae"
    wandb_entity: str = None
    wandb_log_frequency: int = 10

    # Misc
    n_checkpoints: int = 0
    checkpoint_path: str = "checkpoints"
    device: str = "cuda"
    dtype: str = "float32"

    # Post init attributes
    d_sae: int = field(init=False)
    run_name: str = field(init=False)

    def __post_init__(self):
        self.d_sae = self.d_in * self.expansion_factor
        self.device = torch.device(self.device)
        self.dtype = str_to_dtype[self.dtype]

        unique_id = wandb.util.generate_id()
        self.run_name = unique_id
        self.checkpoint_path = f"{self.checkpoint_path}/{unique_id}/{self.model_type}.{self.hook_module_layer}/{self.training_dataset_name}"
        print(f"Checkpoints will be saved at {self.checkpoint_path}")

        total_training_steps = self.total_training_samples // self.batch_size
        print(f"Total training steps: {total_training_steps}")

        total_wandb_updates = total_training_steps // self.wandb_log_frequency
        print(f"Total wandb updates: {total_wandb_updates}")
