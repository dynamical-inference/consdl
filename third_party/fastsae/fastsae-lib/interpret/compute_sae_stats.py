import json
import os
from typing import Dict

from fastsae.utils.run_utils import get_len_dataset
import numpy as np
import torch
from tqdm import tqdm


class StatComputer:
    def __init__(self, sae, backbone_model, dataloader, stat_cfg):
        self.sae = sae
        self.cfg = sae.cfg
        self.backbone_model = backbone_model
        self.dataloader = dataloader
        self.stat_cfg = stat_cfg
        self.model_type = self.cfg.model_type

    def forward_model(self, inputs):
        inputs = inputs.to(self.cfg.device)
        if "clip" in self.model_type:
            return self.backbone_model.vision_model(inputs)
        elif "dino" in self.model_type:
            return self.backbone_model(inputs)
        else:
            raise NotImplementedError

    def get_hooked_module(self):
        if "clip" in self.model_type:
            if self.cfg.hook_module_name == "resid":
                return self.backbone_model.vision_model.encoder.layers[self.cfg.hook_module_layer]
            else:
                raise NotImplementedError
        elif "dino" in self.model_type:
            if self.cfg.hook_module_name == "resid":
                return self.backbone_model.encoder.layer[self.cfg.hook_module_layer]
        else:
            raise NotImplementedError

    def get_sae_inputs(self, clip_inputs):
        hook_component = self.cfg.hook_module_name + "." + str(self.cfg.hook_module_layer)
        module = self.get_hooked_module()

        hooked_output_dict = {}

        def hook(
            module: torch.nn.Module,
            _input,
            outputs,
        ):
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            hooked_output_dict[hook_component] = outputs

        handles = [module.register_forward_hook(hook)]

        try:
            with torch.no_grad():
                self.forward_model(clip_inputs)
        finally:
            for handle in handles:
                handle.remove()

        return hooked_output_dict[hook_component]

    def r2_score(self, x: torch.Tensor, x_hat: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
        """
        https://github.com/KempnerInstitute/overcomplete/blob/main/overcomplete/metrics.py#L316
        Compute the R^2 score (coefficient of determination) for the reconstruction.
        A score of 1 indicates a perfect reconstruction while a score of 0 indicates
        that the reconstruction is as good as the mean.

        """
        assert x.shape == x_hat.shape, "Input tensors must have the same shape"
        assert len(x.shape) == 2, "Input tensors must be 2D"

        ss_res = torch.mean((x - x_hat) ** 2)
        ss_tot = torch.mean((x - x.mean()) ** 2)
        r2 = 1 - (ss_res / (ss_tot + eps))
        return r2

    def compute_stats(self):
        n_iter_samples = 0
        torch.set_float32_matmul_precision("high")

        stats = {
            "mean_act": torch.zeros([self.sae.cfg.d_sae]),
            "sparsity": torch.zeros([self.sae.cfg.d_sae]),
            "sparsity_thr": {thr: torch.zeros([self.sae.cfg.d_sae]) for thr in self.stat_cfg.sparsity_thr},
            "top_act": torch.zeros([self.stat_cfg.n_ref, self.sae.cfg.d_sae]),
            "top_idx": torch.zeros([self.stat_cfg.n_ref, self.sae.cfg.d_sae]),
            "l2_loss": 0.0,
            "r2_score": 0.0,
        }

        total_iter_samples = get_len_dataset(self.dataloader.dataset.dataset)
        n_batch = 0
        pbar = tqdm(total=total_iter_samples, desc="Computing SAE statistics")
        try:
            dataloader_iter = iter(self.dataloader)
            while n_iter_samples < total_iter_samples:
                try:
                    clip_inputs = next(dataloader_iter)
                except StopIteration:
                    break
                sae_inputs = self.get_sae_inputs(clip_inputs)
                sae_inputs = sae_inputs.to(self.sae.cfg.device)

                with torch.no_grad():
                    all_act = self.sae.encode(sae_inputs)  # (batch_size, token_length, sae_dim)
                    recons = self.sae.decode(all_act)  # (batch, tokens, input_dim)
                    avg_act = all_act.mean(1).cpu()  # (batch_size, sae_dim)

                    x_flat = sae_inputs.reshape(-1, sae_inputs.shape[-1])
                    xhat_flat = recons.reshape(-1, recons.shape[-1])
                    r2_val = self.r2_score(x_flat, xhat_flat).item()
                    stats["r2_score"] += r2_val
                    n_batch += 1

                    l2_loss = torch.nn.functional.mse_loss(recons, sae_inputs, reduction="none")
                    l2_loss = l2_loss.mean(dim=(1, 2)).sum().item()  # per sample MSE, summed over batch
                    stats["l2_loss"] += l2_loss

                batch_size = avg_act.size(0)
                stats["mean_act"] += avg_act.sum(dim=0)  # sum over images; (sae_dim, )
                stats["sparsity"] += (avg_act > 0).sum(dim=0)  # sum over images; (sae_dim, )
                for thr in self.stat_cfg.sparsity_thr:
                    stats["sparsity_thr"][thr] += (avg_act > thr).sum(dim=0)

                top_k = min(self.stat_cfg.n_ref, avg_act.size(0))
                batch_act, batch_idx = torch.topk(avg_act, k=top_k, dim=0)  # batch_act (top_k, sae_dim)
                batch_idx += n_iter_samples

                total_act = torch.cat([stats["top_act"], batch_act], dim=0)
                total_idx = torch.cat([stats["top_idx"], batch_idx], dim=0)

                top_act, idx_of_idx = torch.topk(total_act, k=self.stat_cfg.n_ref, dim=0)
                top_idx = torch.gather(total_idx, 0, idx_of_idx)

                stats["top_act"] = top_act
                stats["top_idx"] = top_idx

                n_iter_samples += batch_size

                pbar.set_description(f"[{n_iter_samples}/{total_iter_samples}] images processed")
                pbar.update(batch_size)

                del sae_inputs, all_act
        finally:
            stats["mean_act"] /= stats["sparsity"].clamp(min=1)
            stats["sparsity"] /= n_iter_samples
            stats["top_act"] = stats["top_act"].transpose(0, 1)
            stats["top_idx"] = stats["top_idx"].transpose(0, 1)
            stats["l2_loss"] /= n_iter_samples
            stats["r2_score"] /= n_batch

            for thr in self.stat_cfg.sparsity_thr:
                stats["sparsity_thr"][thr] /= n_iter_samples
                stats[f"l0_{thr}"] = stats["sparsity_thr"][thr].mean().item()

        return stats

    def calculate_entropy(self, top_act, top_label, ignore_label_idx: int = None, eps=1e-9):
        dict_size = top_label.shape[0]
        entropy = torch.zeros(dict_size)

        top_label = torch.as_tensor(top_label)

        for i in range(dict_size):
            unique_labels, counts = top_label[i].unique(return_counts=True)
            if ignore_label_idx is not None:
                counts = counts[unique_labels != ignore_label_idx]
                unique_labels = unique_labels[unique_labels != ignore_label_idx]
            if len(unique_labels) != 0:
                if counts.sum().item() < 10:
                    entropy[i] = -1  # discount as too few datapoints!
                else:
                    summed_probs = torch.zeros_like(unique_labels, dtype=top_act.dtype)
                    for j, label in enumerate(unique_labels):
                        summed_probs[j] = top_act[i][top_label[i] == label].sum().item()
                    summed_probs = summed_probs / summed_probs.sum()
                    entropy[i] = -torch.sum(summed_probs * torch.log(summed_probs + eps))
            else:
                entropy[i] = -1
        return entropy

    def save_results(
        self,
        save_directory: str,
        stats: Dict[str, torch.Tensor],
        dataset_labels_np: np.array = None,
    ) -> None:
        torch.save(stats["sparsity"].detach().cpu(), f"{save_directory}/sparsity.pt")
        torch.save(stats["mean_act"].detach().cpu(), f"{save_directory}/mean_act.pt")

        top_idx = stats["top_idx"].detach().cpu()
        top_act = stats["top_act"].detach().cpu()
        zero_mask = top_act <= 0

        dataset_labels_np = np.array(self.dataloader.dataset.label)
        top_label = dataset_labels_np[top_idx.numpy().astype(int)]
        top_label = torch.as_tensor(top_label)
        top_label[zero_mask] = -1
        torch.save(top_label, f"{save_directory}/top_label.pt")

        top_entropy = self.calculate_entropy(top_act, top_label, ignore_label_idx=-1)
        torch.save(top_entropy.detach().cpu(), f"{save_directory}/top_entropy.pt")

        top_idx[zero_mask] = -1
        top_act[zero_mask] = -1

        torch.save(top_idx, f"{save_directory}/top_idx.pt")
        torch.save(top_act, f"{save_directory}/top_act.pt")

        out_dict = {
            "l2_loss": round(stats["l2_loss"], 4),
            "r2_score": round(stats["r2_score"], 4),
            "l0_0": round(stats["sparsity"].mean().item(), 4),
        }
        for thr in self.stat_cfg.sparsity_thr:
            out_dict[f"l0_{thr}"] = round(stats["sparsity_thr"][thr].mean().item(), 4)

        with open(f"{save_directory}/stats.json", "w") as f:
            json.dump(out_dict, f, indent=4)

        print(f"Results saved to {save_directory}")

    def run(self):
        stats = self.compute_stats()

        sae_stats_path = self.stat_cfg.sae_path.replace(
            "model_with_config.pt",
            f"stats.{self.stat_cfg.stats_dataset_name}.seed.{self.stat_cfg.stats_dataset_seed}",
        )
        os.makedirs(sae_stats_path, exist_ok=False)
        self.save_results(sae_stats_path, stats)
