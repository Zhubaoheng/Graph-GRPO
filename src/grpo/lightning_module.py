"""GRPO PyTorch Lightning module implementing the two-stage Flow-GRPO training architecture."""

import logging
import os
import time
from collections import defaultdict
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig

try:
    import swanlab
except ImportError:
    swanlab = None

from grpo.trainer import GRPOTrainer
from grpo.rewards import create_reward_function
from graph_discrete_flow_model import GraphDiscreteFlowModel
from grpo.core import GRPOCore
import utils

logger = logging.getLogger(__name__)


class GRPOLightningModule(pl.LightningModule):
    """Flow-GRPO PyTorch Lightning module.

    Key features:
        1. Two-stage training: sampling phase + training phase.
        2. Per-configuration statistic tracking.
        3. Reference model soft-update support.
        4. Flow-GRPO compatible gradient accumulation strategy.
    """

    def __init__(
        self,
        cfg: DictConfig,
        datamodule,
        model_kwargs,
        total_steps: int,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(cfg)

        self.cfg = cfg
        self.datamodule = datamodule
        self.model_kwargs = model_kwargs
        self.total_steps = total_steps

        self.model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

        # Initialized lazily in on_fit_start (after device placement).
        self.grpo_trainer = None
        self.reward_function = None

        self.num_batches_per_epoch = cfg.grpo.get('num_batches_per_epoch', 10)
        self.num_inner_epochs = cfg.grpo.get('num_inner_epochs', 1)
        self.train_batch_size = cfg.grpo.get('train_batch_size', 32)

        self.gradient_accumulation_steps = cfg.grpo.gradient_accumulation_steps

        # GRPOTrainer handles optimizer steps manually.
        self.automatic_optimization = False

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load state dict with compatibility handling for different checkpoint sources."""
        metadata = getattr(state_dict, "_metadata", None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        # Remap bare sampling_metrics.* keys to model.sampling_metrics.*
        stray_sampling_keys = [k for k in state_dict.keys() if k.startswith("sampling_metrics.")]
        if stray_sampling_keys:
            for key in stray_sampling_keys:
                new_key = f"model.{key}"
                if new_key not in state_dict:
                    state_dict[new_key] = state_dict[key]
                state_dict.pop(key, None)

        # Ensure sampling statistics match the current dataset shapes
        sampling_metrics = getattr(self.model, "sampling_metrics", None)
        if sampling_metrics is not None:
            current_sampling_state = sampling_metrics.state_dict()
            for key, tensor in current_sampling_state.items():
                full_key = f"model.sampling_metrics.{key}"
                if full_key not in state_dict or state_dict[full_key].shape != tensor.shape:
                    state_dict[full_key] = tensor.detach().cpu()

        # Inject p0_node_dist / p0_edge_dist if missing (pre-trained checkpoint compat)
        if "model.p0_node_dist" not in state_dict and hasattr(self.model, "p0_node_dist"):
            state_dict["model.p0_node_dist"] = self.model.p0_node_dist.detach().cpu()

        if "model.p0_edge_dist" not in state_dict and hasattr(self.model, "p0_edge_dist"):
            state_dict["model.p0_edge_dist"] = self.model.p0_edge_dist.detach().cpu()

        # Inject node_count buffers if missing or shape-mismatched
        if hasattr(self.model, "node_count_prob"):
            key = "model.node_count_prob"
            if key not in state_dict or state_dict[key].shape != self.model.node_count_prob.shape:
                state_dict[key] = self.model.node_count_prob.detach().cpu()

        for buf_name in ("node_count_buffer_rewards", "node_count_buffer_nodes", "node_count_buffer_filled"):
            full_key = f"model.{buf_name}"
            if hasattr(self.model, buf_name):
                current = getattr(self.model, buf_name)
                if full_key not in state_dict or state_dict[full_key].shape != current.shape:
                    state_dict[full_key] = current.detach().cpu()

        return super().load_state_dict(state_dict, strict=strict)

    def _get_forward_steps(self) -> int:
        """Resolve how many forward/inference steps GRPO should run."""
        sample_cfg = getattr(self.cfg, "sample", None)
        default_steps = getattr(sample_cfg, "sample_steps", 100) if sample_cfg is not None else 100
        grpo_steps = self.cfg.grpo.get("forward_steps", None)
        return grpo_steps if grpo_steps is not None else default_steps

    def setup(self, stage: str) -> None:
        """Lightning setup — defer GRPO init to on_fit_start (after GPU placement)."""
        if stage == "fit":
            self._setup_completed = False
            self._restored_trainer_state = None

    def _sync_model_distributions_from_buffers(self) -> None:
        """Sync GraphDiscreteFlowModel internal distributions (p0 / node_count) with loaded buffers.

        In the GRPO scenario the flow model is a sub-module, so its own
        ``on_load_checkpoint`` is never called.  This explicit sync avoids
        stale limit_dist / node_dist after resuming.
        """
        try:
            if (
                hasattr(self.model, "update_limit_dist")
                and hasattr(self.model, "p0_node_dist")
                and hasattr(self.model, "p0_edge_dist")
            ):
                self.model.update_limit_dist(self.model.p0_node_dist, self.model.p0_edge_dist)
            if hasattr(self.model, "update_node_count_dist") and hasattr(self.model, "node_count_prob"):
                self.model.update_node_count_dist(self.model.node_count_prob)
        except Exception as e:
            logger.warning("[Resume] Failed to sync model distributions from buffers: %s", e)

    def on_fit_start(self) -> None:
        """Initialize GRPO components once the model is on the correct device."""
        if not hasattr(self, '_setup_completed') or not self._setup_completed:
            # Sync p0/node_count before creating GRPOTrainer (which copies the reference model).
            self._sync_model_distributions_from_buffers()

            # 1. Create reward function
            model_device = next(self.model.parameters()).device

            ref_metrics = None
            if hasattr(self.model, 'dataset_info') and hasattr(self.model.dataset_info, 'ref_metrics'):
                ref_metrics = self.model.dataset_info.ref_metrics

            self.reward_function = create_reward_function(
                reward_type=self.cfg.grpo.reward_type,
                cfg=self.cfg,
                device=model_device,
                datamodule=self.datamodule,
                model=self.model,
                ref_metrics=ref_metrics,
                name=f"grpo_{self.cfg.grpo.reward_type}",
            )

            # 2. Create Flow-GRPO trainer
            self.grpo_trainer = GRPOTrainer(
                model=self.model,
                reward_function=self.reward_function,
                cfg=self.cfg,
                model_kwargs=self.model_kwargs,
            )

            # Apply restored trainer state if available (from checkpoint resume)
            if hasattr(self, '_restored_trainer_state') and self._restored_trainer_state is not None:
                logger.info("Applying restored GRPO trainer state...")
                self.grpo_trainer.load_state_dict(self._restored_trainer_state)
                self._restored_trainer_state = None

            self._setup_completed = True

    def configure_optimizers(self):
        """Configure AdamW optimizer with optional linear warmup scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.cfg.grpo.learning_rate,
            betas=(
                self.cfg.grpo.get('adam_beta1', 0.9),
                self.cfg.grpo.get('adam_beta2', 0.999),
            ),
            weight_decay=self.cfg.grpo.get('adam_weight_decay', 1e-4),
            eps=self.cfg.grpo.get('adam_epsilon', 1e-8),
        )

        warmup_steps = self.cfg.grpo.get('warmup_steps', 0)
        use_scheduler = self.cfg.grpo.get('use_lr_scheduler', False) or (warmup_steps > 0)

        if use_scheduler:
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }

        return optimizer

    def training_step(self, batch, batch_idx):
        """Run one Flow-GRPO epoch (sampling + policy update).

        The actual training logic lives in ``GRPOTrainer.run_epoch``.
        Returns a dummy loss tensor for Lightning compatibility.
        """
        if self.grpo_trainer is None:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        try:
            optimizer = self.optimizers()
            self.grpo_trainer.run_epoch(optimizer=optimizer)

            # Save intermediate checkpoints at configured intervals
            save_every_n_epoch = self.cfg.grpo.get("save_every_n_epoch", 0)
            if (
                self.trainer is not None
                and self.grpo_trainer is not None
                and save_every_n_epoch > 0
            ):
                current_epoch = self.grpo_trainer.epoch
                if current_epoch > 0 and current_epoch % save_every_n_epoch == 0:
                    last_saved_epoch = getattr(self, "_last_saved_epoch", None)
                    if last_saved_epoch != current_epoch:
                        checkpoint_dir = f"{os.getcwd()}/checkpoint_folder"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        checkpoint_filename = f"flow_grpo_epoch{current_epoch:04d}.ckpt"
                        checkpoint_path = os.path.join(checkpoint_dir, checkpoint_filename)

                        logger.info("Saving Flow-GRPO checkpoint: epoch %d -> %s", current_epoch, checkpoint_path)
                        self.trainer.save_checkpoint(checkpoint_path)
                        self._last_saved_epoch = current_epoch

            return torch.tensor(0.0, device=self.device, requires_grad=True)

        except Exception as e:
            logger.error("Flow-GRPO training step failed: %s", e, exc_info=True)
            return torch.tensor(0.0, device=self.device, requires_grad=True)

    def on_train_epoch_end(self):
        """Log training statistics at the end of each epoch."""
        if self.grpo_trainer:
            metrics = {}

            if hasattr(self.grpo_trainer.grpo_core, 'stat_tracker') and self.grpo_trainer.grpo_core.stat_tracker:
                avg_group_size, num_configs = self.grpo_trainer.grpo_core.stat_tracker.get_statistics_summary()
                metrics['stat_tracker/avg_group_size'] = avg_group_size
                metrics['stat_tracker/num_configs'] = num_configs

            metrics['training/global_step'] = self.grpo_trainer.global_step
            metrics['training/epoch'] = self.grpo_trainer.epoch

            try:
                if swanlab is not None and swanlab.run is not None:
                    swanlab.log(metrics, step=self.grpo_trainer.global_step)
                else:
                    self.log_dict(metrics, on_step=False, on_epoch=True)
            except Exception:
                self.log_dict(metrics, on_step=False, on_epoch=True)

    def validation_step(self, batch, batch_idx):
        """No-op — GRPO uses reward signal instead of traditional validation."""
        pass

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Persist GRPO trainer state alongside the model checkpoint."""
        if self.grpo_trainer:
            checkpoint["grpo_trainer_state"] = self.grpo_trainer.state_dict()
            checkpoint["grpo_epoch"] = self.grpo_trainer.epoch
            checkpoint["grpo_global_step"] = self.grpo_trainer.global_step

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        """Restore GRPO trainer state from the checkpoint."""
        if self.grpo_trainer:
            if "grpo_epoch" in checkpoint:
                self.grpo_trainer.epoch = checkpoint["grpo_epoch"]
            if "grpo_global_step" in checkpoint:
                self.grpo_trainer.global_step = checkpoint["grpo_global_step"]
            if "grpo_trainer_state" in checkpoint:
                self.grpo_trainer.load_state_dict(checkpoint["grpo_trainer_state"])
        else:
            # Store state for deferred initialization in on_fit_start.
            restored = None
            if "grpo_trainer_state" in checkpoint and checkpoint["grpo_trainer_state"] is not None:
                restored = checkpoint["grpo_trainer_state"]
            elif "grpo_epoch" in checkpoint or "grpo_global_step" in checkpoint:
                restored = {
                    "epoch": checkpoint.get("grpo_epoch", 0),
                    "global_step": checkpoint.get("grpo_global_step", 0),
                }
            if restored is not None:
                self._restored_trainer_state = restored

    def forward(self, *args, **kwargs):
        """Delegate to the inner GraphDiscreteFlowModel."""
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def sample_graphs_and_evaluate_rewards(self, num_samples: int = 32) -> Dict[str, float]:
        """Sample graphs and compute reward statistics for evaluation.

        Args:
            num_samples: Number of graphs to sample.

        Returns:
            Dictionary of reward statistics.
        """
        if self.grpo_trainer is None:
            return {}

        self.model.eval()

        graphs, node_mask, *_ = self.grpo_trainer.sample_graphs_with_trajectory_tracking(
            batch_size=num_samples,
            seed=42,
            total_inference_steps=self._get_forward_steps(),
        )

        graph_list = self.grpo_trainer._convert_placeholder_to_graph_list(graphs, node_mask)
        rewards = self.reward_function(graph_list)

        return {
            'eval/reward_mean': rewards.mean().item(),
            'eval/reward_std': rewards.std().item(),
            'eval/reward_min': rewards.min().item(),
            'eval/reward_max': rewards.max().item(),
        }


class FlowGRPODataModule(pl.LightningDataModule):
    """Dummy data module for Flow-GRPO.

    Flow-GRPO does not use real data from a dataloader; this module provides
    placeholder tensors to satisfy the Lightning Trainer interface.
    """

    def __init__(self, num_epochs: int = 100, batch_size: int = 1):
        super().__init__()
        self.num_epochs = num_epochs
        self.batch_size = batch_size

    def setup(self, stage: str = None):
        self.dummy_data = torch.ones(self.num_epochs, 1, dtype=torch.float32)

    def train_dataloader(self):
        from torch.utils.data import DataLoader, TensorDataset
        dataset = TensorDataset(self.dummy_data)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def val_dataloader(self):
        return self.train_dataloader()


def create_grpo_lightning_module(
    cfg: DictConfig,
    model_kwargs: dict,
    datamodule,
    total_steps: int,
) -> GRPOLightningModule:
    """Create and return an initialized GRPO Lightning module.

    Args:
        cfg: Hydra configuration object.
        model_kwargs: Keyword arguments for model initialization.
        datamodule: Data module instance.
        total_steps: Total number of training steps.

    Returns:
        Initialized GRPOLightningModule.
    """
    module = GRPOLightningModule(
        cfg=cfg,
        datamodule=datamodule,
        model_kwargs=model_kwargs,
        total_steps=total_steps,
    )

    logger.info("GRPO Lightning module created successfully")
    return module
