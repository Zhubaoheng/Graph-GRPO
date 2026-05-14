"""Training-phase mixin for GRPOTrainer.

Contains the gradient-accumulation training loop, minibatch iterators,
batch-parallel loss computation, and learning-rate decay logic.
"""

import logging
import math
import time
from collections import defaultdict
from contextlib import nullcontext
from typing import Dict

import numpy as np
import torch
from torch.cuda.amp import autocast

from grpo.trajectory_data import TrajectoryData

try:
    import swanlab
except ImportError:
    swanlab = None

logger = logging.getLogger(__name__)


class TrainingMixin:
    """Mixin supplying the training-phase methods for GRPOTrainer."""

    # ------------------------------------------------------------------
    # Main training loop
    # ------------------------------------------------------------------

    def training_phase(self, training_batch, optimizer):
        """Streaming training phase: recompute and backpropagate gradients on demand."""
        if training_batch is None:
            logger.warning("training_phase received empty batch, skipping")
            return

        if isinstance(training_batch, TrajectoryData) and training_batch.is_empty():
            logger.warning("training_phase received empty TrajectoryData, skipping")
            return
        self.core_model.eval()
        train_start_time = time.time()
        device = next(self.core_model.parameters()).device
        gradient_accumulation_steps = max(1, self.gradient_accumulation_steps)
        shuffle_batches = self.cfg.grpo.get('shuffle_training_batches', False)
        total_samples = (
            len(training_batch)
            if isinstance(training_batch, TrajectoryData)
            else training_batch["old_log_probs"].shape[0]
        )

        for inner_epoch in range(self.num_inner_epochs):
            logger.info("Inner Epoch %d/%d", inner_epoch + 1, self.num_inner_epochs)
            num_mini_batches = max(1, math.ceil(total_samples / self.train_batch_size))
            epoch_losses = defaultdict(list)
            optimizer.zero_grad(set_to_none=True)
            accumulation_counter = 0
            inner_epoch_start = time.time()
            train_loop_start = time.time()

            if isinstance(training_batch, TrajectoryData):
                mini_batch_iter = self._iter_cpu_minibatches(
                    training_batch, shuffle=shuffle_batches
                )
            else:
                mini_batch_iter = self._iter_dict_minibatches(training_batch)

            def _train_step(cpu_batch, mini_idx, num_mini_batches):
                nonlocal accumulation_counter
                if isinstance(cpu_batch, TrajectoryData):
                    batch_on_device = cpu_batch.to(device, non_blocking=True)
                else:
                    batch_on_device = {
                        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
                        for k, v in cpu_batch.items()
                    }

                use_autocast = device.type == "cuda"
                amp_dtype = torch.bfloat16
                autocast_ctx = autocast(enabled=use_autocast, dtype=amp_dtype) if use_autocast else nullcontext()
                with autocast_ctx:
                    loss_dict = self.grpo_core.compute_losses(
                        self.core_model,
                        batch_on_device,
                        self.reference_model,
                        max_steps=self.cfg.grpo.get("train_max_steps"),
                    )

                scaled_loss = loss_dict["total_loss"] / gradient_accumulation_steps
                scaled_loss.backward()
                accumulation_counter += 1

                for key, value in loss_dict.items():
                    if isinstance(value, torch.Tensor):
                        epoch_losses[key].append(value.detach())

                should_update = (
                    accumulation_counter >= gradient_accumulation_steps
                    or mini_idx == num_mini_batches - 1
                )

                if should_update and optimizer is not None:
                    max_grad_norm = self.cfg.grpo.get('max_grad_norm', 1.0)
                    grad_norm_before_clip = torch.nn.utils.clip_grad_norm_(
                        self.core_model.parameters(),
                        max_grad_norm
                    ).item()

                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                    grad_norm_after_clip = min(grad_norm_before_clip, max_grad_norm)
                    accumulation_counter = 0
                    self.global_step += 1

                    if swanlab is not None and swanlab.run is not None:
                        self._log_training_metrics_to_swanlab(
                            epoch_losses=epoch_losses,
                            grad_norm_before_clip=grad_norm_before_clip,
                            grad_norm_after_clip=grad_norm_after_clip,
                            loss_dict=loss_dict,
                            training_batch=training_batch,
                            optimizer=optimizer
                        )

                    if self.global_step % 10 == 0:
                        recent_losses = {
                            k: torch.stack(v[-10:]).mean().item() if v else 0
                            for k, v in epoch_losses.items()
                        }
                        ratio_mean = recent_losses.get("ratio_mean", 0)
                        clipfrac = recent_losses.get("clipfrac", 0)
                        logger.info(
                            "Step %d: Loss=%.4f, KL=%.4f, Entropy=%.4f, Ratio=%.4f, Clip=%.4f, Grad=%.4f->%.4f",
                            self.global_step,
                            recent_losses.get('total_loss', 0),
                            recent_losses.get('kl_loss', 0),
                            recent_losses.get('policy_entropy', 0),
                            ratio_mean, clipfrac,
                            grad_norm_before_clip, grad_norm_after_clip,
                        )

                    self._maybe_run_evaluation()

            for mini_idx, cpu_batch in enumerate(mini_batch_iter):
                _train_step(cpu_batch, mini_idx, num_mini_batches)

            train_loop_time = time.time() - train_loop_start
            avg_losses = {
                k: torch.stack(v).mean().item() for k, v in epoch_losses.items() if v
            }
            inner_epoch_time = time.time() - inner_epoch_start

            logger.info(
                "Inner Epoch %d completed: Loss=%.4f, Policy=%.4f, Entropy=%.4f, "
                "KL=%.4f, Clipfrac=%.4f, time=%.2fs (train_loop=%.2fs)",
                inner_epoch + 1,
                avg_losses.get('total_loss', 0),
                avg_losses.get('policy_loss', 0),
                avg_losses.get('policy_entropy', 0),
                avg_losses.get('kl_loss', 0),
                avg_losses.get('clipfrac', 0),
                inner_epoch_time, train_loop_time,
            )

        train_time = time.time() - train_start_time
        logger.info("Training phase completed, total time: %.2fs", train_time)

    # ------------------------------------------------------------------
    # Minibatch iterators
    # ------------------------------------------------------------------

    def _iter_cpu_minibatches(self, batch: TrajectoryData, shuffle: bool = False):
        dataset_size = len(batch)
        if dataset_size == 0:
            return

        indices = torch.arange(dataset_size)
        if shuffle:
            perm = torch.randperm(dataset_size)
            indices = indices[perm]

        for start in range(0, dataset_size, self.train_batch_size):
            chunk = indices[start:start + self.train_batch_size]
            if chunk.numel() == 0:
                continue
            yield batch[chunk]

    def _iter_dict_minibatches(self, training_batch: Dict[str, torch.Tensor]):
        total_samples = training_batch["old_log_probs"].shape[0]
        for i in range(0, total_samples, self.train_batch_size):
            end = min(i + self.train_batch_size, total_samples)
            yield {
                k: v[i:end] if isinstance(v, torch.Tensor) else v
                for k, v in training_batch.items()
            }

    # ------------------------------------------------------------------
    # Batch-parallel loss
    # ------------------------------------------------------------------

    def _compute_batch_loss_parallel(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        Batch-parallel loss computation across all timesteps - fully vectorized.
        Uses the new vectorized interface to process all batches and timesteps at once.
        """
        return self.grpo_core.compute_losses(
            model=self.core_model,
            batch_data=batch,
            reference_model=self.reference_model,
            max_steps=None,
        )

    # ------------------------------------------------------------------
    # Learning-rate decay
    # ------------------------------------------------------------------

    def _maybe_decay_lr(self, optimizer, training_batch):
        if optimizer is None or self.lr_decay_threshold is None:
            return

        batch_view = training_batch.as_dict() if isinstance(training_batch, TrajectoryData) else training_batch
        if 'rewards' not in batch_view:
            return

        epoch_rewards = batch_view['rewards']
        if isinstance(epoch_rewards, torch.Tensor):
            epoch_mean = epoch_rewards.mean().item()
        else:
            epoch_mean = float(np.mean(epoch_rewards))

        self._lr_decay_history.append(epoch_mean)
        if len(self._lr_decay_history) > self.lr_decay_window * 2:
            self._lr_decay_history = self._lr_decay_history[-self.lr_decay_window * 2:]

        if self._lr_decay_applied:
            return

        if len(self._lr_decay_history) >= self.lr_decay_window:
            recent_means = self._lr_decay_history[-self.lr_decay_window:]
            all_means_high = all(m >= self.lr_decay_threshold for m in recent_means)
            if all_means_high:
                old_lr = optimizer.param_groups[0]['lr']
                new_lr = old_lr * self.lr_decay_factor
                if self.lr_decay_min is not None:
                    new_lr = max(new_lr, self.lr_decay_min)
                for group in optimizer.param_groups:
                    group['lr'] = new_lr
                self._lr_decay_applied = True
                logger.info(
                    "LR decay triggered: mean_reward >= %.3f (window=%d), lr: %g -> %g",
                    self.lr_decay_threshold, self.lr_decay_window, old_lr, new_lr,
                )
                if swanlab is not None and swanlab.run is not None:
                    swanlab.log({
                        'lr_decay/trigger': 1,
                        'lr_decay/old_lr': old_lr,
                        'lr_decay/new_lr': new_lr,
                        'lr_decay/threshold': self.lr_decay_threshold,
                    }, step=self.global_step)
