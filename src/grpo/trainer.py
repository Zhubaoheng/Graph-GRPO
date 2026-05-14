"""GRPOTrainer -- the hub class that composes all training concerns via mixins.

The original monolithic ``GRPOTrainer`` (3 400 lines) has been split into
focused mixin classes.  This file contains only the constructor, the
``run_epoch`` orchestration loop, and state-dict persistence.  All other
methods live in the mixin files and are inherited through multiple
inheritance.
"""

import logging
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.multiprocessing as mp
from multiprocessing import cpu_count

from grpo.core import GRPOCore
from grpo.trajectory_data import TrajectoryData
from grpo.sampling import SamplingMixin
from grpo.training import TrainingMixin
from grpo.reference_model import ReferenceModelMixin
from grpo.reward_workers import (
    RewardWorkerMixin,
    _set_single_thread_env,
    _reward_worker_initializer,
)
from grpo.graph_conversion import GraphConversionMixin
from grpo.evaluation import EvaluationMixin
from grpo.logging_utils import LoggingMixin

try:
    import swanlab
except ImportError:
    swanlab = None

logger = logging.getLogger(__name__)

Graph = Tuple[torch.Tensor, torch.Tensor]


class GRPOTrainer(
    SamplingMixin,
    TrainingMixin,
    ReferenceModelMixin,
    RewardWorkerMixin,
    GraphConversionMixin,
    EvaluationMixin,
    LoggingMixin,
):
    """
    Flow-GRPO Trainer - manages the complete training pipeline.

    Core responsibilities:
    - Manage the two-stage training flow: sampling phase + training phase
    - Execute backpropagation and parameter updates
    - Manage reference model updates

    Division of labor with GRPOCore:
    - GRPOCore: only handles loss computation (PPO loss, KL regularization, advantage function)
    - GRPOTrainer: handles training flow control (gradient accumulation, backpropagation, optimizer steps)

    Two-stage training flow:
    1. Sampling phase:
       - Batch-generate graph trajectories
       - Compute rewards
       - Store data into GRPOCore buffers

    2. Training phase:
       - Retrieve prepared data from GRPOCore
       - Call GRPOCore.compute_losses to compute losses
       - Execute backpropagation and parameter updates
       - Handle gradient accumulation and clipping
    """

    def __init__(
        self,
        model: nn.Module,
        reward_function: Callable,
        cfg: Dict,
        model_kwargs: dict,
    ):
        """
        Initialize Flow-GRPO trainer.

        Args:
            model: Graph generation model
            reward_function: Reward function
            cfg: Configuration dict
            model_kwargs: Model parameters
        """
        # Set PyTorch determinism for reproducibility
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        _set_single_thread_env()

        self.model = model
        self.reward_function = reward_function
        self.reward_kwargs = self._prepare_reward_kwargs(reward_function, cfg)
        self.cfg = cfg
        self.model_kwargs = model_kwargs
        grpo_config = cfg.grpo

        # Avoid excessive file descriptors when passing large tensors via multiprocessing
        try:
            mp.set_sharing_strategy("file_system")
        except RuntimeError:
            pass

        # Initialize GRPO core algorithm
        self.grpo_core = GRPOCore(cfg)

        # Initialize multiprocessing pool for synchronous reward computation
        self.num_reward_workers = grpo_config.get('num_reward_workers', min(cpu_count(), 8))
        self.reward_worker_threads = int(grpo_config.get("reward_worker_threads", 1))

        reward_type = getattr(reward_function, "name", None) or grpo_config.get("reward_type", "default")

        # Create only one process pool; avoid duplicate pools causing process leaks and CPU waste
        reward_type_lower = reward_type.lower() if isinstance(reward_type, str) else "default"
        self.reward_pool = None
        if self.num_reward_workers > 0 and reward_type_lower != "disabled_reward":
            self.reward_pool = mp.get_context('spawn').Pool(
                processes=self.num_reward_workers,
                initializer=_reward_worker_initializer,
                initargs=(self.reward_worker_threads, reward_type, self.reward_kwargs),
            )
        # Evaluation-phase multiprocessing reward computation timeout (seconds)
        self.eval_timeout_seconds = grpo_config.get('eval_timeout_seconds', 600)

        # Flow-GRPO parameters
        self.num_batches_per_epoch = grpo_config.get('num_batches_per_epoch', 1)
        self.train_batch_size = grpo_config.get('train_batch_size', 32)
        self.group_size = grpo_config.get('group_size', 8)
        self.concurrent_sampling_groups = max(1, grpo_config.get('concurrent_sampling_groups', 1))
        self.num_inner_epochs = grpo_config.get('num_inner_epochs', 1)
        self.gradient_accumulation_steps = grpo_config.get('gradient_accumulation_steps', 1)
        self.sample_group_num = grpo_config.get('sample_group_num', 1000)
        self._next_group_id = 0
        self.eval_interval = max(1, grpo_config.get('eval_interval', 5))

        # Learning rate and optimizer parameters
        self.learning_rate = grpo_config.learning_rate
        self.adam_beta1 = grpo_config.get('adam_beta1', 0.9)
        self.adam_beta2 = grpo_config.get('adam_beta2', 0.999)
        self.adam_weight_decay = grpo_config.get('adam_weight_decay', 1e-4)
        self.adam_epsilon = grpo_config.get('adam_epsilon', 1e-8)


        # Reference model updates
        self.ref_model_update_freq = grpo_config.get('ref_model_update_freq', 200)
        self.beta = grpo_config.kl_penalty

        # Sampling parameters
        default_sample_steps = getattr(cfg.sample, 'sample_steps', 100)
        grpo_forward_steps = grpo_config.get('forward_steps', None)
        self.sample_steps = (
            grpo_forward_steps
            if grpo_forward_steps is not None
            else default_sample_steps
        )
        # Train on all steps by default
        self.target_node_count = grpo_config.get('target_node_count', None)
        # Node count range: when target_node_count is None, use [node_count_min, node_count_max]
        # to restrict the node count range in variable-node sampling mode
        self.node_count_min = grpo_config.get('node_count_min', None)
        self.node_count_max = grpo_config.get('node_count_max', 256)


        # Dynamic node-count distribution (p0-like: global top-reward buffer + smooth update of node_dist.prob).
        self.enable_dynamic_node_dist = bool(grpo_config.get("enable_dynamic_node_dist", False))
        self.dynamic_node_dist_alpha = float(grpo_config.get("dynamic_node_dist_alpha", 0.05))
        self.dynamic_node_dist_reward_threshold = float(grpo_config.get("dynamic_node_dist_reward_threshold", 0.001))
        self._pending_node_count_prob_update = None
        self.lr_decay_threshold = grpo_config.get('lr_decay_threshold', None)
        if self.lr_decay_threshold is not None:
            self.lr_decay_threshold = float(self.lr_decay_threshold)
        self.lr_decay_window = int(grpo_config.get('lr_decay_window', 3) or 3)
        self.lr_decay_factor = float(grpo_config.get('lr_decay_factor', 0.5))
        self.lr_decay_min = grpo_config.get('lr_decay_min', None)
        if self.lr_decay_min is not None:
            self.lr_decay_min = float(self.lr_decay_min)
        self._lr_decay_applied = False
        self._lr_decay_history = []
        # Whether to use the GRPO version of step_probs during sampling/log_prob recomputation (can switch back to the pretrain compute_step_probs)
        self.use_grpo_step_probs_for_sampling = grpo_config.get('use_grpo_step_probs_for_sampling', True)

        # State
        self.global_step = 0
        self.epoch = 0

        # [Dynamic p0] Global Buffer to prevent mode collapse
        # Stores (reward, atom_types, edge_types) for best samples seen so far
        self.global_p0_buffer = []
        self.p0_buffer_size = 1000
        self._pending_p0_update = None

        # Initialize components
        self._initialize_training_components()

        if self.target_node_count is not None:
            logger.info("Node count: %s (fixed)", self.target_node_count)
        else:
            logger.info("Node count: sampled (min=%s, max=%s)", self.node_count_min, self.node_count_max)

    def run_epoch(self, optimizer=None):
        epoch_start_time = time.time()
        benchmark_mode = self.cfg.grpo.get("benchmark_mode", False)

        # ============== Phase 1: Sampling phase ==============
        if benchmark_mode:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_sample_start = time.time()

        self.sampling_phase()

        t_sample_end = time.time()
        sample_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if benchmark_mode else 0

        training_batch = self.grpo_core.prepare_training_batch()
        if training_batch is None:
            logger.warning("No sampled data, skipping training phase")
            self.epoch += 1
            return
        torch.cuda.empty_cache()
        # Print statistics
        if self.grpo_core.stat_tracker:
            avg_group_size, num_configs = self.grpo_core.stat_tracker.get_statistics_summary()

        if benchmark_mode:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        t_train_start = time.time()

        self.training_phase(training_batch, optimizer)

        t_train_end = time.time()
        train_mem_gb = torch.cuda.max_memory_allocated() / 1024**3 if benchmark_mode else 0

        if benchmark_mode:
            N = self.cfg.grpo.get("target_node_count", "?")
            T = self.cfg.grpo.get("forward_steps", "?")
            msg = (
                f"[BENCHMARK] epoch={self.epoch} N={N} T={T} "
                f"sample_time={t_sample_end - t_sample_start:.2f}s "
                f"train_time={t_train_end - t_train_start:.2f}s "
                f"sample_mem={sample_mem_gb:.2f}GB "
                f"train_mem={train_mem_gb:.2f}GB "
                f"total_time={t_train_end - t_sample_start:.2f}s"
            )
            print(msg, flush=True)
            logger.warning(
                "[BENCHMARK] epoch=%d N=%s T=%s sample_time=%.2fs train_time=%.2fs "
                "sample_mem=%.2fGB train_mem=%.2fGB total_time=%.2fs",
                self.epoch, N, T,
                t_sample_end - t_sample_start, t_train_end - t_train_start,
                sample_mem_gb, train_mem_gb,
                t_train_end - t_sample_start,
            )

        torch.cuda.empty_cache()

        # Clear buffers
        self.grpo_core.clear_sample_buffer()

        # Apply Delayed p0 Update (Post-Training)
        if hasattr(self, '_pending_p0_update') and self._pending_p0_update is not None:
             updated_node_dist, updated_edge_dist = self._pending_p0_update
             self.core_model.update_limit_dist(updated_node_dist, updated_edge_dist)
             logger.info("Applied pending p0 update for next epoch.")
             self._pending_p0_update = None

        if hasattr(self, "_pending_node_count_prob_update") and self._pending_node_count_prob_update is not None:
            try:
                self.core_model.update_node_count_dist(self._pending_node_count_prob_update)
                logger.info("Applied pending node_dist update for next epoch.")
            except Exception as e:
                logger.warning("Failed to apply pending node_dist update: %s", e)
            self._pending_node_count_prob_update = None

        # Epoch summary
        epoch_time = time.time() - epoch_start_time
        self.epoch += 1

        # Benchmark early exit
        if benchmark_mode:
            max_ep = self.cfg.grpo.get("benchmark_max_epochs", 0)
            if max_ep > 0 and self.epoch >= max_ep:
                print(f"[BENCHMARK] Reached {max_ep} epochs, stopping.", flush=True)
                raise SystemExit(0)

        self._maybe_decay_lr(optimizer, training_batch)
        self._maybe_run_gdpo_eval()
        # (dynamic node-count curriculum removed)

    def state_dict(self) -> Dict:
        """Return trainer state."""
        state = {
            'global_step': self.global_step,
            'epoch': self.epoch,
        }
        if self.grpo_core.stat_tracker:
            state['stat_tracker_stats'] = self.grpo_core.stat_tracker.stats

        # [Persistence] Save Global p0 Buffer
        if hasattr(self, 'global_p0_buffer') and self.global_p0_buffer:
             state['global_p0_buffer'] = self.global_p0_buffer

        return state

    def load_state_dict(self, state_dict: Dict):
        """Load trainer state."""
        self.global_step = state_dict.get('global_step', self.global_step)
        self.epoch = state_dict.get('epoch', self.epoch)
        if 'stat_tracker_stats' in state_dict and self.grpo_core.stat_tracker:
            self.grpo_core.stat_tracker.stats = state_dict['stat_tracker_stats']

        # [Persistence] Load Global p0 Buffer
        if 'global_p0_buffer' in state_dict:
            self.global_p0_buffer = state_dict['global_p0_buffer']
            logger.info("Restored Global p0 Buffer with %d items.", len(self.global_p0_buffer))

    def __del__(self):
        """Clean up resources."""
        reward_pool = getattr(self, "reward_pool", None)
        if reward_pool is not None:
            reward_pool.close()
            reward_pool.join()
