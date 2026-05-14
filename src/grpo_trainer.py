"""Backward-compatibility shim.

Use ``from grpo.trainer import GRPOTrainer`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'grpo_trainer' is deprecated. "
    "Use 'from grpo.trainer import GRPOTrainer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.trainer import GRPOTrainer
from grpo.reward_workers import (
    _set_single_thread_env,
    _reward_worker_initializer,
    _compute_batch_rewards_worker,
    RewardWorkerMixin,
)

__all__ = [
    "GRPOTrainer",
    "_set_single_thread_env",
    "_reward_worker_initializer",
    "_compute_batch_rewards_worker",
    "RewardWorkerMixin",
]
