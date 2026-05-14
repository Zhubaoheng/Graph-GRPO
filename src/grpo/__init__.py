"""GRPO (Group Relative Policy Optimization) for graph generation.

This package organizes the Flow-GRPO training pipeline into logical modules:

- ``core``: Core GRPO algorithm (advantage computation, PPO loss, stat tracking)
- ``trainer``: GRPOTrainer orchestrating sampling and training phases
- ``rewards``: Reward function hierarchy (molecular, graph-structure, docking, etc.)
- ``lightning_module``: PyTorch Lightning wrappers
- ``eval_sampler``: mol_opt evaluation interface (GraphGRPOProposer)
- ``eval_docking``: GDPO docking evaluation utilities
- ``train_utils``: Dataset/model construction helpers for the Hydra entry point
"""

from grpo.core import GRPOCore, PerGraphStatTracker
from grpo.trajectory_data import TrajectoryData
from grpo.trainer import GRPOTrainer
from grpo.rewards import create_reward_function
from grpo.lightning_module import (
    GRPOLightningModule,
    FlowGRPODataModule,
    create_grpo_lightning_module,
)

__all__ = [
    "GRPOCore",
    "PerGraphStatTracker",
    "TrajectoryData",
    "GRPOTrainer",
    "create_reward_function",
    "GRPOLightningModule",
    "FlowGRPODataModule",
    "create_grpo_lightning_module",
]
