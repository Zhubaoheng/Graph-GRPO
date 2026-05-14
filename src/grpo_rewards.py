"""Backward-compatibility shim.

Use ``from grpo.rewards import ...`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'grpo_rewards' is deprecated. "
    "Use 'from grpo.rewards import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.rewards import (
    BaseRewardFunction,
    DefaultRewardFunction,
    GaussianModifier,
    resolve_target_task,
    sascorer,
    PlanarGraphReward,
    SBMGraphReward,
    TreeGraphReward,
    MolecularValidityReward,
    TargetMPOReward,
    TDCOracleReward,
    GDPODockingReward,
    ValsartanSmartsReward,
    create_reward_function,
)

__all__ = [
    "BaseRewardFunction",
    "DefaultRewardFunction",
    "GaussianModifier",
    "resolve_target_task",
    "sascorer",
    "PlanarGraphReward",
    "SBMGraphReward",
    "TreeGraphReward",
    "MolecularValidityReward",
    "TargetMPOReward",
    "TDCOracleReward",
    "GDPODockingReward",
    "ValsartanSmartsReward",
    "create_reward_function",
]
