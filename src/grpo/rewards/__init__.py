"""Reward functions for GRPO training.

All reward classes and the ``create_reward_function`` factory are re-exported
here for convenience::

    from grpo.rewards import create_reward_function, MolecularValidityReward
"""

from grpo.rewards.base import (
    BaseRewardFunction,
    DefaultRewardFunction,
    GaussianModifier,
    resolve_target_task,
    sascorer,
)
from grpo.rewards.graph_rewards import (
    PlanarGraphReward,
    SBMGraphReward,
    TreeGraphReward,
)
from grpo.rewards.molecular_validity import MolecularValidityReward
from grpo.rewards.target_mpo import TargetMPOReward
from grpo.rewards.tdc_oracle import TDCOracleReward
from grpo.rewards.gdpo_docking import GDPODockingReward
from grpo.rewards.valsartan import ValsartanSmartsReward
from grpo.rewards.factory import create_reward_function

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
