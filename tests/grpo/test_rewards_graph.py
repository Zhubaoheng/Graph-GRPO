"""Tests for grpo.rewards.graph_rewards -- TreeGraphReward, PlanarGraphReward, SBMGraphReward."""

import pytest

_SKIP_REASON = "rdkit or other heavy dependencies not available"

try:
    from grpo.rewards.graph_rewards import (
        TreeGraphReward,
        PlanarGraphReward,
        SBMGraphReward,
    )
    from grpo.rewards.base import BaseRewardFunction
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON)


class TestPlanarGraphReward:
    """PlanarGraphReward is the base for graph structure rewards."""

    def test_instantiation(self):
        reward = PlanarGraphReward()
        assert isinstance(reward, BaseRewardFunction)
        assert reward.name == "planar_graph"

    def test_has_fixed_weights(self):
        reward = PlanarGraphReward()
        assert reward.w_valid > 0
        # w_deg and w_clus are 0 when no reference data is provided
        assert reward.w_deg >= 0
        assert reward.w_clus >= 0

    def test_callable(self):
        assert callable(PlanarGraphReward())


class TestSBMGraphReward:
    """SBMGraphReward inherits from PlanarGraphReward."""

    def test_instantiation(self):
        reward = SBMGraphReward()
        assert isinstance(reward, PlanarGraphReward)
        assert reward.name == "sbm_graph"

    def test_is_base_subclass(self):
        assert issubclass(SBMGraphReward, BaseRewardFunction)


class TestTreeGraphReward:
    """TreeGraphReward inherits from PlanarGraphReward."""

    def test_instantiation(self):
        reward = TreeGraphReward()
        assert isinstance(reward, PlanarGraphReward)
        assert reward.name == "tree_graph"

    def test_is_base_subclass(self):
        assert issubclass(TreeGraphReward, BaseRewardFunction)
