"""Tests for grpo.rewards.factory -- create_reward_function factory."""

import pytest

_SKIP_REASON = "rdkit or other heavy dependencies not available"

try:
    from grpo.rewards.factory import create_reward_function
    from grpo.rewards.base import BaseRewardFunction, DefaultRewardFunction
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON)


class TestCreateRewardFunction:
    """create_reward_function should return a BaseRewardFunction subclass."""

    def test_base_type(self):
        rf = create_reward_function("base")
        assert isinstance(rf, BaseRewardFunction)

    def test_default_type(self):
        rf = create_reward_function("default")
        assert isinstance(rf, DefaultRewardFunction)

    def test_tree_type(self):
        rf = create_reward_function("tree")
        assert isinstance(rf, BaseRewardFunction)
        assert rf.name == "tree_graph"

    def test_planar_type(self):
        rf = create_reward_function("planar")
        assert isinstance(rf, BaseRewardFunction)
        assert rf.name == "planar_graph"

    def test_sbm_type(self):
        rf = create_reward_function("sbm")
        assert isinstance(rf, BaseRewardFunction)
        assert rf.name == "sbm_graph"

    def test_unknown_type_returns_default(self):
        rf = create_reward_function("nonexistent_type_xyz")
        assert isinstance(rf, DefaultRewardFunction)

    def test_case_insensitive(self):
        rf_lower = create_reward_function("tree")
        rf_upper = create_reward_function("TREE")
        assert type(rf_lower) is type(rf_upper)
