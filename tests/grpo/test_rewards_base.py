"""Tests for grpo.rewards.base -- BaseRewardFunction, GaussianModifier, etc.

Many of the reward classes have heavy dependencies (rdkit, networkx, etc.).
We test the parts that are self-contained and skip when imports fail.
"""

import math
import pytest

# rdkit / networkx are required by base.py at import time.
_SKIP_REASON = "rdkit or other heavy dependencies not available"

try:
    from grpo.rewards.base import (
        BaseRewardFunction,
        DefaultRewardFunction,
        GaussianModifier,
        resolve_target_task,
        sascorer,
    )
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON)


class TestGaussianModifier:
    """GaussianModifier(mu, sigma)(x) should return exp(-0.5*((x-mu)/sigma)^2)."""

    def test_peak_at_mu(self):
        gm = GaussianModifier(mu=5.0, sigma=1.0)
        assert gm(5.0) == pytest.approx(1.0)

    def test_one_sigma_away(self):
        gm = GaussianModifier(mu=0.0, sigma=1.0)
        expected = math.exp(-0.5)
        assert gm(1.0) == pytest.approx(expected, abs=1e-6)

    def test_symmetric(self):
        gm = GaussianModifier(mu=3.0, sigma=2.0)
        assert gm(1.0) == pytest.approx(gm(5.0), abs=1e-6)

    def test_narrow_sigma(self):
        gm = GaussianModifier(mu=0.0, sigma=0.1)
        # Far from peak should be near zero
        assert gm(10.0) < 1e-10

    def test_wide_sigma(self):
        gm = GaussianModifier(mu=0.0, sigma=100.0)
        # Nearby should still be close to 1
        assert gm(1.0) > 0.99


class TestSascorer:
    """sascorer should be importable (may be None if RDKit contrib is missing)."""

    def test_sascorer_exists(self):
        # sascorer can be None when the RDKit SA_Score contrib is not installed,
        # but the name must be importable from the module.
        # We simply check it was re-exported.
        from grpo.rewards.base import sascorer
        # It is either a module or None -- both are acceptable.
        assert sascorer is None or hasattr(sascorer, "calculateScore")


class TestResolveTargetTask:
    """resolve_target_task extracts grpo.target_task from config."""

    def test_none_cfg_returns_default(self):
        result = resolve_target_task(None)
        assert result == "penalized_logp"

    def test_none_cfg_custom_default(self):
        result = resolve_target_task(None, default="qed")
        assert result == "qed"

    def test_dict_cfg(self):
        cfg = {"grpo": {"target_task": "my_task"}}
        result = resolve_target_task(cfg)
        assert result == "my_task"

    def test_dict_cfg_missing_key(self):
        cfg = {"grpo": {}}
        result = resolve_target_task(cfg)
        assert result == "penalized_logp"


class TestBaseRewardFunction:
    """BaseRewardFunction can be instantiated and has expected attributes."""

    def test_construction(self):
        rf = BaseRewardFunction()
        assert rf.name == "base"

    def test_device_default_cpu(self):
        import torch
        rf = BaseRewardFunction()
        assert rf.device == torch.device("cpu")

    def test_callable(self):
        assert callable(BaseRewardFunction())


class TestDefaultRewardFunction:
    """DefaultRewardFunction subclasses BaseRewardFunction."""

    def test_construction(self):
        rf = DefaultRewardFunction()
        assert rf.name == "default"

    def test_is_base_subclass(self):
        assert issubclass(DefaultRewardFunction, BaseRewardFunction)
