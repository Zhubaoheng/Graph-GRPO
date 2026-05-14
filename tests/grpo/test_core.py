"""Tests for grpo.core -- GRPOCore and PerGraphStatTracker."""

import pytest
import torch
import numpy as np


class TestImports:
    """Verify that the public API is importable."""

    def test_import_grpo_core(self):
        from grpo.core import GRPOCore
        assert GRPOCore is not None

    def test_import_per_graph_stat_tracker(self):
        from grpo.core import PerGraphStatTracker
        assert PerGraphStatTracker is not None


class TestPerGraphStatTracker:
    """PerGraphStatTracker accumulates per-config reward statistics."""

    def test_fresh_tracker_summary(self):
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        avg_group_size, num_configs = tracker.get_statistics_summary()
        assert avg_group_size == 0.0
        assert num_configs == 0

    def test_update_single_config(self):
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        rewards = torch.tensor([1.0, 2.0, 3.0, 4.0])
        configs = ["nodes_5"] * 4
        advantages = tracker.update(configs, rewards)

        assert advantages.shape == rewards.shape
        # After one update, summary should reflect one config
        avg_group_size, num_configs = tracker.get_statistics_summary()
        assert num_configs == 1
        assert avg_group_size > 0

    def test_update_multiple_configs(self):
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        rewards = torch.tensor([1.0, 2.0, 10.0, 20.0])
        configs = ["nodes_5", "nodes_5", "nodes_10", "nodes_10"]
        advantages = tracker.update(configs, rewards)

        assert advantages.shape == (4,)
        _, num_configs = tracker.get_statistics_summary()
        assert num_configs == 2

    def test_update_with_numpy(self):
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        rewards = np.array([0.5, 1.5, 2.5])
        configs = ["cfg_a", "cfg_a", "cfg_a"]
        advantages = tracker.update(configs, rewards)

        assert isinstance(advantages, torch.Tensor)
        assert advantages.shape == (3,)

    def test_intra_group_normalization(self):
        """Within a single config group, advantages should be zero-mean."""
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        rewards = torch.tensor([1.0, 3.0, 5.0, 7.0])
        configs = ["g"] * 4
        advantages = tracker.update(configs, rewards)

        # Mean of advantages within the group should be approximately zero
        assert abs(advantages.mean().item()) < 1e-5

    def test_clear_statistics(self):
        from grpo.core import PerGraphStatTracker

        tracker = PerGraphStatTracker()
        tracker.update(["a", "b"], torch.tensor([1.0, 2.0]))
        tracker.clear_statistics()

        avg_group_size, num_configs = tracker.get_statistics_summary()
        assert avg_group_size == 0.0
        assert num_configs == 0


class TestGRPOCoreConstruction:
    """GRPOCore requires a cfg dict with grpo sub-key."""

    def test_construct_with_minimal_cfg(self):
        from grpo.core import GRPOCore

        cfg = {
            "grpo": {
                "clip_ratio": 0.2,
                "kl_penalty": 0.01,
            }
        }
        core = GRPOCore(cfg)
        assert core.clip_range == 0.2
        assert core.beta == 0.01

    def test_default_hyperparams(self):
        from grpo.core import GRPOCore

        core = GRPOCore({"grpo": {}})
        # Defaults should be sensible numbers
        assert core.clip_range == 0.2
        assert core.entropy_coef == 0.0
        assert core.num_inner_epochs == 1
