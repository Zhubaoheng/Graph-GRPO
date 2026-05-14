"""Tests for backward-compatibility import shims.

Each legacy module (e.g., ``grpo_trainer``, ``grpo_rewards``) should still be
importable, but should emit exactly one ``DeprecationWarning`` on first import.

Because Python caches modules in ``sys.modules``, we use ``importlib`` with
explicit cache invalidation so each test function gets a fresh import.
"""

import importlib
import sys
import warnings

import pytest

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _fresh_import(module_name: str):
    """Import *module_name* after removing it (and its sub-modules) from the
    module cache so the top-level deprecation warning fires again.

    Returns ``(module, caught_warnings)`` where *caught_warnings* is a list of
    ``warnings.WarningMessage`` objects.
    """
    # Evict the module and any children so re-import triggers the top-level
    # warnings.warn() call inside the shim.
    to_remove = [key for key in sys.modules if key == module_name or key.startswith(module_name + ".")]
    for key in to_remove:
        del sys.modules[key]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module(module_name)

    return mod, caught


def _assert_single_deprecation(caught, module_name: str):
    """Assert that exactly one DeprecationWarning was raised and that the
    message mentions the deprecated module name."""
    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecation_warnings) == 1, (
        f"Expected exactly 1 DeprecationWarning for '{module_name}', "
        f"got {len(deprecation_warnings)}: {[str(w.message) for w in deprecation_warnings]}"
    )
    msg = str(deprecation_warnings[0].message)
    # The message should mention the old module being deprecated
    assert "deprecated" in msg.lower(), (
        f"DeprecationWarning message should mention 'deprecated': {msg}"
    )


# ---------------------------------------------------------------------------
# Tests -- one per shim module
# ---------------------------------------------------------------------------

class TestGrpoTrainerShim:
    """``from grpo_trainer import GRPOTrainer`` should work with a warning."""

    @pytest.mark.skipif(
        "grpo.trainer" not in sys.modules and not importlib.util.find_spec("grpo.trainer"),
        reason="grpo.trainer not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("grpo_trainer")
        _assert_single_deprecation(caught, "grpo_trainer")

    @pytest.mark.skipif(
        "grpo.trainer" not in sys.modules and not importlib.util.find_spec("grpo.trainer"),
        reason="grpo.trainer not on sys.path",
    )
    def test_exports_grpo_trainer(self):
        mod, _ = _fresh_import("grpo_trainer")
        assert hasattr(mod, "GRPOTrainer")


class TestGrpoRewardsShim:
    """``from grpo_rewards import create_reward_function`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.rewards"),
        reason="grpo.rewards not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("grpo_rewards")
        _assert_single_deprecation(caught, "grpo_rewards")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.rewards"),
        reason="grpo.rewards not on sys.path",
    )
    def test_exports_create_reward_function(self):
        mod, _ = _fresh_import("grpo_rewards")
        assert hasattr(mod, "create_reward_function")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.rewards"),
        reason="grpo.rewards not on sys.path",
    )
    def test_exports_molecular_validity_reward(self):
        mod, _ = _fresh_import("grpo_rewards")
        assert hasattr(mod, "MolecularValidityReward")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.rewards"),
        reason="grpo.rewards not on sys.path",
    )
    def test_exports_sascorer(self):
        mod, _ = _fresh_import("grpo_rewards")
        assert hasattr(mod, "sascorer")


class TestGrpoCoreShim:
    """``from grpo_core import GRPOCore`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.core"),
        reason="grpo.core not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("grpo_core")
        _assert_single_deprecation(caught, "grpo_core")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.core"),
        reason="grpo.core not on sys.path",
    )
    def test_exports_grpo_core(self):
        mod, _ = _fresh_import("grpo_core")
        assert hasattr(mod, "GRPOCore")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.core"),
        reason="grpo.core not on sys.path",
    )
    def test_exports_per_graph_stat_tracker(self):
        mod, _ = _fresh_import("grpo_core")
        assert hasattr(mod, "PerGraphStatTracker")


class TestGrpoLightningModuleShim:
    """``from grpo_lightning_module import GRPOLightningModule`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.lightning_module"),
        reason="grpo.lightning_module not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("grpo_lightning_module")
        _assert_single_deprecation(caught, "grpo_lightning_module")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.lightning_module"),
        reason="grpo.lightning_module not on sys.path",
    )
    def test_exports_grpo_lightning_module(self):
        mod, _ = _fresh_import("grpo_lightning_module")
        assert hasattr(mod, "GRPOLightningModule")


class TestTrajectoryDataShim:
    """``from trajectory_data import TrajectoryData`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.trajectory_data"),
        reason="grpo.trajectory_data not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("trajectory_data")
        _assert_single_deprecation(caught, "trajectory_data")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.trajectory_data"),
        reason="grpo.trajectory_data not on sys.path",
    )
    def test_exports_trajectory_data(self):
        mod, _ = _fresh_import("trajectory_data")
        assert hasattr(mod, "TrajectoryData")


class TestEvalGrpoSamplerShim:
    """``from eval_grpo_sampler import GraphGRPOProposer`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.eval_sampler"),
        reason="grpo.eval_sampler not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("eval_grpo_sampler")
        _assert_single_deprecation(caught, "eval_grpo_sampler")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.eval_sampler"),
        reason="grpo.eval_sampler not on sys.path",
    )
    def test_exports_graph_grpo_proposer(self):
        mod, _ = _fresh_import("eval_grpo_sampler")
        assert hasattr(mod, "GraphGRPOProposer")


class TestEvalGdpoDockingShim:
    """``from eval_gdpo_docking import gdpo_eval_smiles`` should work with a warning."""

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.eval_docking"),
        reason="grpo.eval_docking not on sys.path",
    )
    def test_deprecation_warning(self):
        mod, caught = _fresh_import("eval_gdpo_docking")
        _assert_single_deprecation(caught, "eval_gdpo_docking")

    @pytest.mark.skipif(
        not importlib.util.find_spec("grpo.eval_docking"),
        reason="grpo.eval_docking not on sys.path",
    )
    def test_exports_gdpo_eval_smiles(self):
        mod, _ = _fresh_import("eval_gdpo_docking")
        assert hasattr(mod, "gdpo_eval_smiles")
