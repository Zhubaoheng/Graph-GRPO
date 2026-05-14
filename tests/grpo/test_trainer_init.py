"""Tests for grpo.trainer -- GRPOTrainer class structure.

GRPOTrainer requires a live model and config to construct, so we only test
the class metadata (inheritance, method presence) without instantiation.
"""

import pytest

_SKIP_REASON = "Heavy dependencies (rdkit, torch, etc.) not available"

try:
    from grpo.trainer import GRPOTrainer
    _HAS_DEPS = True
except ImportError:
    _HAS_DEPS = False

pytestmark = pytest.mark.skipif(not _HAS_DEPS, reason=_SKIP_REASON)


class TestGRPOTrainerImport:
    """Verify the class can be imported from the new location."""

    def test_import(self):
        assert GRPOTrainer is not None

    def test_is_class(self):
        assert isinstance(GRPOTrainer, type)


class TestGRPOTrainerInheritance:
    """GRPOTrainer should inherit from all expected mixins."""

    def _mixin_in_mro(self, name: str) -> bool:
        mro_names = [cls.__name__ for cls in GRPOTrainer.__mro__]
        return name in mro_names

    def test_sampling_mixin(self):
        assert self._mixin_in_mro("SamplingMixin")

    def test_training_mixin(self):
        assert self._mixin_in_mro("TrainingMixin")

    def test_reference_model_mixin(self):
        assert self._mixin_in_mro("ReferenceModelMixin")

    def test_reward_worker_mixin(self):
        assert self._mixin_in_mro("RewardWorkerMixin")

    def test_graph_conversion_mixin(self):
        assert self._mixin_in_mro("GraphConversionMixin")

    def test_evaluation_mixin(self):
        assert self._mixin_in_mro("EvaluationMixin")

    def test_logging_mixin(self):
        assert self._mixin_in_mro("LoggingMixin")


class TestGRPOTrainerMethods:
    """GRPOTrainer should expose the key orchestration methods."""

    def test_has_run_epoch(self):
        assert hasattr(GRPOTrainer, "run_epoch")
        assert callable(getattr(GRPOTrainer, "run_epoch"))

    def test_has_state_dict(self):
        assert hasattr(GRPOTrainer, "state_dict")
        assert callable(getattr(GRPOTrainer, "state_dict"))

    def test_has_load_state_dict(self):
        assert hasattr(GRPOTrainer, "load_state_dict")
        assert callable(getattr(GRPOTrainer, "load_state_dict"))

    def test_has_init(self):
        assert hasattr(GRPOTrainer, "__init__")
