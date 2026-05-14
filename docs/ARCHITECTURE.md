# GRPO Package Architecture

## Overview

The GRPO (Group Relative Policy Optimization) system implements reinforcement
learning for graph generation using discrete flow matching.  The core idea is a
two-stage training loop -- **sampling** then **training** -- where a generative
flow model proposes graphs, a reward function scores them, and the model is
updated with a PPO-clipped objective augmented by group-relative advantages and
KL regularization against a frozen reference model.

The implementation was refactored from several monolithic files (`grpo_core.py`,
`grpo_trainer.py`, `grpo_rewards.py`, etc.) into the `src/grpo/` package.  All
original import paths still work via thin backward-compatibility shims that emit
`DeprecationWarning`.

---

## Package Structure

```
src/grpo/
    __init__.py               # Public API re-exports
    core.py                   # GRPOCore algorithm
    trajectory_data.py        # TrajectoryData container
    trainer.py                # GRPOTrainer hub class (mixin composition)
    sampling.py               # SamplingMixin
    training.py               # TrainingMixin
    reference_model.py        # ReferenceModelMixin
    reward_workers.py         # Multiprocessing reward computation
    graph_conversion.py       # GraphConversionMixin
    evaluation.py             # EvaluationMixin
    logging_utils.py          # LoggingMixin
    lightning_module.py        # GRPOLightningModule, FlowGRPODataModule
    eval_sampler.py           # GraphGRPOProposer (mol_opt integration)
    eval_docking.py           # GDPO docking evaluation utilities
    train_utils.py            # Dataset/model creation helpers
    rewards/                  # Reward function sub-package
        __init__.py           # Re-exports all reward classes + factory
        base.py               # BaseRewardFunction, DefaultRewardFunction,
                              #   GaussianModifier, resolve_target_task, sascorer
        graph_rewards.py      # PlanarGraphReward, SBMGraphReward, TreeGraphReward
        molecular_validity.py # MolecularValidityReward
        target_mpo.py         # TargetMPOReward
        tdc_oracle.py         # TDCOracleReward
        gdpo_docking.py       # GDPODockingReward
        valsartan.py          # ValsartanSmartsReward
        factory.py            # create_reward_function() dispatcher
```

### Module Descriptions

**`core.py`** -- The algorithmic heart.  `GRPOCore` owns the sample buffer,
computes group-relative advantages, PPO-clipped surrogate loss, and optional KL
penalty against the reference model.  `PerGraphStatTracker` maintains
per-configuration (e.g. per-node-count) running reward statistics so that
advantage normalization is independent across graph sizes.

**`trajectory_data.py`** -- `TrajectoryData` is a batch-aligned container
holding both tensor data (log-probs, rewards, etc.) and list data (SMILES,
graph configs).  Supports slicing, concatenation (`union`), and device transfer,
so the training loop can iterate minibatches without manual bookkeeping.

**`trainer.py`** -- `GRPOTrainer` is the hub class.  It inherits from seven
mixins (see Design Decisions below) and adds only the constructor,
`run_epoch()` orchestration, and state-dict persistence.  `run_epoch()` calls
`sampling_phase()` then `training_phase()` and handles p0 distribution updates
between epochs.

**`sampling.py`** -- `SamplingMixin` provides `sampling_phase()`,
`sample_graphs_with_trajectory_tracking()`, and
`refine_candidate_via_denoising()`.  During the sampling phase, graphs are
generated on GPU in batch, transferred to CPU, and scored via the multiprocessing
reward pool.

**`training.py`** -- `TrainingMixin` provides `training_phase()` with
gradient-accumulation, minibatch iteration over the trajectory buffer, and
learning-rate decay logic triggered by reward plateau detection.

**`reference_model.py`** -- `ReferenceModelMixin` handles creation, checkpoint
loading, and periodic hard-copy updates of the frozen reference model used for
KL regularization.

**`reward_workers.py`** -- Module-level functions
(`_set_single_thread_env`, `_reward_worker_initializer`,
`_compute_batch_rewards_worker`) that must live at module scope for
multiprocessing pickle compatibility, plus `RewardWorkerMixin` which manages
the `mp.Pool` lifecycle and dispatches batched reward computation.

**`graph_conversion.py`** -- `GraphConversionMixin` converts between SMILES
strings, RDKit `Mol` objects, and the internal `PlaceHolder` / tensor-pair
graph representation.

**`evaluation.py`** -- `EvaluationMixin` runs periodic evaluation and GDPO
docking assessment, including TDC zero-reward diagnostics.

**`logging_utils.py`** -- `LoggingMixin` encapsulates SwanLab metric logging
and trajectory visualization.

**`lightning_module.py`** -- `GRPOLightningModule` wraps the trainer in a
PyTorch Lightning `LightningModule` so it can be driven by a PL `Trainer`.
`FlowGRPODataModule` provides the dummy data loader needed by PL.
`create_grpo_lightning_module()` is a convenience factory.

**`eval_sampler.py`** -- `GraphGRPOProposer` loads a trained checkpoint and
exposes a `propose()` interface for integration with the mol_opt benchmark
framework.  Also provides CLI subcommands for standalone GDPO evaluation.

**`eval_docking.py`** -- Standalone helper functions for GDPO-style docking
evaluation: SA scoring, similarity filtering, and docking-score aggregation.

**`train_utils.py`** -- Shared helpers extracted from the Hydra entry point:
dataset/model creation, checkpoint validation, and resume logic.

**`rewards/`** -- Hierarchical reward sub-package.  `BaseRewardFunction`
defines the interface.  Concrete implementations cover graph-structural metrics,
molecular validity (QED, SA, ring penalties), multi-parameter optimization
(TargetMPO), TDC oracle wrapping, GDPO docking, and Valsartan SMARTS matching.
`factory.py` provides the `create_reward_function()` dispatcher that maps a
config string to the appropriate class.

---

## Design Decisions

### Mixin Composition for GRPOTrainer

`GRPOTrainer` inherits from seven focused mixins:

```python
class GRPOTrainer(
    SamplingMixin,       # sampling_phase, trajectory tracking, denoising refinement
    TrainingMixin,       # training_phase, minibatch iteration, LR decay
    ReferenceModelMixin, # create/load/update reference model
    RewardWorkerMixin,   # multiprocessing reward pool management
    GraphConversionMixin,# SMILES <-> graph, PlaceHolder conversion
    EvaluationMixin,     # periodic eval, GDPO eval
    LoggingMixin,        # SwanLab metrics
):
```

Each mixin is a single-file class containing a cohesive set of methods.  The
hub class (`trainer.py`) holds only the constructor, `run_epoch()`, and
state-dict methods.  This pattern keeps each file under ~500 lines while
avoiding the coordination overhead of a fully decomposed service architecture.

### Module-Level Worker Functions

Python's `multiprocessing` with the `spawn` start method requires that worker
targets and initializers be picklable.  Instance methods and lambdas are not.
The three worker functions in `reward_workers.py` are therefore defined at
module scope, not as methods of `RewardWorkerMixin`.

### Backward-Compatibility Shims

Each original top-level module (`grpo_core.py`, `grpo_trainer.py`,
`grpo_rewards.py`, `trajectory_data.py`, `grpo_lightning_module.py`,
`eval_grpo_sampler.py`, `eval_gdpo_docking.py`) has been replaced by a thin
shim that:

1. Emits a `DeprecationWarning` on import.
2. Re-exports every public name from its new location inside `grpo/`.

This means all existing scripts, notebooks, and downstream code continue to
work without changes.  For example:

```python
# Old path (still works, emits DeprecationWarning):
from grpo_trainer import GRPOTrainer

# New canonical path:
from grpo.trainer import GRPOTrainer
```

---

## Entry Points

### `src/train_flow_grpo.py` -- Training

Hydra-based entry point.  Loads configuration, builds the
`GRPOLightningModule` via `create_grpo_lightning_module()`, and launches
training through PyTorch Lightning's `Trainer`.  Heavy setup logic lives in
`grpo.train_utils`.

```
python src/train_flow_grpo.py --config-name <config>
```

### `src/eval_grpo_sampler.py` -- Evaluation / Sampling CLI

Backward-compat shim that delegates to `grpo.eval_sampler.main()`.  Provides
subcommands for standalone graph generation and GDPO docking evaluation from a
trained checkpoint.

```
python src/eval_grpo_sampler.py --ckpt_path <path> [--gdpo_eval ...]
```

---

## Backward Compatibility

The shim files live at the original import locations under `src/`:

| Old import path              | New canonical path                     |
|------------------------------|----------------------------------------|
| `grpo_core`                  | `grpo.core`                            |
| `grpo_trainer`               | `grpo.trainer` / `grpo.reward_workers` |
| `grpo_rewards`               | `grpo.rewards`                         |
| `trajectory_data`            | `grpo.trajectory_data`                 |
| `grpo_lightning_module`      | `grpo.lightning_module`                |
| `eval_grpo_sampler`          | `grpo.eval_sampler`                    |
| `eval_gdpo_docking`          | `grpo.eval_docking`                    |

Each shim follows the same pattern:

```python
"""Backward-compatibility shim."""
import warnings as _warnings

_warnings.warn(
    "Importing from 'grpo_core' is deprecated. "
    "Use 'from grpo.core import GRPOCore' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.core import GRPOCore, PerGraphStatTracker

__all__ = ["GRPOCore", "PerGraphStatTracker"]
```

To migrate, replace the old import with the new path.  The shims will be
removed in a future release.

---

## Testing

Tests live in `tests/grpo/`.  The shared `tests/conftest.py` adds `src/` to
`sys.path` so that both package imports (`from grpo.core import ...`) and
legacy bare imports resolve correctly.

Run the test suite from the repository root:

```bash
# All GRPO tests
pytest tests/grpo/ -v

# Specific module
pytest tests/grpo/test_core.py -v
```

Tests should be added alongside new modules following the naming convention
`tests/grpo/test_<module>.py`.
