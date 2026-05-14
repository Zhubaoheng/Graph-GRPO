"""Backward-compatibility shim.

Use ``from grpo.eval_docking import ...`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'eval_gdpo_docking' is deprecated. "
    "Use 'from grpo.eval_docking import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.eval_docking import (
    gdpo_eval_smiles,
    gdpo_get_sim_threshold,
    gdpo_load_train_fps,
)

__all__ = [
    "gdpo_eval_smiles",
    "gdpo_get_sim_threshold",
    "gdpo_load_train_fps",
]
