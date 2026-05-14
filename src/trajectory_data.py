"""Backward-compatibility shim.

Use ``from grpo.trajectory_data import TrajectoryData`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'trajectory_data' is deprecated. "
    "Use 'from grpo.trajectory_data import TrajectoryData' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.trajectory_data import TrajectoryData

__all__ = ["TrajectoryData"]
