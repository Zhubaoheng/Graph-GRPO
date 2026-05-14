"""Backward-compatibility shim.

Use ``from grpo.core import GRPOCore`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'grpo_core' is deprecated. "
    "Use 'from grpo.core import GRPOCore' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.core import GRPOCore, PerGraphStatTracker

__all__ = ["GRPOCore", "PerGraphStatTracker"]
