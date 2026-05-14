"""Backward-compatibility shim.

Use ``from grpo.lightning_module import GRPOLightningModule`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'grpo_lightning_module' is deprecated. "
    "Use 'from grpo.lightning_module import ...' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.lightning_module import (
    GRPOLightningModule,
    FlowGRPODataModule,
    create_grpo_lightning_module,
)

__all__ = [
    "GRPOLightningModule",
    "FlowGRPODataModule",
    "create_grpo_lightning_module",
]
