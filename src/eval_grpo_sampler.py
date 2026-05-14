"""Backward-compatibility shim.

Use ``from grpo.eval_sampler import GraphGRPOProposer`` instead.
"""
import warnings as _warnings

_warnings.warn(
    "Importing from 'eval_grpo_sampler' is deprecated. "
    "Use 'from grpo.eval_sampler import GraphGRPOProposer' instead.",
    DeprecationWarning,
    stacklevel=2,
)

from grpo.eval_sampler import (
    GraphGRPOProposer,
    _compose_cfg_from_repo_defaults,
    _write_gdpo_log,
    _cmd_gdpo_eval,
    main,
)

# Also re-export create_datamodule_and_model_components which was previously
# imported from train_flow_grpo by consumers of this module.
from grpo.train_utils import create_datamodule_and_model_components

__all__ = [
    "GraphGRPOProposer",
    "_compose_cfg_from_repo_defaults",
    "_write_gdpo_log",
    "_cmd_gdpo_eval",
    "main",
    "create_datamodule_and_model_components",
]

if __name__ == "__main__":
    raise SystemExit(main())
