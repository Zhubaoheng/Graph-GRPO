"""Vendored mol_opt core (MIT License, https://github.com/wenhao-gao/mol_opt).

Only the minimal evaluation framework is included:
- Oracle: TDC oracle wrapper with budget tracking and result logging
- BaseOptimizer: Abstract base class for molecular optimization methods
- GraphGRPO_Optimizer: In-process Graph-GRPO optimizer for PMO benchmarks
"""

from mol_opt.optimizer import Oracle, BaseOptimizer, Objdict, top_auc
from mol_opt.graph_grpo import GraphGRPO_Optimizer

__all__ = [
    "Oracle",
    "BaseOptimizer",
    "Objdict",
    "top_auc",
    "GraphGRPO_Optimizer",
]
