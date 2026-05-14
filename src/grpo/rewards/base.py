import logging

import torch
import numpy as np
from typing import List, Tuple, Callable, Optional, Dict, Union
import networkx as nx
from collections import Counter
import hashlib
import pickle
import time
import math

from omegaconf import OmegaConf
from rdkit import Chem, DataStructs
from rdkit import RDLogger
from rdkit.Chem import QED, rdMolDescriptors, AllChem
import sys
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFMCS
from scipy.stats import gmean
import os
from pathlib import Path
import shutil
logger = logging.getLogger(__name__)

try:
    import sascorer  # Standard synthetic accessibility scoring script (Guacamol ecosystem)
except ImportError:
    sascorer = None
    # RDKit Contrib lives under <rdkit_pkg>/Contrib for pip wheels and under
    # $prefix/share/RDKit/Contrib for conda builds; try both.
    _sa_contrib_candidates = []
    try:
        from rdkit.Chem import RDConfig as _RDConfig

        _sa_contrib_candidates.append(os.path.join(_RDConfig.RDContribDir, "SA_Score"))
    except Exception:
        pass
    _sa_contrib_candidates.append(
        os.path.join(sys.prefix, "share", "RDKit", "Contrib", "SA_Score")
    )
    for _contrib_path in _sa_contrib_candidates:
        if not os.path.exists(os.path.join(_contrib_path, "sascorer.py")):
            continue
        if _contrib_path not in sys.path:
            sys.path.append(_contrib_path)
        try:
            import sascorer
            break
        except ImportError as e:
            logger.warning("[GRPO] Found sascorer at %s but import failed: %s", _contrib_path, e)
            sascorer = None
from analysis.rdkit_functions import build_molecule, build_molecule_with_partial_charges
from analysis.lead_opt_oracle import LeadOptOracle
from grpo.eval_docking import gdpo_get_sim_threshold, gdpo_load_train_fps

RDLogger.DisableLog("rdApp.*")
_SA_FALLBACK_WARNED = False


def resolve_target_task(cfg, default: str = "penalized_logp") -> str:
    """
    Resolve the configured target task name.

    Project convention: all GRPO configs must define ``grpo.target_task``.
    """
    if cfg is None:
        return default

    task = None
    try:
        task = OmegaConf.select(cfg, "grpo.target_task", default=None)
    except Exception:
        task = None

    if not task and isinstance(cfg, dict):
        section = cfg.get("grpo", {})
        if isinstance(section, dict):
            task = section.get("target_task")

    return task or default


class GaussianModifier:
    """Gaussian modifier for MPO rewards"""
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: float) -> float:
        return float(np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2)))




class BaseRewardFunction:
    """Base reward function class."""

    def __init__(self, name: str = "base", device: Optional[torch.device] = None):
        self.name = name
        self._cache = {}
        self._cache_size = 1000
        self.device = device if device is not None else torch.device("cpu")

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Compute rewards for a list of graphs. This is a simple reward function
        designed for debugging purposes. Returns 1.0 if the graph is connected
        and planar, otherwise 0.1.

        Args:
            graphs: List of [atom_types, edge_types] pairs

        Returns:
            Tensor of rewards for each graph
        """
        rewards = []
        for atom_types, edge_types in graphs:
            try:
                nx_graph = self._convert_tensor_to_networkx_graph(atom_types, edge_types)
                if nx_graph.number_of_nodes() > 0:
                    is_connected = nx.is_connected(nx_graph)
                    # check_planarity returns a tuple (is_planar, certificate)
                    is_planar, _ = nx.check_planarity(nx_graph)
                    if is_connected and is_planar:
                        reward = 1.0
                    else:
                        reward = 0.1
                else:
                    reward = 0.0
                rewards.append(reward)
            except Exception as e:
                logger.warning("Error computing simple reward: %s", e)
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _convert_tensor_to_networkx_graph(self, atom_types: torch.Tensor, edge_types: torch.Tensor) -> nx.Graph:
        """
        Convert a graph in tensor format to a NetworkX graph object.

        Args:
            atom_types: Node type tensor [num_nodes, num_atom_types]
            edge_types: Edge type tensor [num_nodes, num_nodes, num_edge_types]

        Returns:
            NetworkX undirected graph object
        """
        try:
            n_nodes = atom_types.size(0)

            # Check edge_types dimensions and handle accordingly
            if edge_types.dim() == 3:
                # edge_types shape is [n_nodes, n_nodes, 2]
                # Last dim: [no-edge prob, has-edge prob]
                edge_decisions = torch.argmax(edge_types, dim=-1)  # [n_nodes, n_nodes]
            elif edge_types.dim() == 2:
                # edge_types is already in adjacency matrix format [n_nodes, n_nodes]
                edge_decisions = edge_types
            else:
                raise ValueError(f"Unsupported edge_types dimension: {edge_types.dim()}")

            # Convert to numpy adjacency matrix
            A = edge_decisions.cpu().numpy()

            # Ensure symmetry (undirected graph)
            A = (A + A.T) > 0
            A = A.astype(int)

            # Remove self-loops
            np.fill_diagonal(A, 0)

            # Create NetworkX graph
            nx_graph = nx.from_numpy_array(A)

            return nx_graph

        except Exception as e:
            logger.warning("Error converting to NetworkX graph: %s", e)
            logger.debug("  atom_types shape: %s", atom_types.shape)
            logger.debug("  edge_types shape: %s", edge_types.shape)
            logger.debug("  edge_types dim: %s", edge_types.dim())
            return nx.Graph()

    def _compute_graph_hash_for_caching(self, atom_types: torch.Tensor, edge_types: torch.Tensor) -> str:
        """
        Compute hash of a graph for caching.

        Args:
            atom_types: Node type tensor
            edge_types: Edge type tensor

        Returns:
            Hash string for the graph
        """
        try:
            # Simplified hash computation
            atom_hash = hashlib.md5(atom_types.cpu().numpy().tobytes()).hexdigest()
            edge_hash = hashlib.md5(edge_types.cpu().numpy().tobytes()).hexdigest()
            return f"{atom_hash}_{edge_hash}"
        except Exception:
            return str(hash(str(atom_types) + str(edge_types)))

class DefaultRewardFunction(BaseRewardFunction):
    """Default reward function: encourages connectivity and diversity."""

    def __init__(self, device: Optional[torch.device] = None):
        super().__init__("default", device=device)

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rewards = []

        for atom_types, edge_types in graphs:
            n_nodes = atom_types.size(0)
            n_edges = (edge_types.sum(dim=-1) > 0).sum().item() // 2

            # Encourage reasonable connectivity
            connectivity_reward = min(n_edges / max(1, n_nodes - 1), 1.0)

            # Encourage atom type diversity
            unique_atoms = torch.unique(torch.argmax(atom_types, dim=-1)).size(0)
            diversity_reward = unique_atoms / max(1, n_nodes)

            # Combined reward
            total_reward = (connectivity_reward + diversity_reward) / 2.0
            rewards.append(total_reward)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)
