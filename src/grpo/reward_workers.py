"""
Reward-computation worker helpers and the RewardWorkerMixin.

Module-level functions (_set_single_thread_env, _reward_worker_initializer,
_compute_batch_rewards_worker) MUST remain at module scope so that
multiprocessing can pickle them.
"""

import logging
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.multiprocessing as mp
from multiprocessing import cpu_count, TimeoutError as MPTimeoutError
from rdkit import Chem

from grpo.rewards import create_reward_function, MolecularValidityReward, resolve_target_task
from grpo.eval_docking import gdpo_get_sim_threshold, gdpo_load_train_fps

logger = logging.getLogger(__name__)

Graph = Tuple[torch.Tensor, torch.Tensor]


# ---------------------------------------------------------------------------
# Module-level helpers (must be picklable for multiprocessing)
# ---------------------------------------------------------------------------

def _set_single_thread_env(num_threads: int = 1, *, force: bool = False) -> None:
    """
    Limit CPU thread oversubscription (OpenMP/MKL/BLAS/NumExpr + PyTorch).

    Important: for multiprocessing with the "spawn" start method, call this inside
    each worker process (e.g. Pool initializer). Settings in the parent process
    are not guaranteed to carry over to spawned children.
    """
    num_threads_str = str(int(num_threads))
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "BLIS_NUM_THREADS",
    ):
        if force:
            os.environ[key] = num_threads_str
        else:
            os.environ.setdefault(key, num_threads_str)
    try:
        torch.set_num_threads(int(num_threads))
        torch.set_num_interop_threads(int(num_threads))
    except Exception:
        pass


# Worker-local global variable to persist reward function across batches
_WORKER_REWARD_FUNC = None

def _reward_worker_initializer(num_threads: int = 1, reward_type: str = None, reward_kwargs: dict = None) -> None:
    # 1. Threading and environment setup
    import sys
    _set_single_thread_env(num_threads=num_threads, force=True)
    # Keep worker stdout off the proposer JSONL pipe.
    sys.stdout = sys.stderr

    # 2. Pre-load reward function for this worker
    global _WORKER_REWARD_FUNC
    if reward_type:
        import torch
        from grpo_rewards import create_reward_function
        # Use CPU for docking workers
        device = torch.device('cpu')
        try:
            _WORKER_REWARD_FUNC = create_reward_function(reward_type, device=device, **(reward_kwargs or {}))
        except Exception as e:
            logging.getLogger(__name__).error("[Worker %s] Failed to initialize persistent reward function: %s", os.getpid(), e)

def _compute_batch_rewards_worker(batch_graphs, reward_type: str, device_str: str, reward_kwargs: Optional[Dict] = None):
    """
    Top-level multiprocessing worker function: compute rewards for a batch of graphs.
    """
    import torch
    global _WORKER_REWARD_FUNC

    device = torch.device(device_str)

    # Use persistent worker-local reward function if available and of the right type
    if _WORKER_REWARD_FUNC is not None:
        reward_func = _WORKER_REWARD_FUNC
    else:
        from grpo_rewards import create_reward_function
        reward_func = create_reward_function(reward_type, device=device, **(reward_kwargs or {}))

    processed_graphs = []
    for atom_data, edge_data in batch_graphs:
        if torch.is_tensor(atom_data):
            atom_tensor = atom_data.detach().to(device)
        else:
            atom_tensor = torch.as_tensor(atom_data, device=device)

        if torch.is_tensor(edge_data):
            edge_tensor = edge_data.detach().to(device)
        else:
            edge_tensor = torch.as_tensor(edge_data, device=device)

        processed_graphs.append((atom_tensor, edge_tensor))

    with torch.no_grad():
        rewards = reward_func(processed_graphs)

    if isinstance(rewards, torch.Tensor):
        return rewards.cpu().tolist()
    return rewards


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------

class RewardWorkerMixin:
    """Methods for reward computation, worker management, and serialization."""

    def _prepare_reward_kwargs(self, reward_function: Callable, cfg) -> Dict[str, Any]:
        """Collect reward function init parameters for reuse in worker subprocesses."""
        reward_type = getattr(reward_function, "name", None) or getattr(cfg.grpo, "reward_type", "default")
        reward_type = reward_type.lower() if isinstance(reward_type, str) else "default"
        reward_kwargs: Dict[str, Any] = {}

        # Many rewards (including valsartan/target_mpo/tdc/validity) rely on atom_decoder for graph->RDKit Mol.
        # In the main process, create_reward_function injects from model.dataset_info, but workers only have kwargs;
        # without it, workers fall back to the default decoder, causing atom type mismatch -> invalid molecules -> zero rewards.
        dataset_info = getattr(self.model, "dataset_info", None)
        if dataset_info is not None:
            atom_decoder = getattr(dataset_info, "atom_decoder", None)
            if atom_decoder is not None:
                try:
                    reward_kwargs["atom_decoder"] = list(atom_decoder)
                except Exception:
                    reward_kwargs["atom_decoder"] = atom_decoder

        if reward_type in ("molecular_validity", "guacamol_reward", "gracamol_reward", "gracamol", "validity_connectivity", "valid_connectivity"):
            grpo_cfg = getattr(cfg, "grpo", {})
            try:
                dist_coef = grpo_cfg.get("dist_coef", None)
            except AttributeError:
                dist_coef = getattr(grpo_cfg, "dist_coef", None)
            if dist_coef is None:
                try:
                    dist_coef = grpo_cfg.get("reward_dist_coef", None)
                except AttributeError:
                    dist_coef = getattr(grpo_cfg, "reward_dist_coef", None)
            if dist_coef is not None:
                reward_kwargs["dist_coef"] = float(dist_coef)
            edge_dist_factor = grpo_cfg.get("edge_dist_factor", None) if hasattr(grpo_cfg, "get") else getattr(grpo_cfg, "edge_dist_factor", None)
            if edge_dist_factor is not None:
                reward_kwargs["edge_dist_factor"] = float(edge_dist_factor)

            dataset_info = getattr(self.model, "dataset_info", None)
            if dataset_info is not None:
                node_dist = getattr(dataset_info, "node_types", None)
                edge_dist = getattr(dataset_info, "edge_types", None)
                atom_decoder = getattr(dataset_info, "atom_decoder", None)

                serialized_node = self._serialize_distribution_for_worker(node_dist)
                serialized_edge = self._serialize_distribution_for_worker(edge_dist)
                if serialized_node is not None:
                    reward_kwargs["target_node_dist"] = serialized_node
                if serialized_edge is not None:
                    reward_kwargs["target_edge_dist"] = serialized_edge

        if reward_type in ("target_mpo", "guacamol_mpo", "target_goal"):
            target_task = resolve_target_task(cfg)
            if target_task:
                reward_kwargs["target_task"] = target_task
        if reward_type in ("tdc_oracle", "tdc_pmo", "pmo"):
            grpo_cfg = getattr(cfg, "grpo", {})
            getter = grpo_cfg.get if hasattr(grpo_cfg, "get") else lambda k, d=None: getattr(grpo_cfg, k, d)

            tdc_oracles = getter("tdc_oracles", None)
            tdc_oracle = getter("tdc_oracle", None)
            if tdc_oracles is not None:
                try:
                    reward_kwargs["tdc_oracles"] = list(tdc_oracles)
                except Exception:
                    reward_kwargs["tdc_oracles"] = tdc_oracles
            elif tdc_oracle is not None:
                reward_kwargs["tdc_oracle"] = tdc_oracle

            tdc_aggregation = getter("tdc_aggregation", None)
            if tdc_aggregation is not None:
                reward_kwargs["tdc_aggregation"] = tdc_aggregation
            tdc_weights = getter("tdc_weights", None)
            if tdc_weights is not None:
                try:
                    reward_kwargs["tdc_weights"] = [float(x) for x in list(tdc_weights)]
                except Exception:
                    reward_kwargs["tdc_weights"] = tdc_weights
            tdc_minimize = getter("tdc_minimize", None)
            if tdc_minimize is not None:
                reward_kwargs["tdc_minimize"] = bool(tdc_minimize)
            tdc_invalid_score = getter("tdc_invalid_score", None)
            if tdc_invalid_score is not None:
                reward_kwargs["tdc_invalid_score"] = float(tdc_invalid_score)
            tdc_clip_min = getter("tdc_clip_min", None)
            if tdc_clip_min is not None:
                reward_kwargs["tdc_clip_min"] = float(tdc_clip_min)
            tdc_clip_max = getter("tdc_clip_max", None)
            if tdc_clip_max is not None:
                reward_kwargs["tdc_clip_max"] = float(tdc_clip_max)

            # Ensure PyTDC oracle cache is resolved from project root (not Hydra output dir).
            tdc_home = getter("tdc_home", None)
            try:
                repo_root = Path(__file__).resolve().parents[1]
            except Exception:
                repo_root = None

            if tdc_home is None and repo_root is not None and (repo_root / "oracle").is_dir():
                oracle_names: List[str] = []
                if tdc_oracles is not None:
                    try:
                        oracle_names = [str(x) for x in list(tdc_oracles)]
                    except Exception:
                        oracle_names = [str(tdc_oracles)]
                elif tdc_oracle is not None:
                    oracle_names = [str(tdc_oracle)]

                oracle_dir = repo_root / "oracle"
                has_local_pkl = False
                for name in oracle_names:
                    if (oracle_dir / f"{name}.pkl").is_file() or (oracle_dir / f"{name}_current.pkl").is_file():
                        has_local_pkl = True
                        break
                if has_local_pkl:
                    tdc_home = str(repo_root)

            if tdc_home is not None:
                try:
                    tdc_path = Path(os.path.expanduser(str(tdc_home)))
                    if not tdc_path.is_absolute() and repo_root is not None:
                        tdc_path = repo_root / tdc_path
                    if tdc_path.is_dir() and tdc_path.name == "oracle":
                        tdc_path = tdc_path.parent
                    reward_kwargs["tdc_home"] = str(tdc_path.resolve())
                except Exception:
                    reward_kwargs["tdc_home"] = str(tdc_home)

            dataset_info = getattr(self.model, "dataset_info", None)
            if dataset_info is not None:
                # atom_decoder is injected uniformly at the top of the function
                pass

        if reward_type in ("gdpo_docking", "gdpo"):
            try:
                grpo_cfg = cfg.grpo
            except Exception:
                grpo_cfg = None
            if grpo_cfg is not None:
                try:
                    target_name = grpo_cfg.get("target_name", None)
                except Exception:
                    target_name = getattr(grpo_cfg, "target_name", None)
                if target_name is not None:
                    reward_kwargs["target_name"] = target_name
                dataset_name = getattr(getattr(cfg, "dataset", None), "name", "") or ""
                datadir = getattr(getattr(cfg, "dataset", None), "datadir", None)
                remove_h = getattr(getattr(cfg, "dataset", None), "remove_h", None)
                if dataset_name:
                    reward_kwargs["dataset_name"] = dataset_name
                if datadir is not None:
                    reward_kwargs["datadir"] = datadir
                if remove_h is not None:
                    reward_kwargs["remove_h"] = bool(remove_h)
                sim_override = None
                try:
                    sim_override = grpo_cfg.get("gdpo_sim_threshold", None)
                except Exception:
                    sim_override = getattr(grpo_cfg, "gdpo_sim_threshold", None)
                if sim_override is None:
                    try:
                        sim_override = grpo_cfg.get("gdpo_eval_sim_threshold", None)
                    except Exception:
                        sim_override = getattr(grpo_cfg, "gdpo_eval_sim_threshold", None)
                reward_kwargs["sim_threshold"] = gdpo_get_sim_threshold(
                    dataset_name,
                    override=sim_override,
                )
                try:
                    sa_threshold = grpo_cfg.get("gdpo_sa_threshold", None)
                except Exception:
                    sa_threshold = getattr(grpo_cfg, "gdpo_sa_threshold", None)
                if sa_threshold is not None:
                    reward_kwargs["sa_threshold"] = float(sa_threshold)
                try:
                    dock_exhaustiveness = grpo_cfg.get("gdpo_dock_exhaustiveness", None)
                except Exception:
                    dock_exhaustiveness = getattr(grpo_cfg, "gdpo_dock_exhaustiveness", None)
                if dock_exhaustiveness is not None:
                    reward_kwargs["dock_exhaustiveness"] = int(dock_exhaustiveness)
                try:
                    dock_num_modes = grpo_cfg.get("gdpo_dock_num_modes", None)
                except Exception:
                    dock_num_modes = getattr(grpo_cfg, "gdpo_dock_num_modes", None)
                if dock_num_modes is not None:
                    reward_kwargs["dock_num_modes"] = int(dock_num_modes)
                try:
                    dock_timeout = grpo_cfg.get("gdpo_dock_timeout", None)
                except Exception:
                    dock_timeout = getattr(grpo_cfg, "gdpo_dock_timeout", None)
                if dock_timeout is not None:
                    reward_kwargs["dock_timeout"] = int(dock_timeout)

        # Distribution-matching rewards (planar/sbm/tree) need reference stats in worker processes.
        # Workers cannot access `datamodule`, so we must serialize them from the main-process reward.
        if reward_type in ("planar_graph", "planar", "sbm", "sbm_graph", "tree", "tree_graph"):
            if hasattr(reward_function, "state_dict_for_workers"):
                try:
                    reward_kwargs.update(reward_function.state_dict_for_workers())
                except Exception as e:
                    logger.warning("Failed to serialize distribution-matching reward stats for workers: %s", e)

        return reward_kwargs

    @staticmethod
    def _serialize_distribution_for_worker(dist) -> Optional[List[float]]:
        """Convert various distribution formats to plain lists for pickling."""
        if dist is None:
            return None
        if torch.is_tensor(dist):
            arr = dist.detach().cpu().float()
            total = float(arr.sum())
            if total > 0:
                arr = arr / total
            return arr.tolist()
        if isinstance(dist, np.ndarray):
            arr = dist.astype(float)
            total = float(arr.sum())
            if total > 0:
                arr = arr / total
            return arr.tolist().copy()
        if isinstance(dist, (list, tuple)):
            arr = np.array(dist, dtype=float)
            total = float(arr.sum())
            if total > 0:
                arr = arr / total
            return arr.tolist()
        if isinstance(dist, dict):
            if not dist:
                return None
            max_idx = max(int(k) for k in dist.keys())
            values = [0.0] * (max_idx + 1)
            for k, v in dist.items():
                idx = int(k)
                if idx >= len(values):
                    values.extend([0.0] * (idx - len(values) + 1))
                values[idx] = float(v)
            total = sum(values)
            if total > 0:
                values = [v / total for v in values]
            return values
        return None

    def _compute_rewards_multiprocess_sync(
        self,
        graph_list: List,
        timeout: Optional[float] = None,
        context: str = "reward",
    ) -> torch.Tensor:
        """
        Compute rewards for all graphs synchronously using multiprocessing pool.

        Args:
            graph_list: List of all graphs

        Returns:
            Reward tensor
        """
        num_graphs = len(graph_list)
        if num_graphs == 0:
            return torch.tensor([], dtype=torch.float32)

        if self.reward_pool is None or self.num_reward_workers <= 0:
            with torch.no_grad():
                rewards = self.reward_function(graph_list)
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                return rewards

        # Get reward function info
        reward_func_type = getattr(self.reward_function, 'name', 'default')
        reward_func_type_lower = str(reward_func_type).lower()
        device_str = 'cpu'  # Compute on CPU
        base_reward_kwargs = getattr(self, "reward_kwargs", None) or {}
        reward_kwargs = dict(base_reward_kwargs)

        # === Main process side: convert graph data to pure Python/numpy structures ===
        # This ensures objects passed to multiprocessing pool don't contain Tensors, avoiding PyTorch shared memory/CUDA lock issues.
        py_graphs = []
        is_optimized = False

        # Try batch processing if graph_list elements are PyTorch Tensors
        if graph_list:
            first_item = graph_list[0]
            first_X = first_item[0] if isinstance(first_item, (tuple, list)) else first_item

            try:
                if torch.is_tensor(first_X):
                    # Try batch processing
                    # Assume all graphs have the same shape (typically padded in Dense Flow Matching)
                    all_X = [item[0] for item in graph_list]
                    all_E = [item[1] for item in graph_list]

                    # Stack & Move to CPU (Batch Operation)
                    # FIX: Keep as Tensor for IPC (PyTorch pickling is robust), avoid numpy pickling error
                    batch_X = torch.stack(all_X).cpu()
                    batch_E = torch.stack(all_E).cpu()

                    # Re-zip into list of tuples
                    py_graphs = list(zip(batch_X, batch_E))
                    is_optimized = True
            except Exception as batch_err:
                is_optimized = False

        if not is_optimized:
            # Fallback to original iterative loop
            for item in graph_list:
                if isinstance(item, (tuple, list)) and len(item) == 2:
                    atom_types, edge_types = item
                else:
                    atom_types, edge_types = item

                # FIX: Send as Tensor or List, avoid explicit numpy array if possible for IPC safety
                # But to maintain compat, we send CPU Tensor.
                if torch.is_tensor(atom_types):
                    atom_arr = atom_types.detach().cpu()
                else:
                    atom_arr = torch.tensor(atom_types) # Ensure consistent type

                if torch.is_tensor(edge_types):
                    edge_arr = edge_types.detach().cpu()
                else:
                    edge_arr = torch.tensor(edge_types)

                py_graphs.append((atom_arr, edge_arr))

        # MolecularValidityReward: Compute distribution weights on full graph set in main process, avoid small-batch statistics in subprocesses
        if reward_func_type_lower in ("molecular_validity", "guacamol_reward", "gracamol_reward", "gracamol"):
            try:
                target_node_dist = reward_kwargs.get("target_node_dist")
                target_edge_dist = reward_kwargs.get("target_edge_dist")
                if target_node_dist is None and hasattr(self.reward_function, "target_node_dist"):
                    target_node_dist = getattr(self.reward_function, "target_node_dist")
                if target_edge_dist is None and hasattr(self.reward_function, "target_edge_dist"):
                    target_edge_dist = getattr(self.reward_function, "target_edge_dist")

                scale_factor = reward_kwargs.get("scale_factor") or reward_kwargs.get("dist_scale_factor")
                clip_range = reward_kwargs.get("clip_range") or reward_kwargs.get("dist_clip_range")
                if scale_factor is None and hasattr(self.reward_function, "scale_factor"):
                    scale_factor = getattr(self.reward_function, "scale_factor")
                if clip_range is None and hasattr(self.reward_function, "clip_range"):
                    clip_range = getattr(self.reward_function, "clip_range")

                node_weights, edge_weights = MolecularValidityReward.compute_distribution_weights(
                    py_graphs,
                    target_node_dist=target_node_dist,
                    target_edge_dist=target_edge_dist,
                    scale_factor=scale_factor,
                    clip_range=clip_range,
                )
                reward_kwargs["precomputed_node_weights"] = node_weights
                reward_kwargs["precomputed_edge_weights"] = edge_weights

                if target_node_dist is not None and "target_node_dist" not in reward_kwargs:
                    reward_kwargs["target_node_dist"] = target_node_dist
                if target_edge_dist is not None and "target_edge_dist" not in reward_kwargs:
                    reward_kwargs["target_edge_dist"] = target_edge_dist
                if scale_factor is not None:
                    reward_kwargs["scale_factor"] = float(scale_factor)
                if clip_range is not None:
                    reward_kwargs["clip_range"] = float(clip_range)
            except Exception as weight_err:
                logger.warning("Global distribution weight pre-computation failed, falling back to per-shard computation: %s", weight_err)

        # === SMILES Level De-duplication (Optimization) ===
        # If there are many identical molecules in the batch, only dock the unique ones.
        unique_smiles_to_indices = defaultdict(list)
        graph_to_smiles = []

        # We need atom_decoder to convert graphs to SMILES in the main process
        dataset_info = getattr(self.model, "dataset_info", None)
        atom_decoder = getattr(dataset_info, "atom_decoder", None) if dataset_info is not None else None

        from analysis.rdkit_functions import build_molecule

        valid_graphs_for_docking = []
        original_idx_to_unique_idx = {}

        for i, (at, et) in enumerate(py_graphs):
            try:
                # Basic graph-to-smiles to find duplicates
                at_types = torch.argmax(at, dim=-1) if at.dim() == 2 else at
                et_types = torch.argmax(et, dim=-1) if et.dim() == 3 else et
                mol = build_molecule(at_types, et_types, atom_decoder)
                if mol:
                    smi = Chem.MolToSmiles(mol)
                    if smi:
                        if smi not in unique_smiles_to_indices:
                            unique_smiles_to_indices[smi].append(i)
                            valid_graphs_for_docking.append((at, et))
                            original_idx_to_unique_idx[i] = len(valid_graphs_for_docking) - 1
                        else:
                            # It's a duplicate
                            first_idx = unique_smiles_to_indices[smi][0]
                            original_idx_to_unique_idx[i] = original_idx_to_unique_idx[first_idx]
                        continue
            except Exception:
                pass
            # If fail to build SMILES, keep as unique to be safe
            valid_graphs_for_docking.append((at, et))
            original_idx_to_unique_idx[i] = len(valid_graphs_for_docking) - 1

        num_unique = len(valid_graphs_for_docking)
        if num_unique < num_graphs:
            logger.info("[GRPO] SMILES deduplication: %d -> %d unique molecules", num_graphs, num_unique)

        # Decide batch processing strategy
        batch_size = max(1, num_unique // (self.num_reward_workers * 4))
        batch_size = min(batch_size, 2000)

        # Create batches
        batches = []
        for i in range(0, num_unique, batch_size):
            batch = valid_graphs_for_docking[i:min(i + batch_size, num_unique)]
            batches.append((batch, reward_func_type, device_str, reward_kwargs))

        try:
            # Compute in parallel using multiprocessing pool
            if timeout is not None and timeout > 0:
                async_result = self.reward_pool.starmap_async(_compute_batch_rewards_worker, batches, chunksize=1)
                try:
                    results = async_result.get(timeout=timeout)
                except MPTimeoutError:
                    logger.warning("%s Multiprocessing reward computation did not complete within %s s, skipping this batch.", context, timeout)
                    self.reward_pool.terminate()
                    self.reward_pool.join()
                    self.reward_pool = mp.get_context('spawn').Pool(
                        processes=self.num_reward_workers,
                        initializer=_reward_worker_initializer,
                        initargs=(self.reward_worker_threads, reward_func_type, reward_kwargs),
                    )
                    return torch.tensor([], dtype=torch.float32)
            else:
                results = self.reward_pool.starmap(_compute_batch_rewards_worker, batches, chunksize=1)

            # Merge unique molecule results
            unique_rewards = []
            for batch_rewards in results:
                unique_rewards.extend(batch_rewards)

            # Map back to original full list
            all_rewards = []
            for i in range(num_graphs):
                u_idx = original_idx_to_unique_idx[i]
                all_rewards.append(unique_rewards[u_idx])

            return torch.tensor(all_rewards, dtype=torch.float32)

        except Exception as e:
            logger.warning("Multiprocessing reward computation failed (%s): %s", context, e)
            logger.info("Falling back to single-process computation...")
            with torch.no_grad():
                rewards = self.reward_function(graph_list)
                if not isinstance(rewards, torch.Tensor):
                    rewards = torch.tensor(rewards, dtype=torch.float32)
                return rewards

    @staticmethod
    def _compute_single_reward_worker(graph_data, reward_type: str, device_str: str):
        """
        Compute reward for a single graph in a worker process (kept for compatibility).

        Args:
            graph_data: Single graph data [atom_types, edge_types]
            reward_type: Reward function type
            device_str: Device string

        Returns:
            Single reward value
        """
        import torch
        from grpo_rewards import create_reward_function

        device = torch.device(device_str)
        reward_func = create_reward_function(reward_type, device=device)

        rewards = reward_func([graph_data])
        return rewards[0].item()
