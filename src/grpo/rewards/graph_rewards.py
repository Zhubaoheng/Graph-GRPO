import logging

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict, Union
import networkx as nx
from collections import Counter
import hashlib
import pickle
import os

from grpo.rewards.base import BaseRewardFunction

logger = logging.getLogger(__name__)


class PlanarGraphReward(BaseRewardFunction):
    """
    Planar graph reward for distribution matching against the training set.

    Components (per-graph):
    - `valid`: connected + planar (0/1)
    - `deg`: similarity of degree histogram vs train reference
    - `clus`: similarity of clustering-coefficient histogram vs train reference
    - `orb`: similarity of ORCA orbit-count vector vs train reference
    - `train`: weighted sum of `deg/clus/orb`

    Final reward (normalized to [0, 1]):
        `R = valid * (w_valid + train) / (w_valid + w_deg + w_clus + w_orb)`
    """

    _WARNED_ORCA = False
    _FIXED_BINS = 100
    _FIXED_W_VALID = 1.0
    _FIXED_W_DEG = 0.2
    _FIXED_W_CLUS = 0.2
    _FIXED_W_ORB = 0.2
    _FIXED_DEG_SCALE = 5.0
    _FIXED_CLUS_SCALE = 5.0
    _FIXED_ORB_SCALE = 1.0
    _FIXED_USE_ORB = True

    def __init__(
        self,
        device: Optional[torch.device] = None,
        *,
        datamodule=None,
        ref_degree_dist: Optional[Union[np.ndarray, List[float]]] = None,
        ref_clustering_hist: Optional[Union[np.ndarray, List[float]]] = None,
        ref_orbit_mean: Optional[Union[np.ndarray, List[float]]] = None,
    ):
        super().__init__("planar_graph", device=device)
        self.bins = int(self._FIXED_BINS)
        self.w_valid = float(self._FIXED_W_VALID)
        self.w_deg = float(self._FIXED_W_DEG)
        self.w_clus = float(self._FIXED_W_CLUS)
        self.w_orb = float(self._FIXED_W_ORB)
        self.deg_scale = float(self._FIXED_DEG_SCALE)
        self.clus_scale = float(self._FIXED_CLUS_SCALE)
        self.orb_scale = float(self._FIXED_ORB_SCALE)
        self.use_orb = bool(self._FIXED_USE_ORB)
        self._cache_size = 2000

        # Reference statistics are loaded from the dataset's `ref_metrics.pkl`
        need_orb = bool(self.use_orb and ref_orbit_mean is None)
        if ref_degree_dist is None or ref_clustering_hist is None or need_orb:
            if datamodule is not None:
                stats = self._load_reference_stats_from_ref_metrics(datamodule)
                if ref_degree_dist is None: ref_degree_dist = stats.get("ref_degree_dist")
                if ref_clustering_hist is None: ref_clustering_hist = stats.get("ref_clustering_hist")
                if ref_orbit_mean is None: ref_orbit_mean = stats.get("ref_orbit_mean")

                # If still missing, COMPUTE from datamodule (User Requested: "Load all graphs into memory and compute")
                if ref_degree_dist is None or ref_clustering_hist is None:
                    logger.info("[PlanarGraphReward] Reference stats missing from file. Computing from training set (InMemory)...")
                    stats_computed = self._compute_stats_from_datamodule(datamodule)
                    if ref_degree_dist is None: ref_degree_dist = stats_computed.get("ref_degree_dist") # Already normalized
                    if ref_clustering_hist is None: ref_clustering_hist = stats_computed.get("ref_clustering_hist")
                    if self.use_orb and ref_orbit_mean is None:
                        ref_orbit_mean = stats_computed.get("ref_orbit_mean")

        # Graceful handling of missing stats
        if ref_degree_dist is None:
            logger.warning("[PlanarGraphReward] Missing 'ref_degree_dist'. Disabling degree reward.")
            self.w_deg = 0.0
            self.ref_degree_dist = np.zeros(1) # dummy
        else:
            self.ref_degree_dist = self._safe_normalize(np.asarray(ref_degree_dist, dtype=np.float64))

        if ref_clustering_hist is None:
            logger.warning("[PlanarGraphReward] Missing 'ref_clustering_hist'. Disabling clustering reward.")
            self.w_clus = 0.0
            self.ref_clustering_hist = np.zeros(1) # dummy
        else:
            self.ref_clustering_hist = self._safe_normalize(np.asarray(ref_clustering_hist, dtype=np.float64))

        if self.use_orb:
            if ref_orbit_mean is None:
                logger.warning("[PlanarGraphReward] Missing 'ref_orbit_mean'. Disabling orbit reward.")
                self.w_orb = 0.0
                self.use_orb = False
                self.ref_orbit_mean = None
            else:
                 self.ref_orbit_mean = np.asarray(ref_orbit_mean, dtype=np.float64)
        else:
            self.ref_orbit_mean = None

    def state_dict_for_workers(self) -> Dict[str, object]:
        use_orb = bool(self.use_orb and self.ref_orbit_mean is not None)
        return {
            "ref_degree_dist": self.ref_degree_dist.astype(np.float64).tolist(),
            "ref_clustering_hist": self.ref_clustering_hist.astype(np.float64).tolist(),
            "ref_orbit_mean": None if not use_orb else self.ref_orbit_mean.astype(np.float64).tolist(),
        }

    @staticmethod
    def _safe_normalize(vec: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float64)
        s = float(vec.sum())
        if not np.isfinite(s) or s <= 0:
            return np.zeros_like(vec, dtype=np.float64)
        out = vec / (s + eps)
        out = np.clip(out, 0.0, 1.0)
        out = out / max(eps, float(out.sum()))
        return out

    @staticmethod
    def _js_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
        if p.shape != q.shape: return 1.0 # mismatch
        p = np.asarray(p, dtype=np.float64) + eps
        q = np.asarray(q, dtype=np.float64) + eps
        p = p / float(p.sum())
        q = q / float(q.sum())
        m = 0.5 * (p + q)
        kl_pm = float(np.sum(p * (np.log(p) - np.log(m))))
        kl_qm = float(np.sum(q * (np.log(q) - np.log(m))))
        return 0.5 * (kl_pm + kl_qm)

    def _degree_hist(self, G: nx.Graph) -> np.ndarray:
        hist = np.asarray(nx.degree_histogram(G), dtype=np.float64)
        target_len = int(self.ref_degree_dist.shape[0])
        if hist.shape[0] < target_len:
            hist = np.pad(hist, (0, target_len - hist.shape[0]))
        elif hist.shape[0] > target_len:
            hist = hist[:target_len]
        return self._safe_normalize(hist)

    def _clustering_hist(self, G: nx.Graph) -> np.ndarray:
        coeffs = list(nx.clustering(G).values())
        hist, _ = np.histogram(coeffs, bins=self.bins, range=(0.0, 1.0), density=False)
        hist = np.asarray(hist, dtype=np.float64)
        target_len = int(self.ref_clustering_hist.shape[0])
        if hist.shape[0] < target_len:
            hist = np.pad(hist, (0, target_len - hist.shape[0]))
        elif hist.shape[0] > target_len:
            hist = hist[:target_len]
        return self._safe_normalize(hist)

    def _orbit_vec(self, G: nx.Graph) -> Optional[np.ndarray]:
        if not self.use_orb:
            return None
        try:
            from analysis.spectre_utils import orca as _orca
        except Exception:
            return None

        try:
            counts = _orca(G)
            counts = np.asarray(counts, dtype=np.float64)
            if counts.ndim != 2 or counts.shape[0] <= 0:
                return None
            vec = np.sum(counts, axis=0) / float(G.number_of_nodes())
            if self.ref_orbit_mean is not None:
                ref_len = int(self.ref_orbit_mean.shape[0])
                if vec.shape[0] < ref_len:
                    vec = np.pad(vec, (0, ref_len - vec.shape[0]))
                elif vec.shape[0] > ref_len:
                    vec = vec[:ref_len]
            return vec
        except Exception:
            return None

    @staticmethod
    def _graph_hash(edge_types: torch.Tensor) -> str:
        if edge_types.dim() == 3:
            edge_idx = edge_types.detach().to("cpu").argmax(dim=-1).to(torch.uint8).numpy()
        else:
            edge_idx = edge_types.detach().to("cpu").to(torch.uint8).numpy()
        return hashlib.md5(edge_idx.tobytes()).hexdigest()

    def compute_components(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        valid_scores: List[float] = []
        deg_scores: List[float] = []
        clus_scores: List[float] = []
        orb_scores: List[float] = []
        train_scores: List[float] = []
        total_scores: List[float] = []

        for atom_types, edge_types in graphs:
            try:
                key = self._graph_hash(edge_types)
                cached = self._cache.get(key)
                if cached is not None:
                    v, d, c, o, t, r = cached
                    valid_scores.append(v)
                    deg_scores.append(d)
                    clus_scores.append(c)
                    orb_scores.append(o)
                    train_scores.append(t)
                    total_scores.append(r)
                    continue

                G = self._convert_tensor_to_networkx_graph(atom_types, edge_types)
                if G.number_of_nodes() <= 0 or G.number_of_edges() <= 0:
                    valid = 0.0
                else:
                    try:
                        valid = 1.0 if (nx.is_connected(G) and nx.check_planarity(G)[0]) else 0.0
                    except Exception:
                        valid = 0.0

                deg_sim = 0.0
                clus_sim = 0.0
                orb_sim = 0.0

                if valid > 0.0:
                    if self.w_deg > 0:
                        p_deg = self._degree_hist(G)
                        deg_sim = float(np.exp(-self.deg_scale * self._js_divergence(p_deg, self.ref_degree_dist)))

                    if self.w_clus > 0:
                        p_clus = self._clustering_hist(G)
                        clus_sim = float(np.exp(-self.clus_scale * self._js_divergence(p_clus, self.ref_clustering_hist)))

                    if self.use_orb and self.w_orb > 0:
                        vec = self._orbit_vec(G)
                        if vec is not None and self.ref_orbit_mean is not None:
                            dist = float(np.mean(np.abs(vec - self.ref_orbit_mean)))
                            orb_sim = float(np.exp(-self.orb_scale * dist))

                train_reward = float(self.w_deg * deg_sim + self.w_clus * clus_sim + self.w_orb * orb_sim)
                denom = float(self.w_valid + self.w_deg + self.w_clus + self.w_orb)
                if denom <= 0.0:
                    denom = 1.0
                total = float(valid * (self.w_valid + train_reward) / denom)

                valid_scores.append(valid)
                deg_scores.append(deg_sim)
                clus_scores.append(clus_sim)
                orb_scores.append(orb_sim)
                train_scores.append(train_reward)
                total_scores.append(total)

                if len(self._cache) >= self._cache_size:
                    try:
                        self._cache.pop(next(iter(self._cache)))
                    except Exception:
                        self._cache.clear()
                self._cache[key] = (valid, deg_sim, clus_sim, orb_sim, train_reward, total)

            except Exception:
                valid_scores.append(0.0)
                deg_scores.append(0.0)
                clus_scores.append(0.0)
                orb_scores.append(0.0)
                train_scores.append(0.0)
                total_scores.append(0.0)

        def _to(x: List[float]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        return {
            "valid": _to(valid_scores),
            "deg": _to(deg_scores),
            "clus": _to(clus_scores),
            "orb": _to(orb_scores),
            "train": _to(train_scores),
            "total": _to(total_scores),
        }

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        return self.compute_components(graphs)["total"]

    @staticmethod
    def _as_1d_float_array(x) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim != 1:
            raise ValueError(f"expected 1D array, got shape={arr.shape}")
        return arr

    def _load_reference_stats_from_ref_metrics(self, datamodule) -> Dict[str, np.ndarray]:
        """
        Load planar reward reference statistics from the dataset's `ref_metrics.pkl`
        Returns empty dict if not found, instead of crashing.
        """
        try:
            root = datamodule.train_dataloader().dataset.root
        except Exception:
             # Can't resolve root
             return {}

        ref_metrics_path = os.path.join(root, "ref_metrics.pkl")
        if hasattr(datamodule, "remove_h"):
            try:
                if bool(datamodule.remove_h):
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_no_h.pkl")
                else:
                    ref_metrics_path = ref_metrics_path.replace(".pkl", "_h.pkl")
            except Exception:
                pass

        if not os.path.exists(ref_metrics_path):
            return {}

        try:
            with open(ref_metrics_path, "rb") as f:
                payload = pickle.load(f)
        except Exception:
            return {}

        if not isinstance(payload, dict):
            return {}

        candidates: List[dict] = []
        if isinstance(payload.get("planar_reward_stats"), dict):
            candidates.append(payload["planar_reward_stats"])
        if isinstance(payload.get("train"), dict):
            candidates.append(payload["train"])
        candidates.append(payload)

        for cand in candidates:
            try:
                deg_raw = cand.get("ref_degree_dist", cand.get("degree_dist", cand.get("degree_hist")))
                clus_raw = cand.get("ref_clustering_hist", cand.get("clustering_hist"))
                orb_raw = cand.get("ref_orbit_mean", cand.get("orbit_mean"))

                if deg_raw is None or clus_raw is None:
                    continue

                # Be lenient: if they are scalars, ignore them (continue searching)
                if isinstance(deg_raw, (float, int)) or isinstance(clus_raw, (float, int)):
                     continue

                deg = self._as_1d_float_array(deg_raw)
                clus = self._as_1d_float_array(clus_raw)

                out: Dict[str, np.ndarray] = {
                    "ref_degree_dist": deg,
                    "ref_clustering_hist": clus,
                }
                if orb_raw is not None and not isinstance(orb_raw, (float, int)):
                     out["ref_orbit_mean"] = self._as_1d_float_array(orb_raw)

                return out
            except Exception:
                continue

        # Not found
        return {}

    def _compute_stats_from_datamodule(self, datamodule) -> Dict[str, np.ndarray]:
        """
        Manually compute reference statistics from the full training dataset.
        This provides a fallback when the pickle file only contains scalar MMD scores.
        """
        try:
            from analysis.spectre_utils import degree_worker, clustering_worker, orca
        except ImportError:
             logger.warning("[PlanarGraphReward] analysis.spectre_utils not found. Cannot compute stats.")
             return {}

        import networkx as nx
        import numpy as np

        # Helper to convert tensor to NX
        def _to_nx_local(X_dense, E_dense):
            if E_dense.dim() == 3:
                adj = E_dense.argmax(dim=-1).float()
            else:
                adj = E_dense.float()
            adj_np = adj.cpu().numpy()
            G = nx.from_numpy_array(adj_np)
            G.remove_edges_from(nx.selfloop_edges(G))
            return G

        logger.info("[Compute] Accessing training data loader...")
        try:
             loader = datamodule.train_dataloader()
        except Exception:
             return {}

        graphs_list = []
        from utils import to_dense

        # Iterate over all training batches
        for batch in loader:
             dense_data, node_mask = to_dense(batch.x, batch.edge_index, batch.edge_attr, batch.batch)
             # dense_data is (B, N, F)
             X, E = dense_data.X, dense_data.E
             B = X.size(0)
             for i in range(B):
                 # We must respect the mask to create correct graphs
                 mask_i = node_mask[i]
                 valid_nodes = mask_i.sum().item()
                 # Slice the submatrix
                 if E.dim() == 4:
                     E_sub = E[i, :valid_nodes, :valid_nodes, :]
                 else:
                     E_sub = E[i, :valid_nodes, :valid_nodes]

                 G = _to_nx_local(None, E_sub)
                 if G.number_of_nodes() > 0:
                     graphs_list.append(G)

        logger.info("[Compute] Collected %d valid graphs. Calculating statistics...", len(graphs_list))

        out = {}
        # 1. Degree
        deg_hists = [degree_worker(G) for G in graphs_list]
        max_len = max([len(h) for h in deg_hists] + [1])
        deg_sum = np.zeros(max_len)
        for h in deg_hists:
            deg_sum[:len(h)] += h
        if deg_sum.sum() > 0:
            out["ref_degree_dist"] = deg_sum / deg_sum.sum()
        else:
             out["ref_degree_dist"] = np.zeros(1)

        # 2. Clustering
        clus_hists = [clustering_worker((G, self.bins)) for G in graphs_list]
        if clus_hists:
            clus_sum = np.sum(clus_hists, axis=0)
            if clus_sum.sum() > 0:
                out["ref_clustering_hist"] = clus_sum / clus_sum.sum()
            else:
                 out["ref_clustering_hist"] = np.zeros(self.bins)
        else:
             out["ref_clustering_hist"] = np.zeros(self.bins)

        # 3. Orbit
        if self.use_orb:
            orb_vecs = []
            for G in graphs_list:
                try:
                    cnts = orca(G)
                    if G.number_of_nodes() > 0:
                         vec = cnts.sum(axis=0) / G.number_of_nodes()
                         orb_vecs.append(vec)
                except Exception:
                     pass
            if orb_vecs:
                out["ref_orbit_mean"] = np.mean(orb_vecs, axis=0)

        logger.info("[Compute] Done. ref_degree_dist len=%d", len(out.get('ref_degree_dist', [])))
        return out


class SBMGraphReward(PlanarGraphReward):
    """
    SBM Graph Reward (Stochastic Block Model).
    Inherits distribution matching logic from PlanarGraphReward.
    Validity Definition: Connectivity Only.
    """
    def __init__(self, **kwargs):
        # Override name for clarity
        super().__init__(**kwargs)
        self.name = "sbm_graph"

    def compute_components(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        valid_scores: List[float] = []
        deg_scores: List[float] = []
        clus_scores: List[float] = []
        orb_scores: List[float] = []
        train_scores: List[float] = []
        total_scores: List[float] = []

        for atom_types, edge_types in graphs:
            try:
                key = self._graph_hash(edge_types)
                cached = self._cache.get(key)
                if cached is not None:
                    v, d, c, o, t, r = cached
                    valid_scores.append(v)
                    deg_scores.append(d)
                    clus_scores.append(c)
                    orb_scores.append(o)
                    train_scores.append(t)
                    total_scores.append(r)
                    continue

                G = self._convert_tensor_to_networkx_graph(atom_types, edge_types)
                if G.number_of_nodes() <= 0 or G.number_of_edges() <= 0:
                    valid = 0.0
                    conn_score = 0.0
                else:
                    try:
                        # Partial Score for SBM: Connectivity Ratio
                        ccs = list(nx.connected_components(G))
                        max_cc_len = max(len(c) for c in ccs)
                        conn_score = float(max_cc_len) / float(G.number_of_nodes())
                    except Exception:
                        conn_score = 0.0

                    # SBM Validity: Just connectivity
                    valid = 1.0 if conn_score >= 0.999 else 0.0

                deg_sim = 0.0
                clus_sim = 0.0
                orb_sim = 0.0

                # Base Reward: Connectivity Ratio directly
                base_reward = conn_score

                if valid > 0.0:
                    if self.w_deg > 0:
                        p_deg = self._degree_hist(G)
                        deg_sim = float(np.exp(-self.deg_scale * self._js_divergence(p_deg, self.ref_degree_dist)))

                    if self.w_clus > 0:
                        p_clus = self._clustering_hist(G)
                        clus_sim = float(np.exp(-self.clus_scale * self._js_divergence(p_clus, self.ref_clustering_hist)))

                    if self.use_orb and self.w_orb > 0:
                        vec = self._orbit_vec(G)
                        if vec is not None and self.ref_orbit_mean is not None:
                            dist = float(np.mean(np.abs(vec - self.ref_orbit_mean)))
                            orb_sim = float(np.exp(-self.orb_scale * dist))

                train_reward = float(self.w_deg * deg_sim + self.w_clus * clus_sim + self.w_orb * orb_sim)

                # Total Reward
                total = float(base_reward + valid * train_reward)

                valid_scores.append(conn_score) # Log connectivity ratio as 'valid' score for visibility
                deg_scores.append(deg_sim)
                clus_scores.append(clus_sim)
                orb_scores.append(orb_sim)
                train_scores.append(train_reward)
                total_scores.append(total)

                if len(self._cache) >= self._cache_size:
                    try: self._cache.pop(next(iter(self._cache)))
                    except Exception: self._cache.clear()
                self._cache[key] = (conn_score, deg_sim, clus_sim, orb_sim, train_reward, total)

            except Exception:
                valid_scores.append(0.0)
                deg_scores.append(0.0)
                clus_scores.append(0.0)
                orb_scores.append(0.0)
                train_scores.append(0.0)
                total_scores.append(0.0)

        def _to(x: List[float]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        return {
            "valid": _to(valid_scores),
            "deg": _to(deg_scores),
            "clus": _to(clus_scores),
            "orb": _to(orb_scores),
            "train": _to(train_scores),
            "total": _to(total_scores),
        }


class TreeGraphReward(PlanarGraphReward):
    """
    Tree Graph Reward.
    Inherits distribution matching logic from PlanarGraphReward.
    Validity Definition: Tree (Connected + Acyclic).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "tree_graph"

    def compute_components(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        valid_scores: List[float] = []
        deg_scores: List[float] = []
        clus_scores: List[float] = []
        orb_scores: List[float] = []
        train_scores: List[float] = []
        total_scores: List[float] = []

        for atom_types, edge_types in graphs:
            try:
                key = self._graph_hash(edge_types)
                cached = self._cache.get(key)
                if cached is not None:
                    v, d, c, o, t, r = cached
                    valid_scores.append(v)
                    deg_scores.append(d)
                    clus_scores.append(c)
                    orb_scores.append(o)
                    train_scores.append(t)
                    total_scores.append(r)
                    continue

                G = self._convert_tensor_to_networkx_graph(atom_types, edge_types)
                if G.number_of_nodes() <= 0:
                    conn_score = 0.0
                    is_tree = False
                    valid = 0.0
                else:
                    # 1. Connectivity Score
                    try:
                        ccs = list(nx.connected_components(G))
                        max_cc_len = max(len(c) for c in ccs)
                        conn_score = float(max_cc_len) / float(G.number_of_nodes())
                    except Exception:
                        conn_score = 0.0

                    # 2. Tree Check (Acyclic + Connected)
                    # Note: is_tree() requires connectivity. If not connected, it is False.
                    # We also want to reward being a Forest (acyclic) if not connected?
                    # Keep it simple: Tree = Connected + No Cycles.
                    try:
                        is_tree = nx.is_tree(G)
                    except Exception:
                        is_tree = False

                    valid = 1.0 if is_tree else 0.0

                deg_sim = 0.0
                clus_sim = 0.0
                orb_sim = 0.0

                # Base Reward:
                # Strict validity: non-tree -> 0.0, tree -> connectivity score (1.0).
                base_reward = conn_score if is_tree else 0.0

                if valid > 0.0:
                    if self.w_deg > 0:
                        p_deg = self._degree_hist(G)
                        deg_sim = float(np.exp(-self.deg_scale * self._js_divergence(p_deg, self.ref_degree_dist)))

                    if self.w_clus > 0:
                        p_clus = self._clustering_hist(G)
                        clus_sim = float(np.exp(-self.clus_scale * self._js_divergence(p_clus, self.ref_clustering_hist)))

                    if self.use_orb and self.w_orb > 0:
                         # Orbit is crucial for Trees to distinguish chain vs star etc.
                        vec = self._orbit_vec(G)
                        if vec is not None and self.ref_orbit_mean is not None:
                            dist = float(np.mean(np.abs(vec - self.ref_orbit_mean)))
                            orb_sim = float(np.exp(-self.orb_scale * dist))

                train_reward = float(self.w_deg * deg_sim + self.w_clus * clus_sim + self.w_orb * orb_sim)
                total = float(base_reward + valid * train_reward)

                valid_scores.append(float(valid))
                deg_scores.append(deg_sim)
                clus_scores.append(clus_sim)
                orb_scores.append(orb_sim)
                train_scores.append(train_reward)
                total_scores.append(total)

                if len(self._cache) >= self._cache_size:
                    try: self._cache.pop(next(iter(self._cache)))
                    except Exception: self._cache.clear()
                self._cache[key] = (float(valid), deg_sim, clus_sim, orb_sim, train_reward, total)

            except Exception:
                valid_scores.append(0.0)
                deg_scores.append(0.0)
                clus_scores.append(0.0)
                orb_scores.append(0.0)
                train_scores.append(0.0)
                total_scores.append(0.0)

        def _to(x: List[float]) -> torch.Tensor:
            return torch.tensor(x, dtype=torch.float32, device=self.device)

        return {
            "valid": _to(valid_scores),
            "deg": _to(deg_scores),
            "clus": _to(clus_scores),
            "orb": _to(orb_scores),
            "train": _to(train_scores),
            "total": _to(total_scores),
        }
