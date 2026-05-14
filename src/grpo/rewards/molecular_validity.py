import logging
import time
import math

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from collections import Counter

from rdkit import Chem
from rdkit.Chem import QED, AllChem

import sys
import os

try:
    import sascorer  # Standard synthetic accessibility scoring script (Guacamol ecosystem)
except ImportError:
    sascorer = None
    conda_prefix = sys.prefix
    contrib_path = os.path.join(conda_prefix, "share", "RDKit", "Contrib", "SA_Score")
    candidate = os.path.join(contrib_path, "sascorer.py")
    if os.path.exists(candidate):
        if contrib_path not in sys.path:
            sys.path.append(contrib_path)
        try:
            import sascorer
        except ImportError as e:
            logging.getLogger(__name__).warning("[GRPO] Found sascorer but import failed: %s", e)
            sascorer = None

from grpo.rewards.base import BaseRewardFunction
from analysis.rdkit_functions import build_molecule, build_molecule_with_partial_charges

logger = logging.getLogger(__name__)

_SA_FALLBACK_WARNED = False


class MolecularValidityReward(BaseRewardFunction):
    """
    Reward function for molecular datasets such as Guacamol (streamlined version).

    Features:
    1. Removed redundant constant configuration.
    2. Invalid molecules skip distribution and conformer reward computation.
    3. Removed reason string return.
    """

    # Only keep necessary distribution defaults
    _DEFAULT_ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    _DEFAULT_TARGET_NODE_DIST = {
        0: 0.74, 1: 0.11, 2: 0.11, 3: 0.014,
        4: 0.0, 5: 0.002, 6: 0.008, 7: 0.0, 8: 0.001, 9: 0.015, 10: 0.0, 11: 0.0
    }

    _DEFAULT_TARGET_EDGE_DIST = {
        0: 0.925, 1: 0.036, 2: 0.005, 3: 0.0002, 4: 0.033
    }

    def __init__(
        self,
        atom_decoder: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        target_node_dist: Optional[Dict[int, float]] = None,
        target_edge_dist: Optional[Dict[int, float]] = None,
        dist_coef: float = 0.0,
        scale_factor: float = 10.0,
        clip_range: float = 2.0,
        edge_dist_factor: float = 1.0,
        precomputed_node_weights: Optional[Dict[int, float]] = None,
        precomputed_edge_weights: Optional[Dict[int, float]] = None,
        conformer_weight: float = 0.5,
        conformer_num: int = 5,
        conformer_eref: float = 1.0,
        conformer_deref: float = 5.0,
        conformer_s1: float = 0.5,
        conformer_s2: float = 2.0,
    ):
        super().__init__("molecular_validity", device=device)

        if build_molecule is None:
            raise ImportError("analysis.rdkit_functions was not imported correctly.")

        self.atom_decoder = atom_decoder or self._DEFAULT_ATOM_DECODER

        # Distribution configuration
        # avoid tensor truth-value ambiguity when caller passes torch.Tensor
        node_dist_in = target_node_dist if target_node_dist is not None else self._DEFAULT_TARGET_NODE_DIST
        edge_dist_in = target_edge_dist if target_edge_dist is not None else self._DEFAULT_TARGET_EDGE_DIST
        self.target_node_dist = self._to_distribution_dict(node_dist_in)
        self.target_edge_dist = self._to_distribution_dict(edge_dist_in)

        self.dist_coef = dist_coef
        self.scale_factor = scale_factor
        self.clip_range = clip_range
        self.edge_dist_factor = edge_dist_factor

        self.precomputed_node_weights = self._sanitize_weight_dict(precomputed_node_weights)
        self.precomputed_edge_weights = self._sanitize_weight_dict(precomputed_edge_weights)

        # Conformer parameters
        self.conformer_weight = conformer_weight
        self.conformer_num = conformer_num
        self.conformer_eref = conformer_eref
        self.conformer_deref = conformer_deref
        self.conformer_s1 = conformer_s1
        self.conformer_s2 = conformer_s2

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        """
        Computation pipeline:
        1. Parse graphs -> 2. Base reward (Validity) -> 3. Conformer (valid only)
        -> 4. Distribution (valid only) -> 5. Aggregate
        """
        batch_size = len(graphs)
        t_start = time.time()

        # 1. Preprocessing
        t0 = time.time()
        batch_indices, mols, invalid_mask = self._preprocess_batch(graphs)
        t_pre = time.time() - t0

        # 2. Base chemistry reward (update invalid_mask if < 0)
        t0 = time.time()
        base_rewards, qed_scores, sa_scores = self._compute_chem_metrics(mols)
        for i, r in enumerate(base_rewards):
            if r < 0:
                invalid_mask[i] = True
        t_chem = time.time() - t0

        # 3. Conformer score (skip invalid)
        t0 = time.time()
        conf_scores = self._compute_batch_conformer_scores(mols, invalid_mask)
        t_conf = time.time() - t0

        # 4. Distribution score (skip invalid for per-molecule computation, but weight stats include all)
        t0 = time.time()
        dist_scores = self._compute_batch_distribution_scores(batch_indices, invalid_mask)
        t_dist = time.time() - t0

        # 5. Aggregate
        t0 = time.time()
        final_rewards = self._aggregate_final_rewards(
            base_rewards, dist_scores, conf_scores, invalid_mask,
            debug_info=(qed_scores, sa_scores)
        )
        t_aggr = time.time() - t0

        total = time.time() - t_start
        eps = 1e-12
        if total > 0:
            def pct(v):
                return 100.0 * v / max(total, eps)

            logger.debug(
                "[Profile] Batch=%d | Total=%.3fs\n"
                "   >> Preprocess: %.3fs (%.1f%%)\n"
                "   >> Chem Metrics: %.3fs (%.1f%%)\n"
                "   >> Conformer: %.3fs (%.1f%%)\n"
                "   >> Dist Scores: %.3fs (%.1f%%)\n"
                "   >> Aggregate: %.3fs (%.1f%%)",
                batch_size, total,
                t_pre, pct(t_pre),
                t_chem, pct(t_chem),
                t_conf, pct(t_conf),
                t_dist, pct(t_dist),
                t_aggr, pct(t_aggr),
            )

        return torch.tensor(final_rewards, dtype=torch.float32, device=self.device)

    # --------------------------------------------------------------------------
    # Core logic
    # --------------------------------------------------------------------------

    def _preprocess_batch(self, graphs):
        batch_indices = []
        mols = []
        invalid_mask = []
        for atom_types, edge_types in graphs:
            # Extract indices
            idx_pair = self._extract_graph_indices(atom_types, edge_types)
            batch_indices.append(idx_pair)
            # Build molecule
            mol = self._graph_to_mol(atom_types, edge_types)
            mols.append(mol)
            invalid_mask.append(mol is None)
        return batch_indices, mols, invalid_mask

    def _compute_chem_metrics(self, mols: List[Optional["Chem.Mol"]]):
        """Compute base reward (Valid/QED/SA) without returning a reason string."""
        base_rewards, qed_scores, sa_scores = [], [], []
        for mol in mols:
            r, qed, sa = self._compute_base_reward_single(mol)
            base_rewards.append(r)
            qed_scores.append(qed)
            sa_scores.append(sa)
        return base_rewards, qed_scores, sa_scores

    def _compute_batch_conformer_scores(self, mols, invalid_mask):
        scores = []
        if self.conformer_weight <= 0:
            return [(0.0, 0.0, 0.0)] * len(mols)

        t_start = time.time()
        timing_acc = {"add_h": 0.0, "embed": 0.0, "optimize": 0.0, "scoring": 0.0}
        valid_calls = 0

        for i, mol in enumerate(mols):
            if invalid_mask[i]:
                scores.append((0.0, 0.0, 0.0))
            else:
                score_tuple, timing = self._compute_conformer_stability(mol)
                scores.append(score_tuple)
                if timing is not None:
                    valid_calls += 1
                    for key in timing_acc:
                        timing_acc[key] += timing.get(key, 0.0)

        total = time.time() - t_start
        if total > 0 and valid_calls > 0:
            def pct(v):
                return 100.0 * v / max(total, 1e-12)

            logger.debug(
                "[Conformer Profile] Batch=%d | Valid=%d | Total=%.3fs\n"
                "   >> AddHs: %.3fs (%.1f%%)\n"
                "   >> Embed: %.3fs (%.1f%%)\n"
                "   >> Optimize: %.3fs (%.1f%%)\n"
                "   >> Scoring: %.3fs (%.1f%%)",
                len(mols), valid_calls, total,
                timing_acc['add_h'], pct(timing_acc['add_h']),
                timing_acc['embed'], pct(timing_acc['embed']),
                timing_acc['optimize'], pct(timing_acc['optimize']),
                timing_acc['scoring'], pct(timing_acc['scoring']),
            )

        return scores

    def _compute_batch_distribution_scores(self, batch_indices, invalid_mask):
        """Compute distribution scores; invalid molecules get 0 to save computation."""
        # A. Determine weights (aggregate over full batch so weights reflect true generation distribution)
        if self.precomputed_node_weights and self.precomputed_edge_weights:
            node_weights = self.precomputed_node_weights
            edge_weights = self.precomputed_edge_weights
        else:
            node_weights, edge_weights = self._calculate_dynamic_weights(batch_indices)

        # B. Compute per-molecule scores
        dist_scores = []
        for i, (atom_indices, edge_indices_flat) in enumerate(batch_indices):
            # Optimization: skip detailed distribution scoring for invalid molecules
            if invalid_mask[i]:
                dist_scores.append((0.0, 0.0))
                continue

            # Node score
            n_score = 0.0
            n_nodes = len(atom_indices)
            if n_nodes > 0:
                n_score = sum(node_weights.get(int(idx), -self.clip_range) for idx in atom_indices) / n_nodes

            # Edge score
            e_score = 0.0
            if len(edge_indices_flat) > 0:
                # Geometric adaptive scaling: normalize by node count, not N^2, to prevent sparse edge signals from being overwhelmed
                norm_edges = max(1, n_nodes)
                edge_sum = sum(edge_weights.get(int(idx), -self.clip_range) for idx in edge_indices_flat)
                e_score = edge_sum / norm_edges

            dist_scores.append((n_score, e_score))
        return dist_scores

    def _calculate_dynamic_weights(self, batch_indices):
        nc, ec = Counter(), Counter()
        tn, te = 0, 0
        for atom_indices, edge_indices_flat in batch_indices:
            nc.update(int(i) for i in atom_indices)
            ec.update(int(i) for i in edge_indices_flat)
            tn += len(atom_indices)
            te += len(edge_indices_flat)
        return self._compute_weights_from_counts(nc, ec, tn, te)

    def _aggregate_final_rewards(self, base_rewards, dist_scores, conf_scores, invalid_mask, debug_info):
        final_rewards = []
        qed_s, sa_s = debug_info

        for i in range(len(base_rewards)):
            base_r = base_rewards[i]

            if invalid_mask[i]:
                # Invalid molecules directly return the base reward (typically -1.0)
                final_rewards.append(base_r)
                continue

            n_score, e_score = dist_scores[i]
            c_score, _, _ = conf_scores[i]

            dist_term = self.dist_coef * (n_score + self.edge_dist_factor * e_score)
            conf_term = self.conformer_weight * c_score

            total_r = base_r + dist_term + conf_term

            logger.debug(
                "[Reward] idx=%d, Base=%.2f, Qed=%.2f, Sa=%.2f, "
                "D_n=%.2f, D_e=%.2f, C=%.2f, Tot=%.2f",
                i, base_r, qed_s[i], sa_s[i],
                n_score, e_score, c_score, total_r,
            )
            final_rewards.append(total_r)

        return final_rewards

    # --------------------------------------------------------------------------
    # Helper methods
    # --------------------------------------------------------------------------

    def _graph_to_mol(self, atom_types, edge_types):
        if build_molecule is None: return None
        try:
            at = torch.as_tensor(atom_types).long().cpu()
            et = torch.as_tensor(edge_types).long().cpu()
            if at.dim() == 2: at = at.argmax(dim=-1)
            if et.dim() == 3: et = et.argmax(dim=-1)
            if at.numel() == 0: return None
            mol = build_molecule(at, et, self.atom_decoder)
            return mol if (mol and mol.GetNumAtoms() > 0) else None
        except Exception: return None

    @staticmethod
    def _compute_base_reward_single(mol) -> Tuple[float, float, float]:
        """
        Returns: (reward, qed, sa_normalized)
        Reward Range: [-1.0, 1.0]
        """
        # Hardcoded bounds
        MIN_R, MAX_R = -1.0, 1.0

        if mol is None: return MIN_R, 0.0, 0.0

        # 1. Connectivity
        try:
            if len(Chem.GetMolFrags(mol)) > 1: return MIN_R, 0.0, 0.0
        except Exception: return MIN_R, 0.0, 0.0

        # 2. Carbon/hydrogen-only check
        has_hetero = any(a.GetAtomicNum() not in (6, 1) for a in mol.GetAtoms())
        if not has_hetero: return -0.2, 0.0, 0.0

        # 3. Valence
        try: Chem.SanitizeMol(mol)
        except Exception: return -0.5, 0.0, 0.0

        # 4. Metrics
        try: qed = float(QED.qed(mol))
        except Exception: qed = 0.0

        global _SA_FALLBACK_WARNED
        raw_sa = 10.0
        if sascorer:
            try: raw_sa = float(sascorer.calculateScore(mol))
            except Exception: pass
        sa_norm = float(np.clip(1.0 - (raw_sa - 1.0) / 9.0, 0.0, 1.0))

        final_reward = 0.6 * qed + 0.4 * sa_norm
        return float(np.clip(final_reward, MIN_R, MAX_R)), qed, sa_norm

    def _compute_conformer_stability(self, mol) -> Tuple[Tuple[float, float, float], Optional[Dict[str, float]]]:
        """
        Compute conformer stability score for a molecule and return timing statistics.
        """
        if self.conformer_weight <= 0 or mol is None:
            return (0.0, 0.0, 0.0), None

        try:
            from rdkit.Chem import AllChem
        except ImportError:
            logger.debug("AllChem not available")
            return (0.0, 0.0, 0.0), None

        timing = {"add_h": 0.0, "embed": 0.0, "optimize": 0.0, "scoring": 0.0}

        try:
            # 1. Preprocessing: add hydrogens
            t0 = time.time()
            mol_h = Chem.AddHs(Chem.Mol(mol), addCoords=True)
            timing["add_h"] += time.time() - t0

            # 2. Generate initial 3D conformers (Embed)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            params.useRandomCoords = True
            params.maxIterations = 100
            # Disable some expensive advanced corrections
            params.useSmallRingTorsions = False

            t0 = time.time()
            cids = AllChem.EmbedMultipleConfs(mol_h, numConfs=self.conformer_num, params=params)
            timing["embed"] += time.time() - t0

            if not cids:
                return (0.0, 0.0, 0.0), timing

            # 3. Force field optimization
            t0 = time.time()
            try:
                if AllChem.MMFFHasAllMoleculeParams(mol_h):
                    res = AllChem.MMFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=500)
                else:
                    res = AllChem.UFFOptimizeMoleculeConfs(mol_h, numThreads=1, maxIters=500)
            except Exception:
                timing["optimize"] += time.time() - t0
                return (0.0, 0.0, 0.0), timing
            timing["optimize"] += time.time() - t0

            # 4. Check convergence
            t0 = time.time()
            energies = [float(r[1]) for r in res if r[0] == 0]

            if not energies:
                return (0.0, 0.0, 0.0), timing

            # 5. Compute scores
            E_min, E_max = min(energies), max(energies)
            n_heavy = mol_h.GetNumHeavyAtoms() or 1

            S_energy = math.exp(-self.conformer_s1 * max(0.0, (E_min/n_heavy) - self.conformer_eref))
            S_range = math.exp(-self.conformer_s2 * max(0.0, (E_max - E_min) - self.conformer_deref))
            timing["scoring"] += time.time() - t0

            return (0.7 * S_energy + 0.3 * S_range, S_energy, S_range), timing

        except Exception:
            return (0.0, 0.0, 0.0), timing

    @staticmethod
    def _extract_graph_indices(atom_types, edge_types):
        if torch.is_tensor(atom_types):
            if atom_types.dim() == 2: atom_types = atom_types.argmax(-1)
            a_idx = atom_types.detach().cpu().numpy()
        else: a_idx = np.array(atom_types)

        if torch.is_tensor(edge_types):
            if edge_types.dim() == 3: edge_types = edge_types.argmax(-1)
            e_idx = edge_types.detach().cpu().numpy()
        else: e_idx = np.array(edge_types)
        return a_idx, e_idx.flatten()

    @staticmethod
    def _to_distribution_dict(d):
        # guard against tensor truth-value ambiguity
        if d is None:
            return {}
        if torch.is_tensor(d):
            if d.numel() == 0:
                return {}
            d = d.detach().cpu().numpy()
        if isinstance(d, (list, tuple, np.ndarray)):
            if len(d) == 0:
                return {}
            d = {i: float(v) for i, v in enumerate(d)}
        if isinstance(d, dict):
            if len(d) == 0:
                return {}
        total = sum(d.values())
        return {int(k): v/total for k, v in d.items()} if total > 0 else {}

    @staticmethod
    def _sanitize_weight_dict(d):
        return {int(k): float(v) for k, v in d.items()} if d else None

    @staticmethod
    def compute_distribution_weights(
        graphs,
        target_node_dist=None,
        target_edge_dist=None,
        scale_factor=10.0,
        clip_range=2.0,
    ):
        """
        Pre-compute node/edge distribution weights based on all generated graphs.
        Intended for a single pass in the main process, avoiding bias from
        small-batch statistics in subprocesses.
        """
        from collections import Counter

        eps = 1e-6
        tnd = MolecularValidityReward._to_distribution_dict(
            target_node_dist if target_node_dist is not None else MolecularValidityReward._DEFAULT_TARGET_NODE_DIST
        )
        ted = MolecularValidityReward._to_distribution_dict(
            target_edge_dist if target_edge_dist is not None else MolecularValidityReward._DEFAULT_TARGET_EDGE_DIST
        )

        nc, ec = Counter(), Counter()
        tn = te = 0
        for atom_types, edge_types in graphs:
            at_idx, ed_idx = MolecularValidityReward._extract_graph_indices(atom_types, edge_types)
            nc.update(int(i) for i in at_idx)
            ec.update(int(i) for i in ed_idx)
            tn += len(at_idx)
            te += len(ed_idx)

        def calc_w(counts, total, target_dist):
            w = {}
            for idx in set(target_dist) | set(counts):
                p_tgt = target_dist.get(idx, 0.0)
                p_batch = counts.get(idx, 0) / max(1, total)
                val = np.log(p_tgt + eps) - np.log(p_batch + eps)
                w[idx] = float(np.clip(val * scale_factor, -clip_range, clip_range))
            return w

        return calc_w(nc, tn, tnd), calc_w(ec, te, ted)

    def _compute_weights_from_counts(self, nc, ec, tn, te):
        eps = 1e-6
        sf, cr = self.scale_factor, self.clip_range
        tnd, ted = self.target_node_dist, self.target_edge_dist

        def calc_w(counts, total, target_dist):
            w = {}
            for idx in set(target_dist) | set(counts):
                p_tgt = target_dist.get(idx, 0.0)
                p_batch = counts.get(idx, 0) / max(1, total)
                val = np.log(p_tgt + eps) - np.log(p_batch + eps)
                w[idx] = float(np.clip(val * sf, -cr, cr))
            return w
        return calc_w(nc, tn, tnd), calc_w(ec, te, ted)
