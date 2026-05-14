import logging

import torch
import numpy as np
from typing import List, Tuple, Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

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

from grpo.rewards.base import BaseRewardFunction, GaussianModifier, resolve_target_task
from analysis.rdkit_functions import build_molecule

logger = logging.getLogger(__name__)


class TargetMPOReward(BaseRewardFunction):
    """
    [DEPRECATED] Goal-Directed MPO Reward Function (Generic Target).

    This class is deprecated. Please use specific reward classes (e.g.,
    ValsartanSmartsReward) or TDCOracleReward instead.
    Legacy: Supports specific tasks from Guacamol Benchmark or other target-based objectives.
    """
    _DEFAULT_ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    def __init__(self, target_task: str = "penalized_logp", atom_decoder: Optional[List[str]] = None, device: Optional[torch.device] = None):
        logger.warning("[WARNING] TargetMPOReward is DEPRECATED. Please switch to specific reward classes like ValsartanSmartsReward.")
        super().__init__("target_mpo", device=device)
        self.atom_decoder = atom_decoder or self._DEFAULT_ATOM_DECODER
        self.target_task = target_task.lower()

        # Lazy import to avoid errors when the library is not installed
        try:
            from guacamol import standard_benchmarks as _standard_benchmarks  # noqa: F401
        except ImportError:
            raise ImportError("Guacamol is not installed. Please install guacamol to use this reward.")

        # Task initialization
        self.objective = None
        self._init_task()

    def _init_task(self):
        """ Initialize specific task objective function """
        # 1. Penalized LogP (Custom Implementation)
        if self.target_task == "penalized_logp":
            self.score_metric = self._score_penalized_logp

        # 2. Aripiprazole Similarity
        elif self.target_task == "aripiprazole_similarity":
            from guacamol.standard_benchmarks import aripiprazole_similarity
            benchmark = aripiprazole_similarity()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 3. QED Maximization
        elif self.target_task == "qed":
            from guacamol.standard_benchmarks import qed_benchmark
            benchmark = qed_benchmark()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 4. Osimertinib MPO (Target MW, TPSA, LogP + Similarity)
        elif self.target_task == "osimertinib_mpo":
            from guacamol.standard_benchmarks import hard_osimertinib
            benchmark = hard_osimertinib()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 5. Fexofenadine MPO (Max LogP, Min TPSA, Min RotBonds + Similarity)
        elif self.target_task == "fexofenadine_mpo":
            from guacamol.standard_benchmarks import hard_fexofenadine
            benchmark = hard_fexofenadine()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 6. Ranolazine MPO (Make Ranolazine more polar + add fluorine)
        elif self.target_task == "ranolazine_mpo":
            from guacamol.standard_benchmarks import ranolazine_mpo
            benchmark = ranolazine_mpo()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 7. Perindopril MPO (similarity + aromatic rings constraint)
        elif self.target_task == "perindopril_mpo":
            from guacamol.standard_benchmarks import perindopril_rings
            benchmark = perindopril_rings()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 8. Amlodipine MPO (similarity + ring count constraint)
        elif self.target_task == "amlodipine_mpo":
            from guacamol.standard_benchmarks import amlodipine_rings
            benchmark = amlodipine_rings()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 9. Sitagliptin MPO (property matching with dissimilarity + isomer constraint)
        elif self.target_task == "sitagliptin_mpo":
            from guacamol.standard_benchmarks import sitagliptin_replacement
            benchmark = sitagliptin_replacement()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        # 10. Zaleplon MPO (similarity + formula constraint)
        elif self.target_task == "zaleplon_mpo":
            from guacamol.standard_benchmarks import zaleplon_with_other_formula
            benchmark = zaleplon_with_other_formula()
            self.objective = benchmark.objective
            self.score_metric = self._score_guacamol_objective

        else:
            raise ValueError(
                f"Unknown target task: {self.target_task}. Supported: penalized_logp, aripiprazole_similarity, qed, "
                "osimertinib_mpo, fexofenadine_mpo, ranolazine_mpo, perindopril_mpo, amlodipine_mpo, sitagliptin_mpo, zaleplon_mpo"
            )

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rewards = []
        for atom_types, edge_types in graphs:
            mol = self._graph_to_mol(atom_types, edge_types)

            # 1. Basic existence check
            if mol is None:
                rewards.append(0.0)
                continue

            # 2. RDKit Sanitize (strict mode attempt)
            is_valid = False
            try:
                Chem.SanitizeMol(mol)
                is_valid = True
            except Exception:
                is_valid = False

            # 3. Compute score
            try:
                if is_valid:
                    frags = Chem.GetMolFrags(mol, asMols=True)
                    if len(frags) > 1:
                        # Soft Reward Strategy:
                        # 1. Find largest fragment
                        largest_mol = max(frags, key=lambda m: m.GetNumAtoms())
                        # 2. Score largest fragment
                        try:
                             base_score = self.score_metric(largest_mol)
                        except Exception:
                             base_score = 0.0

                        rewards.append(float(base_score))
                    else:
                        score = self.score_metric(mol)
                        rewards.append(float(score))
                else:
                    # Invalid molecule -> Strict Reward: 0.0
                    rewards.append(0.0)
            except Exception:
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _score_guacamol_objective(self, mol) -> float:
        """Score using Guacamol Objective."""
        smi = Chem.MolToSmiles(mol)
        if not smi:
            return 0.0
        return self.objective.score(smi)

    def _score_penalized_logp(self, mol) -> float:
        """
        Penalized LogP = LogP - SA - CycleScore
        Ref: GCPN, MolDQN, etc.
        """
        # 1. LogP
        try:
            logp = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        except Exception:
            return -10.0 # Penalty

        # 2. SA Score
        sa = 10.0
        if sascorer:
            try:
                sa = sascorer.calculateScore(mol)
            except Exception:
                pass

        # 3. Cycle Score (Ring penalty for large rings > 6)
        cycle_score = 0.0
        try:
             cycle_list = mol.GetRingInfo().AtomRings()
             for ring in cycle_list:
                 if len(ring) > 6:
                     cycle_score += 1.0
        except Exception:
            pass

        # Formula: LogP - SA - Cycle
        # Note: Often normalized or clipped in literature, but here we perform raw optimization first
        # Usually we want to maximize this.
        return logp - sa - cycle_score

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
