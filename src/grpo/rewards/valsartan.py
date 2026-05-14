import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdFMCS

from grpo.rewards.base import BaseRewardFunction

from analysis.rdkit_functions import build_molecule

logger = logging.getLogger(__name__)


class GaussianModifier:
    """Gaussian modifier for MPO rewards"""
    def __init__(self, mu: float, sigma: float):
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x: float) -> float:
        return float(np.exp(-0.5 * np.power((x - self.mu) / self.sigma, 2)))


class ValsartanSmartsReward(BaseRewardFunction):
    """
    Valsartan Smarts Reward (Full & Easy/Curricular)
    """
    _DEFAULT_ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]

    def __init__(self, mode: str = "full", atom_decoder: Optional[List[str]] = None, device: Optional[torch.device] = None):
        super().__init__("valsartan_smarts", device=device)
        self.mode = mode  # "full" or "easy"
        self.atom_decoder = atom_decoder or self._DEFAULT_ATOM_DECODER

        # Initialize resources
        self._init_resources()

    def _init_resources(self):
        # Valsartan core SMARTS
        self.valsartan_smarts = "CN(C=O)Cc1ccc(c2ccccc2)cc1"
        # Use MolFromSmiles to ensure concrete atom types for MCS comparison
        # (MolFromSmarts creates query atoms which can confuse rdFMCS)
        self.valsartan_mol = Chem.MolFromSmiles(self.valsartan_smarts)
        if self.valsartan_mol is None:
             # Fallback if sanitization fails (unlikely for this string)
             self.valsartan_mol = Chem.MolFromSmarts(self.valsartan_smarts)
        # Query mol used for strict substructure (SMARTS) matching bonus.
        # Note: this is a *substructure* match, not full-molecule equality.
        self.valsartan_query = Chem.MolFromSmarts(self.valsartan_smarts)

        self.valsartan_num_atoms = self.valsartan_mol.GetNumHeavyAtoms()

        # Target properties from Sitagliptin
        sitagliptin_smiles = "NC(CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F"
        sitagliptin_mol = Chem.MolFromSmiles(sitagliptin_smiles)

        target_logp = Descriptors.MolLogP(sitagliptin_mol)
        target_tpsa = Descriptors.TPSA(sitagliptin_mol)
        target_bertz = Descriptors.BertzCT(sitagliptin_mol)

        self.valsartan_logp_modifier = GaussianModifier(mu=target_logp, sigma=0.2)
        self.valsartan_tpsa_modifier = GaussianModifier(mu=target_tpsa, sigma=5)
        self.valsartan_bertz_modifier = GaussianModifier(mu=target_bertz, sigma=30)

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        rewards = []
        for atom_types, edge_types in graphs:
            mol = self._graph_to_mol(atom_types, edge_types)
            if mol is None:
                rewards.append(0.0)
                continue

            # RDKit ops below (MCS/Descriptors) are much more reliable on sanitized Mol objects.
            # If sanitization fails, treat as invalid and give 0 reward.
            try:
                # build_molecule returns an RWMol; normalize to Mol for descriptor stability
                if hasattr(mol, "GetMol"):
                    mol = mol.GetMol()
                Chem.SanitizeMol(mol)

                # Use the largest fragment to ensure metrics (MCS, descriptors) are consistent.
                frags = Chem.GetMolFrags(mol, asMols=True)
                if frags and len(frags) > 1:
                    mol = max(frags, key=lambda m: int(m.GetNumAtoms()))
            except Exception as e:
                smi = None
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    smi = None
                logger.warning("[ValsartanSmartsReward] SanitizeMol failed: %s: %s; smiles=%s", type(e).__name__, e, smi)
                rewards.append(0.0)
                continue

            try:
                # Strict SMARTS substructure match (aligned with the original `valsartan_smarts()`):
                #   valsartan_mol = Chem.MolFromSmarts(smarts)
                #   matches = molecule.GetSubstructMatches(valsartan_mol)
                #   smarts_score = 1.0 if len(matches)>0 else 0.0
                smarts_match = False
                try:
                    if self.valsartan_query is not None:
                        smarts_match = len(mol.GetSubstructMatches(self.valsartan_query)) > 0
                    elif self.valsartan_mol is not None:
                        smarts_match = len(mol.GetSubstructMatches(self.valsartan_mol)) > 0
                except Exception:
                    smarts_match = False

                try:
                    # MCS calculation (stricter than before):
                    # - Compare bond order (not "any")
                    # - Match valences
                    # - Rings match rings only; MCS must include complete rings
                    mcs_res = rdFMCS.FindMCS(
                        [mol, self.valsartan_mol],
                        bondCompare=rdFMCS.BondCompare.CompareOrder,
                        atomCompare=rdFMCS.AtomCompare.CompareElements,
                        matchValences=True,
                        ringMatchesRingOnly=True,
                        completeRingsOnly=True,
                        timeout=1,
                    )
                    mcs_atoms = int(getattr(mcs_res, "numAtoms", 0) or 0)
                    if bool(getattr(mcs_res, "canceled", False)):
                        logger.warning(
                            "[ValsartanSmartsReward] FindMCS canceled/timeout: "
                            "mcs_atoms=%d, template_atoms=%d",
                            mcs_atoms, self.valsartan_num_atoms,
                        )

                    # Base Ratio
                    ratio = mcs_atoms / max(1, self.valsartan_num_atoms)

                    # Power transformation
                    structural_score = ratio ** 2.0

                except Exception as e:
                    smi = None
                    try:
                        smi = Chem.MolToSmiles(mol)
                    except Exception:
                        smi = None
                    logger.warning("[ValsartanSmartsReward] FindMCS failed: %s: %s; smiles=%s", type(e).__name__, e, smi)
                    structural_score = 0.0

                # Calculate Properties
                logp_val = Descriptors.MolLogP(mol)
                tpsa_val = Descriptors.TPSA(mol)
                bertz_val = Descriptors.BertzCT(mol)

                logp_score = self.valsartan_logp_modifier(logp_val)
                tpsa_score = self.valsartan_tpsa_modifier(tpsa_val)
                bertz_score = self.valsartan_bertz_modifier(bertz_val)

                props_avg = (logp_score + tpsa_score + bertz_score) / 3.0

                # Final Reward
                if smarts_match:
                    reward = 2.0
                else:
                    reward = structural_score + 0.1 * props_avg
                rewards.append(float(reward))

            except Exception as e:
                smi = None
                try:
                    smi = Chem.MolToSmiles(mol)
                except Exception:
                    smi = None
                logger.warning("[ValsartanSmartsReward] Reward computation failed: %s: %s; smiles=%s", type(e).__name__, e, smi)
                rewards.append(0.0)

        return torch.tensor(rewards, dtype=torch.float32, device=self.device)

    def _graph_to_mol(self, atom_types, edge_types):
        # Code duplication from other rewards, could be refactored but safe to copy for now
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
