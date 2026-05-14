"""Mixin providing graph ↔ SMILES conversion and PlaceHolder utilities."""

import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from rdkit import Chem
from omegaconf import ListConfig

import utils

logger = logging.getLogger(__name__)

Graph = Tuple[torch.Tensor, torch.Tensor]


class GraphConversionMixin:
    """Methods for converting between SMILES strings, RDKit molecules, and
    internal graph representations (PlaceHolder / tensor pairs)."""

    # ------------------------------------------------------------------
    # SMILES helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize_smiles_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple, ListConfig)):
            out = [str(x).strip() for x in val if str(x).strip()]
            return out
        s = str(val).strip()
        if not s:
            return []
        if "," in s:
            return [item.strip() for item in s.split(",") if item.strip()]
        return [s]

    @staticmethod
    def _get_cfg_value(cfg: Dict, key: str) -> Optional[Any]:
        cur: Any = cfg
        for part in str(key).split("."):
            if part == "":
                continue
            try:
                cur = cur.get(part)
            except Exception:
                cur = getattr(cur, part, None)
            if cur is None:
                return None
        return cur

    # ------------------------------------------------------------------
    # SMILES → Graph
    # ------------------------------------------------------------------

    def _smiles_to_graph(self, smiles: str) -> Optional["Graph"]:
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
            mol = Chem.RemoveHs(mol)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                return None
            dataset_info = getattr(self.model, "dataset_info", None)
            atom_encoder = getattr(dataset_info, "atom_encoder", None) if dataset_info is not None else None
            if not isinstance(atom_encoder, dict):
                return None
            x_idx: List[int] = []
            for atom in mol.GetAtoms():
                sym = atom.GetSymbol()
                if sym not in atom_encoder:
                    return None
                x_idx.append(int(atom_encoder[sym]))
            n = len(x_idx)
            if n <= 0:
                return None
            e_idx = torch.zeros((n, n), dtype=torch.long)
            bt = Chem.rdchem.BondType
            for bond in mol.GetBonds():
                a = int(bond.GetBeginAtomIdx())
                b = int(bond.GetEndAtomIdx())
                t = bond.GetBondType()
                if t == bt.SINGLE:
                    v = 1
                elif t == bt.DOUBLE:
                    v = 2
                elif t == bt.TRIPLE:
                    v = 3
                elif t == bt.AROMATIC and int(self.model.input_dims.get("E", 0) or 0) > 4:
                    v = 4
                else:
                    return None
                e_idx[a, b] = v
                e_idx[b, a] = v
            return torch.tensor(x_idx, dtype=torch.long), e_idx
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Graph → SMILES
    # ------------------------------------------------------------------

    def _graph_to_smiles(self, graph: Graph) -> Optional[str]:
        try:
            from analysis.rdkit_functions import build_molecule, mol2smiles
        except Exception:
            return None

        dataset_info = getattr(self.model, "dataset_info", None)
        atom_decoder = getattr(dataset_info, "atom_decoder", None) if dataset_info is not None else None
        if not atom_decoder:
            return None

        X, E = graph
        if torch.is_tensor(X) and X.dim() > 1:
            X = torch.argmax(X, dim=-1)
        if torch.is_tensor(E) and E.dim() > 2:
            E = torch.argmax(E, dim=-1)
        if not torch.is_tensor(X):
            X = torch.tensor(X, dtype=torch.long)
        if not torch.is_tensor(E):
            E = torch.tensor(E, dtype=torch.long)

        try:
            mol = build_molecule(X, E, atom_decoder)
        except Exception:
            return None
        if mol is None:
            return None
        try:
            smi = mol2smiles(mol)
        except Exception:
            smi = None
        if not smi:
            return None
        try:
            mol_frags = Chem.rdmolops.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
            largest_mol = max(mol_frags, default=mol, key=lambda m: m.GetNumAtoms())
            smi = mol2smiles(largest_mol)
        except Exception:
            return None
        return smi or None

    # ------------------------------------------------------------------
    # PlaceHolder → graph list conversions
    # ------------------------------------------------------------------

    def _convert_placeholder_to_graph_list_cpu(self, graphs: utils.PlaceHolder, node_mask: torch.Tensor, as_tensor: bool = False) -> List:
        """Convert PlaceHolder graphs to list or tensor format."""
        graph_list = []
        X, E = graphs.X, graphs.E

        # Move to CPU first
        X_cpu = X.cpu()
        E_cpu = E.cpu()
        node_mask_cpu = node_mask.cpu()

        for i in range(X.size(0)):
            n_nodes = node_mask_cpu[i].sum().item()
            atom_tensor = X_cpu[i, :n_nodes].contiguous()
            edge_tensor = E_cpu[i, :n_nodes, :n_nodes].contiguous()

            if as_tensor:
                graph_list.append((atom_tensor, edge_tensor))
            else:
                atom_types = torch.argmax(atom_tensor, dim=-1)
                edge_types = torch.argmax(edge_tensor, dim=-1)
                graph_list.append([
                    atom_types.to(torch.int64).tolist(),
                    edge_types.to(torch.int64).tolist(),
                ])

        return graph_list

    @staticmethod
    def _convert_placeholder_to_graph_list(graphs: utils.PlaceHolder, node_mask: torch.Tensor) -> List:
        """Convert PlaceHolder graphs to list format (move to CPU only when needed)."""
        graph_list = []
        X, E = graphs.X, graphs.E

        for i in range(X.size(0)):
            n_nodes = node_mask[i].sum().item()

            # Convert to discrete labels
            if X.dim() == 3:
                atom_types = torch.argmax(X[i, :n_nodes], dim=-1)
            else:
                atom_types = X[i, :n_nodes]

            if E.dim() == 4:
                edge_types = torch.argmax(E[i, :n_nodes, :n_nodes], dim=-1)
            else:
                edge_types = E[i, :n_nodes, :n_nodes]

            graph_list.append([atom_types, edge_types])

        return graph_list
