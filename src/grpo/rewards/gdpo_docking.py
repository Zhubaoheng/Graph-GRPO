import logging
import os
import pickle
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import QED, rdMolDescriptors

from grpo.rewards.base import BaseRewardFunction, sascorer

from analysis.rdkit_functions import build_molecule, build_molecule_with_partial_charges
from analysis.lead_opt_oracle import LeadOptOracle
from grpo.eval_docking import gdpo_get_sim_threshold, gdpo_load_train_fps

logger = logging.getLogger(__name__)

_SA_FALLBACK_WARNED = False


class GDPODockingReward(BaseRewardFunction):
    """
    GDPO docking reward:
        r = 0.1 * (r_qed + r_sa) + 0.3 * r_nov + 0.5 * r_ds
    """

    _DEFAULT_ATOM_DECODER = ["C", "N", "O", "F", "B", "Br", "Cl", "I", "P", "S", "Se", "Si"]
    _FP_RADIUS = 2
    _FP_BITS = 1024

    @staticmethod
    def _project_root() -> Optional[Path]:
        try:
            cur = Path(__file__).resolve()
        except Exception:
            return None
        for parent in cur.parents:
            if (parent / "configs").is_dir() and (parent / "src").is_dir():
                return parent
        return None

    @property
    def _DEFAULT_TRAIN_PT_PATH(self) -> Path:
        root = self._project_root() or Path.cwd()
        return root / "data" / "zinc" / "full" / "processed" / "train.pt"

    @property
    def _DEFAULT_FPS_CACHE_PATH(self) -> Path:
        root = self._project_root() or Path.cwd()
        return root / "data" / "zinc" / "full" / "processed" / "train.pt.fps.pkl"
    _WARNED_DECODER_MISMATCH = False

    def __init__(
        self,
        target_name: str,
        atom_decoder: Optional[List[str]] = None,
        device: Optional[torch.device] = None,
        sa_threshold: Optional[float] = None,
        sim_threshold: Optional[float] = None,
        dock_exhaustiveness: Optional[int] = None,
        dock_num_modes: Optional[int] = None,
        dock_timeout: Optional[int] = None,
        dataset_name: Optional[str] = None,
        datadir: Optional[str] = None,
        remove_h: Optional[bool] = None,
    ):
        super().__init__("gdpo_docking", device=device)
        if not target_name:
            raise ValueError("GDPODockingReward requires target_name")
        self.target_name = str(target_name)
        self.atom_decoder = atom_decoder or self._DEFAULT_ATOM_DECODER
        self.sa_threshold = float(sa_threshold) if sa_threshold is not None else (10.0 - 5.0) / 9.0
        if sim_threshold is None:
            sim_threshold = gdpo_get_sim_threshold(dataset_name or "")
        self.sim_threshold = float(sim_threshold)
        self.dataset_name = str(dataset_name) if dataset_name else None
        self.datadir = str(datadir) if datadir else None
        self.remove_h = bool(remove_h) if remove_h is not None else True
        self.repo_root = self._project_root() or Path.cwd()
        self._dock_cache: Dict[str, float] = {}
        self.oracle = LeadOptOracle(
            target_name=self.target_name,
            exhaustiveness=dock_exhaustiveness,
            num_modes=dock_num_modes,
            dock_timeout=dock_timeout,
        )
        self.train_fps_cache_path = None
        self._train_fps = self._load_train_fps()

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if len(graphs) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        valid_indices: List[int] = []
        valid_smiles: List[str] = []
        components: List[Tuple[float, float, float, float]] = []

        for i, (atom_types, edge_types) in enumerate(graphs):
            mol = self._graph_to_mol(atom_types, edge_types)
            if mol is None:
                continue
            try:
                if hasattr(mol, "GetMol"):
                    mol = mol.GetMol()
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            try:
                frags = Chem.GetMolFrags(mol, asMols=True)
                if frags and len(frags) > 1:
                    mol = max(frags, key=lambda m: int(m.GetNumAtoms()))
            except Exception:
                pass
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                smi = None
            if not smi:
                continue

            r_qed = 0.0
            try:
                r_qed = 1.0 if float(QED.qed(mol)) > 0.5 else 0.0
            except Exception:
                r_qed = 0.0

            r_sa = 0.0
            if sascorer is None:
                global _SA_FALLBACK_WARNED
                if not _SA_FALLBACK_WARNED:
                    logger.warning("[GRPO] sascorer is not available, SA reward set to zero.")
                    _SA_FALLBACK_WARNED = True
            else:
                try:
                    sa = float(sascorer.calculateScore(mol))
                    r_sa = (10.0 - sa) / 9.0
                except Exception:
                    r_sa = 0.0

            r_nov = 1.0
            max_sim = 0.0
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self._FP_RADIUS, nBits=self._FP_BITS
                )
                if self._train_fps:
                    sims = DataStructs.BulkTanimotoSimilarity(fp, self._train_fps)
                    max_sim = float(max(sims)) if sims else 0.0
                else:
                    max_sim = 0.0
                r_nov = 1.0 - max_sim
            except Exception:
                r_nov = 0.0

            valid_indices.append(i)
            valid_smiles.append(smi)
            components.append((r_qed, r_sa, r_nov, max_sim))

        out = np.zeros((len(graphs),), dtype=np.float32)
        if valid_smiles:
            # PRUNING LOGIC: If QED is too low, don't waste time docking.
            # DockingVina in genmol eval script also filters QED >= 0.6.
            # Here we skip docking if r_qed == 0 (which means qed <= 0.5).
            to_dock_smiles = []
            to_dock_map = []

            for i, (r_qed, r_sa, r_nov, max_sim) in enumerate(components):
                if r_qed <= 0:
                    continue
                if sascorer is not None and r_sa < self.sa_threshold:
                    continue
                if self.sim_threshold is not None and max_sim >= self.sim_threshold:
                    continue
                smi = valid_smiles[i]
                if smi in self._dock_cache:
                    continue
                to_dock_smiles.append(smi)
                to_dock_map.append(i)

            energies = np.zeros(len(valid_smiles), dtype=np.float32)
            for i, smi in enumerate(valid_smiles):
                cached = self._dock_cache.get(smi)
                if cached is not None:
                    energies[i] = float(cached)
            if to_dock_smiles:
                actual_energies = self.oracle.score(to_dock_smiles)
                for i, energy in enumerate(actual_energies):
                    mapped_idx = to_dock_map[i]
                    energies[mapped_idx] = float(energy)
                    self._dock_cache[to_dock_smiles[i]] = float(energy)

            for (idx, (r_qed, r_sa, r_nov, _max_sim), energy) in zip(valid_indices, components, energies):
                # energy is affinity (negative means better binding).
                # clipping to [-20, 0] ensures energy is non-positive.
                # r_ds will be in [0, 1].
                r_ds = -1.0 * float(np.clip(energy, -20.0, 0.0)) / 20.0
                reward = 0.1 * (r_qed + r_sa) + 0.3 * r_nov + 0.5 * r_ds
                out[idx] = float(reward)

        return torch.tensor(out, dtype=torch.float32, device=self.device)

    def _load_train_fps(self) -> List:
        if self.dataset_name and self.datadir:
            try:
                return gdpo_load_train_fps(
                    dataset_name=self.dataset_name,
                    datadir=self.datadir,
                    remove_h=self.remove_h,
                    repo_root=self.repo_root,
                )
            except Exception as exc:
                logger.warning("[GRPO] GDPO train fps load failed: %s. Falling back to train.pt.", exc)

        path = self._DEFAULT_TRAIN_PT_PATH
        cache_path = self._resolve_cache_path(path)
        if cache_path is not None and cache_path.is_file():
            try:
                if cache_path.stat().st_size >= 16:
                    with open(cache_path, "rb") as handle:
                        cached = pickle.load(handle)
                    if isinstance(cached, list) and cached:
                        return cached
            except Exception:
                pass

        if not path.is_file():
            raise FileNotFoundError(f"train.pt not found: {path}")

        logger.info("[GRPO] Computing fingerprints from train.pt: %s", path)
        fps = self._load_train_fps_from_processed(path)
        logger.info("[GRPO] Fingerprint computation complete, count: %d", len(fps))

        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(cache_path, "wb") as handle:
                    pickle.dump(fps, handle)
            except Exception:
                pass

        return fps

    def _resolve_cache_path(self, path: Path) -> Optional[Path]:
        if self._DEFAULT_FPS_CACHE_PATH.is_file():
            return self._DEFAULT_FPS_CACHE_PATH
        return Path(f"{path}.fps.pkl")

    def _load_train_fps_from_processed(self, path: Path) -> List:
        try:
            from torch_geometric.data import InMemoryDataset
            from torch_geometric.data.data import Data as PyGData
        except Exception as exc:
            raise ImportError("torch_geometric is required to load processed train.pt") from exc

        data_obj = None
        slices = None
        try:
            from torch.serialization import safe_globals

            with safe_globals([PyGData]):
                data_obj, slices = torch.load(path, map_location="cpu")
        except Exception:
            try:
                data_obj, slices = torch.load(path, map_location="cpu", weights_only=False)
            except TypeError:
                data_obj, slices = torch.load(path, map_location="cpu")

        if data_obj is None or slices is None:
            return []
        if not isinstance(slices, dict) or not slices:
            return []

        dataset = InMemoryDataset.__new__(InMemoryDataset)
        dataset.data = data_obj
        dataset.slices = slices

        first_key = next(iter(slices))
        total = int(slices[first_key].numel() - 1)
        logger.info("[GRPO] train.pt slices=%s total=%d", list(slices.keys()), total)
        fps = []
        skipped = 0
        failed = 0
        fp_error_logged = False
        debug_samples = 3
        for idx in range(total):
            try:
                data = InMemoryDataset.get(dataset, idx)
            except Exception:
                failed += 1
                continue
            mol = self._data_to_mol(data)
            if mol is None:
                skipped += 1
                if debug_samples > 0:
                    try:
                        x_shape = tuple(getattr(data, "x", torch.empty(0)).shape)
                        edge_attr = getattr(data, "edge_attr", None)
                        edge_shape = tuple(edge_attr.shape) if edge_attr is not None else None
                        edge_index = getattr(data, "edge_index", None)
                        edge_index_shape = tuple(edge_index.shape) if edge_index is not None else None
                        logger.warning(
                            "[GRPO] data->mol failed idx=%d x=%s "
                            "edge_attr=%s edge_index=%s",
                            idx, x_shape, edge_shape, edge_index_shape,
                        )
                    except Exception:
                        pass
                    debug_samples -= 1
                continue
            try:
                fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(
                    mol, self._FP_RADIUS, nBits=self._FP_BITS
                )
                fps.append(fp)
            except Exception as exc:
                failed += 1
                if not fp_error_logged:
                    try:
                        smi = Chem.MolToSmiles(mol) if mol is not None else None
                    except Exception:
                        smi = None
                    logger.warning("[GRPO] Fingerprint computation failed: %s smiles=%s", exc, smi)
                    fp_error_logged = True
                continue
        logger.info(
            "[GRPO] train.pt conversion stats: total=%d fps=%d "
            "mol_none=%d fp_fail=%d",
            total, len(fps), skipped, failed,
        )
        return fps

    def _data_to_mol(self, data) -> Optional[Chem.Mol]:
        if data is None or not hasattr(data, "x"):
            return None
        try:
            atom_types = torch.argmax(data.x, dim=-1).long().cpu()
        except Exception:
            return None

        n_nodes = int(atom_types.numel())
        if n_nodes == 0:
            return None

        atom_dim = int(getattr(data.x, "size", lambda *_: 0)(-1))
        atom_decoder = self.atom_decoder
        if atom_dim and len(atom_decoder) != atom_dim:
            if atom_dim == 9:
                atom_decoder = ["C", "N", "O", "F", "P", "S", "Cl", "Br", "I"]
            elif atom_dim == 12:
                atom_decoder = self._DEFAULT_ATOM_DECODER
            else:
                if not self._WARNED_DECODER_MISMATCH:
                    logger.warning(
                        "[GRPO] atom_decoder size mismatch: decoder=%d "
                        "data.x=%d. Skipping molecules.",
                        len(self.atom_decoder), atom_dim,
                    )
                    self._WARNED_DECODER_MISMATCH = True
                return None
            if not self._WARNED_DECODER_MISMATCH:
                logger.warning(
                    "[GRPO] atom_decoder size mismatch: decoder=%d "
                    "data.x=%d. Using fallback decoder.",
                    len(self.atom_decoder), atom_dim,
                )
                self._WARNED_DECODER_MISMATCH = True

        edge_types = torch.zeros((n_nodes, n_nodes), dtype=torch.long)
        if hasattr(data, "edge_index") and data.edge_index is not None:
            edge_index = data.edge_index
            edge_attr = getattr(data, "edge_attr", None)
            if edge_attr is not None:
                if edge_attr.dim() > 1:
                    edge_vals = torch.argmax(edge_attr, dim=-1)
                else:
                    edge_vals = edge_attr
            else:
                edge_vals = None
            for k in range(edge_index.size(1)):
                i = int(edge_index[0, k].item())
                j = int(edge_index[1, k].item())
                if edge_vals is not None:
                    bond = int(edge_vals[k].item())
                else:
                    bond = 1
                edge_types[i, j] = bond

        try:
            mol = build_molecule_with_partial_charges(atom_types, edge_types, atom_decoder)
            if hasattr(mol, "GetMol"):
                mol = mol.GetMol()
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                return None
        except Exception as exc:
            if not self._WARNED_DECODER_MISMATCH:
                logger.warning("[GRPO] build_molecule_with_partial_charges failed: %s", exc)
                self._WARNED_DECODER_MISMATCH = True
            return None
        return mol if (mol and mol.GetNumAtoms() > 0) else None

    def _graph_to_mol(self, atom_types, edge_types):
        if build_molecule is None:
            return None
        try:
            at = torch.as_tensor(atom_types).long().cpu()
            et = torch.as_tensor(edge_types).long().cpu()
            if at.dim() == 2:
                at = at.argmax(dim=-1)
            if et.dim() == 3:
                et = et.argmax(dim=-1)
            if at.numel() == 0:
                return None
            mol = build_molecule(at, et, self.atom_decoder)
            return mol if (mol and mol.GetNumAtoms() > 0) else None
        except Exception:
            return None
