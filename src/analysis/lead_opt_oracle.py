from __future__ import annotations

import csv
import logging
import os
import re
import shutil
import subprocess
import tempfile
import pickle
import fcntl
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

try:
    from meeko import MoleculePreparation, PDBQTMolecule
except Exception:
    MoleculePreparation = None
    PDBQTMolecule = None
try:
    from openbabel import pybel
except Exception:
    pybel = None

try:
    import sascorer  # Guacamol SA scorer
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
            logger.warning("[LeadOptOracle] Found sascorer but import failed: %s", e)
            sascorer = None

_sascorer = sascorer

RDLogger.DisableLog("rdApp.*")


class LeadOptOracle:
    """
    Docking energy calculator backed by Vina.
    """

    _VINA_RESULT_RE = re.compile(r"REMARK\s+VINA\s+RESULT:\s*([-+]?\d+(?:\.\d+)?)")
    _VINA_TABLE_RE = re.compile(r"^\s*\d+\s+([-+]?\d+(?:\.\d+)?)\b")
    _BOX_BY_TARGET = {
        "fa7": {"center": (10.131, 41.879, 32.097), "size": (20.673, 20.198, 21.362)},
        "parp1": {"center": (26.413, 11.282, 27.238), "size": (18.521, 17.479, 19.995)},
        "5ht1b": {"center": (-26.602, 5.277, 17.898), "size": (22.5, 22.5, 22.5)},
        "jak2": {"center": (114.758, 65.496, 11.345), "size": (19.033, 17.929, 20.283)},
        "braf": {"center": (84.194, 6.949, -7.081), "size": (22.032, 19.211, 14.106)},
    }

    def __init__(
        self,
        target_name: str,
        seed_idx: Optional[int] = None,
        sim_threshold: Optional[float] = None,
        exhaustiveness: Optional[int] = None,
        num_modes: Optional[int] = None,
        dock_timeout: Optional[int] = None,
    ):
        self.target_name = str(target_name)
        self.seed_idx = int(seed_idx) if seed_idx is not None else None
        self.sim_threshold = float(sim_threshold) if sim_threshold is not None else None
        self._seed_smiles: Optional[str] = None
        self._cache = {}

        if MoleculePreparation is None or PDBQTMolecule is None:
            raise ImportError(
                "meeko is required for LeadOptOracle. Please install it to enable PDBQT preparation."
            )

        project_root = self._project_root() or Path.cwd()
        self._data_dir = project_root / "data" / "lead_opt" / "docking"
        self._receptor_path = self._data_dir / f"{self.target_name}.pdbqt"
        self._vina_path = self._resolve_vina_binary(project_root)
        self._box = self._BOX_BY_TARGET.get(self.target_name)
        if self._box is None:
            raise ValueError(f"Docking box not configured for target '{self.target_name}'.")
        default_exhaustiveness = int(os.environ.get("LEAD_OPT_DOCK_EXHAUSTIVENESS", "1"))
        default_num_modes = int(os.environ.get("LEAD_OPT_DOCK_NUM_MODES", "5"))
        self._exhaustiveness = int(exhaustiveness) if exhaustiveness is not None else default_exhaustiveness
        self._num_modes = int(num_modes) if num_modes is not None else default_num_modes
        self._num_cpu = 1
        # Allow env override to trim long-tail docking latency.
        default_timeout = int(os.environ.get("LEAD_OPT_DOCK_TIMEOUT", "6"))
        self._dock_timeout = int(dock_timeout) if dock_timeout is not None else default_timeout

        if not self._data_dir.is_dir():
            raise FileNotFoundError(f"LeadOptOracle data dir not found: {self._data_dir}")
        if not self._receptor_path.is_file():
            raise FileNotFoundError(f"Receptor pdbqt not found: {self._receptor_path}")
        if self._vina_path is None:
            raise FileNotFoundError(
                "No docking binary found. Put qvina02 at bin/qvina02 or install AutoDock Vina "
                "so the `vina` executable is available in PATH."
            )

        if self.seed_idx is not None:
            try:
                self._seed_smiles = self._load_seed_smiles(self.seed_idx)
            except Exception as e:
                logger.warning("[LeadOptOracle] Failed to load seed SMILES: %s", e)

        # Shared disk-backed cache configuration
        project_root = self._project_root() or Path.cwd()
        self._cache_dir = project_root / "data" / "zinc"
        self._cache_path = self._cache_dir / f"{self.target_name}_docking_cache.pkl"
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    @staticmethod
    def _resolve_vina_binary(project_root: Path) -> Optional[Path]:
        candidates = [project_root / "bin" / "qvina02"]
        for name in ("qvina02", "vina"):
            resolved = shutil.which(name)
            if resolved:
                candidates.append(Path(resolved))
        for candidate in candidates:
            try:
                if candidate.is_file() and os.access(candidate, os.X_OK):
                    return candidate
            except Exception:
                continue
        return None

    def _load_seed_smiles(self, seed_idx: int) -> str:
        actives_path = self._data_dir / "actives.csv"
        if not actives_path.is_file():
            raise FileNotFoundError(f"LeadOptOracle actives.csv not found: {actives_path}")
        rows = []
        with open(actives_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if (row.get("target") or "").strip() == self.target_name:
                    rows.append(row)
        if not rows:
            raise ValueError(f"No seed entries found for target '{self.target_name}' in {actives_path}")
        if seed_idx < 0 or seed_idx >= len(rows):
            raise IndexError(f"seed_idx {seed_idx} out of range for target '{self.target_name}'")
        seed_smiles = (rows[seed_idx].get("smiles") or "").strip()
        if not seed_smiles:
            raise ValueError(f"Empty seed SMILES for target '{self.target_name}' seed_idx={seed_idx}")
        return seed_smiles

    def _load_cache(self) -> dict:
        if not self._cache_path.is_file():
            return {}
        try:
            with open(self._cache_path, "rb") as f:
                # Use shared lock for reading
                fcntl.flock(f, fcntl.LOCK_SH)
                data = pickle.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save_cache(self, new_results: dict):
        if not new_results:
            return
        # Use exclusive lock for updating
        try:
            # We open with "a+b" or similar to ensure we can lock and read-modify-write
            # But the simplest is to open for reading/writing
            mode = "rb+" if self._cache_path.is_file() else "wb+"
            with open(self._cache_path, mode) as f:
                fcntl.flock(f, fcntl.LOCK_EX)
                
                existing = {}
                if mode == "rb+":
                    try:
                        f.seek(0)
                        existing = pickle.load(f)
                        if not isinstance(existing, dict): existing = {}
                    except Exception:
                        existing = {}
                
                existing.update(new_results)
                
                f.seek(0)
                f.truncate()
                pickle.dump(existing, f)
                f.flush()
                fcntl.flock(f, fcntl.LOCK_UN)
            
            # Also update local memory cache
            self._cache.update(new_results)
        except Exception as e:
            logger.warning("[LeadOptOracle] Error saving cache: %s", e)

    def score(self, smiles_list: Iterable[str]) -> List[float]:
        if isinstance(smiles_list, str):
            smiles_list = [smiles_list]
        smiles_list = list(smiles_list)
        
        # 1. Sync local cache with disk before processing (optional but good for visibility)
        disk_cache = self._load_cache()
        self._cache.update(disk_cache)

        scores: List[float] = []
        debug = os.environ.get("LEAD_OPT_DEBUG", "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        stats = {
            "total": 0,
            "invalid": 0,
            "dock_fail": 0,
            "dock_ok": 0,
            "cache_hit": 0,
        }
        
        new_dock_results = {}
        for smi in smiles_list:
            if smi in self._cache:
                scores.append(self._cache[smi])
                stats["cache_hit"] += 1
                continue
            
            score, reason = self._score_single_with_reason(smi)
            self._cache[smi] = score
            new_dock_results[smi] = score
            scores.append(score)
            
            if debug:
                stats["total"] += 1
                if reason in stats:
                    stats[reason] += 1
                elif reason == "ok":
                    stats["dock_ok"] += 1
                elif reason:
                    stats["invalid"] += 1
        
        # 2. Persist new results back to disk
        if new_dock_results:
            self._save_cache(new_dock_results)

        if debug and stats["total"] > 0:
            logger.warning(
                "[LeadOptOracle] batch stats: "
                f"total={stats['total']} "
                f"cache_hit={stats['cache_hit']} "
                f"invalid={stats['invalid']} "
                f"dock_fail={stats['dock_fail']} "
                f"dock_ok={stats['dock_ok']}"
            )
        return scores

    def _score_single_with_reason(self, smiles: str) -> tuple[float, str]:
        mol = self._smiles_to_mol(smiles)
        if mol is None:
            return 100.0, "invalid"

        affinity, reason = self._dock_molecule(mol)
        if affinity is None:
            return 100.0, reason or "dock_fail"
        return float(affinity), "ok"

    def _dock_molecule(self, mol: Chem.Mol) -> Tuple[Optional[float], str]:
        debug_dock = os.environ.get("LEAD_OPT_DEBUG_DOCK", "").strip().lower() in {"1", "true", "t", "yes", "y", "on"}
        mol_3d = Chem.AddHs(Chem.Mol(mol))
        if not self._embed_molecule(mol_3d):
            if debug_dock:
                logger.warning("[LeadOptOracle] embed failed")
            return None, "embed_fail"

        pdbqt_string = self._mol_to_pdbqt(mol_3d, debug=debug_dock)
        if not pdbqt_string:
            pdbqt_string = self._mol_to_pdbqt_openbabel(mol_3d, debug=debug_dock)
        if not pdbqt_string:
            if debug_dock:
                logger.warning("[LeadOptOracle] PDBQT conversion failed")
            return None, "pdbqt_fail"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            ligand_path = tmpdir_path / "ligand.pdbqt"
            out_path = tmpdir_path / "out.pdbqt"

            ligand_path.write_text(pdbqt_string)

            cmd = [
                str(self._vina_path),
                "--receptor",
                str(self._receptor_path),
                "--ligand",
                str(ligand_path),
                "--out",
                str(out_path),
                "--center_x",
                str(self._box["center"][0]),
                "--center_y",
                str(self._box["center"][1]),
                "--center_z",
                str(self._box["center"][2]),
                "--size_x",
                str(self._box["size"][0]),
                "--size_y",
                str(self._box["size"][1]),
                "--size_z",
                str(self._box["size"][2]),
                "--cpu",
                str(self._num_cpu),
                "--num_modes",
                str(self._num_modes),
                "--exhaustiveness",
                str(self._exhaustiveness),
            ]
            if debug_dock:
                logger.info("[LeadOptOracle] docking cmd: %s", ' '.join(cmd))
            try:
                proc = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=self._dock_timeout,
                )
            except subprocess.TimeoutExpired:
                if debug_dock:
                    logger.warning("[LeadOptOracle] docking timeout")
                return None, "dock_timeout"
            except Exception:
                if debug_dock:
                    logger.warning("[LeadOptOracle] docking subprocess failed")
                return None, "dock_subprocess_fail"

            if proc.returncode != 0:
                if debug_dock:
                    logger.info("[LeadOptOracle] docking returncode=%s", proc.returncode)
                    if proc.stdout:
                        logger.info("[LeadOptOracle] docking stdout:\n%s", proc.stdout)
                    if proc.stderr:
                        logger.info("[LeadOptOracle] docking stderr:\n%s", proc.stderr)
                return None, "dock_returncode"

            affinity = self._extract_affinity(out_path, proc.stdout, proc.stderr)
            if debug_dock and affinity is None:
                logger.info("[LeadOptOracle] docking affinity not found")
                if proc.stdout:
                    logger.info("[LeadOptOracle] docking stdout:\n%s", proc.stdout)
                if proc.stderr:
                    logger.info("[LeadOptOracle] docking stderr:\n%s", proc.stderr)
            if affinity is None:
                return None, "affinity_not_found"
            return affinity, "ok"

    @staticmethod
    def _project_root() -> Optional[Path]:
        try:
            return Path(__file__).resolve().parents[2]
        except Exception:
            return None

    @classmethod
    def _smiles_to_mol(cls, smiles: str) -> Optional[Chem.Mol]:
        if not smiles:
            return None
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None
        try:
            frags = Chem.GetMolFrags(mol, asMols=True)
            if frags and len(frags) > 1:
                mol = max(frags, key=lambda m: int(m.GetNumAtoms()))
        except Exception:
            pass
        return mol

    @staticmethod
    def _embed_molecule(mol: Chem.Mol) -> bool:
        try:
            params = AllChem.ETKDGv3()
        except Exception:
            params = AllChem.ETKDG()
        params.randomSeed = 0xF00D
        try:
            status = AllChem.EmbedMolecule(mol, params)
        except Exception:
            status = -1
        if status == -1:
            try:
                status = AllChem.EmbedMolecule(mol, randomSeed=0xF00D, useRandomCoords=True)
            except Exception:
                status = -1
        return status != -1

    @staticmethod
    def _mol_to_pdbqt(mol: Chem.Mol, *, debug: bool = False) -> Optional[str]:
        prep = MoleculePreparation()
        try:
            setups = prep.prepare(mol)
        except Exception:
            if debug:
                logger.warning("[LeadOptOracle] meeko prepare failed")
            return None
        if not setups:
            if debug:
                logger.info("[LeadOptOracle] meeko prepare returned 0 setups")
            return None

        # Prefer the newer writer API to avoid meeko deprecation warnings.
        try:
            from meeko import PDBQTWriterLegacy
        except Exception:
            PDBQTWriterLegacy = None
        if PDBQTWriterLegacy is not None:
            try:
                ret = PDBQTWriterLegacy.write_string(setups[0])
                if isinstance(ret, tuple):
                    return ret[0]
                return ret
            except Exception:
                pass

        try:
            pdbqt = PDBQTMolecule(setups[0])
            if hasattr(pdbqt, "to_pdbqt_string"):
                return pdbqt.to_pdbqt_string()
            if hasattr(pdbqt, "write_pdbqt_string"):
                return pdbqt.write_pdbqt_string()
        except Exception:
            pdbqt = None

        if hasattr(prep, "write_pdbqt_string"):
            try:
                return prep.write_pdbqt_string(setups[0])
            except Exception:
                pass

        return None

    @staticmethod
    def _mol_to_pdbqt_openbabel(mol: Chem.Mol, *, debug: bool = False) -> Optional[str]:
        if pybel is None:
            if debug:
                logger.info("[LeadOptOracle] openbabel/pybel not available for PDBQT fallback")
            return None
        try:
            mol_block = Chem.MolToMolBlock(mol)
        except Exception:
            if debug:
                logger.warning("[LeadOptOracle] RDKit MolToMolBlock failed for OpenBabel fallback")
            return None
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                mol_path = tmpdir_path / "ligand.mol"
                pdbqt_path = tmpdir_path / "ligand.pdbqt"
                mol_path.write_text(mol_block)
                ob_mols = list(pybel.readfile("mol", str(mol_path)))
                if not ob_mols:
                    if debug:
                        logger.info("[LeadOptOracle] OpenBabel readfile produced 0 molecules")
                    return None
                ob_mols[0].write("pdbqt", str(pdbqt_path), overwrite=True)
                return pdbqt_path.read_text()
        except Exception:
            if debug:
                logger.warning("[LeadOptOracle] OpenBabel PDBQT conversion failed")
            return None

    def _extract_affinity(
        self,
        out_path: Path,
        stdout: Optional[str],
        stderr: Optional[str],
    ) -> Optional[float]:
        candidates: List[float] = []

        if out_path.is_file():
            try:
                text = out_path.read_text()
                candidates.extend(self._extract_affinities_from_text(text))
            except Exception:
                pass

        candidates.extend(self._extract_affinities_from_text(stdout))
        candidates.extend(self._extract_affinities_from_text(stderr))

        if not candidates:
            return None
        return float(min(candidates))

    def _extract_affinities_from_text(self, text: Optional[str]) -> List[float]:
        if not text:
            return []
        vals: List[float] = []
        for line in text.splitlines():
            match = self._VINA_RESULT_RE.search(line)
            if match:
                try:
                    vals.append(float(match.group(1)))
                except Exception:
                    pass
                continue
            match = self._VINA_TABLE_RE.match(line)
            if match:
                try:
                    vals.append(float(match.group(1)))
                except Exception:
                    pass
        return vals
