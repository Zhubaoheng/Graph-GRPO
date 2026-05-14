import logging
import os
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from rdkit import Chem

from grpo.rewards.base import BaseRewardFunction, sascorer

from analysis.rdkit_functions import build_molecule
from grpo.tdc_compat import patch_tdc_legacy_sklearn_pickles

logger = logging.getLogger(__name__)


class TDCOracleReward(BaseRewardFunction):
    """
    Uses TDC (PyTDC) Oracle as a goal-directed / PMO scoring function.

    - Default: invalid molecules (cannot build / sanitize / no SMILES) receive invalid_score.
    - Supports single-objective (single oracle) or multi-objective (multiple oracles + aggregation).
    """

    def __init__(
        self,
        oracle_names: Union[str, List[str]],
        atom_decoder: Optional[List[str]] = None,
        aggregation: str = "mean",
        weights: Optional[List[float]] = None,
        minimize: bool = False,
        invalid_score: float = 0.0,
        clip_min: Optional[float] = None,
        clip_max: Optional[float] = None,
        tdc_home: Optional[str] = None,
        device: Optional[torch.device] = None,
    ):
        super().__init__("tdc_oracle", device=device)
        self.atom_decoder = atom_decoder
        self.oracle_names = [oracle_names] if isinstance(oracle_names, str) else list(oracle_names)
        if not self.oracle_names:
            raise ValueError("TDCOracleReward: oracle_names must not be empty")

        self.aggregation = (aggregation or "mean").lower()
        self.weights = weights
        self.minimize = bool(minimize)
        self.invalid_score = float(invalid_score)
        self.clip_min = clip_min
        self.clip_max = clip_max
        self._tdc_home = tdc_home

        self._maybe_configure_tdc_home(self._tdc_home)
        patch_tdc_legacy_sklearn_pickles()

        try:
            from tdc import Oracle  # PyTDC
        except ImportError as e:
            raise ImportError(
                "TDC (PyTDC) is not installed. Please install it via `pip install PyTDC` and ensure `tdc` is importable."
            ) from e

        self._oracles = [Oracle(name=name) for name in self.oracle_names]

    @staticmethod
    def _project_root() -> Optional[Path]:
        """
        Resolve repo/project root (independent of Hydra chdir).

        This file lives in `<root>/src/grpo/rewards/tdc_oracle.py`.
        """
        try:
            cur = Path(__file__).resolve()
        except Exception:
            return None
        for parent in cur.parents:
            if (parent / "configs").is_dir() and (parent / "src").is_dir():
                return parent
        return None

    @classmethod
    def _resolve_tdc_home(cls, tdc_home: Optional[str]) -> Optional[Path]:
        """
        Resolve TDC_HOME to an absolute path.

        Priority:
        1) explicit `tdc_home`
        2) env `TDC_HOME`
        3) repo root (if `<root>/oracle` exists)
        """
        if tdc_home is None:
            tdc_home = os.environ.get("TDC_HOME") or None

        if tdc_home is not None:
            raw = Path(os.path.expanduser(str(tdc_home)))
            if raw.is_absolute():
                candidate = raw
            else:
                root = cls._project_root()
                candidate = (root / raw) if root is not None else raw

            # Users sometimes pass the oracle directory itself (e.g. ".../oracle");
            # PyTDC expects TDC_HOME whose child is "oracle/".
            if candidate.is_dir() and candidate.name == "oracle":
                return candidate.parent
            return candidate

        root = cls._project_root()
        if root is not None and (root / "oracle").is_dir():
            return root

        return None

    @staticmethod
    def _safe_link_or_copy(src: Path, dst: Path) -> None:
        """
        Make `dst` point to `src` (symlink/hardlink/copy).
        Best-effort and race-safe: if `dst` appears concurrently, do nothing.
        """
        if dst.exists():
            return
        try:
            dst.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Prefer symlink within the same directory (portable across moves inside the folder).
        try:
            dst.symlink_to(src.name)
            return
        except FileExistsError:
            return
        except Exception:
            pass

        # Fallback: hardlink (same filesystem).
        try:
            os.link(str(src), str(dst))
            return
        except FileExistsError:
            return
        except Exception:
            pass

        # Last resort: copy (large but always works).
        try:
            tmp = dst.with_suffix(dst.suffix + ".tmp")
            shutil.copy2(str(src), str(tmp))
            os.replace(str(tmp), str(dst))
        except FileExistsError:
            return
        except Exception:
            try:
                if tmp.exists():
                    tmp.unlink()
            except Exception:
                pass

    @classmethod
    def _is_html_corrupt(cls, p: Path) -> bool:
        """Check if a file starts with '<' (likely an HTML error page instead of a pkl)."""
        if not p.is_file():
            return False
        try:
            with open(p, "rb") as f:
                head = f.read(16)
                return len(head) > 0 and head.startswith(b"<")
        except Exception:
            return False

    @classmethod
    def _ensure_oracle_pkls_present(cls, tdc_home: Path, oracle_names: List[str]) -> None:
        """
        Ensure `<TDC_HOME>/oracle/<name>.pkl` exists and is valid for each oracle.
        Also handles version-specific aliases like `_current.pkl`.
        """
        oracle_dir = tdc_home / "oracle"
        try:
            oracle_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            return

        for name in oracle_names:
            expected = oracle_dir / f"{name}.pkl"

            # Special aliases PyTDC looks for (drd2 -> drd2_current)
            aliases = [f"{name}_current.pkl", f"{name}_latest.pkl"]

            # Check for corruption if it exists
            all_files = [expected] + [oracle_dir / a for a in aliases]
            for p in all_files:
                if p.exists() and cls._is_html_corrupt(p):
                    corrupt_path = p.with_suffix(".pkl.corrupt")
                    logger.warning("[TDC] Found corrupt HTML file at %s, renaming to %s", p, corrupt_path.name)
                    try:
                        if corrupt_path.exists():
                            corrupt_path.unlink()
                        p.rename(corrupt_path)
                    except Exception as e:
                        logger.warning("[TDC] Failed to rename corrupt file: %s", e)

            # If any valid version exists, ensure all necessary aliases exist as symlinks
            valid_existing = [p for p in all_files if p.is_file() and not cls._is_html_corrupt(p)]

            if valid_existing:
                chosen = valid_existing[0]
                for target in all_files:
                    if not target.exists():
                        cls._safe_link_or_copy(chosen, target)
                continue

            # If none of them exist, look for other pattern-matched pkls
            candidates = sorted(oracle_dir.glob(f"{name}*.pkl"))
            valid_candidates = [p for p in candidates if p.is_file() and not cls._is_html_corrupt(p)]

            if not valid_candidates:
                continue

            # Prefer the newest by mtime etc.
            chosen = max(valid_candidates, key=lambda p: p.stat().st_mtime)
            # Only log restoration if explicitly debugging
            if os.environ.get("GRPO_DEBUG_TDC_HOME", "0") == "1":
                logger.info("[TDC] Restoring %s and aliases from %s", expected.name, chosen.name)

            for target in all_files:
                if not target.exists():
                    cls._safe_link_or_copy(chosen, target)

    def _maybe_configure_tdc_home(self, tdc_home: Optional[str]) -> None:
        resolved = self._resolve_tdc_home(tdc_home)
        if resolved is None:
            return

        # Force for this reward instance.
        os.environ["TDC_HOME"] = str(resolved)

        # PyTDC (certain versions) uses hardcoded "./oracle" relative path.
        # To bypass Hydra's chdir, we create a symlink in the current working directory.
        cwd_oracle = Path("oracle").resolve()
        real_oracle = resolved / "oracle"

        if real_oracle.is_dir() and not cwd_oracle.exists():
            try:
                # Use relative symlink if possible, else absolute.
                cwd_oracle.symlink_to(real_oracle)
                if os.environ.get("GRPO_DEBUG_TDC_HOME", "0") == "1":
                    logger.info("[TDC] Created symlink in CWD: ./oracle -> %s", real_oracle)
            except Exception as e:
                if os.environ.get("GRPO_DEBUG_TDC_HOME", "0") == "1":
                    logger.warning("[TDC] Failed to create CWD symlink: %s", e)

        # Ensure all necessary pkls and version-specific aliases exist in the source dir.
        try:
            self._ensure_oracle_pkls_present(resolved, self.oracle_names)
        except Exception:
            pass

        if os.environ.get("GRPO_DEBUG_TDC_HOME", "").strip().lower() in ("1", "true", "yes", "y", "on"):
            oracle_dir = resolved / "oracle"
            expected = [oracle_dir / f"{name}.pkl" for name in self.oracle_names]
            details = []
            for p in expected:
                status = "EXISTS" if p.exists() else "MISSING"
                if p.exists():
                    is_html = self._is_html_corrupt(p)
                    status += " (HTML Corrupt!!)" if is_html else " (Valid pkl)"
                details.append(f"{p.name}: {status}")

            logger.debug(
                "[TDC] TDC_HOME=%s\n"
                "   oracle_dir=%s\n"
                "   details: %s",
                resolved, oracle_dir, details,
            )

    def __call__(self, graphs: List[Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        if len(graphs) == 0:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        valid_indices: List[int] = []
        valid_smiles: List[str] = []

        for i, (atom_types, edge_types) in enumerate(graphs):
            mol = self._graph_to_mol(atom_types, edge_types)
            if mol is None:
                continue
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                continue
            # Many generated graphs are disconnected; TDC oracles are typically trained on single molecules.
            # Use the largest fragment to avoid SMILES with '.' dominating the score distribution.
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
            valid_indices.append(i)
            valid_smiles.append(smi)

        out = np.full((len(graphs),), self.invalid_score, dtype=np.float32)
        if not valid_smiles:
            return torch.tensor(out, dtype=torch.float32, device=self.device)

        scores = np.zeros((len(valid_smiles), len(self._oracles)), dtype=np.float32)
        for j, oracle in enumerate(self._oracles):
            s = None
            try:
                s = oracle(valid_smiles)
            except Exception:
                s = None

            # Some oracles do NOT support list input; they may return a scalar or a mismatched shape.
            # In that case, fall back to per-SMILES calls.
            try:
                s_arr = np.asarray(s)
            except Exception:
                s_arr = np.asarray([])

            needs_fallback = (
                s_arr.ndim == 0
                or (s_arr.ndim >= 1 and int(s_arr.shape[0]) != int(len(valid_smiles)))
            )
            if needs_fallback:
                s_arr = np.asarray([oracle(smi) for smi in valid_smiles], dtype=np.float32)
            else:
                s_arr = s_arr.astype(np.float32, copy=False)

            scores[:, j] = s_arr.reshape(-1)

        if self.minimize:
            scores = -scores

        if self.clip_min is not None or self.clip_max is not None:
            scores = np.clip(
                scores,
                a_min=self.clip_min if self.clip_min is not None else -np.inf,
                a_max=self.clip_max if self.clip_max is not None else np.inf,
            )

        if self.weights is not None:
            w = np.asarray(self.weights, dtype=np.float32)
            if w.shape[0] != scores.shape[1]:
                raise ValueError(
                    f"TDCOracleReward: weights length ({w.shape[0]}) does not match number of oracles ({scores.shape[1]})"
                )
        else:
            w = np.ones((scores.shape[1],), dtype=np.float32)

        agg = self.aggregation
        if agg in ("mean", "avg", "average"):
            agg_scores = (scores * w[None, :]).sum(axis=1) / max(float(w.sum()), 1e-12)
        elif agg in ("sum",):
            agg_scores = (scores * w[None, :]).sum(axis=1)
        elif agg in ("min",):
            agg_scores = scores.min(axis=1)
        elif agg in ("max",):
            agg_scores = scores.max(axis=1)
        elif agg in ("geometric_mean", "gmean", "geo"):
            eps = 1e-12
            safe = np.clip(scores, a_min=0.0, a_max=None) + eps
            agg_scores = np.exp((np.log(safe) * w[None, :]).sum(axis=1) / max(float(w.sum()), eps))
        else:
            raise ValueError(f"TDCOracleReward: unsupported aggregation='{self.aggregation}'")

        for i, idx in enumerate(valid_indices):
            out[idx] = float(agg_scores[i])

        return torch.tensor(out, dtype=torch.float32, device=self.device)

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
