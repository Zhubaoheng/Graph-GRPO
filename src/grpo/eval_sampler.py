from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
from pathlib import Path
import re
import sys

# Ensure both src/ and repo root are on sys.path
_src_dir = str(Path(__file__).resolve().parents[1])
_repo_root = str(Path(__file__).resolve().parents[2])
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Avoid Intel OpenMP shared-memory usage in restricted environments (macOS sandbox).
os.environ.setdefault("KMP_DISABLE_SHM", "1")
os.environ.setdefault("KMP_USE_SHM", "0")

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from rdkit import Chem

from analysis.visualization import MolecularVisualization
from grpo.eval_docking import (
    gdpo_eval_smiles,
    gdpo_get_sim_threshold,
    gdpo_load_train_fps,
)
from graph_discrete_flow_model import GraphDiscreteFlowModel
from grpo.trainer import GRPOTrainer
from grpo.train_utils import create_datamodule_and_model_components

Graph = Tuple[torch.Tensor, torch.Tensor]


class _DisabledReward:
    name = "disabled_reward"

    def __call__(self, _graphs: List[Graph]):
        raise RuntimeError(
            "GraphGRPOProposer disables reward/oracle calls inside your_repo; "
            "scores must come from mol_opt's oracle wrapper."
        )


@dataclass(frozen=True)
class _TopEntry:
    score: float
    smiles: str
    graph: Graph


class GraphGRPOProposer:
    """
    Graph-GRPO proposer (candidate generator + refinement engine).

    This class must NOT call any real oracle. It only:
    - samples graphs
    - refines graphs with fixed noise_fraction=0.3
    - converts graph -> SMILES
    - applies local validity/fragment filters
    - maintains a Top-10 pool based on externally-provided scores (observe)
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        self.cfg = cfg
        self.device = device

        try:
            self.cfg.general.test_only = True
        except Exception:
            pass

        # ================= 1) datamodule / dataset_infos (migrated) =================
        datamodule, model_kwargs = create_datamodule_and_model_components(self.cfg)
        try:
            datamodule.setup(stage="fit")
        except Exception:
            pass

        self.datamodule = datamodule
        self.model_kwargs = model_kwargs
        self.dataset_infos = model_kwargs["dataset_infos"]

        visualization_tools = model_kwargs.get("visualization_tools")
        if visualization_tools is None:
            visualization_tools = MolecularVisualization(
                getattr(self.cfg.dataset, "remove_h", False),
                dataset_infos=self.dataset_infos,
            )
        self.visualization_tools = visualization_tools

        # ================= 2) model + ckpt loading (migrated) =================
        self.model = GraphDiscreteFlowModel(cfg=self.cfg, **model_kwargs).to(self.device)
        self._load_checkpoint_if_available()

        # [Ablation] Force static p0: revert to dataset-derived default distribution
        _force_static = self._get_bool_cfg_or_env(
            cfg=self.cfg,
            key="grpo.eval_force_static_p0",
            env_key="GRAPH_GRPO_FORCE_STATIC_P0",
            default=False,
        )
        if _force_static:
            from flow_matching.noise_distribution import NoiseDistribution
            static_limit = NoiseDistribution(
                self.cfg.model.transition, self.model.dataset_info
            ).get_limit_dist()
            self.model.update_limit_dist(
                static_limit.X.squeeze(), static_limit.E.squeeze(),
            )
            logger.info(
                "[Ablation] Forced static p0: node=%s edge=%s",
                static_limit.X.squeeze().tolist(),
                static_limit.E.squeeze().tolist(),
            )

        self.model.eval()

        # ================= 3) trainer init (migrated) =================
        self.trainer = GRPOTrainer(
            model=self.model,
            reward_function=_DisabledReward(),
            cfg=self.cfg,
            model_kwargs=model_kwargs,
        )

        steps = int(getattr(self.cfg.grpo, "forward_steps", 100) or 100)
        self.trainer.sample_steps = steps

        self._pending_graph_by_smiles: Dict[str, Graph] = {}
        self._best_by_smiles: Dict[str, _TopEntry] = {}
        self._topk: List[_TopEntry] = []
        self.round0_samples = int(getattr(self.cfg.grpo, "round0_samples", 500) or 500)
        self.refine_topk = int(getattr(self.cfg.grpo, "refine_topk", 10) or 10)
        self.refine_topk_early = int(getattr(self.cfg.grpo, "refine_topk_early", self.refine_topk) or self.refine_topk)
        self.refine_topk_late = int(getattr(self.cfg.grpo, "refine_topk_late", self.refine_topk) or self.refine_topk)
        self.refine_num_vars_early = int(getattr(self.cfg.grpo, "refine_num_vars_early", 200) or 200)
        self.refine_num_vars_late = int(getattr(self.cfg.grpo, "refine_num_vars_late", 200) or 200)
        self.refine_switch_budget = int(getattr(self.cfg.grpo, "refine_switch_budget", 2000) or 2000)
        self.noise_fraction_early = float(getattr(self.cfg.grpo, "noise_fraction_early", 0.3) or 0.3)
        self.noise_fraction_late = float(getattr(self.cfg.grpo, "noise_fraction_late", 0.3) or 0.3)
        self.noise_switch_budget = int(getattr(self.cfg.grpo, "noise_switch_budget", 2000) or 2000)
        self.disable_refine = self._get_bool_cfg_or_env(
            cfg=self.cfg,
            key="grpo.disable_refine",
            env_key="GRAPH_GRPO_DISABLE_REFINE",
            default=False,
        )
        self._debug_noise = self._get_bool_cfg_or_env(
            cfg=self.cfg,
            key="grpo.debug_noise",
            env_key="GRAPH_GRPO_DEBUG_NOISE",
            default=False,
        )
        if self._debug_noise:
            logger.debug(
                "[GraphGRPOProposer] debug_noise=on "
                "noise_fraction_early=%s noise_fraction_late=%s "
                "noise_switch_budget=%s disable_refine=%s",
                self.noise_fraction_early,
                self.noise_fraction_late,
                self.noise_switch_budget,
                self.disable_refine,
            )

        # Default batch size logic: if refine is disabled, we can afford much larger batches (e.g. 2048).
        # Otherwise default to round0_samples (often 64-500).
        default_batch_size = int(self.round0_samples)
        if self.disable_refine:
            default_batch_size = max(default_batch_size, 2048)
        self.eval_batch_size = self._get_int_cfg_or_env(
            cfg=self.cfg,
            key="grpo.eval_batch_size",
            env_key="GRAPH_GRPO_EVAL_BATCH_SIZE",
            default=default_batch_size,
        )


        cfg_screen_mode = self._get_cfg_value(self.cfg, "grpo.screen_mode")
        force_disable_screen = False
        if cfg_screen_mode is not None:
            cfg_screen_mode_norm = self._normalize_bool(cfg_screen_mode)
            if cfg_screen_mode_norm is False:
                force_disable_screen = True

        self.screen_mode = False
        self.screen_csv_path = ""
        self.screen_score_column = ""
        self.screen_topk = 0

        if not force_disable_screen:
            self.screen_mode = self._get_bool_cfg_or_env(
                cfg=self.cfg,
                key="grpo.screen_mode",
                env_key="GRAPH_GRPO_SCREEN_MODE",
                default=False,
            )
            self.screen_csv_path = self._get_str_cfg_or_env(
                cfg=self.cfg,
                key="grpo.screen_csv_path",
                env_key="GRAPH_GRPO_SCREEN_CSV",
                default="",
            )
            self.screen_score_column = self._get_str_cfg_or_env(
                cfg=self.cfg,
                key="grpo.screen_score_column",
                env_key="GRAPH_GRPO_SCREEN_COLUMN",
                default=str(getattr(getattr(self.cfg, "grpo", None), "target_task", "") or ""),
            )
            self.screen_topk = self._get_int_cfg_or_env(
                cfg=self.cfg,
                key="grpo.screen_topk",
                env_key="GRAPH_GRPO_SCREEN_TOPK",
                default=int(self.round0_samples),
            )
        if self.screen_mode and int(self.round0_samples) <= 0 and int(self.screen_topk) <= 0:
            self.screen_topk = 1
            logger.warning(
                "[GraphGRPOProposer] round0_samples<=0; forcing screen_topk=1"
            )
        self.screen_cache_dir = self._get_str_cfg_or_env(
            cfg=self.cfg,
            key="grpo.screen_cache_dir",
            env_key="GRAPH_GRPO_SCREEN_CACHE_DIR",
            default="",
        )
        self._refine_seed_smiles = self._normalize_smiles_list(
            self._get_cfg_value(self.cfg, "grpo.refine_seed_smiles")
        )
        self._refine_seed_smiles_set = set(self._refine_seed_smiles)
        self._refine_seed_graphs = self._build_refine_seed_graphs(self._refine_seed_smiles)
        if self._refine_seed_graphs:
            logger.info(
                "[GraphGRPOProposer] refine seed bases loaded: %d",
                len(self._refine_seed_graphs),
            )
        self._screen_smiles_ranked: List[str] = []
        self._screen_best_smiles: Optional[str] = None
        self._screen_best_score: Optional[float] = None
        self._screen_best_injected = False
        self._screen_best_enqueue_attempted = False
        if self.screen_mode:
            self._screen_smiles_ranked = self._load_screen_smiles_ranked()
            if self._screen_best_smiles is None:
                self._screen_best_smiles, self._screen_best_score = self._load_screen_best_entry()

    @staticmethod
    def _get_str_cfg_or_env(cfg: DictConfig, key: str, env_key: str, default: str = "") -> str:
        val = os.environ.get(env_key)
        if val is not None and str(val).strip() != "":
            return str(val).strip()
        cur: Any = cfg
        for part in str(key).split("."):
            if part == "":
                continue
            try:
                cur = cur.get(part)
            except Exception:
                cur = getattr(cur, part, None)
            if cur is None:
                return str(default)
        return str(cur) if cur is not None else str(default)

    @staticmethod
    def _get_cfg_value(cfg: DictConfig, key: str) -> Optional[Any]:
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

    @staticmethod
    def _normalize_smiles_list(val: Any) -> List[str]:
        if val is None:
            return []
        if isinstance(val, (list, tuple)):
            out = [str(x).strip() for x in val if str(x).strip()]
            return out
        s = str(val).strip()
        if not s:
            return []
        if "," in s:
            return [item.strip() for item in s.split(",") if item.strip()]
        return [s]

    def _build_refine_seed_graphs(self, smiles_list: List[str]) -> List[Graph]:
        graphs: List[Graph] = []
        for smi in smiles_list:
            g = self._smiles_to_graph(smi)
            if g is None:
                logger.warning(
                    "[GraphGRPOProposer] refine seed SMILES cannot be converted to graph: %s",
                    smi,
                )
                continue
            if not self.valid_filter(smi, g):
                logger.warning(
                    "[GraphGRPOProposer] refine seed SMILES filtered by MolFilter: %s",
                    smi,
                )
                continue
            graphs.append(g)
        return graphs

    @staticmethod
    def _normalize_bool(val: Any) -> Optional[bool]:
        if isinstance(val, bool):
            return val
        if val is None:
            return None
        s = str(val).strip().lower()
        if s in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if s in {"0", "false", "f", "no", "n", "off"}:
            return False
        return None

    @classmethod
    def _get_int_cfg_or_env(cls, cfg: DictConfig, key: str, env_key: str, default: int) -> int:
        val = os.environ.get(env_key)
        if val is not None and str(val).strip() != "":
            try:
                return int(float(str(val).strip()))
            except Exception:
                return int(default)
        try:
            return int(float(cls._get_str_cfg_or_env(cfg, key, env_key="__unused__", default=str(default))))
        except Exception:
            return int(default)

    @classmethod
    def _get_bool_cfg_or_env(cls, cfg: DictConfig, key: str, env_key: str, default: bool) -> bool:
        val = os.environ.get(env_key)
        if val is not None and str(val).strip() != "":
            v = str(val).strip().lower()
            return v in {"1", "true", "t", "yes", "y", "on"}
        cfg_val = cls._get_cfg_value(cfg, key)
        cfg_norm = cls._normalize_bool(cfg_val)
        if cfg_norm is True:
            return True
        if cfg_norm is False:
            return False
        return bool(default)

    def _screen_cache_path(self) -> Optional[str]:
        if not self.screen_csv_path or not self.screen_score_column:
            return None
        if int(self.screen_topk) <= 0:
            return None
        csv_path = os.path.abspath(os.path.expanduser(str(self.screen_csv_path)))
        cache_dir = str(self.screen_cache_dir).strip()
        if cache_dir:
            cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
        else:
            cache_dir = os.path.join(os.path.dirname(csv_path), ".graph_grpo_screen_cache")
        try:
            os.makedirs(cache_dir, exist_ok=True)
        except Exception:
            return None
        col = self._safe_filename(self.screen_score_column, max_len=64)
        return os.path.join(cache_dir, f"top_smiles_{col}_k{int(self.screen_topk)}.txt")

    def _load_screen_smiles_ranked(self) -> List[str]:
        """
        Load top-K SMILES ranked by the CSV score column.

        CSV expected columns:
          - smiles
          - <task columns>, e.g. median2, valsartan_smarts, ...
        """
        if not self.screen_csv_path:
            logger.warning("[GraphGRPOProposer] screen_mode enabled but GRAPH_GRPO_SCREEN_CSV/grpo.screen_csv_path is empty")
            return []
        if not self.screen_score_column:
            logger.warning("[GraphGRPOProposer] screen_mode enabled but GRAPH_GRPO_SCREEN_COLUMN/grpo.screen_score_column is empty")
            return []

        cache_path = self._screen_cache_path()
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path, "r", encoding="utf-8") as f:
                    smiles = [ln.strip() for ln in f if ln.strip()]
                if smiles:
                    return smiles[: int(self.screen_topk)]
            except Exception:
                pass

        csv_path = os.path.abspath(os.path.expanduser(str(self.screen_csv_path)))
        if not os.path.exists(csv_path):
            logger.warning("[GraphGRPOProposer] screen CSV not found: %s", csv_path)
            return []

        k = int(self.screen_topk) if int(self.screen_topk) > 0 else int(self.round0_samples)
        if k <= 0:
            return []
        n_rows = 0
        col = str(self.screen_score_column)

        rows: list[tuple[float, str]] = []
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    raise RuntimeError("CSV has no header row")

                try:
                    idx_smiles = header.index("smiles")
                except ValueError as e:
                    raise RuntimeError("CSV missing required column: smiles") from e
                try:
                    idx_score = header.index(col)
                except ValueError as e:
                    raise RuntimeError(f"CSV missing required column: {col}") from e

                for row in reader:
                    n_rows += 1
                    if not row:
                        continue
                    if idx_smiles >= len(row) or idx_score >= len(row):
                        continue
                    smi = (row[idx_smiles] or "").strip()
                    if not smi:
                        continue
                    try:
                        score = float(row[idx_score])
                        if not math.isfinite(score):
                            continue
                    except Exception:
                        continue
                    rows.append((score, smi))
        except Exception as e:
            logger.warning("[GraphGRPOProposer] Failed to load screen CSV: %s", e)
            return []

        rows.sort(key=lambda x: x[0], reverse=True)
        smiles_ranked = [smi for _, smi in rows[:k]]
        if rows and self._screen_best_smiles is None:
            self._screen_best_score = float(rows[0][0])
            self._screen_best_smiles = str(rows[0][1])

        if cache_path:
            try:
                with open(cache_path, "w", encoding="utf-8") as f:
                    f.write("\n".join(smiles_ranked) + "\n")
            except Exception:
                pass

        logger.info(
            "[GraphGRPOProposer] screen_mode loaded top-%d/%d seeds from %s (column=%s, rows=%d)",
            len(smiles_ranked), k, csv_path, col, n_rows,
        )
        return smiles_ranked

    def _load_screen_best_entry(self) -> Tuple[Optional[str], Optional[float]]:
        """
        Load the single best (smiles, score) from the screen CSV column.
        """
        if not self.screen_csv_path:
            logger.warning("[GraphGRPOProposer] screen_mode enabled but GRAPH_GRPO_SCREEN_CSV/grpo.screen_csv_path is empty")
            return None, None
        if not self.screen_score_column:
            logger.warning("[GraphGRPOProposer] screen_mode enabled but GRAPH_GRPO_SCREEN_COLUMN/grpo.screen_score_column is empty")
            return None, None

        csv_path = os.path.abspath(os.path.expanduser(str(self.screen_csv_path)))
        if not os.path.exists(csv_path):
            logger.warning("[GraphGRPOProposer] screen CSV not found: %s", csv_path)
            return None, None

        col = str(self.screen_score_column)
        best_score: Optional[float] = None
        best_smiles: Optional[str] = None
        n_rows = 0
        try:
            with open(csv_path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if not header:
                    raise RuntimeError("CSV has no header row")

                try:
                    idx_smiles = header.index("smiles")
                except ValueError as e:
                    raise RuntimeError("CSV missing required column: smiles") from e
                try:
                    idx_score = header.index(col)
                except ValueError as e:
                    raise RuntimeError(f"CSV missing required column: {col}") from e

                for row in reader:
                    n_rows += 1
                    if not row:
                        continue
                    if idx_smiles >= len(row) or idx_score >= len(row):
                        continue
                    smi = (row[idx_smiles] or "").strip()
                    if not smi:
                        continue
                    try:
                        score = float(row[idx_score])
                        if not math.isfinite(score):
                            continue
                    except Exception:
                        continue
                    if best_score is None or score > best_score:
                        best_score = score
                        best_smiles = smi
        except Exception as e:
            logger.warning("[GraphGRPOProposer] Failed to load screen CSV for best seed: %s", e)
            return None, None

        if best_smiles is not None and best_score is not None:
            logger.info(
                "[GraphGRPOProposer] screen_mode best seed loaded (score=%s, rows=%d, column=%s)",
                best_score, n_rows, col,
            )
        return best_smiles, best_score

    def _maybe_inject_screen_best(self) -> None:
        if self._screen_best_injected or not self.screen_mode:
            return
        if self._screen_best_smiles is None or self._screen_best_score is None:
            self._screen_best_injected = True
            return

        if not self._topk:
            top1_score = None
        else:
            top1_score = float(self._topk[0].score)

        if top1_score is not None and self._screen_best_score <= top1_score:
            logger.info(
                "[GraphGRPOProposer] screen best not injected: score=%s <= top1=%s",
                self._screen_best_score, top1_score,
            )
            self._screen_best_injected = True
            return

        existing = self._best_by_smiles.get(self._screen_best_smiles)
        if existing is not None:
            if self._screen_best_score > existing.score:
                self._best_by_smiles[self._screen_best_smiles] = _TopEntry(
                    score=float(self._screen_best_score),
                    smiles=self._screen_best_smiles,
                    graph=existing.graph,
                )
                self._topk = sorted(self._best_by_smiles.values(), key=lambda e: e.score, reverse=True)[: self.refine_topk]
                logger.info(
                    "[GraphGRPOProposer] screen best updated topk (score=%s, prev=%s)",
                    self._screen_best_score, existing.score,
                )
            else:
                logger.info(
                    "[GraphGRPOProposer] screen best already present with higher/equal score "
                    "(screen=%s, existing=%s)",
                    self._screen_best_score, existing.score,
                )
            self._screen_best_injected = True
            return

        g = self._smiles_to_graph(self._screen_best_smiles)
        if g is None:
            logger.warning(
                "[GraphGRPOProposer] screen best SMILES cannot be converted to graph, skip: %s",
                self._screen_best_smiles,
            )
            self._screen_best_injected = True
            return
        if not self.valid_filter(self._screen_best_smiles, g):
            logger.warning(
                "[GraphGRPOProposer] screen best seed filtered by MolFilter, skip: %s",
                self._screen_best_smiles,
            )
            self._screen_best_injected = True
            return

        self._best_by_smiles[self._screen_best_smiles] = _TopEntry(
            score=float(self._screen_best_score),
            smiles=self._screen_best_smiles,
            graph=g,
        )
        self._topk = sorted(self._best_by_smiles.values(), key=lambda e: e.score, reverse=True)[: self.refine_topk]
        self._screen_best_injected = True
        logger.info(
            "[GraphGRPOProposer] Injected screen best into topk (score=%s, smiles=%s)",
            self._screen_best_score, self._screen_best_smiles,
        )

    def _maybe_enqueue_screen_best_for_scoring(
        self,
        out: List[str],
        seen: set[str],
        stats: Dict[str, int],
    ) -> None:
        if not self.screen_mode or self._screen_best_enqueue_attempted:
            return
        self._screen_best_enqueue_attempted = True

        if self._screen_best_smiles is None or self._screen_best_score is None:
            return

        if self._screen_best_smiles in seen:
            return

        g = self._smiles_to_graph(self._screen_best_smiles)
        if g is None:
            logger.warning(
                "[GraphGRPOProposer] screen best not queued for scoring (bad graph): %s",
                self._screen_best_smiles,
            )
            return
        if not self.valid_filter(self._screen_best_smiles, g):
            logger.warning(
                "[GraphGRPOProposer] screen best not queued for scoring (filtered): %s",
                self._screen_best_smiles,
            )
            return

        seen.add(self._screen_best_smiles)
        self._pending_graph_by_smiles[self._screen_best_smiles] = g
        out.insert(0, self._screen_best_smiles)
        if "smiles_kept" in stats:
            stats["smiles_kept"] += 1
        logger.info(
            "[GraphGRPOProposer] Queued screen best for oracle scoring (score=%s, smiles=%s)",
            self._screen_best_score, self._screen_best_smiles,
        )

    def _smiles_to_graph(self, smiles: str) -> Optional[Graph]:
        """
        Convert a SMILES string into this model's discrete graph tensors (indices or one-hot).
        Used by screen mode to seed refinement from a pre-ranked library.
        """
        try:
            mol = Chem.MolFromSmiles(smiles, sanitize=False)
            if mol is None:
                return None
            mol = Chem.RemoveHs(mol)
            try:
                Chem.Kekulize(mol, clearAromaticFlags=True)
            except Exception:
                return None

            atom_encoder = getattr(self.dataset_infos, "atom_encoder", None)
            if not isinstance(atom_encoder, dict):
                return None

            x_idx: list[int] = []
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

    def _load_checkpoint_if_available(self) -> None:
        ckpt_path = None
        try:
            ckpt_path = self.cfg.grpo.get("resume_from_checkpoint")
        except Exception:
            ckpt_path = getattr(getattr(self.cfg, "grpo", None), "resume_from_checkpoint", None)

        if not ckpt_path:
            ckpt_path = os.environ.get("GRAPH_GRPO_CKPT")

        # Backward/bridge compatibility: many configs + the mol_opt bridge set
        # `grpo.pretrained_checkpoint`, not `resume_from_checkpoint`.
        if not ckpt_path:
            try:
                ckpt_path = self.cfg.grpo.get("pretrained_checkpoint")
            except Exception:
                ckpt_path = getattr(getattr(self.cfg, "grpo", None), "pretrained_checkpoint", None)

        if not ckpt_path:
            return

        ckpt_path = os.path.expanduser(str(ckpt_path))
        if not os.path.exists(ckpt_path):
            logger.warning("[GraphGRPOProposer] Checkpoint not found: %s. Using random weights.", ckpt_path)
            return

        logger.info("[GraphGRPOProposer] Loading checkpoint: %s", ckpt_path)
        try:
            checkpoint = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        except TypeError:
            checkpoint = torch.load(ckpt_path, map_location=self.device)

        state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
        # Skip buffers with shape mismatch (e.g. node_count_prob varies by dataset split)
        model_state = self.model.state_dict()
        filtered = {k: v for k, v in state_dict.items()
                    if k not in model_state or model_state[k].shape == v.shape}
        self.model.load_state_dict(filtered, strict=False)
        # Sync p0/node_count buffers into internal distributions (Lightning hook not called here).
        try:
            if hasattr(self.model, "on_load_checkpoint"):
                self.model.on_load_checkpoint(checkpoint if isinstance(checkpoint, dict) else {})
        except Exception as exc:
            logger.warning("[GraphGRPOProposer] Failed to sync checkpoint buffers: %s", exc)

    def _graph_to_indices(self, graph: Graph) -> Tuple[torch.Tensor, torch.Tensor]:
        X, E = graph
        if torch.is_tensor(X) and X.dim() > 1:
            X = torch.argmax(X, dim=-1)
        if torch.is_tensor(E) and E.dim() > 2:
            E = torch.argmax(E, dim=-1)
        X = torch.as_tensor(X, device="cpu").contiguous()
        E = torch.as_tensor(E, device="cpu").contiguous()
        return X, E

    def sample_graphs(self, n: int, seed: int) -> List[Graph]:
        """
        Use trainer.sample_graphs_with_trajectory_tracking to generate graphs only (no scoring).
        """
        with torch.no_grad():
            graphs, node_mask, *_ = self.trainer.sample_graphs_with_trajectory_tracking(
                batch_size=int(n),
                seed=int(seed),
                total_inference_steps=int(self.trainer.sample_steps),
                force_same_start=False,
                group_size_for_same_start=1,
                return_probs=False,
            )
        return self.trainer._convert_placeholder_to_graph_list_cpu(graphs, node_mask, as_tensor=True)

    def refine_graph(
        self,
        base_graph: Graph,
        num_vars: int = 200,
        seed: int = 0,
        noise_fraction: float = 0.3,
    ) -> List[Graph]:
        """
        Call trainer.refine_candidate_via_denoising with configurable noise_fraction (no scoring).
        """
        nf = 0.3
        try:
            nf = float(noise_fraction)
            if nf < 0.0 or nf > 1.0:
                raise ValueError("noise_fraction must be in [0, 1]")
        except Exception:
            pass

        base_X, base_E = base_graph
        with torch.no_grad():
            refined_X, refined_E = self.trainer.refine_candidate_via_denoising(
                init_X=base_X,
                init_E=base_E,
                num_variations=int(num_vars),
                noise_fraction=nf,
                total_inference_steps=int(self.trainer.sample_steps),
                seed=int(seed),
            )

        out: List[Graph] = []
        for i in range(int(num_vars)):
            out.append((refined_X[i].detach().cpu().contiguous(), refined_E[i].detach().cpu().contiguous()))
        return out

    def graph_to_smiles(self, graph: Graph) -> Optional[str]:
        """
        Convert graph -> SMILES via RDKit; return None on failure.
        """
        try:
            return self.trainer._graph_to_smiles(graph)
        except Exception:
            return None

    def valid_filter(self, smiles: str, _graph: Graph) -> bool:
        """
        Local filter (does not spend oracle budget):
        - RDKit parsable
        - fragment count < 3 (i.e. allow 1-2 fragments)
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return False

        return mol

    @staticmethod
    def _safe_filename(text: str, *, max_len: int = 64) -> str:
        text = re.sub(r"[^a-zA-Z0-9_.-]+", "_", str(text))
        text = re.sub(r"_+", "_", text).strip("._")
        if not text:
            text = "item"
        return text[: int(max_len)]

    def save_topk_visualizations(self, output_dir: str, *, topk: int = 10) -> None:
        """
        Save Top-K molecules as PNG images into `output_dir`.
        Intended to be called after the benchmark finishes (e.g. on server shutdown).
        """
        out_dir = os.path.abspath(os.path.expanduser(str(output_dir)))
        os.makedirs(out_dir, exist_ok=True)

        entries = list(self._topk[: int(topk)])
        if not entries:
            logger.warning("[GraphGRPOProposer] No top-k entries to visualize (output_dir=%s)", out_dir)
            return

        molecules: List[Graph] = []
        filenames: List[str] = []
        lines: List[str] = []
        for i, e in enumerate(entries, start=1):
            molecules.append(e.graph)
            smi_short = self._safe_filename(e.smiles, max_len=32)
            filenames.append(f"top{i:02d}_score{e.score:.4f}_{smi_short}.png")
            lines.append(f"{i}\tscore={e.score:.6f}\t{e.smiles}")

        try:
            self.visualization_tools.visualize(
                out_dir,
                molecules,
                num_molecules_to_visualize=len(molecules),
                log=None,
                filenames=filenames,
            )
            with open(os.path.join(out_dir, "top10_smiles.txt"), "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
            logger.info(
                "[GraphGRPOProposer] Saved top-%d images to %s",
                len(molecules), out_dir,
            )
        except Exception as e:
            logger.warning("[GraphGRPOProposer] Failed to visualize top-k: %s", e)

    def propose(self, batch_size: int, replay: Any, state: Dict[str, Any]) -> List[str]:
        """
        Return SMILES (deduped + locally filtered). Must NOT call any oracle.

        Round 0 (warmup): sample `round0_samples` graphs.
        Round >= 1: refine Top-K bases with budget-based refine schedules and
        noise_fraction switching from `noise_fraction_early` to `noise_fraction_late`.
        """
        _ = batch_size, replay

        base_seed = int(state.get("seed", 0))
        propose_idx = int(state.get("propose_idx", 0))
        state["propose_idx"] = propose_idx + 1

        round_idx = int(state.get("round_idx", 0))

        self._pending_graph_by_smiles = {}
        out: List[str] = []
        seen: set[str] = set()
        stats = {
            "graphs_total": 0,
            "smiles_none": 0,
            "smiles_dup": 0,
            "smiles_filtered": 0,
            "smiles_kept": 0,
        }

        if round_idx == 0:
            if self.screen_mode and int(self.round0_samples) <= 0:
                out = []
                seen = set()
                stats = {
                    "smiles_total": 0,
                    "smiles_dup": 0,
                    "smiles_bad_graph": 0,
                    "smiles_filtered": 0,
                    "smiles_kept": 0,
                }
                seeds: List[str] = []
                if self._screen_best_smiles:
                    seeds.append(self._screen_best_smiles)
                seeds.extend(self._screen_smiles_ranked)
                k = int(self.screen_topk) if int(self.screen_topk) > 0 else len(seeds)
                for smi in seeds[:k]:
                    stats["smiles_total"] += 1
                    if smi in seen:
                        stats["smiles_dup"] += 1
                        continue
                    g = self._smiles_to_graph(smi)
                    if g is None:
                        stats["smiles_bad_graph"] += 1
                        continue
                    if not self.valid_filter(smi, g):
                        stats["smiles_filtered"] += 1
                        continue
                    seen.add(smi)
                    self._pending_graph_by_smiles[smi] = g
                    out.append(smi)
                    stats["smiles_kept"] += 1

                state["round_idx"] = round_idx + 1
                logger.info(
                    "[Proposer] screen_mode seeds (round0_samples<=0): "
                    "Kept=%d/%d (Dup=%d, BadGraph=%d, Filtered=%d)",
                    stats['smiles_kept'], stats['smiles_total'],
                    stats['smiles_dup'], stats['smiles_bad_graph'],
                    stats['smiles_filtered'],
                )
                if out:
                    return out
                logger.warning(
                    "[Proposer] screen_mode seeds produced 0 SMILES; falling back to sampling 1 graph"
                )
                graphs = self.sample_graphs(n=1, seed=base_seed + propose_idx)
            else:
                graphs = self.sample_graphs(n=self.round0_samples, seed=base_seed + propose_idx)
        else:
            if self.disable_refine:
                graphs = self.sample_graphs(n=self.eval_batch_size, seed=base_seed + propose_idx)
            else:
                n_oracle = int(state.get("n_oracle", 0) or 0)
                if n_oracle < self.refine_switch_budget:
                    refine_topk = self.refine_topk_early
                    refine_num_vars = self.refine_num_vars_early
                else:
                    refine_topk = self.refine_topk_late
                    refine_num_vars = self.refine_num_vars_late

                bases: List[Graph] = []
                if self._refine_seed_graphs:
                    bases.extend(self._refine_seed_graphs)
                if self._topk:
                    for entry in self._topk:
                        if entry.smiles in self._refine_seed_smiles_set:
                            continue
                        bases.append(entry.graph)
                        if len(bases) >= refine_topk:
                            break
                bases = bases[:refine_topk]
                if not bases:
                    graphs = self.sample_graphs(n=self.round0_samples, seed=base_seed + propose_idx)
                else:
                    noise_fraction = (
                        self.noise_fraction_early
                        if n_oracle < self.noise_switch_budget
                        else self.noise_fraction_late
                    )
                    if self._debug_noise:
                        logger.debug(
                            "[GraphGRPOProposer] round=%d n_oracle=%d "
                            "noise_fraction=%s (early=%s late=%s switch=%d) "
                            "refine_topk=%d refine_num_vars=%d",
                            round_idx, n_oracle, noise_fraction,
                            self.noise_fraction_early, self.noise_fraction_late,
                            self.noise_switch_budget, refine_topk, refine_num_vars,
                        )
                    graphs = []
                    for i, base in enumerate(bases[:refine_topk]):
                        graphs.extend(
                            self.refine_graph(
                                base_graph=base,
                                num_vars=refine_num_vars,
                                seed=base_seed + propose_idx * 1000 + i,
                                noise_fraction=noise_fraction,
                            )
                        )

        for g in graphs:
            stats["graphs_total"] += 1
            smi = self.graph_to_smiles(g)
            if smi is None:
                stats["smiles_none"] += 1
                continue
            if smi in seen:
                stats["smiles_dup"] += 1
                continue
            if not self.valid_filter(smi, g):
                stats["smiles_filtered"] += 1
                continue
            seen.add(smi)
            self._pending_graph_by_smiles[smi] = g
            out.append(smi)
            stats["smiles_kept"] += 1

        self._maybe_enqueue_screen_best_for_scoring(out, seen, stats)

        state["round_idx"] = round_idx + 1
        if not out:
            logger.warning(
                "[Proposer] Request produced 0 SMILES! Stats: %s (round=%d, propose_idx=%d)",
                stats, round_idx, propose_idx,
            )
        else:
            yield_pct = 100.0 * stats["smiles_kept"] / max(1, stats["graphs_total"])
            logger.info(
                "[Proposer] Yield: %.1f%% | Total=%d, Kept=%d, Dup=%d, Filtered=%d, None=%d",
                yield_pct, stats['graphs_total'], stats['smiles_kept'],
                stats['smiles_dup'], stats['smiles_filtered'], stats['smiles_none'],
            )
        return out

    def observe(self, smiles: List[str], scores: Any, replay: Any, state: Dict[str, Any]) -> None:
        """
        Receive externally-computed scores (from mol_opt oracle wrapper) and update top-k pool.
        """
        _ = replay, state

        if isinstance(scores, torch.Tensor):
            scores_list = scores.detach().cpu().tolist()
        elif isinstance(scores, np.ndarray):
            scores_list = scores.tolist()
        else:
            scores_list = list(scores) if isinstance(scores, list) else [scores]

        n = min(len(smiles), len(scores_list))
        for smi, sc in zip(smiles[:n], scores_list[:n]):
            if smi is None:
                continue
            try:
                score_f = float(sc)
            except Exception:
                continue

            if self.screen_mode and self._screen_best_smiles and smi == self._screen_best_smiles:
                logger.info(
                    "[GraphGRPOProposer] screen best scored by oracle: %s (smiles=%s)",
                    score_f, self._screen_best_smiles,
                )

            g = self._pending_graph_by_smiles.get(smi)
            if g is None:
                continue

            prev = self._best_by_smiles.get(smi)
            if prev is None or score_f > prev.score:
                self._best_by_smiles[smi] = _TopEntry(score=score_f, smiles=smi, graph=g)

        self._topk = sorted(self._best_by_smiles.values(), key=lambda e: e.score, reverse=True)[:self.refine_topk]

        if self.screen_mode and not self._screen_best_injected:
            round_idx = int(state.get("round_idx", 0) or 0)
            if round_idx == 1:
                self._maybe_inject_screen_best()


def _load_yaml(path: str) -> DictConfig:
    return OmegaConf.load(os.path.abspath(os.path.expanduser(path)))


def _compose_cfg_from_repo_defaults(*, base_config: str, override_paths: List[str]) -> DictConfig:
    """
    Lightweight Hydra-like config composition without requiring hydra-core.

    - Loads `configs/config.yaml` and its `defaults` entries.
    - Merges in each override YAML (typically one of `configs/grpo/*.yaml`).
    """
    base_config = os.path.abspath(os.path.expanduser(base_config))
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    configs_dir = os.path.join(repo_root, "configs")

    base = _load_yaml(base_config)
    defaults = base.get("defaults", []) if isinstance(base, DictConfig) else []

    merged: DictConfig = OmegaConf.create({})
    for item in list(defaults):
        if item in (None, "_self_"):
            continue
        if isinstance(item, str):
            if item == "_self_":
                continue
            continue
        if isinstance(item, (dict, DictConfig)):
            # e.g. {"general": "general_default"}
            if len(item) != 1:
                continue
            group, name = next(iter(item.items()))
            if not group or not name:
                continue
            rel = os.path.join(configs_dir, str(group), f"{name}.yaml")
            if os.path.exists(rel):
                group_cfg = _load_yaml(rel)
                # Mimic Hydra config groups by nesting under the group key.
                merged = OmegaConf.merge(merged, OmegaConf.create({str(group): group_cfg}))

    # Merge base itself last so non-default keys in config.yaml (e.g. hydra.run.dir) are present.
    merged = OmegaConf.merge(merged, base)

    # Auto-apply experiment config before grpo overrides, mirroring the training
    # command pattern: +experiment=zinc dataset=zinc +grpo=<task>
    # This ensures model architecture dims match the checkpoint.
    for p in override_paths:
        _ov = _load_yaml(p)
        ds_name = str(OmegaConf.select(_ov, "dataset.name", default="") or "").lower()
        for key in ("zinc", "moses", "guacamol"):
            if key in ds_name:
                exp_path = os.path.join(configs_dir, "experiment", f"{key}.yaml")
                if os.path.exists(exp_path):
                    merged = OmegaConf.merge(merged, _load_yaml(exp_path))
                break

    for p in override_paths:
        merged = OmegaConf.merge(merged, _load_yaml(p))
    return merged


def _write_gdpo_log(out_dir: str, dataset_name: str, log_entry: Dict[str, Any]) -> str:
    log_suffix = "moses" if "moses" in dataset_name.lower() else "zinc"
    log_path = os.path.join(out_dir, f"evaluation_dict{log_suffix}.log")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry) + "\n")
    return log_path


def _cmd_gdpo_eval(argv: List[str]) -> int:
    p = argparse.ArgumentParser(
        prog="eval_grpo_sampler.py gdpo_eval",
        description="GDPO-style docking evaluation for a trained Graph-GRPO checkpoint.",
    )
    p.add_argument("--ckpt", required=True, help="Checkpoint path to evaluate.")
    p.add_argument(
        "--grpo-config",
        required=True,
        help="Path to a GRPO config YAML (e.g. configs/grpo/lead_opt_parp1.yaml).",
    )
    p.add_argument(
        "--extra-config",
        action="append",
        default=[],
        help="Additional config YAMLs to merge before the GRPO config (repeatable).",
    )
    p.add_argument("--base-config", default="configs/config.yaml", help="Repo base config with defaults (default: configs/config.yaml).")
    p.add_argument("--target", default=None, help="Override target_name (default: from grpo config).")
    p.add_argument("--num-samples", type=int, default=None, help="Number of molecules to sample (default: gdpo_eval_samples or 512).")
    p.add_argument("--sim-threshold", type=float, default=None, help="Override novelty sim threshold (default: dataset-based).")
    p.add_argument("--seed", type=int, default=0, help="Base RNG seed for sampling (default: 0).")
    p.add_argument("--out-dir", default="gdpo_eval_results", help="Output directory for evaluation log.")
    p.add_argument("--device", default=None, help="torch device string (default: cuda if available else cpu).")
    args = p.parse_args(argv)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
    base_config = args.base_config
    if not os.path.isabs(base_config):
        base_config = os.path.join(repo_root, base_config)

    grpo_config = args.grpo_config
    if not os.path.isabs(grpo_config):
        grpo_config = os.path.join(repo_root, grpo_config)

    extra_configs = []
    for cfg_path in list(args.extra_config or []):
        path = cfg_path
        if not os.path.isabs(path):
            path = os.path.join(repo_root, path)
        extra_configs.append(path)
    cfg = _compose_cfg_from_repo_defaults(
        base_config=base_config,
        override_paths=[*extra_configs, grpo_config],
    )

    # Override checkpoint for model weights.
    try:
        cfg.grpo.pretrained_checkpoint = os.path.abspath(os.path.expanduser(args.ckpt))
    except Exception:
        pass

    target_name = args.target or str(
        getattr(getattr(cfg, "grpo", None), "target_name", "")
        or getattr(getattr(cfg, "grpo", None), "target_task", "")
        or ""
    )
    if not target_name:
        raise ValueError("Missing target_name; provide --target or set grpo.target_name in config.")
    try:
        cfg.grpo.target_name = str(target_name)
        cfg.grpo.target_task = str(target_name)
    except Exception:
        pass

    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    proposer = GraphGRPOProposer(cfg=cfg, device=torch_device)

    num_samples = args.num_samples
    if num_samples is None:
        num_samples = int(getattr(getattr(cfg, "grpo", None), "gdpo_eval_samples", 512) or 512)

    seed_base = int(args.seed)
    samples_left = int(num_samples)
    batch_size = max(1, int(proposer.eval_batch_size))
    graphs: List[Graph] = []
    batch_idx = 0
    while samples_left > 0:
        cur_bs = min(samples_left, batch_size)
        graphs.extend(proposer.sample_graphs(cur_bs, seed=seed_base + batch_idx))
        samples_left -= cur_bs
        batch_idx += 1

    smiles_list = [proposer.graph_to_smiles(g) for g in graphs]
    valid_smiles = [s for s in smiles_list if s]
    valid_r = len(valid_smiles) / (len(smiles_list) + 1e-8) if smiles_list else 0.0
    uniq_r = len(set(valid_smiles)) / (len(valid_smiles) + 1e-8) if valid_smiles else 0.0

    dataset_name = str(getattr(getattr(cfg, "dataset", None), "name", "") or "")
    sim_threshold = gdpo_get_sim_threshold(
        dataset_name,
        override=args.sim_threshold or getattr(getattr(cfg, "grpo", None), "gdpo_eval_sim_threshold", None),
    )
    train_fps = gdpo_load_train_fps(
        dataset_name=dataset_name,
        datadir=str(getattr(getattr(cfg, "dataset", None), "datadir", "") or ""),
        remove_h=bool(getattr(getattr(cfg, "dataset", None), "remove_h", False)),
        repo_root=Path(repo_root),
    )
    result = gdpo_eval_smiles(
        target_name=str(target_name),
        smiles=valid_smiles,
        train_fps=train_fps,
        sim_threshold=float(sim_threshold),
        repo_root=Path(repo_root),
        dock_exhaustiveness=getattr(getattr(cfg, "grpo", None), "gdpo_dock_exhaustiveness", None),
        dock_num_modes=getattr(getattr(cfg, "grpo", None), "gdpo_dock_num_modes", None),
        dock_timeout=getattr(getattr(cfg, "grpo", None), "gdpo_dock_timeout", None),
        dock_num_workers=int(getattr(getattr(cfg, "grpo", None), "num_reward_workers", 1) or 1),
        dock_cpu_per_worker=int(getattr(getattr(cfg, "grpo", None), "gdpo_dock_cpu_per_worker", 1) or 1),
    )

    top_ds_mean, top_ds_std = result.get("top_ds", (float("nan"), float("nan")))
    log_entry = {
        "seed": seed_base,
        "dataset": dataset_name,
        "target_prop": str(target_name),
        "VALID": round(100 * valid_r, 4),
        "UNIQ": round(100 * uniq_r, 4),
        "novelty": result.get("novelty", 0.0),
        "top_ds": [top_ds_mean, top_ds_std],
        "avgscore": result.get("avgscore", 0.0),
        "hit": result.get("hit", 0.0),
        "avgds": round(result.get("avgds", 0.0), 4),
        "avgqed": round(result.get("avgqed", 0.0), 4),
        "avgsa": round(result.get("avgsa", 0.0), 4),
        "sim_threshold": float(sim_threshold),
        "samples": int(num_samples),
    }

    out_dir = os.path.abspath(os.path.expanduser(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)
    log_path = _write_gdpo_log(out_dir, dataset_name, log_entry)

    logger.info(
        "[GDPO Eval] VALID=%s UNIQ=%s Novelty=%.4f Top-DS=%.4f+/-%.4f "
        "Hit=%.4f AvgDS=%.4f Log=%s",
        log_entry['VALID'], log_entry['UNIQ'], log_entry['novelty'],
        top_ds_mean, top_ds_std, log_entry['hit'], log_entry['avgds'], log_path,
    )
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        logger.error("Usage: python src/eval_grpo_sampler.py gdpo_eval --ckpt ... --grpo-config configs/grpo/lead_opt_parp1.yaml")
        return 2
    cmd = argv[0].strip().lower()
    if cmd in {"gdpo_eval", "gdpo"}:
        return _cmd_gdpo_eval(argv[1:])
    logger.error("Unknown command: %s", cmd)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
