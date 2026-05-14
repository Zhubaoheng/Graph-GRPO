"""Mixin providing evaluation and diagnostic methods for GRPOTrainer."""

import json
import logging
import os
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from grpo.eval_docking import gdpo_eval_smiles, gdpo_get_sim_threshold, gdpo_load_train_fps
from grpo.trajectory_data import TrajectoryData

try:
    import swanlab
except ImportError:
    swanlab = None

logger = logging.getLogger(__name__)

Graph = Tuple[torch.Tensor, torch.Tensor]


class EvaluationMixin:
    """Evaluation and diagnostic methods extracted from GRPOTrainer."""

    # ------------------------------------------------------------------
    # TDC zero-reward diagnostics
    # ------------------------------------------------------------------

    def _debug_tdc_zero_reward_batch(self, graph_list: List, *, max_samples: int = 32) -> None:
        """
        Quick diagnostic when TDC rewards are all-zero for a batch.
        Checks whether graph -> RDKit Mol conversion succeeds and
        Chem.SanitizeMol / MolToSmiles success rate, to distinguish
        "oracle truly outputs 0" from "all molecules judged invalid".
        """
        try:
            from rdkit import Chem
            from analysis.rdkit_functions import build_molecule
        except Exception as e:
            logger.debug("[TDC Debug] Cannot import RDKit/build_molecule: %s", e)
            return

        dataset_info = getattr(self.model, "dataset_info", None)
        atom_decoder = getattr(dataset_info, "atom_decoder", None) if dataset_info is not None else None
        if not atom_decoder:
            logger.debug("[TDC Debug] Missing dataset_info.atom_decoder, cannot perform graph->SMILES check")
            return

        n_total = min(int(max_samples), len(graph_list))
        if n_total <= 0:
            return

        built_ok = 0
        sanitize_ok = 0
        smiles_ok = 0
        err_counter: Dict[str, int] = defaultdict(int)
        example_smiles: List[str] = []

        for at, et in graph_list[:n_total]:
            try:
                at_t = at.detach().cpu() if torch.is_tensor(at) else torch.as_tensor(at)
                et_t = et.detach().cpu() if torch.is_tensor(et) else torch.as_tensor(et)
                if at_t.dim() == 2:
                    at_t = at_t.argmax(dim=-1)
                if et_t.dim() == 3:
                    et_t = et_t.argmax(dim=-1)
                at_t = at_t.to(dtype=torch.long)
                et_t = et_t.to(dtype=torch.long)
            except Exception as e:
                err_counter[f"tensor_cast:{type(e).__name__}"] += 1
                continue

            try:
                mol = build_molecule(at_t, et_t, atom_decoder)
                built_ok += 1
            except Exception as e:
                err_counter[f"build_molecule:{type(e).__name__}"] += 1
                continue

            try:
                Chem.SanitizeMol(mol)
                sanitize_ok += 1
            except Exception as e:
                err_counter[f"sanitize:{type(e).__name__}"] += 1
                continue

            try:
                smi = Chem.MolToSmiles(mol)
                if smi:
                    smiles_ok += 1
                    if len(example_smiles) < 5:
                        example_smiles.append(smi)
            except Exception as e:
                err_counter[f"smiles:{type(e).__name__}"] += 1
                continue

        common_err = None
        if err_counter:
            common_err = max(err_counter.items(), key=lambda kv: kv[1])[0]

        logger.info(
            "[TDC Debug] All rewards in this batch are 0; graph->SMILES diagnostic: "
            "n=%d, built=%d, sanitize=%d, smiles=%d%s",
            n_total, built_ok, sanitize_ok, smiles_ok,
            f", top_err={common_err}" if common_err else "",
        )
        if example_smiles:
            logger.info("   Examples (first %d): %s", len(example_smiles), example_smiles)

        # Optional: if PyTDC is installed, call the oracle on a few SMILES to check if outputs are truly 0
        try:
            if example_smiles:
                from tdc import Oracle

                oracle_name = None
                try:
                    oracle_name = self.cfg.grpo.get("tdc_oracle", None)
                except Exception:
                    oracle_name = getattr(self.cfg.grpo, "tdc_oracle", None)
                if oracle_name:
                    oracle = Oracle(name=str(oracle_name))
                    raw = oracle(example_smiles[: min(5, len(example_smiles))])
                    arr = np.asarray(raw)
                    if arr.ndim == 0:
                        vals = [float(arr)]
                    else:
                        vals = [float(x) for x in arr.reshape(-1).tolist()]
                    logger.debug("   PyTDC oracle('%s') output: %s", oracle_name, vals)
        except Exception as e:
            logger.debug("   PyTDC oracle call failed (ignored): %s", e)

    # ------------------------------------------------------------------
    # Periodic evaluation rollout
    # ------------------------------------------------------------------

    def _maybe_run_evaluation(self):
        if self.eval_interval <= 0:
            return
        if self.global_step <= 0 or self.global_step % self.eval_interval != 0:
            return
        eval_rewards = self._run_evaluation_rollout()
        if eval_rewards.numel() == 0:
            return
        reward_mean = eval_rewards.mean().item()
        reward_std = eval_rewards.std().item()
        reward_min = eval_rewards.min().item()
        reward_max = eval_rewards.max().item()
        logger.info(
            "Eval @ step %d: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
            self.global_step, reward_mean, reward_std, reward_min, reward_max,
        )
        # [Monitoring] Calculate Validity and Positive Reward Stats
        valid_mask = eval_rewards > 0.01
        num_valid = valid_mask.sum().item()
        valid_rate = num_valid / eval_rewards.numel() if eval_rewards.numel() > 0 else 0.0

        if num_valid > 0:
            avg_valid_reward = eval_rewards[valid_mask].mean().item()
        else:
            avg_valid_reward = 0.0

        if swanlab is not None and swanlab.run is not None:
            swanlab.log({
                'eval/reward_mean': reward_mean,
                'eval/reward_std': reward_std,
                'eval/reward_min': reward_min,
                'eval/reward_max': reward_max,
                'eval/valid_rate': valid_rate,          # New Metric
                'eval/avg_valid_reward': avg_valid_reward # New Metric
            }, step=self.global_step)
        logger.info("  [Stats] Mean Reward: %.4f | Max Reward: %.4f | Valid Rate: %.2f%% | Avg Valid Reward: %.4f", reward_mean, reward_max, valid_rate * 100, avg_valid_reward)

    def _run_evaluation_rollout(self) -> torch.Tensor:
        """Simplified Evaluation: Sample a group, calc VUN & Score."""
        eval_graphs = []
        self.core_model.eval()
        try:
            # Sample a group of graphs (e.g., 2048 total, split into batches)
            target_samples = 2048
            batch_size = 128 # Keep batch size small for memory
            num_batches = (target_samples + batch_size - 1) // batch_size

            logger.info("Sampling %d graphs for evaluation in %d batches...", target_samples, num_batches)

            for i in range(num_batches):
                current_batch_size = min(batch_size, target_samples - len(eval_graphs))
                if current_batch_size <= 0:
                    break

                graphs, node_mask, _, _, _, _, _ = self.sample_graphs_with_trajectory_tracking(
                    batch_size=current_batch_size,
                    seed=int(time.time() * 1000) % (2**32) + self.global_step + i,
                    total_inference_steps=self.sample_steps,
                    force_same_start=False,
                )

                batch_graphs = self._convert_placeholder_to_graph_list_cpu(graphs, node_mask, as_tensor=True)
                eval_graphs.extend(batch_graphs)
                logger.info("   - Batch %d/%d: Collected %d/%d", i + 1, num_batches, len(eval_graphs), target_samples)
        finally:
            self.core_model.eval()

        if not eval_graphs:
            return torch.tensor([], dtype=torch.float32)

        logger.info("Starting evaluation on %d samples...", len(eval_graphs))

        # 1. Calc Score (Reward)
        eval_rewards = self._compute_rewards_multiprocess_sync(
            eval_graphs,
            timeout=1800,
            context="eval",
        )

        return eval_rewards

    # ------------------------------------------------------------------
    # GDPO docking evaluation
    # ------------------------------------------------------------------

    def _maybe_run_gdpo_eval(self):
        every = int(
            self._get_cfg_value(self.cfg, "grpo.gdpo_eval_every_n_epochs")
            or self._get_cfg_value(self.cfg, "grpo.lead_eval_every_n_epochs")
            or 0
        )
        if every <= 0:
            return
        if self.epoch <= 0 or self.epoch % every != 0:
            return

        reward_type = str(self._get_cfg_value(self.cfg, "grpo.reward_type") or "").lower()
        if reward_type not in ("gdpo_docking", "gdpo"):
            return

        target_name = (
            self._get_cfg_value(self.cfg, "grpo.target_name")
            or self._get_cfg_value(self.cfg, "grpo.target_task")
            or ""
        )
        if not target_name:
            logger.warning("[GDPO Eval] Missing grpo.target_name; skipping docking eval.")
            return

        dataset_name = str(self._get_cfg_value(self.cfg, "dataset.name") or "")
        sim_threshold = gdpo_get_sim_threshold(
            dataset_name,
            override=self._get_cfg_value(self.cfg, "grpo.gdpo_eval_sim_threshold"),
        )
        target_samples = 2048
        eval_exhaustiveness = self._get_cfg_value(self.cfg, "grpo.gdpo_dock_exhaustiveness") or 8
        eval_num_modes = self._get_cfg_value(self.cfg, "grpo.gdpo_dock_num_modes")
        eval_timeout = self._get_cfg_value(self.cfg, "grpo.gdpo_dock_timeout")
        eval_workers = int(self._get_cfg_value(self.cfg, "grpo.num_reward_workers") or 1)
        eval_workers = max(1, eval_workers)
        eval_cpu_per_worker =  1
        out_dir = self._get_cfg_value(self.cfg, "grpo.gdpo_eval_out_dir") or "gdpo_eval_results"
        out_dir = os.path.abspath(os.path.expanduser(str(out_dir)))
        os.makedirs(out_dir, exist_ok=True)

        # Preserve RNG states to avoid perturbing training randomness.
        py_state = random.getstate()
        np_state = np.random.get_state()
        torch_state = torch.get_rng_state()
        cuda_states = None
        if torch.cuda.is_available():
            try:
                cuda_states = torch.cuda.get_rng_state_all()
            except Exception:
                cuda_states = None

        was_training = self.core_model.training
        self.core_model.eval()

        try:
            logger.info("[GDPO Eval] Sampling %d graphs for docking evaluation...", target_samples)
            graphs, node_mask, _, _, _, _, _ = self.sample_graphs_with_trajectory_tracking(
                batch_size=target_samples,
                seed=int(time.time() * 1000) % (2**32) + self.global_step,
                total_inference_steps=self.sample_steps,
                force_same_start=False,
            )
            eval_graphs: List[Graph] = self._convert_placeholder_to_graph_list_cpu(
                graphs, node_mask, as_tensor=True
            )

            smiles_list = [self._graph_to_smiles(g) for g in eval_graphs]
            valid_smiles = [s for s in smiles_list if s]
            valid_r = len(valid_smiles) / (len(smiles_list) + 1e-8) if smiles_list else 0.0
            uniq_r = len(set(valid_smiles)) / (len(valid_smiles) + 1e-8) if valid_smiles else 0.0

            repo_root = Path(__file__).resolve().parents[1]
            train_fps = gdpo_load_train_fps(
                dataset_name=dataset_name,
                datadir=str(self._get_cfg_value(self.cfg, "dataset.datadir") or ""),
                remove_h=bool(self._get_cfg_value(self.cfg, "dataset.remove_h")),
                repo_root=repo_root,
            )
            result = gdpo_eval_smiles(
                target_name=str(target_name),
                smiles=valid_smiles,
                train_fps=train_fps,
                sim_threshold=float(sim_threshold),
                repo_root=repo_root,
                dock_exhaustiveness=eval_exhaustiveness,
                dock_num_modes=int(eval_num_modes) if eval_num_modes is not None else None,
                dock_num_workers=eval_workers,
                dock_cpu_per_worker=eval_cpu_per_worker,
                dock_timeout=int(eval_timeout) if eval_timeout is not None else None,
            )

            top_ds_mean, top_ds_std = result.get("top_ds", (float("nan"), float("nan")))
            log_entry = {
                "epoch": int(self.epoch),
                "global_step": int(self.global_step),
                "dataset": str(self._get_cfg_value(self.cfg, "dataset.name") or ""),
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
                "samples": int(target_samples),
            }

            log_suffix = "moses" if "moses" in dataset_name.lower() else "zinc"
            log_path = os.path.join(out_dir, f"evaluation_dict{log_suffix}.log")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")

            logger.info(
                "[GDPO Eval] VALID=%s UNIQ=%s Novelty=%.4f Top-DS=%.4f+/-%.4f Hit=%.4f AvgDS=%.4f",
                log_entry['VALID'], log_entry['UNIQ'], log_entry['novelty'],
                top_ds_mean, top_ds_std, log_entry['hit'], log_entry['avgds'],
            )

            if swanlab is not None and swanlab.run is not None:
                swanlab.log(
                    {
                        "gdpo_eval/valid_percent": log_entry["VALID"],
                        "gdpo_eval/uniq_percent": log_entry["UNIQ"],
                        "gdpo_eval/novelty": log_entry["novelty"],
                        "gdpo_eval/top_ds_mean": top_ds_mean,
                        "gdpo_eval/top_ds_std": top_ds_std,
                        "gdpo_eval/hit": log_entry["hit"],
                        "gdpo_eval/avgds": log_entry["avgds"],
                        "gdpo_eval/avgqed": log_entry["avgqed"],
                        "gdpo_eval/avgsa": log_entry["avgsa"],
                    },
                    step=self.global_step,
                )
        finally:
            # Restore RNG states
            try:
                random.setstate(py_state)
                np.random.set_state(np_state)
                torch.set_rng_state(torch_state)
                if cuda_states is not None:
                    torch.cuda.set_rng_state_all(cuda_states)
            except Exception:
                pass
            self.core_model.train(was_training)
