"""In-process GraphGRPO optimizer for mol_opt benchmarks.

Based on mol_opt/main/graph_grpo/run.py (MIT License), simplified to
run entirely in-process without JSONL subprocess bridges.
"""
from __future__ import annotations

import os
import re
from typing import Any, Dict

from mol_opt.optimizer import BaseOptimizer


def _oracle_to_grpo_cfg_name(oracle_name: str) -> str:
    """Map TDC oracle name to GRPO config filename."""
    name = str(oracle_name).strip().lower()
    name = name.replace(" ", "_").replace("-", "_").replace("/", "_")
    name = re.sub(r"__+", "_", name)
    if name.endswith("_current"):
        name = name[:-8]
    if name.endswith("_latest"):
        name = name[:-7]
    if name in {"median_1", "median_2"}:
        name = name.replace("_", "")
    return name


def _infer_experiment(config_dir: str, cfg_name: str) -> str:
    """Infer the experiment name from the grpo config's dataset.name.

    This mirrors the training command pattern where users specify
    ``+experiment=zinc dataset=zinc +grpo=<task>``.  The experiment
    config sets the correct model architecture dimensions.
    """
    import yaml

    grpo_path = os.path.join(config_dir, "grpo", f"{cfg_name}.yaml")
    if not os.path.isfile(grpo_path):
        return ""
    with open(grpo_path) as f:
        raw = yaml.safe_load(f) or {}
    ds_name = (raw.get("dataset") or {}).get("name", "").lower()
    for key in ("zinc", "moses", "guacamol"):
        if key in ds_name:
            return key
    return ""


def _abspath_if_relative(root: str, path: Any) -> Any:
    if not path:
        return path
    path = os.path.expanduser(str(path))
    return path if os.path.isabs(path) else os.path.join(root, path)


class GraphGRPO_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "graph_grpo"

    def _compose_grpo_cfg(self, oracle_name: str, config: Dict[str, Any]):
        """Compose Hydra config for the given oracle task."""
        from grpo.eval_sampler import GraphGRPOProposer  # noqa: F401

        # Locate repo root from this file's position: src/mol_opt/graph_grpo.py -> repo root
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
        config_dir = os.path.join(repo_root, "configs")

        cfg_name = config.get("grpo_cfg") or _oracle_to_grpo_cfg_name(oracle_name)

        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra

        if GlobalHydra.instance().is_initialized():
            GlobalHydra.instance().clear()

        with initialize_config_dir(version_base="1.3", config_dir=config_dir):
            # Mirror the training command pattern:
            #   +experiment=zinc dataset=zinc +grpo=<task>
            # The experiment override sets model architecture dims to match checkpoints.
            experiment = config.get("experiment") or _infer_experiment(config_dir, cfg_name)
            overrides = []
            if experiment:
                overrides.append(f"+experiment={experiment}")
            overrides.append(f"+grpo={cfg_name}")

            cfg = compose(config_name="config", overrides=overrides)

        # Resolve relative paths against repo root
        try:
            cfg.dataset.datadir = _abspath_if_relative(repo_root, cfg.dataset.datadir)
        except Exception:
            pass

        ckpt_override = config.get("checkpoint_path") or os.environ.get("GRAPH_GRPO_CKPT")
        if ckpt_override:
            try:
                cfg.grpo.pretrained_checkpoint = _abspath_if_relative(repo_root, ckpt_override)
            except Exception:
                pass

        # Proposer never computes rewards locally
        try:
            cfg.grpo.num_reward_workers = 5
        except Exception:
            pass

        try:
            cfg.general.test_only = True
        except Exception:
            pass

        return cfg

    def _optimize(self, oracle, config):
        """Run the propose/observe optimization loop in-process."""
        self.oracle.assign_evaluator(oracle)

        oracle_name = getattr(oracle, "name", None) or getattr(self.args, "oracles", ["unknown"])[0]

        import torch
        from grpo.eval_sampler import GraphGRPOProposer

        cfg = self._compose_grpo_cfg(oracle_name, config)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        proposer = GraphGRPOProposer(cfg=cfg, device=device)

        replay = {}
        state = {"seed": int(getattr(self, "seed", 0)), "round_idx": 0, "propose_idx": 0}
        max_empty_rounds = int(config.get("max_empty_rounds", 50))
        empty_rounds = 0

        while not self.finish:
            smiles = proposer.propose(
                batch_size=int(config.get("batch_size", 0) or 0),
                replay=replay,
                state=state,
            )
            if not smiles:
                empty_rounds += 1
                if empty_rounds >= max_empty_rounds:
                    print(f"[GraphGRPO] Stopping: {max_empty_rounds} consecutive rounds produced 0 SMILES.")
                    break
                continue
            empty_rounds = 0
            scores = self.oracle(smiles)
            state["n_oracle"] = len(self.oracle.mol_buffer)
            proposer.observe(smiles, scores, replay=replay, state=state)
            if self.finish:
                break
