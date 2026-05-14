#!/usr/bin/env python3
"""
Sweep training runs over a fixed list of TDC/PMO oracle tasks.

Typical usage:
  python scripts/sweep_tdc_oracles.py --ckpt /path/to/zinc250k.ckpt --cuda 0

Notes:
  - Uses existing Hydra config: +grpo=tdc_pmo
  - Forces swanlab disabled by default (avoid network dependency)
  - If an oracle name matches an existing Guacamol MPO config in configs/grpo,
    reuses its grpo.target_node_count for node-count control.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

from omegaconf import OmegaConf


ORACLES: List[str] = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazone_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo",
]


def _load_optional_target_node_count(repo_root: Path, oracle: str) -> Optional[int]:
    cfg_path = repo_root / "configs" / "grpo" / f"{oracle}.yaml"
    if not cfg_path.exists():
        return None
    try:
        cfg = OmegaConf.load(cfg_path)
    except Exception:
        return None
    try:
        val = OmegaConf.select(cfg, "grpo.target_node_count")
    except Exception:
        val = None
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _env_for_subprocess(cuda_visible_devices: Optional[str]) -> Dict[str, str]:
    env = os.environ.copy()
    if cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    # Workarounds for some restricted /dev/shm environments (safe no-ops elsewhere)
    env.setdefault("KMP_USE_SHM", "0")
    env.setdefault("KMP_DISABLE_SHARED_MEMORY", "1")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    return env


def _format_cmd(cmd: List[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to zinc250k checkpoint (.ckpt).")
    parser.add_argument("--cuda", default=None, help="CUDA_VISIBLE_DEVICES value (e.g., '0' or '0,1').")
    parser.add_argument("--experiment", default="guacamol", help="Hydra experiment name (default: guacamol).")
    parser.add_argument("--grpo-config", default="tdc_pmo", help="Hydra grpo config (default: tdc_pmo).")
    parser.add_argument("--swanlab", default="disabled", help="general.swanlab override (default: disabled).")
    parser.add_argument("--only", nargs="*", default=None, help="Run only these oracle names.")
    parser.add_argument("--skip", nargs="*", default=None, help="Skip these oracle names.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands only.")
    args = parser.parse_args()

    ckpt_path = Path(os.path.expanduser(args.ckpt))
    if not ckpt_path.exists():
        raise SystemExit(f"Checkpoint not found: {ckpt_path}")

    selected = ORACLES
    if args.only:
        only = set(args.only)
        selected = [o for o in selected if o in only]
    if args.skip:
        skip = set(args.skip)
        selected = [o for o in selected if o not in skip]
    if not selected:
        raise SystemExit("No oracles selected.")

    env = _env_for_subprocess(args.cuda)

    for oracle in selected:
        target_node_count = _load_optional_target_node_count(repo_root, oracle)

        cmd = [
            "python",
            "src/train_flow_grpo.py",
            f"+experiment={args.experiment}",
            f"+grpo={args.grpo_config}",
            f"general.name={oracle}",
            f"general.swanlab={args.swanlab}",
            f"grpo.pretrained_checkpoint={str(ckpt_path)}",
            f"grpo.tdc_oracle={oracle}",
        ]
        if target_node_count is not None:
            cmd.append(f"grpo.target_node_count={target_node_count}")

        logger.info("%s", _format_cmd(cmd))
        if args.dry_run:
            continue

        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True)


if __name__ == "__main__":
    main()

