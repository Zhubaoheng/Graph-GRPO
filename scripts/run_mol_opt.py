#!/usr/bin/env python
"""Unified mol_opt benchmark runner for Graph-GRPO.

Runs mol_opt evaluation in a single process, single conda environment,
without JSONL subprocess bridges or external shell scripts.

Usage:
    # Single oracle
    python scripts/run_mol_opt.py --oracle DRD2 --ckpt /path/to/ckpt

    # Multiple oracles
    python scripts/run_mol_opt.py --oracle QED DRD2 GSK3B --ckpt /path/to/ckpt --seed 0 1 2

    # All 23 PMO benchmark tasks
    python scripts/run_mol_opt.py --batch --ckpt /path/to/ckpt

    # Per-task checkpoints from a directory (Graph-RL mode)
    python scripts/run_mol_opt.py --batch --ckpt-dir /path/to/checkpoint_dir

    # Ablation: disable refinement
    python scripts/run_mol_opt.py --batch --ckpt /path/to/ckpt --disable-refine

    # Ablation: with screen mode
    python scripts/run_mol_opt.py --batch --ckpt /path/to/ckpt --screen-mode --screen-csv /path/to/zinc250k.csv

    # Ablation: force static p0
    python scripts/run_mol_opt.py --batch --ckpt-dir /path/to/ckpts --force-static-p0

Requirements:
    - conda env with tdc, torch, rdkit (e.g. graph-grpo)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

# Add src/ to sys.path
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))
sys.path.insert(0, _REPO_ROOT)

from mol_opt.graph_grpo import GraphGRPO_Optimizer, _oracle_to_grpo_cfg_name  # noqa: E402
from grpo.tdc_compat import patch_tdc_legacy_sklearn_pickles  # noqa: E402

# All 23 PMO benchmark tasks
ALL_ORACLES = [
    "Albuterol_Similarity", "Amlodipine_MPO", "Celecoxib_Rediscovery",
    "Deco_Hop", "DRD2", "Fexofenadine_MPO", "GSK3B",
    "Isomers_C7H8N2O2", "Isomers_C9H10N2O2PF2Cl",
    "JNK3", "Median 1", "Median 2", "Mestranol_Similarity",
    "Osimertinib_MPO", "Perindopril_MPO", "QED", "Ranolazine_MPO",
    "Scaffold_Hop", "Sitagliptin_MPO", "Thiothixene_Rediscovery",
    "Troglitazone_Rediscovery", "Valsartan_Smarts", "Zaleplon_MPO",
]


def _resolve_ckpt(ckpt_dir: str, oracle_name: str) -> str | None:
    """Resolve per-task checkpoint from a directory.

    Looks for:  <dir>/<cfg_name>.ckpt, <dir>/<cfg_name>/*.ckpt (latest),
                <dir>/<oracle_name>.ckpt
    """
    cfg_name = _oracle_to_grpo_cfg_name(oracle_name)
    for name in (cfg_name, oracle_name):
        path = os.path.join(ckpt_dir, f"{name}.ckpt")
        if os.path.isfile(path):
            return path
        subdir = os.path.join(ckpt_dir, name)
        if os.path.isdir(subdir):
            ckpts = sorted(glob.glob(os.path.join(subdir, "*.ckpt")),
                           key=os.path.getmtime, reverse=True)
            if ckpts:
                return ckpts[0]
    return None


def build_args(parsed: argparse.Namespace) -> argparse.Namespace:
    """Build the args namespace that BaseOptimizer.__init__ expects."""
    return argparse.Namespace(
        method="graph_grpo",
        smi_file=None,
        n_jobs=-1,
        output_dir=parsed.output_dir,
        max_oracle_calls=parsed.max_oracle_calls,
        freq_log=parsed.freq_log,
        log_results=parsed.log_results,
        oracles=[],
        seed=parsed.seed,
        task="simple",
        config_default="hparams_default.yaml",
        pickle_directory=None,
        wandb="disabled",
    )


def run_single(oracle_name: str, args: argparse.Namespace, ckpt: str, seed: int):
    """Run a single oracle evaluation."""
    patch_tdc_legacy_sklearn_pickles()
    from tdc import Oracle as TDCOracle

    if ckpt:
        os.environ["GRAPH_GRPO_CKPT"] = os.path.abspath(ckpt)

    # Per-task output directory
    run_output_dir = os.path.join(args.output_dir, f"{oracle_name}_{seed}")
    os.makedirs(run_output_dir, exist_ok=True)
    args.output_dir = run_output_dir
    os.environ["GRAPH_GRPO_OUTPUT_DIR"] = run_output_dir

    print(f"\n{'='*60}")
    print(f"Oracle: {oracle_name} | Seed: {seed} | Budget: {args.max_oracle_calls}")
    print(f"Output: {run_output_dir}")
    print(f"{'='*60}\n")

    config = {}
    if ckpt:
        config["checkpoint_path"] = os.path.abspath(ckpt)

    oracle = TDCOracle(name=oracle_name)
    optimizer = GraphGRPO_Optimizer(args=args)

    t0 = time.time()
    optimizer.optimize(oracle=oracle, config=config, seed=seed)
    elapsed = time.time() - t0

    print(f"\n[Done] {oracle_name} seed={seed} in {elapsed/60:.1f} min")
    final_buffer = getattr(optimizer, "last_mol_buffer", optimizer.mol_buffer)
    print(f"  Oracle calls: {len(final_buffer)}")
    if final_buffer:
        import numpy as np
        top_scores = sorted(final_buffer.values(), key=lambda x: x[0], reverse=True)
        print(f"  Top-1: {top_scores[0][0]:.4f}")
        if len(top_scores) >= 10:
            print(f"  Top-10 avg: {np.mean([s[0] for s in top_scores[:10]]):.4f}")

    return elapsed


def main():
    parser = argparse.ArgumentParser(
        description="Unified mol_opt benchmark runner for Graph-GRPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--oracle", nargs="+", default=None,
                        help="Oracle name(s), e.g. DRD2 QED GSK3B")
    parser.add_argument("--batch", action="store_true",
                        help="Run all 23 PMO benchmark tasks")
    parser.add_argument("--ckpt", type=str, default=None,
                        help="Path to a single checkpoint (used for all tasks)")
    parser.add_argument("--ckpt-dir", type=str, default=None,
                        help="Directory of per-task checkpoints (Graph-RL mode)")
    parser.add_argument("--seed", type=int, nargs="+", default=[0],
                        help="Random seed(s)")
    parser.add_argument("--max-oracle-calls", type=int, default=10000,
                        help="Maximum oracle calls per task (default: 10000)")
    parser.add_argument("--freq-log", type=int, default=100,
                        help="Logging frequency (default: 100)")
    parser.add_argument("--output-dir", type=str, default="results/mol_opt",
                        help="Output directory (default: results/mol_opt)")
    parser.add_argument("--log-results", action="store_true",
                        help="Log detailed results table")

    # Ablation flags (passed as env vars to GraphGRPOProposer)
    parser.add_argument("--disable-refine", action="store_true",
                        help="Disable refinement (ablation)")
    parser.add_argument("--screen-mode", action="store_true",
                        help="Enable screen mode (ablation)")
    parser.add_argument("--screen-csv", type=str, default=None,
                        help="Path to screen CSV (e.g. zinc250k_scores.csv)")
    parser.add_argument("--force-static-p0", action="store_true",
                        help="Force static p0 distribution (ablation)")
    parsed = parser.parse_args()

    if not parsed.oracle and not parsed.batch:
        parser.error("Specify --oracle <name> or --batch")
    if not parsed.ckpt and not parsed.ckpt_dir:
        parser.error("Specify --ckpt or --ckpt-dir")
    if parsed.screen_mode and not parsed.screen_csv:
        parser.error("--screen-mode requires --screen-csv")

    # Set ablation env vars (read by GraphGRPOProposer)
    if parsed.disable_refine:
        os.environ["GRAPH_GRPO_DISABLE_REFINE"] = "1"
    if parsed.screen_mode:
        os.environ["GRAPH_GRPO_SCREEN_MODE"] = "1"
    if parsed.screen_csv:
        os.environ["GRAPH_GRPO_SCREEN_CSV"] = os.path.abspath(parsed.screen_csv)
    if parsed.force_static_p0:
        os.environ["GRAPH_GRPO_FORCE_STATIC_P0"] = "1"

    oracles = ALL_ORACLES if parsed.batch else parsed.oracle
    base_output_dir = parsed.output_dir

    # Header
    ablation_flags = []
    if parsed.disable_refine:
        ablation_flags.append("disable_refine")
    if parsed.screen_mode:
        ablation_flags.append("screen_mode")
    if parsed.force_static_p0:
        ablation_flags.append("force_static_p0")

    print(f"Repo: {_REPO_ROOT}")
    print(f"Checkpoint: {parsed.ckpt or parsed.ckpt_dir or '(none)'}")
    print(f"Mode: {'ckpt-dir (per-task)' if parsed.ckpt_dir else 'single ckpt'}")
    print(f"Tasks: {len(oracles)} oracle(s)")
    print(f"Seeds: {parsed.seed}")
    if ablation_flags:
        print(f"Ablation: {', '.join(ablation_flags)}")

    # Batch summary CSV
    summary_path = os.path.join(base_output_dir, "summary.csv")
    os.makedirs(base_output_dir, exist_ok=True)
    summary_rows = []
    skipped = 0
    had_error = False

    for oracle_name in oracles:
        for seed in parsed.seed:
            # Resolve checkpoint
            if parsed.ckpt_dir:
                ckpt = _resolve_ckpt(parsed.ckpt_dir, oracle_name)
                if not ckpt:
                    print(f"\n[SKIP] {oracle_name}: no checkpoint found in {parsed.ckpt_dir}",
                          file=sys.stderr)
                    skipped += 1
                    had_error = True
                    continue
            else:
                ckpt = parsed.ckpt

            args = build_args(parsed)
            args.output_dir = base_output_dir
            args.oracles = [oracle_name]
            try:
                elapsed = run_single(oracle_name, args, ckpt, seed)
                summary_rows.append({
                    "oracle": oracle_name, "seed": seed,
                    "ckpt": ckpt, "elapsed_min": f"{elapsed / 60:.1f}",
                    "status": "ok",
                })
            except Exception as e:
                print(f"\n[ERROR] {oracle_name} seed={seed}: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                summary_rows.append({
                    "oracle": oracle_name, "seed": seed,
                    "ckpt": ckpt or "", "elapsed_min": "",
                    "status": f"error: {e}",
                })
                had_error = True
                continue

    # Write batch summary
    if summary_rows:
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["oracle", "seed", "ckpt", "elapsed_min", "status"])
            writer.writeheader()
            writer.writerows(summary_rows)
        print(f"\nBatch summary: {summary_path}")

    if skipped:
        print(f"WARNING: {skipped} task(s) skipped (checkpoint not found).", file=sys.stderr)

    print(f"\nAll done. Results in: {base_output_dir}/")
    if had_error:
        sys.exit(1)


if __name__ == "__main__":
    main()
