#!/bin/bash
# P1: Dynamic Prior Ablation - 23 tasks, seed=0
# Condition A: dynamic p0 (as saved in RL checkpoint)
# Condition B: static p0 (force override to dataset default)
#
# Usage:
#   bash scripts/run_p1_dynamic_p0_ablation.sh [A|B|AB]

set -euo pipefail

CKPT_DIR="checkpoint"
BUDGET=10000
SEED=0

run_condition() {
  local cond="$1"
  local force_static="$2"
  local tag="p1_${cond}"
  local out_dir="results/${tag}_seed${SEED}_budget${BUDGET}"

  echo "========================================"
  echo " P1 Ablation - Condition $cond"
  echo " force_static_p0=$force_static"
  echo " output: $out_dir"
  echo "========================================"

  local extra_flags=()
  if [[ "$force_static" == "1" ]]; then
    extra_flags+=(--force-static-p0)
  fi

  python scripts/run_mol_opt.py \
    --batch \
    --ckpt-dir "$CKPT_DIR" \
    --max-oracle-calls "$BUDGET" \
    --seed "$SEED" \
    --output-dir "$out_dir" \
    "${extra_flags[@]}"
}

MODE="${1:-AB}"

case "$MODE" in
  A|a)
    run_condition "A_dynamic_p0" "0"
    ;;
  B|b)
    run_condition "B_static_p0" "1"
    ;;
  AB|ab|Ab)
    run_condition "A_dynamic_p0" "0"
    run_condition "B_static_p0" "1"
    ;;
  *)
    echo "Usage: $0 [A|B|AB]"
    exit 1
    ;;
esac

echo "P1 ablation done."
