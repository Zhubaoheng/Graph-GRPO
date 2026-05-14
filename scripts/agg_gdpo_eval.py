#!/usr/bin/env python3
import argparse
import json
import logging
import math
import statistics
from collections import defaultdict
from typing import Any, Dict, Iterable, List

logger = logging.getLogger(__name__)


def _load_entries(path: str) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(entry, dict):
                entries.append(entry)
    return entries


def _filter_entries(entries: Iterable[Dict[str, Any]], target: str) -> List[Dict[str, Any]]:
    if not target:
        return list(entries)
    target = target.lower()
    keep: List[Dict[str, Any]] = []
    for e in entries:
        t = str(e.get("target_prop", "")).lower()
        if t == target:
            keep.append(e)
    return keep


def _stat(values: List[float]) -> Dict[str, float]:
    if not values:
        return {"mean": float("nan"), "std": float("nan")}
    mean = statistics.mean(values)
    if len(values) > 1:
        std = statistics.stdev(values)
    else:
        std = 0.0
    return {"mean": mean, "std": std}


def main() -> int:
    p = argparse.ArgumentParser(description="Aggregate all numeric indicators in eval logs.")
    p.add_argument("--log", required=True, help="Path to evaluation_dict*.log")
    p.add_argument("--target", default="", help="Filter by target_prop (e.g. parp1)")
    p.add_argument("--last", type=int, default=3, help="Use only last N matching entries (default: 3)")
    args = p.parse_args()

    entries = _load_entries(args.log)
    entries = _filter_entries(entries, args.target)
    if args.last and args.last > 0:
        entries = entries[-args.last :]

    if not entries:
        print("No entries found.")
        return 0

    metrics = defaultdict(list)
    seeds = []
    targets = set()

    for e in entries:
        if "seed" in e:
            seeds.append(e["seed"])
        if "target_prop" in e:
            targets.add(str(e["target_prop"]))
        
        for k, v in e.items():
            if k in ["seed", "samples", "sim_threshold", "target_prop", "dataset"]:
                continue
            
            # Special handling for top_ds
            if k == "top_ds" and isinstance(v, (list, tuple)) and len(v) > 0:
                metrics["top_ds_mean"].append(float(v[0]))
                if len(v) > 1:
                    metrics["top_ds_std"].append(float(v[1]))
                continue

            if isinstance(v, (int, float)):
                metrics[k].append(float(v))

    target_str = ", ".join(sorted(targets)) if targets else "N/A"
    print(f"Aggregated {len(entries)} entries. Target: {target_str}. Seeds: {seeds}")
    print("-" * 60)
    print(f"{'Metric':<20} | {'Mean':<12} | {'Std':<12}")
    print("-" * 60)
    
    # Sort metrics for consistent output
    for k in sorted(metrics.keys()):
        stat = _stat(metrics[k])
        print(f"{k:<20} | {stat['mean']:<12.6f} | {stat['std']:<12.6f}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
