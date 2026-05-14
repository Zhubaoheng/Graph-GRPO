from __future__ import annotations

import csv
import logging
import multiprocessing as mp
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, QED

from analysis.lead_opt_oracle import LeadOptOracle

_GDPO_DOCK_CTX: Dict[str, Any] = {}


def _load_sascorer():
    try:
        from grpo.rewards.base import sascorer as _sascorer
    except ImportError:
        _sascorer = None
    if _sascorer is None:
        raise ImportError(
            "sascorer is required for GDPO-style docking evaluation. "
            "Install RDKit Contrib SA_Score or provide analysis.sascorer."
        )
    return _sascorer


def gdpo_get_sim_threshold(dataset_name: str, override: Optional[float] = None) -> float:
    if override is not None:
        try:
            return float(override)
        except Exception:
            pass
    name = str(dataset_name or "").lower()
    if "moses" in name:
        return 0.6
    return 0.4


def _resolve_datadir(datadir: str, repo_root: Path) -> Path:
    datadir_path = Path(datadir)
    if not datadir_path.is_absolute():
        datadir_path = (repo_root / datadir_path).resolve()
    return datadir_path


def gdpo_load_train_fps(
    *,
    dataset_name: str,
    datadir: str,
    remove_h: bool,
    repo_root: Path,
    cache_name: str = "train_fps_r2_1024.pkl",
) -> List[Any]:
    datadir_path = _resolve_datadir(datadir, repo_root)
    cache_path = datadir_path / cache_name
    if cache_path.is_file():
        try:
            import pickle

            with open(cache_path, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass

    train_smiles: List[str] = []
    name = str(dataset_name or "").lower()
    if "zinc" in name:
        fname = "train_smiles_no_h.npy" if remove_h else "train_smiles_h.npy"
        smiles_path = datadir_path / fname
        if not smiles_path.is_file():
            raise FileNotFoundError(f"Train SMILES not found: {smiles_path}")
        arr = np.load(smiles_path, allow_pickle=True)
        train_smiles = [str(s) for s in arr.tolist()]
    elif "moses" in name:
        smiles_path = datadir_path / "new_train.smiles"
        if smiles_path.is_file():
            with open(smiles_path, "r", encoding="utf-8") as f:
                train_smiles = [ln.strip() for ln in f if ln.strip()]
        else:
            smiles_path = datadir_path / "train_moses.csv"
            if not smiles_path.is_file():
                raise FileNotFoundError(f"Train SMILES not found: {smiles_path}")
            with open(smiles_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                train_smiles = [row.get("SMILES", "").strip() for row in reader]
    else:
        raise ValueError(f"Unsupported dataset for GDPO docking eval: {dataset_name}")

    fps: List[Any] = []
    for smi in train_smiles:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024))
        except Exception:
            continue

    if not fps:
        raise ValueError("No valid train fingerprints computed for GDPO docking eval.")

    try:
        import pickle

        with open(cache_path, "wb") as f:
            pickle.dump(fps, f)
    except Exception:
        pass

    return fps


def gdpo_max_sims(mols: Iterable[Chem.Mol], train_fps: List[Any]) -> List[float]:
    if not train_fps:
        return [0.0 for _ in mols]
    sims: List[float] = []
    for mol in mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, 1024)
            vals = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
            sims.append(float(max(vals)) if vals else 0.0)
        except Exception:
            sims.append(0.0)
    return sims


def gdpo_hit_threshold(target_name: str) -> float:
    if target_name == "parp1":
        return 10.0
    if target_name == "fa7":
        return 8.5
    if target_name == "5ht1b":
        return 8.7845
    if target_name == "jak2":
        return 9.1
    if target_name == "braf":
        return 10.3
    raise ValueError(f"Unknown target protein for GDPO eval: {target_name}")

def _prepare_pdbqt_from_mol(protonated: Chem.Mol) -> Optional[str]:
    try:
        from meeko import MoleculePreparation, PDBQTMolecule
    except Exception:
        return None

    prep = MoleculePreparation()
    try:
        setups = prep.prepare(protonated)
    except Exception:
        return None
    if not setups:
        return None

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
        pass

    if hasattr(prep, "write_pdbqt_string"):
        try:
            return prep.write_pdbqt_string(setups[0])
        except Exception:
            pass

    return None

def _gdpo_dock_worker_init(
    target_name: str,
    receptor_path: str,
    box: Dict[str, Tuple[float, float, float]],
    exhaustiveness: int,
    n_poses: int,
    cpu_per_worker: int,
    dock_timeout: Optional[int],
) -> None:
    global _GDPO_DOCK_CTX
    oracle = None
    try:
        oracle = LeadOptOracle(
            target_name=str(target_name),
            exhaustiveness=int(exhaustiveness),
            num_modes=int(n_poses),
            dock_timeout=int(dock_timeout) if dock_timeout is not None else None,
        )
        try:
            oracle._num_cpu = int(cpu_per_worker)
        except Exception:
            pass
    except Exception as exc:
        oracle = exc
    _GDPO_DOCK_CTX = {
        "oracle": oracle,
        "target_name": target_name,
        "receptor_path": receptor_path,
        "box": box,
        "exhaustiveness": int(exhaustiveness),
        "n_poses": int(n_poses),
        "cpu_per_worker": int(cpu_per_worker),
        "dock_timeout": dock_timeout,
    }

def _gdpo_dock_worker(smiles: str) -> Tuple[float, str]:
    ctx = _GDPO_DOCK_CTX
    oracle = ctx.get("oracle")
    if oracle is None or isinstance(oracle, Exception):
        return -1.0, "dock_fail"
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return -1.0, "invalid"
    try:
        score, reason = oracle._score_single_with_reason(smiles)
    except Exception:
        return -1.0, "dock_fail"
    if reason != "ok":
        return -1.0, reason or "dock_fail"
    return -float(score), "ok"


def gdpo_vina_dock_smiles(
    *,
    target_name: str,
    smiles_list: List[str],
    repo_root: Path,
    receptor_path: Optional[Path] = None,
    box_by_target: Optional[Dict[str, Dict[str, Tuple[float, float, float]]]] = None,
    exhaustiveness: Optional[int] = None,
    n_poses: Optional[int] = None,
    num_workers: Optional[int] = None,
    cpu_per_worker: Optional[int] = None,
    dock_timeout: Optional[int] = None,
) -> List[float]:
    box_map = box_by_target or LeadOptOracle._BOX_BY_TARGET
    box = box_map.get(str(target_name))
    if box is None:
        raise ValueError(f"Docking box not configured for target '{target_name}'.")

    if receptor_path is None:
        receptor_path = repo_root / "data" / "lead_opt" / "docking" / f"{target_name}.pdbqt"
    if not receptor_path.is_file():
        raise FileNotFoundError(f"Receptor pdbqt not found: {receptor_path}")

    scores: List[float] = []
    total = len(smiles_list)
    log_every = int(os.environ.get("GDPO_DOCK_LOG_EVERY", "100") or 0)
    if log_every < 0:
        log_every = 0
    start_time = time.time()
    invalid = 0
    dock_fail = 0
    dock_ok = 0
    reason_counts: Counter[str] = Counter()

    ex = int(exhaustiveness) if exhaustiveness is not None else 16
    poses = int(n_poses) if n_poses is not None else 20
    workers = int(num_workers) if num_workers is not None else 1
    cpu = int(cpu_per_worker) if cpu_per_worker is not None else 1
    if total <= 0:
        if reason_counts:
            summary = ", ".join(f"{k}={v}" for k, v in reason_counts.most_common())
            logger.info("[GDPO Eval] Docking reasons: %s", summary)
        return scores
    workers = max(1, min(workers, total))
    cpu = max(1, cpu)

    if workers <= 1:
        _gdpo_dock_worker_init(
            target_name=str(target_name),
            receptor_path=str(receptor_path),
            box=box,
            exhaustiveness=ex,
            n_poses=poses,
            cpu_per_worker=cpu,
            dock_timeout=dock_timeout,
        )
        for idx, smi in enumerate(smiles_list, start=1):
            score, reason = _gdpo_dock_worker(smi)
            scores.append(float(score))
            reason_counts[reason] += 1
            if reason == "ok":
                dock_ok += 1
            elif reason == "invalid":
                invalid += 1
            else:
                dock_fail += 1

            if log_every and (idx % log_every == 0 or idx == total):
                elapsed = time.time() - start_time
                logger.info(
                    "[GDPO Eval] Docking %d/%d (%.1f%%) ok=%d invalid=%d fail=%d elapsed=%.1fs",
                    idx, total, 100.0 * idx / total, dock_ok, invalid, dock_fail, elapsed,
                )
        return scores

    ctx = mp.get_context("spawn")
    with ctx.Pool(
        processes=workers,
        initializer=_gdpo_dock_worker_init,
        initargs=(str(target_name), str(receptor_path), box, ex, poses, cpu, dock_timeout),
    ) as pool:
        for idx, (score, reason) in enumerate(pool.imap(_gdpo_dock_worker, smiles_list), start=1):
            scores.append(float(score))
            reason_counts[reason] += 1
            if reason == "ok":
                dock_ok += 1
            elif reason == "invalid":
                invalid += 1
            else:
                dock_fail += 1

            if log_every and (idx % log_every == 0 or idx == total):
                elapsed = time.time() - start_time
                logger.info(
                    "[GDPO Eval] Docking %d/%d (%.1f%%) ok=%d invalid=%d fail=%d elapsed=%.1fs",
                    idx, total, 100.0 * idx / total, dock_ok, invalid, dock_fail, elapsed,
                )
    if reason_counts:
        summary = ", ".join(f"{k}={v}" for k, v in reason_counts.most_common())
        logger.info("[GDPO Eval] Docking reasons: %s", summary)
    return scores


def gdpo_eval_smiles(
    *,
    target_name: str,
    smiles: List[str],
    train_fps: List[Any],
    sim_threshold: float,
    repo_root: Path,
    receptor_path: Optional[Path] = None,
    dock_exhaustiveness: Optional[int] = None,
    dock_num_modes: Optional[int] = None,
    dock_num_workers: Optional[int] = None,
    dock_cpu_per_worker: Optional[int] = None,
    dock_timeout: Optional[int] = None,
) -> Dict[str, Any]:
    num_mols = len(smiles)
    if num_mols == 0:
        return {
            "validity": 0,
            "uniqueness": 0,
            "novelty": 0,
            "top_ds": (0, 0),
            "hit": 0,
            "avgscore": 0,
            "avgds": 0,
            "avgqed": 0,
            "avgsa": 0,
        }

    filtered = [s for s in smiles if s]
    validity = len(filtered) / (num_mols + 1e-8)
    mols = [Chem.MolFromSmiles(s) for s in filtered]
    uniqueness = len(set(filtered)) / (len(filtered) + 1e-8) if filtered else 0.0

    sims = gdpo_max_sims(mols, train_fps)
    novelty = (
        sum(1 for s in sims if s < sim_threshold) / len(sims)
        if sims
        else 0.0
    )

    # Deduplicate before docking, like GDPO.
    uniq_smiles: List[str] = []
    uniq_mols: List[Chem.Mol] = []
    uniq_sims: List[float] = []
    seen = set()
    for smi, mol, sim in zip(filtered, mols, sims):
        if smi in seen:
            continue
        seen.add(smi)
        uniq_smiles.append(smi)
        uniq_mols.append(mol)
        uniq_sims.append(sim)

    before_num = len(uniq_smiles)
    ds_scores = gdpo_vina_dock_smiles(
        target_name=target_name,
        smiles_list=uniq_smiles,
        repo_root=repo_root,
        receptor_path=receptor_path,
        exhaustiveness=dock_exhaustiveness,
        n_poses=dock_num_modes,
        num_workers=dock_num_workers,
        cpu_per_worker=dock_cpu_per_worker,
        dock_timeout=dock_timeout,
    )
    keep_smiles: List[str] = []
    keep_mols: List[Chem.Mol] = []
    keep_ds: List[float] = []
    keep_sims: List[float] = []
    for smi, mol, ds, sim in zip(uniq_smiles, uniq_mols, ds_scores, uniq_sims):
        if ds == -1:
            continue
        keep_smiles.append(smi)
        keep_mols.append(mol)
        keep_ds.append(float(ds))
        keep_sims.append(float(sim))

    after_num = len(keep_smiles)
    num_mols = num_mols - (before_num - after_num)

    sascorer = _load_sascorer()
    qed_vals = []
    sa_vals = []
    for mol in keep_mols:
        try:
            qed_vals.append(float(QED.qed(mol)))
        except Exception:
            qed_vals.append(0.0)
        try:
            sa_raw = float(sascorer.calculateScore(mol))
            sa_vals.append(float((10.0 - sa_raw) / 9.0))
        except Exception:
            sa_vals.append(0.0)

    if keep_ds:
        avgscore = float(np.mean((np.array(keep_ds) / 10.0) * np.array(qed_vals) * np.array(sa_vals)))
        avgds = float(np.mean(np.array(keep_ds) / 10.0))
        avgqed = float(np.mean(qed_vals))
        avgsa = float(np.mean(sa_vals))
    else:
        avgscore = 0.0
        avgds = 0.0
        avgqed = 0.0
        avgsa = 0.0

    # Apply GDPO filters.
    filtered_rows = []
    for ds, qed, sa, sim in zip(keep_ds, qed_vals, sa_vals, keep_sims):
        if qed <= 0.5:
            continue
        if sa <= (10 - 5) / 9:
            continue
        if sim >= sim_threshold:
            continue
        filtered_rows.append(ds)

    filtered_rows = sorted(filtered_rows, reverse=True)
    num_top5 = int(num_mols * 0.05)
    top_slice = filtered_rows[:num_top5] if num_top5 > 0 else []
    if top_slice:
        top_mean = float(np.mean(top_slice))
        top_std = float(np.std(top_slice, ddof=1)) if len(top_slice) > 1 else float("nan")
    else:
        top_mean = float("nan")
        top_std = float("nan")

    hit_thr = gdpo_hit_threshold(target_name)
    hit = (
        sum(1 for ds in filtered_rows if ds > hit_thr) / (num_mols + 1e-6)
        if num_mols > 0
        else 0.0
    )

    return {
        "validity": validity,
        "uniqueness": uniqueness,
        "novelty": novelty,
        "top_ds": (top_mean, top_std),
        "hit": hit,
        "avgscore": avgscore,
        "avgds": avgds,
        "avgqed": avgqed,
        "avgsa": avgsa,
    }
