import logging
from typing import Dict, List, Optional, Tuple

import torch

from grpo.rewards.base import BaseRewardFunction, DefaultRewardFunction, resolve_target_task
from grpo.rewards.graph_rewards import PlanarGraphReward, SBMGraphReward, TreeGraphReward
from grpo.rewards.molecular_validity import MolecularValidityReward
from grpo.rewards.target_mpo import TargetMPOReward
from grpo.rewards.tdc_oracle import TDCOracleReward
from grpo.rewards.gdpo_docking import GDPODockingReward
from grpo.rewards.valsartan import ValsartanSmartsReward

from grpo.eval_docking import gdpo_get_sim_threshold

logger = logging.getLogger(__name__)


def create_reward_function(
    reward_type: str,
    cfg=None,
    device=None,
    **kwargs
) -> BaseRewardFunction:
    """
    Reward function factory.

    Args:
        reward_type: Type of reward function
        cfg: Full configuration object
        device: Device
        **kwargs: Additional parameters for backward compatibility

    Returns:
        Reward function instance of the corresponding type
    """
    reward_type = reward_type.lower()

    datamodule = kwargs.get('datamodule')
    model = kwargs.get('model')
    ref_metrics = kwargs.get('ref_metrics')
    name = kwargs.get('name')
    atom_decoder = kwargs.get('atom_decoder')
    target_node_dist = kwargs.get('target_node_dist')
    target_edge_dist = kwargs.get('target_edge_dist')
    dist_coef = kwargs.get('dist_coef', None)
    dist_scale = kwargs.get('scale_factor') or kwargs.get('dist_scale_factor')
    dist_clip = kwargs.get('clip_range') or kwargs.get('dist_clip_range')
    edge_dist_factor = kwargs.get('edge_dist_factor')
    precomputed_node_weights = kwargs.get('precomputed_node_weights')
    precomputed_edge_weights = kwargs.get('precomputed_edge_weights')
    sa_threshold = kwargs.get("sa_threshold")
    sim_threshold = kwargs.get("sim_threshold")
    dock_exhaustiveness = kwargs.get("dock_exhaustiveness")
    dock_num_modes = kwargs.get("dock_num_modes")
    dock_timeout = kwargs.get("dock_timeout")
    dataset_name = kwargs.get("dataset_name")
    datadir = kwargs.get("datadir")
    remove_h = kwargs.get("remove_h")

    # TDC/PMO Oracle parameters (kwargs take precedence, then cfg.grpo)
    tdc_oracle = kwargs.get("tdc_oracle")
    tdc_oracles = kwargs.get("tdc_oracles")
    tdc_aggregation = kwargs.get("tdc_aggregation")
    tdc_weights = kwargs.get("tdc_weights")
    tdc_minimize = kwargs.get("tdc_minimize")
    tdc_invalid_score = kwargs.get("tdc_invalid_score")
    tdc_clip_min = kwargs.get("tdc_clip_min")
    tdc_clip_max = kwargs.get("tdc_clip_max")
    tdc_home = kwargs.get("tdc_home")

    # [Fix] Unpack ref_metrics if provided (e.g. from dataset_info)
    if ref_metrics is not None and isinstance(ref_metrics, dict):
        # We only set them if they are NOT already in kwargs (kwargs takes precedence)
        if "ref_degree_dist" not in kwargs:
             kwargs["ref_degree_dist"] = ref_metrics.get("ref_degree_dist")
        if "ref_clustering_hist" not in kwargs:
             kwargs["ref_clustering_hist"] = ref_metrics.get("ref_clustering_hist")
        if "ref_orbit_mean" not in kwargs:
             kwargs["ref_orbit_mean"] = ref_metrics.get("ref_orbit_mean")

    # Target Task Name
    target_task = kwargs.get("target_task")
    if target_task is None:
        target_task = resolve_target_task(cfg)

    lead_target_name = kwargs.get("target_name")

    # Distribution coefficient from config takes priority (aligned with GRPO hyperparameters)
    if dist_coef is None and cfg is not None and hasattr(cfg, "grpo"):
        try:
            dist_coef = cfg.grpo.get("dist_coef", None)
        except AttributeError:
            dist_coef = getattr(cfg.grpo, "dist_coef", None)
        if dist_coef is None:
            try:
                dist_coef = cfg.grpo.get("reward_dist_coef", None)
            except AttributeError:
                dist_coef = getattr(cfg.grpo, "reward_dist_coef", None)

    # Default to using the dataset distribution from the model as target
    if target_node_dist is None and model is not None and hasattr(model, 'dataset_info'):
        target_node_dist = getattr(model.dataset_info, 'node_types', None)
    if target_edge_dist is None and model is not None and hasattr(model, 'dataset_info'):
        target_edge_dist = getattr(model.dataset_info, 'edge_types', None)

    if atom_decoder is None and model is not None and hasattr(model, 'dataset_info'):
        atom_decoder = getattr(model.dataset_info, 'atom_decoder', None)

    if reward_type == "base":
        logger.info("Creating base debug reward function (connectivity/planarity)")
        return BaseRewardFunction(device=device)

    if reward_type == "default":
        logger.info("Creating default reward function")
        return DefaultRewardFunction(device=device)

    elif reward_type in ("planar_graph", "planar"):
        return PlanarGraphReward(
            device=device,
            datamodule=datamodule,
            ref_degree_dist=kwargs.get("ref_degree_dist"),
            ref_clustering_hist=kwargs.get("ref_clustering_hist"),
            ref_orbit_mean=kwargs.get("ref_orbit_mean"),
        )

    elif reward_type in ("sbm", "sbm_graph"):
        return SBMGraphReward(
            device=device,
            datamodule=datamodule,
            ref_degree_dist=kwargs.get("ref_degree_dist"),
            ref_clustering_hist=kwargs.get("ref_clustering_hist"),
            ref_orbit_mean=kwargs.get("ref_orbit_mean"),
        )

    elif reward_type in ("tree", "tree_graph"):
        return TreeGraphReward(
            device=device,
            datamodule=datamodule,
            ref_degree_dist=kwargs.get("ref_degree_dist"),
            ref_clustering_hist=kwargs.get("ref_clustering_hist"),
            ref_orbit_mean=kwargs.get("ref_orbit_mean"),
        )

    elif reward_type in ("guacamol_mpo", "guacamol_goal", "target_mpo", "target_goal"):
        return TargetMPOReward(
            target_task=target_task if target_task else "penalized_logp",
            atom_decoder=atom_decoder,
            device=device
        )
    elif reward_type in ("tdc_oracle", "tdc_pmo", "pmo"):
        if cfg is not None and hasattr(cfg, "grpo"):
            try:
                tdc_oracle = tdc_oracle or cfg.grpo.get("tdc_oracle", None)
                tdc_oracles = tdc_oracles or cfg.grpo.get("tdc_oracles", None)
                tdc_aggregation = tdc_aggregation or cfg.grpo.get("tdc_aggregation", None)
                tdc_weights = tdc_weights or cfg.grpo.get("tdc_weights", None)
                if tdc_minimize is None:
                    tdc_minimize = cfg.grpo.get("tdc_minimize", None)
                if tdc_invalid_score is None:
                    tdc_invalid_score = cfg.grpo.get("tdc_invalid_score", None)
                if tdc_clip_min is None:
                    tdc_clip_min = cfg.grpo.get("tdc_clip_min", None)
                if tdc_clip_max is None:
                    tdc_clip_max = cfg.grpo.get("tdc_clip_max", None)
                if tdc_home is None:
                    tdc_home = cfg.grpo.get("tdc_home", None)
            except AttributeError:
                pass

        oracle_names = tdc_oracles or tdc_oracle
        if oracle_names is None:
            raise ValueError("TDC reward requires grpo.tdc_oracle or grpo.tdc_oracles to be set")

        return TDCOracleReward(
            oracle_names=oracle_names,
            atom_decoder=atom_decoder,
            aggregation=tdc_aggregation or "mean",
            weights=tdc_weights,
            minimize=bool(tdc_minimize) if tdc_minimize is not None else False,
            invalid_score=float(tdc_invalid_score) if tdc_invalid_score is not None else 0.0,
            clip_min=tdc_clip_min,
            clip_max=tdc_clip_max,
            tdc_home=tdc_home,
            device=device,
        )
    elif reward_type in ("gdpo_docking", "gdpo"):
        if cfg is not None and hasattr(cfg, "grpo"):
            try:
                lead_target_name = lead_target_name or cfg.grpo.get("target_name", None)
            except AttributeError:
                try:
                    lead_target_name = lead_target_name or getattr(cfg.grpo, "target_name", None)
                except Exception:
                    pass
            if sim_threshold is None:
                sim_override = None
                try:
                    sim_override = cfg.grpo.get("gdpo_sim_threshold", None)
                except Exception:
                    sim_override = getattr(cfg.grpo, "gdpo_sim_threshold", None)
                if sim_override is None:
                    try:
                        sim_override = cfg.grpo.get("gdpo_eval_sim_threshold", None)
                    except Exception:
                        sim_override = getattr(cfg.grpo, "gdpo_eval_sim_threshold", None)
                dataset_name = dataset_name or getattr(getattr(cfg, "dataset", None), "name", None)
                sim_threshold = gdpo_get_sim_threshold(dataset_name or "", override=sim_override)

        if dataset_name is None and cfg is not None:
            dataset_name = getattr(getattr(cfg, "dataset", None), "name", None)
        if datadir is None and cfg is not None:
            datadir = getattr(getattr(cfg, "dataset", None), "datadir", None)
        if remove_h is None and cfg is not None:
            remove_h = getattr(getattr(cfg, "dataset", None), "remove_h", None)

        if lead_target_name is None:
            raise ValueError("GDPODockingReward requires grpo.target_name or target_name in kwargs.")

        return GDPODockingReward(
            target_name=str(lead_target_name),
            atom_decoder=atom_decoder,
            device=device,
            sa_threshold=sa_threshold,
            sim_threshold=sim_threshold,
            dock_exhaustiveness=dock_exhaustiveness,
            dock_num_modes=dock_num_modes,
            dock_timeout=dock_timeout,
            dataset_name=dataset_name,
            datadir=datadir,
            remove_h=remove_h,
        )
    elif reward_type in ("molecular_validity", "guacamol_reward", "gracamol_reward", "gracamol"):
        return MolecularValidityReward(
            atom_decoder=atom_decoder,
            device=device,
            target_node_dist=target_node_dist,
            target_edge_dist=target_edge_dist,
            dist_coef=dist_coef if dist_coef is not None else 0.1,
            scale_factor=dist_scale if dist_scale is not None else 10.0,
            clip_range=dist_clip if dist_clip is not None else 2.0,
            edge_dist_factor=edge_dist_factor if edge_dist_factor is not None else 1.0,
            precomputed_node_weights=precomputed_node_weights,
            precomputed_edge_weights=precomputed_edge_weights,
        )

    elif reward_type in ("valsartan_smarts", "valsartan_smarts_easy", "valsartan"):
        return ValsartanSmartsReward(
            mode="easy", # Force easy mode as requested
            atom_decoder=atom_decoder,
            device=device
        )

    else:
        logger.warning("Unknown reward function type: %s, using default reward function", reward_type)
        return DefaultRewardFunction(device=device)
