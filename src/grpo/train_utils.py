"""Helper functions extracted from the Hydra entry point ``train_flow_grpo.py``.

This module contains dataset/model creation logic and checkpoint handling
utilities that are shared between the training entry point and the evaluation
sampler.
"""

import logging
import os
import pathlib
import time

import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


def _should_strict_resume(cfg: DictConfig) -> bool:
    try:
        return bool(cfg.grpo.get("strict_resume", True))
    except Exception:
        return True


def _validate_resume_checkpoint(flow_grpo_module: pl.LightningModule, ckpt_path: str, cfg: DictConfig) -> None:
    """Pre-check a resume checkpoint for parameter key/shape compatibility.

    Tolerates missing keys that are expected to be injected or rebuilt at
    runtime (e.g. dataset-specific buffers from older checkpoints).

    Args:
        flow_grpo_module: The current GRPO Lightning module.
        ckpt_path: Path to the checkpoint file.
        cfg: Hydra configuration object.

    Raises:
        ValueError: If checkpoint keys/shapes are incompatible.
    """
    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(ckpt_path, map_location="cpu")

    state_dict = checkpoint.get("state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if isinstance(state_dict, dict):
        # Reject pretrained DeFoG ckpts (single "model." prefix) here so the
        # user is redirected to grpo.pretrained_checkpoint, which has the
        # prefix-remapping logic. Lightning's trainer.fit(ckpt_path=...) does
        # not auto-remap, so silent acceptance would fail downstream anyway.
        looks_like_pretrained = (
            all(k.startswith("model.") for k in state_dict.keys())
            and not any(k.startswith("model.model.") for k in state_dict.keys())
        )
        if looks_like_pretrained:
            raise ValueError(
                f"Checkpoint '{ckpt_path}' looks like a pretrained DeFoG flow model "
                f"(state_dict keys start with a single 'model.' prefix). "
                f"For cold-start RL, pass it via 'grpo.pretrained_checkpoint=...' "
                f"instead of 'grpo.resume_from_checkpoint=...'. "
                f"'resume_from_checkpoint' is reserved for resuming a previous "
                f"GRPO Lightning training run (keys prefixed with 'model.model.')."
            )

        normalized_state_dict = dict(state_dict)
        for key in list(state_dict.keys()):
            if key.startswith("model."):
                continue
            prefixed = f"model.{key}"
            if prefixed not in normalized_state_dict:
                normalized_state_dict[prefixed] = state_dict[key]
        state_dict = normalized_state_dict
    model_state = flow_grpo_module.state_dict()

    model_keys = set(model_state.keys())
    ckpt_keys = set(state_dict.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)

    compat_missing = {
        "model.p0_node_dist",
        "model.p0_edge_dist",
        "model.node_count_prob",
        "model.node_count_buffer_rewards",
        "model.node_count_buffer_nodes",
        "model.node_count_buffer_filled",
    }

    def _is_compat_key(key: str) -> bool:
        if key in compat_missing:
            return True
        if key.startswith("model.sampling_metrics."):
            return True
        return False

    missing = [key for key in missing if not _is_compat_key(key)]

    shape_mismatch = []
    for key in sorted(model_keys & ckpt_keys):
        v_ckpt = state_dict.get(key)
        v_model = model_state.get(key)
        if hasattr(v_ckpt, "shape") and hasattr(v_model, "shape"):
            if tuple(v_ckpt.shape) != tuple(v_model.shape):
                shape_mismatch.append(key)

    shape_mismatch = [key for key in shape_mismatch if not _is_compat_key(key)]

    if missing or unexpected or shape_mismatch:
        raise ValueError(
            "Resume checkpoint is incompatible with current model. "
            f"missing={len(missing)}, unexpected={len(unexpected)}, shape_mismatch={len(shape_mismatch)}. "
            f"Example missing={missing[:5]}, unexpected={unexpected[:5]}, shape_mismatch={shape_mismatch[:5]}"
        )

    if "grpo_trainer_state" not in checkpoint:
        logger.warning(
            "Resume checkpoint missing 'grpo_trainer_state'; "
            "GRPO trainer buffers will be reinitialized."
        )


def create_datamodule_and_model_components(cfg: DictConfig):
    """Create the datamodule and model components.

    Mirrors the logic in the original ``main.py`` entry point to ensure
    consistent dataset / model initialisation.

    Args:
        cfg: Hydra configuration object containing dataset and model settings.

    Returns:
        Tuple of (datamodule, model_kwargs).
    """
    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        from analysis.visualization import NonMolecularVisualization
        from datasets.spectre_dataset import (
            SpectreGraphDataModule,
            SpectreDatasetInfos,
        )
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
            SBMSamplingMetrics,
            Comm20SamplingMetrics,
            TreeSamplingMetrics,
        )
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures

        datamodule = SpectreGraphDataModule(cfg)
        if dataset_config["name"] == "sbm":
            sampling_metrics = SBMSamplingMetrics(datamodule)
        elif dataset_config["name"] == "comm20":
            sampling_metrics = Comm20SamplingMetrics(datamodule)
        elif dataset_config["name"] == "planar":
            sampling_metrics = PlanarSamplingMetrics(datamodule)
        elif dataset_config["name"] == "tree":
            sampling_metrics = TreeSamplingMetrics(datamodule)
        else:
            raise NotImplementedError(
                f"Dataset {dataset_config['name']} not implemented"
            )

        dataset_infos = SpectreDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    elif dataset_config["name"] in ["qm9", "guacamol", "guacamol_mpo", "moses"] or "zinc" in dataset_config["name"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
        from models.extra_features import ExtraFeatures
        from models.extra_features_molecular import ExtraMolecularFeatures
        from analysis.visualization import MolecularVisualization

        if "qm9" in dataset_config["name"]:
            from datasets import qm9_dataset

            datamodule = qm9_dataset.QM9DataModule(cfg)
            dataset_infos = qm9_dataset.QM9infos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = qm9_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )

        elif dataset_config["name"] in ["guacamol", "guacamol_mpo"]:
            # Choose Guacamol split implementation based on cfg.dataset.split
            split = getattr(cfg.dataset, "split", "ood")
            if split == "ood":
                from datasets import guacamol_dataset_ood as guacamol_dataset
            else:
                from datasets import guacamol_dataset as guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)

            # For MPO tasks (empty dataset), skip loading smiles or provide dummy
            if dataset_config.get("empty", False) or dataset_config["name"] == "guacamol_mpo":
                 dataset_smiles = {"train": [], "val": [], "test": []}
            else:
                dataset_smiles = guacamol_dataset.get_smiles(
                    raw_dir=datamodule.train_dataset.raw_dir,
                    filter_dataset=cfg.dataset.filter,
                )
        elif dataset_config.name == "moses":
            from datasets import moses_dataset

            datamodule = moses_dataset.MosesDataModule(cfg)
            dataset_infos = moses_dataset.MOSESinfos(datamodule, cfg)
            dataset_smiles = moses_dataset.get_smiles(
                raw_dir=datamodule.train_dataset.raw_dir,
                filter_dataset=cfg.dataset.filter,
            )
        elif "zinc" in dataset_config["name"]:
            from datasets import zinc_dataset

            use_empty = bool(getattr(cfg.dataset, "empty", False))
            if use_empty:
                # Avoid triggering real data download: use a minimal mock batch
                # to initialise model dimensions.
                from torch_geometric.data import Data
                from torch_geometric.data import InMemoryDataset

                class _MockZINCDataset(InMemoryDataset):
                    def __init__(self, stage: str, root: str):
                        self.stage = stage
                        super().__init__(root)
                        self.data, self.slices = None, None

                    def len(self):
                        return 1

                    def get(self, idx):
                        # ZINC (no aromatic): 9 atom types, 4 edge types ([no bond, single, double, triple])
                        x = torch.zeros((2, 9), dtype=torch.float)
                        x[:, 0] = 1.0
                        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
                        edge_attr = torch.zeros((2, 4), dtype=torch.float)
                        edge_attr[:, 1] = 1.0
                        y = torch.zeros((1, 0), dtype=torch.float)
                        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, idx=0)

                    @property
                    def raw_file_names(self):
                        return []

                    @property
                    def processed_file_names(self):
                        return []

                    def download(self):
                        pass

                    def process(self):
                        pass

                from datasets.abstract_dataset import MolecularDataModule

                base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
                root_path = os.path.join(base_path, getattr(cfg.dataset, "datadir", "data/zinc/"))
                datasets = {
                    "train": _MockZINCDataset("train", root=root_path),
                    "val": _MockZINCDataset("val", root=root_path),
                    "test": _MockZINCDataset("test", root=root_path),
                }
                datamodule = MolecularDataModule(cfg, datasets)
                dataset_smiles = {"train": [], "val": [], "test": []}
            else:
                datamodule = zinc_dataset.ZINCDataModule(cfg)
                dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
                dataset_smiles = zinc_dataset.get_smiles(
                    cfg=cfg,
                    datamodule=datamodule,
                    dataset_infos=dataset_infos,
                    evaluate_datasets=False,
                )

            if use_empty:
                dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
        else:
            raise ValueError("Dataset not implemented")

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = ExtraMolecularFeatures(dataset_infos=dataset_infos)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )

        train_metrics = TrainMolecularMetricsDiscrete(dataset_infos)
        add_virtual_states = "absorbing" == cfg.model.transition
        sampling_metrics = SamplingMolecularMetrics(
            dataset_infos, dataset_smiles, cfg, add_virtual_states=add_virtual_states
        )
        visualization_tools = MolecularVisualization(
            cfg.dataset.remove_h, dataset_infos=dataset_infos
        )
    elif dataset_config["name"] == "tls":
        from datasets import tls_dataset
        from metrics.tls_metrics import TLSSamplingMetrics
        from analysis.visualization import NonMolecularVisualization
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures

        datamodule = tls_dataset.TLSDataModule(cfg)
        dataset_infos = tls_dataset.TLSInfos(datamodule=datamodule)

        train_metrics = TrainAbstractMetricsDiscrete()
        extra_features = (
            ExtraFeatures(
                cfg.model.extra_features,
                cfg.model.rrwp_steps,
                dataset_info=dataset_infos,
            )
            if cfg.model.extra_features is not None
            else DummyExtraFeatures()
        )
        domain_features = DummyExtraFeatures()

        sampling_metrics = TLSSamplingMetrics(datamodule)
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    elif dataset_config["name"] == "my_tree":
        from datasets.my_tree_dataset import (
            MyTreeGraphDataModule,
            MyTreeDatasetInfos,
        )
        from analysis.visualization import NonMolecularVisualization
        from analysis.spectre_utils import TreeSamplingMetrics
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures

        datamodule = MyTreeGraphDataModule(cfg)
        sampling_metrics = TreeSamplingMetrics(datamodule)

        dataset_infos = MyTreeDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    elif dataset_config["name"] == "my_planar":
        from datasets.my_planar_dataset import (
            MyPlanarGraphDataModule,
            MyPlanarDatasetInfos,
        )
        from analysis.visualization import NonMolecularVisualization
        from analysis.spectre_utils import PlanarSamplingMetrics
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from models.extra_features import DummyExtraFeatures, ExtraFeatures

        datamodule = MyPlanarGraphDataModule(cfg)
        sampling_metrics = PlanarSamplingMetrics(datamodule)

        dataset_infos = MyPlanarDatasetInfos(datamodule, dataset_config)
        train_metrics = TrainAbstractMetricsDiscrete()
        visualization_tools = NonMolecularVisualization(dataset_name=cfg.dataset.name)

        extra_features = ExtraFeatures(
            cfg.model.extra_features,
            cfg.model.rrwp_steps,
            dataset_info=dataset_infos,
        )
        domain_features = DummyExtraFeatures()

        dataset_infos.compute_input_output_dims(
            datamodule=datamodule,
            extra_features=extra_features,
            domain_features=domain_features,
        )
    else:
        raise NotImplementedError("Unknown dataset {}".format(cfg["dataset"]))

    # Goal-directed/PMO training does not rely on reference_metrics;
    # empty datasets should not trigger real data statistics either.
    if not bool(getattr(cfg.dataset, "empty", False)):
        dataset_infos.compute_reference_metrics(
            datamodule=datamodule,
            sampling_metrics=sampling_metrics,
        )
    else:
        dataset_infos.ref_metrics = {"val": None, "test": None}

    model_kwargs = {
        "dataset_infos": dataset_infos,
        "train_metrics": train_metrics,
        "sampling_metrics": sampling_metrics,
        "visualization_tools": visualization_tools,
        "extra_features": extra_features,
        "domain_features": domain_features,
        "test_labels": (
            datamodule.test_labels
            if ("qm9" in cfg.dataset.name and cfg.general.conditional)
            else None
        ),
    }

    return datamodule, model_kwargs


def _initialize_distributed_environment_for_checkpoint_loading():
    """Initialise the distributed environment for loading distributed checkpoints.

    Checkpoints saved during distributed training contain process-group
    information and require an active distributed environment to load
    correctly, even when running on a single GPU.
    """
    if not torch.distributed.is_initialized():
        if "WORLD_SIZE" in os.environ:
            # Multi-GPU environment set up by torchrun / SLURM launcher
            try:
                torch.distributed.init_process_group(backend="nccl")
            except Exception:
                torch.distributed.init_process_group(backend="gloo")
        else:
            # Single-GPU run: create a minimal single-process group
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "12355"
            os.environ["RANK"] = "0"
            os.environ["WORLD_SIZE"] = "1"
            try:
                torch.distributed.init_process_group(backend="gloo", rank=0, world_size=1)
            except Exception:
                pass


def _load_pretrained_weights_into_grpo_module(grpo_module, pretrained_path: str):
    """Load pretrained model weights into the GRPO Lightning module.

    Handles the structural difference between a plain
    ``GraphDiscreteFlowModel`` checkpoint (keys like ``layer.weight``) and the
    wrapped ``GRPOLightningModule`` (keys like ``model.layer.weight``).

    Args:
        grpo_module: GRPO Lightning module instance.
        pretrained_path: Path to the pretrained checkpoint.
    """
    checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)

    # Support both:
    # 1) GraphDiscreteFlowModel checkpoint (keys like "model.*", "p0_node_dist", ...)
    # 2) Flow-GRPO/GRPO Lightning checkpoint (already prefixed with "model.*" at top-level)
    raw_state = checkpoint.get("state_dict", checkpoint)
    if not isinstance(raw_state, dict):
        raise ValueError("Invalid checkpoint format: missing state_dict dict")

    raw_keys = list(raw_state.keys())
    already_grpo_prefixed = any(
        k.startswith("model.model.")
        or k.startswith("model.p0_")
        or k.startswith("model.node_count_")
        for k in raw_keys
    )

    if already_grpo_prefixed:
        # Keys already match GRPOLightningModule; do NOT add another "model." prefix.
        remapped_state_dict = dict(raw_state)
        logger.info("Detected GRPO-compatible checkpoint keys (already contain 'model.' prefix), skipping remapping.")
    else:
        # Remap state_dict keys:
        # GraphDiscreteFlowModel -> GRPOLightningModule.model = GraphDiscreteFlowModel
        # Add a top-level 'model.' prefix.
        remapped_state_dict = {}
        for k, v in raw_state.items():
            new_key = f"model.{k}"
            remapped_state_dict[new_key] = v

    # Drop sampling-metric parameters tied to dataset statistics to avoid shape mismatches
    keys_to_drop = []
    for k in remapped_state_dict.keys():
        if k.startswith("model.sampling_metrics."):
            keys_to_drop.append(k)
    if keys_to_drop:
        logger.warning("Ignoring pretrained sampling-metric parameters (shapes may differ from current dataset):")
        for k in keys_to_drop:
            logger.warning("   - %s", k)
            remapped_state_dict.pop(k, None)

    grpo_module.load_state_dict(remapped_state_dict, strict=False)

    # Ensure all parameters require gradients for GRPO training
    for param in grpo_module.parameters():
        param.requires_grad = True
    logger.info("All parameters set to requires_grad=True")


def _run_flow_grpo_test_only(cfg: DictConfig):
    """Run test-only evaluation using GRPO sampling and standard metrics.

    Sampling uses ``sample_graphs_with_trajectory_tracking`` with independent
    random starting points (no same-start grouping). Evaluation calls
    ``GraphDiscreteFlowModel.evaluate_samples`` for consistency with
    ``main.py`` test_only mode.

    Args:
        cfg: Hydra configuration object.

    Returns:
        Dict of evaluation metrics.
    """
    import random

    from grpo.lightning_module import create_grpo_lightning_module
    from grpo.rewards import resolve_target_task

    random_seed = int(time.time()) % (2**31)
    pl.seed_everything(random_seed)
    logger.info("test_only using random seed: %d", random_seed)

    ckpt_path = cfg.general.get("test_only")
    if not ckpt_path:
        raise ValueError("general.test_only did not provide a ckpt path; cannot execute test_only.")
    ckpt_path = os.path.expanduser(ckpt_path)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    # Prepare data and model
    datamodule, model_kwargs = create_datamodule_and_model_components(cfg)
    try:
        datamodule.setup(stage="fit")
    except Exception:
        pass
    flow_grpo_module = create_grpo_lightning_module(
        cfg=cfg,
        model_kwargs=model_kwargs,
        datamodule=datamodule,
        total_steps=cfg.grpo.total_steps,
    )

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("state_dict", checkpoint)
    has_double_model_prefix = any(k.startswith("model.model.") for k in state_dict.keys())
    if has_double_model_prefix:
        # Flow-GRPO Lightning / compatible ckpt: keys are already aligned
        flow_grpo_module.load_state_dict(state_dict, strict=False)
        logger.info("Loaded Flow-GRPO/compatible checkpoint (with 'model.model.' prefix): %s", ckpt_path)
    else:
        # Treat as a raw flow pretrained model (GraphDiscreteFlowModel ckpt);
        # reuse the training-stage loading logic for consistency
        logger.info(
            "Detected GraphDiscreteFlowModel checkpoint (no 'model.model.' prefix), "
            "loading as pretrained flow model: %s", ckpt_path
        )
        _load_pretrained_weights_into_grpo_module(flow_grpo_module, ckpt_path)

    # [Ablation] Force static p0: revert p0 to the dataset-derived default,
    # undoing any dynamic p0 learned during RL training.
    if cfg.grpo.get("eval_force_static_p0", False):
        inner_model = flow_grpo_module.model  # GraphDiscreteFlowModel
        # Re-derive the static p0 from NoiseDistribution (dataset statistics)
        static_limit = inner_model.noise_dist.__class__(
            cfg.model.transition, inner_model.dataset_info
        ).get_limit_dist()
        static_node = static_limit.X.squeeze()
        static_edge = static_limit.E.squeeze()
        inner_model.update_limit_dist(static_node, static_edge)
        logger.info(
            "[Ablation] Forced static p0:\n  node=%s\n  edge=%s",
            static_node.tolist(), static_edge.tolist(),
        )

    # Move to device
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    device = torch.device("cuda") if use_gpu else torch.device("cpu")
    flow_grpo_module.to(device)
    flow_grpo_module.eval()

    # Manually initialise GRPO components (on_fit_start is normally called by Trainer)
    flow_grpo_module.on_fit_start()

    # Match the sample count from main.py test_only logic
    total_to_generate = (
        cfg.general.final_model_samples_to_generate * cfg.general.num_sample_fold
    )
    samples_left = total_to_generate
    batch_size = 2 * cfg.train.batch_size
    total_steps = flow_grpo_module._get_forward_steps()
    logger.info("test_only sampling steps: %d", total_steps)

    all_graphs = []

    # Batch sampling loop (mirrors main.py shard logic)
    while samples_left > 0:
        cur_bs = min(samples_left, batch_size)
        graphs, node_mask, *_ = flow_grpo_module.grpo_trainer.sample_graphs_with_trajectory_tracking(
            batch_size=cur_bs,
            seed=None,
            total_inference_steps=total_steps,
            force_same_start=False,
            group_size_for_same_start=None,
        )
        batch_graphs = flow_grpo_module.grpo_trainer._convert_placeholder_to_graph_list_cpu(
            graphs, node_mask, as_tensor=True
        )
        all_graphs.extend(batch_graphs)
        samples_left -= cur_bs

    graph_list = all_graphs[:total_to_generate]

    # Compute GRPO rewards (consistent with training stage)
    reward_type_str = str(cfg.grpo.reward_type).lower()
    is_goal_directed_reward = any(k in reward_type_str for k in ("guacamol", "tdc", "pmo"))
    reward_stats = {}
    if getattr(flow_grpo_module, "reward_function", None) is not None:
        try:
            rewards = flow_grpo_module.grpo_trainer._compute_rewards_multiprocess_sync(
                graph_list,
                timeout=getattr(flow_grpo_module.grpo_trainer, "eval_timeout_seconds", 1800),
                context="test_only",
            )
            if rewards.numel() > 0:
                rewards_cpu = rewards.detach().cpu()
                reward_stats = {
                    "grpo/reward_mean": float(rewards_cpu.mean().item()),
                    "grpo/reward_std": float(rewards_cpu.std().item()) if rewards_cpu.numel() > 1 else 0.0,
                    "grpo/reward_min": float(rewards_cpu.min().item()),
                    "grpo/reward_max": float(rewards_cpu.max().item()),
                }
                logger.info("Test-only GRPO reward statistics:")
                for k, v in reward_stats.items():
                    logger.info("   %s: %s", k, v)

                # Goal-directed (Guacamol/TDC/PMO) Top-K Analysis
                if is_goal_directed_reward:
                    import numpy as np
                    logger.info("Performing goal-directed Top-K analysis...")
                    from analysis.rdkit_functions import mol2smiles, build_molecule

                    # Convert all graphs to SMILES
                    smiles_list = []
                    atom_decoder = datamodule.dataset_infos.atom_decoder
                    for G in graph_list:
                        atom_types, edge_types = G
                        if isinstance(atom_types, torch.Tensor): atom_types = atom_types.cpu()
                        if isinstance(edge_types, torch.Tensor): edge_types = edge_types.cpu()

                        mol = build_molecule(atom_types, edge_types, atom_decoder)
                        smi = mol2smiles(mol)
                        smiles_list.append(smi)

                    # Pair with rewards
                    scored_mols = []
                    for s, r in zip(smiles_list, rewards_cpu.tolist()):
                        if s: # Filter out None/Invalid SMILES
                            scored_mols.append((s, r))

                    # Sort by reward descending
                    scored_mols.sort(key=lambda x: x[1], reverse=True)

                    # Compute Top-K Stats
                    top_k_stats = {}
                    if len(scored_mols) > 0:
                        top_k_stats["grpo/top1_score"] = scored_mols[0][1]
                        top_k_stats["grpo/top10_mean"] = np.mean([x[1] for x in scored_mols[:10]])
                        top_k_stats["grpo/top100_mean"] = np.mean([x[1] for x in scored_mols[:100]])

                        logger.info("   Top-1 Score: %s", top_k_stats["grpo/top1_score"])
                        logger.info("   Top-10 Mean: %s", top_k_stats["grpo/top10_mean"])
                        logger.info("   Top-100 Mean: %s", top_k_stats["grpo/top100_mean"])

                        reward_stats.update(top_k_stats)

                        # Save Best Molecules
                        task_for_filename = resolve_target_task(cfg, default=None)
                        if not task_for_filename:
                            tdc_tag = OmegaConf.select(cfg, "grpo.tdc_oracle") or OmegaConf.select(cfg, "grpo.tdc_oracles")
                            if tdc_tag is not None:
                                if isinstance(tdc_tag, (list, tuple)):
                                    task_for_filename = "_".join(str(x) for x in tdc_tag)
                                else:
                                    try:
                                        task_for_filename = "_".join(str(x) for x in list(tdc_tag))
                                    except Exception:
                                        task_for_filename = str(tdc_tag)
                        if not task_for_filename:
                            task_for_filename = "goal_directed"
                        for ch in [" ", "/", "\\", ":", ";", ","]:
                            task_for_filename = task_for_filename.replace(ch, "_")
                        task_for_filename = task_for_filename[:80]
                        best_mols_file = os.path.join(os.getcwd(), f"best_molecules_{task_for_filename}.txt")
                        with open(best_mols_file, "w") as f:
                            f.write(f"Rank\tScore\tSMILES\n")
                            for i, (s, r) in enumerate(scored_mols[:100]):
                                f.write(f"{i+1}\t{r:.4f}\t{s}\n")
                        logger.info("Top 100 molecules saved to: %s", best_mols_file)

            else:
                logger.warning("Test-only GRPO reward computation returned empty results")
        except Exception as e:
            logger.warning("Test-only GRPO reward computation failed: %s", e)
    else:
        logger.warning("GRPO reward function not initialised; skipping reward computation")

    # Evaluation: choose strategy based on task type
    to_log = {}
    is_guacamol_mpo = is_goal_directed_reward

    if is_guacamol_mpo:
        logger.info("Goal-directed mode: skipping full distribution metrics, computing VUN + Score only.")

        from analysis.rdkit_functions import mol2smiles, build_molecule

        valid_smiles = []
        atom_decoder = datamodule.dataset_infos.atom_decoder

        # Build molecules and check validity
        for G in graph_list:
            atom_types, edge_types = G
            if isinstance(atom_types, torch.Tensor): atom_types = atom_types.cpu()
            if isinstance(edge_types, torch.Tensor): edge_types = edge_types.cpu()

            mol = build_molecule(atom_types, edge_types, atom_decoder)
            smi = mol2smiles(mol)
            if smi:
                valid_smiles.append(smi)

        validity = len(valid_smiles) / len(graph_list) if len(graph_list) > 0 else 0
        unique_smiles = set(valid_smiles)
        uniqueness = len(unique_smiles) / len(valid_smiles) if len(valid_smiles) > 0 else 0
        novelty = 1.0 # MPO task typically doesn't track novelty against a training set

        to_log["Validity"] = validity
        to_log["Uniqueness"] = uniqueness
        to_log["Novelty"] = novelty

        logger.info("   Validity: %.4f", validity)
        logger.info("   Uniqueness: %.4f", uniqueness)
        logger.info("   Novelty: %.4f", novelty)

    else:
        # Standard distribution matching evaluation
        model = flow_grpo_module.model
        model.sampling_metrics.reset()
        to_log = model.evaluate_samples(
            samples=graph_list,
            labels=None,
            is_test=True,
        )

    # Merge reward statistics into the log for unified reporting
    if reward_stats:
        to_log.update(reward_stats)

    # Write results to disk (consistent with main.py test_only)
    filename = os.path.join(
        os.getcwd(),
        f"test_epoch{flow_grpo_module.current_epoch}_res_{cfg.sample.eta}_{cfg.sample.rdb}.txt",
    )
    with open(filename, "w") as file:
        for key, value in to_log.items():
            file.write(f"{key}: {value}\n")

    logger.info(
        "Test-only sampling and evaluation complete. Mode: %s",
        "Goal-Directed" if is_guacamol_mpo else "Distribution",
    )
    for k, v in to_log.items():
        logger.info("   %s: %s", k, v)

    return to_log
