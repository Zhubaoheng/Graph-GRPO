"""Entry point for pretraining the discrete flow model (without GRPO).

For GRPO fine-tuning, use ``train_flow_grpo.py`` instead.
"""
import logging
import os
import pathlib
import sys
import time

# Ensure both src/ and repo root are on sys.path
_src_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_src_dir)
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import (
    ensure_legacy_aliases,
    get_logger,
    patch_torch_load_weights_only_default,
    setup_logging,
    suppress_noisy_warnings,
)
from src import utils
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
from graph_discrete_flow_model import GraphDiscreteFlowModel
from models.extra_features import DummyExtraFeatures, ExtraFeatures

suppress_noisy_warnings()
ensure_legacy_aliases()

os.environ.setdefault("PL_DISABLE_WANDB", "1")
torch.cuda.empty_cache()

logger = get_logger(__name__)


def _align_sampling_metrics_checkpoint(ckpt_path, sampling_metrics):
    """
    Ensure that the checkpoint stores the same sampling metric statistics as the
    currently instantiated metrics object. This is needed when we evaluate an
    OOD split whose target distributions differ from the ones used during
    training.
    """
    if not ckpt_path or sampling_metrics is None:
        return ckpt_path

    if not os.path.isfile(ckpt_path):
        logger.warning("Checkpoint %s not found. Skipping alignment.", ckpt_path)
        return ckpt_path

    try:
        checkpoint = torch.load(ckpt_path, map_location="cpu")
    except Exception as exc:
        logger.warning("Failed to load checkpoint %s: %s", ckpt_path, exc)
        return ckpt_path

    state_dict = checkpoint.get("state_dict")
    if state_dict is None:
        logger.warning("state_dict missing in checkpoint %s.", ckpt_path)
        return ckpt_path

    needs_save = False
    sampling_state = sampling_metrics.state_dict()
    for key, tensor in sampling_state.items():
        full_key = f"sampling_metrics.{key}"
        new_tensor = tensor.detach().cpu()
        old_tensor = state_dict.get(full_key)
        if (
            old_tensor is None
            or old_tensor.shape != new_tensor.shape
            or not torch.equal(old_tensor, new_tensor)
        ):
            state_dict[full_key] = new_tensor
            needs_save = True

    if needs_save:
        torch.save(checkpoint, ckpt_path)
        logger.info(
            "Updated sampling metric distributions stored in %s "
            "to match the current dataset statistics.",
            ckpt_path,
        )

    return ckpt_path


@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    patch_torch_load_weights_only_default()
    # Training / testing flow
    # Seed setup
    if not cfg.general.test_only:
        # Use fixed seed during training
        train_seed = getattr(cfg, 'train', None)
        if train_seed is not None and hasattr(train_seed, 'seed'):
            pl.seed_everything(cfg.train.seed)
        else:
            # Fall back to default seed
            pl.seed_everything(42)
    else:
        # Use random seed during testing for varied results
        random_seed = int(time.time()) % (2**31)
        pl.seed_everything(random_seed)
        logger.info("Using random seed for testing: %d", random_seed)

    dataset_config = cfg["dataset"]

    if dataset_config["name"] in [
        "sbm",
        "comm20",
        "planar",
        "tree",
    ]:
        import graph_tool  # noqa: F401  (needed for spectre_utils at import time)

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

    elif dataset_config["name"] in ["qm9", "guacamol", "moses", "zinc"]:
        from metrics.molecular_metrics import (
            TrainMolecularMetrics,
            SamplingMolecularMetrics,
        )
        from metrics.molecular_metrics_discrete import TrainMolecularMetricsDiscrete
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
        elif dataset_config["name"] == "guacamol":
            # Select Guacamol split implementation based on config
            split = getattr(cfg.dataset, "split", "ood")
            if split == "ood":
                from src.datasets import guacamol_dataset_ood as guacamol_dataset
            else:
                from src.datasets import guacamol_dataset as guacamol_dataset

            datamodule = guacamol_dataset.GuacamolDataModule(cfg)
            dataset_infos = guacamol_dataset.Guacamolinfos(datamodule, cfg)
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

            datamodule = zinc_dataset.ZINCDataModule(cfg)
            dataset_infos = zinc_dataset.ZINCinfos(datamodule=datamodule, cfg=cfg)
            dataset_smiles = zinc_dataset.get_smiles(
                cfg=cfg,
                datamodule=datamodule,
                dataset_infos=dataset_infos,
                evaluate_datasets=False,
            )
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

        # We do not evaluate novelty during training
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
        import graph_tool  # noqa: F401  (needed for spectre_utils at import time)

        from datasets.my_tree_dataset import (
            MyTreeGraphDataModule,
            MyTreeDatasetInfos,
        )
        from analysis.visualization import NonMolecularVisualization
        from analysis.spectre_utils import (
            TreeSamplingMetrics,
        )

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
        import graph_tool  # noqa: F401  (needed for spectre_utils at import time)

        from datasets.my_planar_dataset import (
            MyPlanarGraphDataModule,
            MyPlanarDatasetInfos,
        )
        from analysis.visualization import NonMolecularVisualization
        from analysis.spectre_utils import (
            PlanarSamplingMetrics,
        )

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

    dataset_infos.compute_reference_metrics(
        datamodule=datamodule,
        sampling_metrics=sampling_metrics,
    )

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

    utils.create_folders(cfg)

    # Normal training/testing
    model = GraphDiscreteFlowModel(cfg=cfg, **model_kwargs)

    callbacks = []
    if cfg.train.save_model:
        checkpoint_callback = ModelCheckpoint(
            dirpath=f"checkpoints/{cfg.general.name}",
            filename="{epoch}",
            save_top_k=-1,
            every_n_epochs=cfg.general.sample_every_val
            * cfg.general.check_val_every_n_epochs,
        )
        callbacks.append(checkpoint_callback)


    name = cfg.general.name
    if name == "debug":
        logger.warning("Run is called 'debug' -- it will run with fast_dev_run.")

    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    trainer = Trainer(
        gradient_clip_val=cfg.train.clip_grad,
        strategy="ddp_find_unused_parameters_true",  # Needed to load old checkpoints
        accelerator="gpu" if use_gpu else "cpu",
        devices=cfg.general.gpus if use_gpu else 1,
        max_epochs=cfg.train.n_epochs,
        check_val_every_n_epoch=cfg.general.check_val_every_n_epochs,
        fast_dev_run=name == "debug",
        enable_progress_bar=False,
        callbacks=callbacks,
        log_every_n_steps=50 if name != "debug" else 1,
        logger=[],
    )

    if not cfg.general.test_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.general.resume)
    else:
        # Start by evaluating test_only_path
        first_ckpt = _align_sampling_metrics_checkpoint(
            cfg.general.test_only, sampling_metrics
        )
        trainer.test(model, datamodule=datamodule, ckpt_path=first_ckpt)
        if cfg.general.evaluate_all_checkpoints:
            directory = pathlib.Path(cfg.general.test_only).parents[0]
            logger.info("Checkpoint directory: %s", directory)
            files_list = os.listdir(directory)
            for file in files_list:
                if ".ckpt" in file:
                    ckpt_path = os.path.join(directory, file)
                    if ckpt_path == cfg.general.test_only:
                        continue
                    logger.info("Loading checkpoint %s", ckpt_path)
                    aligned_ckpt = _align_sampling_metrics_checkpoint(
                        ckpt_path, sampling_metrics
                    )
                    trainer.test(
                        model, datamodule=datamodule, ckpt_path=aligned_ckpt
                    )


if __name__ == "__main__":
    main()
