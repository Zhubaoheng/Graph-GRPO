"""Flow-GRPO training entry point.

Uses Hydra for configuration management. Heavy helper functions live in
``grpo.train_utils``; this file contains only the pipeline orchestration
and the ``@hydra.main`` entry point.
"""

import logging
import os
import sys
import time

# Ensure both src/ (for grpo.*, utils, etc.) and repo root (for src.datasets.*)
# are on sys.path regardless of how the script is invoked.
_src_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_src_dir)
for _p in (_src_dir, _repo_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from datetime import datetime

import hydra
import numpy as np
import pytorch_lightning as pl
import scipy
import torch
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LearningRateMonitor, Callback

from utils import (
    ensure_legacy_aliases,
    get_logger,
    patch_torch_load_weights_only_default,
    setup_logging,
    suppress_noisy_warnings,
)

suppress_noisy_warnings()
ensure_legacy_aliases()

if not hasattr(scipy, 'errstate'):
    scipy.errstate = np.errstate

os.environ.setdefault("PL_DISABLE_WANDB", "1")
torch.set_float32_matmul_precision('medium')

try:
    import swanlab
except ImportError:
    swanlab = None

from grpo.lightning_module import (
    GRPOLightningModule,
    FlowGRPODataModule,
    create_grpo_lightning_module,
)
from graph_discrete_flow_model import GraphDiscreteFlowModel
from grpo.rewards import resolve_target_task
from grpo.train_utils import (
    create_datamodule_and_model_components,
    _should_strict_resume,
    _validate_resume_checkpoint,
    _initialize_distributed_environment_for_checkpoint_loading,
    _load_pretrained_weights_into_grpo_module,
    _run_flow_grpo_test_only,
)

logger = get_logger(__name__)


def run_flow_grpo_training_pipeline(cfg: DictConfig):
    """Run the full Flow-GRPO training pipeline.

    Implements a two-phase training architecture: sampling phase followed by
    a policy-gradient training phase.

    Args:
        cfg: Hydra configuration object containing model, data, and training
            parameters.

    Returns:
        The trained Flow-GRPO Lightning module.
    """
    # Restore torch.load default to weights_only=False (project checkpoints are trusted)
    patch_torch_load_weights_only_default()

    logger.info("Dataset: %s", cfg.dataset.name)
    logger.info("Reward function: %s", cfg.grpo.reward_type)

    # Initialise logging backend
    swanlab_mode = cfg.general.get('swanlab', 'disabled')
    if swanlab is not None and swanlab_mode != 'disabled':
        try:
            config_dict = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
            swanlab.init(
                project=f"Flow-GRPO-{cfg.dataset.name}",
                experiment_name=cfg.general.name,
                config=config_dict,
                mode=swanlab_mode,
            )
        except Exception as e:
            logger.warning("SwanLab initialization failed: %s", e)

    # Set random seed
    pl.seed_everything(cfg.train.seed)

    # GPU setup
    use_gpu = cfg.general.gpus > 0 and torch.cuda.is_available()
    if use_gpu:
        available_gpus = min(cfg.general.gpus, torch.cuda.device_count())
        for i in range(available_gpus):
            logger.info("   GPU %d: %s", i, torch.cuda.get_device_name(i))
    else:
        logger.info("Using CPU")
        available_gpus = 0

    # 1. Create datamodule and model components
    try:
        datamodule, model_kwargs = create_datamodule_and_model_components(cfg)
    except Exception as e:
        logger.error("Failed to create datamodule and model components: %s", e)
        raise e

    # Test-only shortcut
    if cfg.general.get("test_only"):
        logger.info("Detected general.test_only, entering test-only mode (GRPO sampling + evaluation)")
        return _run_flow_grpo_test_only(cfg)

    # 2. Create Flow-GRPO Lightning module
    try:
        flow_grpo_module = create_grpo_lightning_module(
            cfg=cfg,
            model_kwargs=model_kwargs,
            datamodule=datamodule,
            total_steps=cfg.grpo.total_steps,
        )
    except Exception as e:
        logger.error("Failed to create Flow-GRPO Lightning module: %s", e)
        raise e

    # 3. Create Flow-GRPO datamodule (dummy data)
    flow_grpo_datamodule = FlowGRPODataModule(
        num_epochs=cfg.grpo.total_steps,
        batch_size=1
    )

    # 4. Create callbacks
    callbacks = []

    # 5. Create Lightning Trainer
    trainer_kwargs = {
        'max_steps': cfg.grpo.total_steps,
        'max_epochs': -1,  # Controlled by max_steps
        'accumulate_grad_batches': 1,  # Manual optimisation handles accumulation

        'check_val_every_n_epoch': None,
        'val_check_interval': None,
        'num_sanity_val_steps': 0,

        'log_every_n_steps': 10,
        'enable_progress_bar': True,
        'enable_model_summary': True,

        'callbacks': callbacks,

        'fast_dev_run': cfg.general.name == "debug",

        'deterministic': False,  # GRPO requires stochasticity
        'benchmark': True,
        'logger': False,  # Disable logger to avoid LearningRateMonitor errors
    }

    # GPU configuration
    if use_gpu:
        trainer_kwargs.update({
            'accelerator': "gpu",
            'devices': available_gpus,
            'precision': cfg.get('mixed_precision', '32'),
            'strategy': 'ddp' if available_gpus > 1 else 'auto'
        })
    else:
        trainer_kwargs.update({
            'accelerator': "cpu",
            'devices': 1,
        })

    trainer = Trainer(**trainer_kwargs)

    # 6. Start training
    try:
        # Check whether to resume from a checkpoint
        ckpt_path = cfg.grpo.get('resume_from_checkpoint')
        if ckpt_path and os.path.exists(ckpt_path):
            if _should_strict_resume(cfg):
                _validate_resume_checkpoint(flow_grpo_module, ckpt_path, cfg)
            logger.info("Resuming training from checkpoint: %s", ckpt_path)
            logger.info("   Restoring full training state (optimizer, lr schedule, step count, etc.)")
            trainer.fit(
                model=flow_grpo_module,
                datamodule=flow_grpo_datamodule,
                ckpt_path=ckpt_path
            )
        else:
            # Initialise distributed environment (needed for loading distributed checkpoints)
            _initialize_distributed_environment_for_checkpoint_loading()

            # Load pretrained model if provided
            pretrained_path = cfg.grpo.get('pretrained_checkpoint')
            if pretrained_path and os.path.exists(pretrained_path):
                logger.info("Loading pretrained model: %s", pretrained_path)
                _load_pretrained_weights_into_grpo_module(flow_grpo_module, pretrained_path)

            trainer.fit(
                model=flow_grpo_module,
                datamodule=flow_grpo_datamodule
            )

        logger.info("Flow-GRPO training complete.")

        # Save final model to the Hydra output directory
        save_dir = HydraConfig.get().runtime.output_dir
        current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_checkpoint_path = os.path.join(save_dir, f"final_model_{current_time_str}.ckpt")
        trainer.save_checkpoint(final_checkpoint_path)
        logger.info("Final model saved to: %s", final_checkpoint_path)

    except Exception as e:
        logger.error("Flow-GRPO training failed: %s", e, exc_info=True)
        raise e

    finally:
        # Clean up resources
        if use_gpu:
            torch.cuda.empty_cache()
        try:
            if swanlab is not None and swanlab.run is not None:
                swanlab.finish()
        except Exception:
            pass

    return flow_grpo_module


# Use the shared baseline config; select dataset/experiment or grpo variant via overrides
# e.g. `python src/train_flow_grpo.py --config-name config experiment=planar grpo=grpo_planar`
@hydra.main(version_base="1.3", config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Flow-GRPO training entry point."""
    try:
        model = run_flow_grpo_training_pipeline(cfg)

    except Exception as e:
        logger.error("Flow-GRPO execution failed: %s", e)
        raise e


if __name__ == "__main__":
    main()
