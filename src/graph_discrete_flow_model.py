import time
import os
import logging

# Handle optional logging system import
try:
    import swanlab
except ImportError:
    swanlab = None

import numpy as np
import pickle
from tqdm import tqdm
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.cuda.amp import autocast
from torch.distributions.categorical import Categorical

from models.transformer_model import GraphTransformer

from metrics.train_metrics import TrainLossDiscrete
import utils
from flow_matching.noise_distribution import NoiseDistribution
from flow_matching.time_distorter import TimeDistorter
from flow_matching.rate_matrix_continuous import ContinuousRateMatrixDesigner
from flow_matching.utils import p_xt_g_x1
from flow_matching import flow_matching_utils
from datasets.dataset_utils import DistributionNodes

logger = logging.getLogger(__name__)


class GraphDiscreteFlowModel(pl.LightningModule):
    def __init__(
        self,
        cfg,
        dataset_infos,
        train_metrics,
        sampling_metrics,
        visualization_tools,
        extra_features,
        domain_features,
        test_labels=None,
    ):
        super().__init__()

        self.cfg = cfg
        self.name = f"{cfg.dataset.name}_{cfg.general.name}"
        self.model_dtype = torch.float32
        self.conditional = cfg.general.conditional
        self.test_labels = test_labels

        # Add device attribute (PyTorch Lightning manages devices automatically)
        self.register_buffer('_device_buffer', torch.zeros(1))

        # number of steps used for sampling
        self.sample_T = cfg.sample.sample_steps

        self.input_dims = dataset_infos.input_dims
        self.output_dims = dataset_infos.output_dims
        self.dataset_info = dataset_infos
        self.node_dist = dataset_infos.nodes_dist

        self.train_metrics = train_metrics
        self.sampling_metrics = sampling_metrics

        self.visualization_tools = visualization_tools
        self.extra_features = extra_features
        self.domain_features = domain_features

        self.noise_dist = NoiseDistribution(cfg.model.transition, dataset_infos)
        self.limit_dist = self.noise_dist.get_limit_dist()

        # add virtual class when absorbing state refers to a new class
        self.noise_dist.update_input_output_dims(self.input_dims)
        self.noise_dist.update_dataset_infos(self.dataset_info)

        # [Persistence] Register buffers for p0 distribution to save them in state_dict
        # Initialize with the default distribution
        self.register_buffer('p0_node_dist', self.limit_dist.X.clone().detach())
        self.register_buffer('p0_edge_dist', self.limit_dist.E.clone().detach())

        # [Persistence] Node-count distribution (learnable online during GRPO).
        # Store the categorical probabilities and an optional Top-K buffer for resumable updates.
        try:
            node_count_prob_pre = torch.as_tensor(self.node_dist.prob, dtype=torch.float32).detach()
        except Exception:
            node_count_prob_pre = torch.ones(1, dtype=torch.float32)

        # Scale node_count_prob to node_count_max if requested to allow exploration beyond training set
        grpo_cfg = getattr(cfg, "grpo", {})
        nc_max = int(grpo_cfg.get("node_count_max", len(node_count_prob_pre) - 1) or (len(node_count_prob_pre) - 1))
        nc_min = int(grpo_cfg.get("node_count_min", 0) or 0)

        # Ensure we have at least up to nc_max
        target_size = max(len(node_count_prob_pre), nc_max + 1)
        node_count_prob = torch.zeros(target_size, dtype=torch.float32)
        node_count_prob[:len(node_count_prob_pre)] = node_count_prob_pre

        # Check for initial uniform distribution request
        if grpo_cfg.get("initial_node_dist", None) == "uniform":
            logger.info("[GRPO] Initializing node distribution uniformly between %d and %d", nc_min, nc_max)
            node_count_prob.zero_()
            # nc_min and nc_max bounds are inclusive
            m_min = max(0, nc_min)
            m_max = min(target_size - 1, nc_max)
            node_count_prob[m_min : m_max + 1] = 1.0
            if node_count_prob.sum() > 0:
                node_count_prob /= node_count_prob.sum()
            else:
                node_count_prob.fill_(1.0 / target_size)

        self.register_buffer("node_count_prob", node_count_prob.clone())
        # Re-sync DistributionNodes to the (potentially resized) initial state
        self.update_node_count_dist(node_count_prob)

        try:
            node_count_buf_size = int(getattr(cfg, "grpo", {}).get("dynamic_node_dist_buffer_size", 1000))
        except Exception:
            node_count_buf_size = 1000
        node_count_buf_size = max(1, int(node_count_buf_size))
        self.register_buffer("node_count_buffer_rewards", torch.full((node_count_buf_size,), -1e9, dtype=torch.float32))
        self.register_buffer("node_count_buffer_nodes", torch.zeros((node_count_buf_size,), dtype=torch.long))
        self.register_buffer("node_count_buffer_filled", torch.zeros((1,), dtype=torch.long))

        self.train_loss = TrainLossDiscrete(
            self.cfg.model.lambda_train,
        )

        self.model = GraphTransformer(
            n_layers=cfg.model.n_layers,
            input_dims=self.input_dims,
            hidden_mlp_dims=cfg.model.hidden_mlp_dims,
            hidden_dims=cfg.model.hidden_dims,
            output_dims=self.output_dims,
            act_fn_in=nn.ReLU(),
            act_fn_out=nn.ReLU(),
        )

        self.save_hyperparameters(
            ignore=[
                "train_metrics",
                "sampling_metrics",
            ],
        )

        # logging
        self.start_epoch_time = None
        self.train_iterations = None
        self.val_iterations = None
        self.log_every_steps = cfg.general.log_every_steps
        self.number_chain_steps = cfg.general.number_chain_steps
        self.val_counter = 0
        self.adapt_counter = 0

        # time distortor for both training and sampling steps
        self.time_distorter = TimeDistorter(
            train_distortion=cfg.train.time_distortion,
            sample_distortion=cfg.sample.time_distortion,
            alpha=1,
            beta=1,
        )

        # rate matrix designer
        # using ContinuousRateMatrixDesigner for differentiable training as requested
        self.rate_matrix_designer = ContinuousRateMatrixDesigner(
            limit_dist=self.limit_dist,
        )

    @property
    def device(self):
        """Return the device where the model resides."""
        return self._device_buffer.device

    def get_rate_matrix_designer(self):
        return self.rate_matrix_designer

    # [Persistence] Hook to sync p0 from buffers after loading checkpoint
    def on_load_checkpoint(self, checkpoint: dict) -> None:
        """
        Called by Lightning when loading a checkpoint.
        We use this to restore the limit_dist (p0) from the saved buffers.
        """
        # Note: buffers 'p0_node_dist' and 'p0_edge_dist' are automatically loaded by Lightning/PyTorch
        # into self.p0_node_dist and self.p0_edge_dist BEFORE this hook is called.

        if hasattr(self, 'p0_node_dist') and hasattr(self, 'p0_edge_dist'):
            logger.info("[Checkpoint] Restoring p0 distribution from checkpoint...")
            # Update internal components with the loaded buffer values
            self.update_limit_dist(self.p0_node_dist, self.p0_edge_dist)
        else:
            logger.warning("[Checkpoint] No p0 buffers found in checkpoint, using default initialization.")

        if hasattr(self, "node_count_prob"):
            try:
                self.update_node_count_dist(self.node_count_prob)
            except Exception as e:
                logger.warning("[Checkpoint] Failed to restore node_count_prob: %s", e)

    def update_limit_dist(self, node_dist=None, edge_dist=None):
        """
        Update the model's limit distribution (p0) based on new data.
        Args:
            node_dist: tensor of shape (x_num_classes,) or (1, x_num_classes)
            edge_dist: tensor of shape (e_num_classes,) or (1, e_num_classes)
        """
        # 1. Update NoiseDistribution
        self.noise_dist.update_limit_dist(node_dist, edge_dist)

        # 2. Sync local limit_dist
        self.limit_dist = self.noise_dist.get_limit_dist()

        # 3. Update RateMatrixDesigners
        self.rate_matrix_designer.update_limit_dist(self.limit_dist)

        # 4. [Persistence] Update buffers so they are saved in checkpoint
        if node_dist is not None:
            if hasattr(self, 'p0_node_dist'):
                # Ensure same device and dtype
                self.p0_node_dist.copy_(self.limit_dist.X.squeeze().to(self.p0_node_dist.device))

        if edge_dist is not None:
             if hasattr(self, 'p0_edge_dist'):
                self.p0_edge_dist.copy_(self.limit_dist.E.squeeze().to(self.p0_edge_dist.device))

    def update_node_count_dist(self, node_count_prob: torch.Tensor) -> None:
        """
        Update the node-count categorical distribution used by `self.node_dist.sample_n`.

        This is separate from p0 (atom/bond types). It controls how many nodes are sampled.
        """
        if node_count_prob is None:
            return
        p = torch.as_tensor(node_count_prob, dtype=torch.float32).detach()
        if p.dim() != 1 or p.numel() == 0:
            raise ValueError(f"update_node_count_dist expects 1D prob, got shape={tuple(p.shape)}")

        # Normalize (avoid zeros-only).
        s = float(p.sum().item())
        if s <= 0:
            p = torch.ones_like(p)
            s = float(p.sum().item())
        p = p / s

        # Persist and Sync shape.
        # If the model has a larger buffer (e.g. from cfg.grpo.node_count_max), keep its size.
        if hasattr(self, "node_count_prob"):
            target_sz = self.node_count_prob.numel()
            if p.numel() == target_sz:
                self.node_count_prob.copy_(p.to(self.node_count_prob.device))
            else:
                # Shape mismatch (likely from checkpoint or expanded config).
                # We prioritize the current buffer's capacity if it's larger.
                new_p = torch.zeros(target_sz, dtype=torch.float32, device=self.node_count_prob.device)
                copy_sz = min(target_sz, p.numel())
                new_p[:copy_sz] = p[:copy_sz].to(self.node_count_prob.device)
                # Renormalize if necessary
                if new_p.sum() > 0:
                    new_p /= new_p.sum()
                else:
                    new_p.fill_(1.0 / target_sz)
                self.node_count_prob.copy_(new_p)
                p = new_p # Use the padded one for rebuilding DistributionNodes
        else:
            self.register_buffer("node_count_prob", p.clone())

        # Rebuild sampling distribution object on CPU.
        try:
            p_cpu = p.detach().cpu()
            if isinstance(self.node_dist, DistributionNodes):
                self.node_dist.update_prob(p_cpu)
            else:
                self.node_dist = DistributionNodes(p_cpu)
        except Exception as e:
            raise RuntimeError(f"Failed to update node_dist DistributionNodes: {e}") from e

    def training_step(self, data, i):
        if data.edge_index.numel() == 0:
            logger.warning("Found a batch with no edges. Skipping.")
            return

        if self.conditional:
            if torch.rand(1) < 0.1:
                data.y = torch.ones_like(data.y, device=self.device) * -1

        dense_data, node_mask = utils.to_dense(
            data.x,
            data.edge_index,
            data.edge_attr,
            data.batch,
        )

        dense_data = dense_data.mask(node_mask)
        X, E = dense_data.X, dense_data.E
        noisy_data = self.apply_noise(X, E, data.y, node_mask)
        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)

        loss = self.train_loss(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            pred_y=pred.y,
            true_X=X,
            true_E=E,
            true_y=data.y,
            log=i % self.log_every_steps == 0,
        )

        self.train_metrics(
            masked_pred_X=pred.X,
            masked_pred_E=pred.E,
            true_X=X,
            true_E=E,
            log=i % self.log_every_steps == 0,
        )

        return {"loss": loss}

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(),
            lr=self.cfg.train.lr,
            amsgrad=True,
            weight_decay=self.cfg.train.weight_decay,
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Handle checkpoints from different sources:
        1. GRPO checkpoints may have a double 'model.' prefix that needs remapping.
        2. During OOD testing, override sampling_metrics stats with current dataset stats.
        """

        def _override_sampling_stats(target_state_dict):
            if getattr(self.cfg.general, "test_only", None):
                sampling_state = self.sampling_metrics.state_dict()
                for key, tensor in sampling_state.items():
                    target_state_dict[f"sampling_metrics.{key}"] = tensor.detach().cpu()
            return target_state_dict

        # Base DeFoG checkpoints predate the GRPO persistence buffers; backfill any
        # missing one from the freshly-initialized model value so a pretrained base
        # checkpoint still loads under strict=True.
        _grpo_persistence_buffers = (
            "p0_node_dist",
            "p0_edge_dist",
            "node_count_prob",
            "node_count_buffer_rewards",
            "node_count_buffer_nodes",
            "node_count_buffer_filled",
        )

        def _backfill_grpo_buffers(target_state_dict):
            for name in _grpo_persistence_buffers:
                if name not in target_state_dict and hasattr(self, name):
                    target_state_dict[name] = getattr(self, name).detach().cpu().clone()
                    logger.warning(
                        "Checkpoint missing GRPO buffer '%s', using freshly-initialized default",
                        name,
                    )
            return target_state_dict

        has_double_prefix = any(k.startswith("model.model.") for k in state_dict.keys())

        if has_double_prefix:
            logger.info("Detected GRPO checkpoint, remapping keys...")
            remapped_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("model.model."):
                    new_key = k[6:]  # 'model.model.xxx' -> 'model.xxx'
                    remapped_state_dict[new_key] = v
                elif k == "model._device_buffer":
                    remapped_state_dict["_device_buffer"] = v
                elif k.startswith("model.") and not k.startswith("model.model."):
                    new_key = k[6:]  # 'model.xxx' -> 'xxx'
                    remapped_state_dict[new_key] = v
                else:
                    remapped_state_dict[k] = v

            logger.info(
                "Key remapping complete. Example: %s -> %s",
                list(state_dict.keys())[0],
                list(remapped_state_dict.keys())[0],
            )

            if "_device_buffer" not in remapped_state_dict:
                remapped_state_dict["_device_buffer"] = torch.zeros_like(
                    self._device_buffer.cpu()
                )
                logger.warning("Checkpoint missing _device_buffer, added default value")

            remapped_state_dict = _override_sampling_stats(remapped_state_dict)
            remapped_state_dict = _backfill_grpo_buffers(remapped_state_dict)
            return super().load_state_dict(remapped_state_dict, strict=strict)

        # Normal checkpoint; override sampling_metrics stats if needed
        state_dict = dict(state_dict)
        state_dict = _override_sampling_stats(state_dict)
        if "_device_buffer" not in state_dict:
            state_dict["_device_buffer"] = torch.zeros_like(self._device_buffer.cpu())
            logger.warning("Checkpoint missing _device_buffer, added default value")
        state_dict = _backfill_grpo_buffers(state_dict)
        return super().load_state_dict(state_dict, strict=strict)

    def on_fit_start(self) -> None:
        self.train_iterations = len(self.trainer.datamodule.train_dataloader())
        logger.info(
            "Size of the input features: X=%s, E=%s, y=%s",
            self.input_dims["X"],
            self.input_dims["E"],
            self.input_dims["y"],
        )
        if self.local_rank == 0:
            if swanlab is not None:
                utils.setup_swanlab(self.cfg)

    def on_train_epoch_start(self) -> None:
        logger.info("Starting train epoch...")
        self.start_epoch_time = time.time()
        self.train_loss.reset()
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        to_log = self.train_loss.log_epoch_metrics()
        logger.info(
            "Epoch %d: X_CE: %.3f -- E_CE: %.3f -- y_CE: %.3f -- %.1fs",
            self.current_epoch,
            to_log['train_epoch/x_CE'],
            to_log['train_epoch/E_CE'],
            to_log['train_epoch/y_CE'],
            time.time() - self.start_epoch_time,
        )
        epoch_at_metrics, epoch_bond_metrics = self.train_metrics.log_epoch_metrics()
        logger.info(
            "Epoch %d: %s -- %s",
            self.current_epoch, epoch_at_metrics, epoch_bond_metrics,
        )
        try:
            if swanlab is not None and swanlab.run:
                swanlab.log({"epoch": self.current_epoch})
        except Exception:
            pass

    def on_validation_epoch_start(self) -> None:
        logger.info("Starting validation...")
        self.sampling_metrics.reset()

    def validation_step(self, data, i):
        return

    def on_validation_epoch_end(self) -> None:
        self.val_counter += 1
        if self.val_counter % self.cfg.general.sample_every_val == 0:
            logger.info("Starting to sample")
            samples, labels = self.sample(
                is_test=False, save_samples=False, save_visualization=True
            )
            to_log = self.evaluate_samples(
                samples=samples, labels=labels, is_test=False
            )

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"val_epoch{self.current_epoch}_res.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

        logger.info("Finished validation.")

    def on_test_epoch_start(self) -> None:
        logger.info("Starting test...")
        self.sampling_metrics.reset()
        if self.local_rank == 0:
            if swanlab is not None:
                utils.setup_swanlab(self.cfg)

    def test_step(self, data, i):
        return

    def on_test_epoch_end(self) -> None:

        if self.cfg.sample.search:
            logger.info("Starting sampling optimization...")
            self.search_hyperparameters()
        else:
            logger.info("Starting to sample")
            samples, labels = self.sample(
                is_test=True,
                save_samples=self.cfg.general.save_samples,
                save_visualization=True,
            )
            to_log = self.evaluate_samples(samples=samples, labels=labels, is_test=True)

            # Store results
            filename = os.path.join(
                os.getcwd(),
                f"test_epoch{self.current_epoch}_res.txt",
            )
            with open(filename, "w") as file:
                for key, value in to_log.items():
                    file.write(f"{key}: {value}\n")

            # Additional evaluation: compute molecular reward stats on generated samples during test_only
            enable_reward_eval = getattr(self.cfg.general, "enable_reward_eval", False)
            if enable_reward_eval:
                try:
                    if "guacamol" in self.cfg.dataset.name:
                        from grpo_rewards import MolecularValidityReward

                        reward_func = MolecularValidityReward(
                            atom_decoder=getattr(self.dataset_info, "atom_decoder", None),
                            device=torch.device("cpu"),
                        )
                        # samples is a list of [atom_types, edge_types]; pass directly to reward function
                        rewards = reward_func(samples)
                        if rewards.numel() > 0:
                            mean_r = rewards.mean().item()
                            std_r = rewards.std().item()
                            min_r = rewards.min().item()
                            max_r = rewards.max().item()
                            logger.info(
                                "[Test Molecular Reward] mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                                mean_r, std_r, min_r, max_r,
                            )
                except Exception as e:
                    logger.error("[Test Molecular Reward] evaluation failed: %s", e)

            logger.info("Finished testing.")

    def sample(self, is_test, save_samples, save_visualization):

        # Load generated samples if they exist
        if self.cfg.general.generated_path:
            logger.info("Loading generated samples...")
            with open(self.cfg.general.generated_path, "rb") as f:
                samples = pickle.load(f)
            # Set labels to None
            labels = [None] * len(samples)
            return samples, None

        # Otherwise, generate new samples
        if is_test:
            samples_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_generate = (
                self.cfg.general.final_model_samples_to_generate
                * self.cfg.general.num_sample_fold
            )
            samples_left_to_save = self.cfg.general.final_model_samples_to_save
            chains_left_to_save = self.cfg.general.final_model_chains_to_save

        else:
            samples_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_generate = self.cfg.general.samples_to_generate
            samples_left_to_save = self.cfg.general.samples_to_save
            chains_left_to_save = self.cfg.general.chains_to_save

        samples = []
        labels = []
        graph_id = 0
        while samples_left_to_generate > 0:
            logger.info(
                "Samples left to generate: %d/%d",
                samples_left_to_generate,
                samples_to_generate,
            )
            bs = 2 * self.cfg.train.batch_size
            to_generate = min(samples_left_to_generate, bs)
            to_save = min(samples_left_to_save, bs)
            chains_save = min(chains_left_to_save, bs)
            num_chain_steps = min(self.number_chain_steps, self.sample_T)
            cur_samples, cur_labels = self.sample_batch(
                graph_id,
                to_generate,
                num_nodes=None,
                save_final=to_save,
                keep_chain=chains_save,
                number_chain_steps=num_chain_steps,
                save_visualization=save_visualization,
            )
            samples.extend(cur_samples)
            labels.extend(cur_labels)

            graph_id += to_generate
            samples_left_to_save -= to_save
            samples_left_to_generate -= to_generate
            chains_left_to_save -= chains_save

        if save_samples:
            logger.info("Saving the generated graphs")

            # saving in txt version
            filename = "graphs.txt"
            with open(filename, "w") as f:
                for item in samples:
                    f.write(f"N={item[0].shape[0]}\n")
                    atoms = item[0].tolist()
                    f.write("X: \n")
                    for at in atoms:
                        f.write(f"{at} ")
                    f.write("\n")
                    f.write("E: \n")
                    for bond_list in item[1]:
                        for bond in bond_list:
                            f.write(f"{bond} ")
                        f.write("\n")
                    f.write("\n")

            # saving in pkl version
            with open(f"generated_samples_rank{self.local_rank}.pkl", "wb") as f:
                pickle.dump(samples, f)

            logger.info("Generated graphs saved.")

        return samples, labels

    def evaluate_samples(
        self,
        samples,
        labels,
        is_test,
        save_filename="",
    ):
        logger.info("Computing sampling metrics...")

        to_log = {}
        samples_to_evaluate = self.cfg.general.final_model_samples_to_generate
        if is_test:
            for i in range(self.cfg.general.num_sample_fold):
                cur_samples = samples[
                    i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                ]
                if labels is not None:
                    cur_labels = labels[
                        i * samples_to_evaluate : (i + 1) * samples_to_evaluate
                    ]
                else:
                    cur_labels = None

                cur_to_log = self.sampling_metrics.forward(
                    cur_samples,
                    ref_metrics=self.dataset_info.ref_metrics,
                    name=f"self.name_{i}",
                    current_epoch=self.current_epoch,
                    val_counter=-1,
                    test=is_test,
                    local_rank=self.local_rank,
                    labels=cur_labels if self.conditional and cur_labels is not None else None,
                )

                if i == 0:
                    to_log = {i: [cur_to_log[i]] for i in cur_to_log}
                else:
                    to_log = {i: to_log[i] + [cur_to_log[i]] for i in cur_to_log}

                filename = os.path.join(
                    os.getcwd(),
                    f"epoch{self.current_epoch}_res_fold{i}_{save_filename}.txt",
                )
                with open(filename, "w") as file:
                    for key, value in cur_to_log.items():
                        file.write(f"{key}: {value}\n")

            to_log = {
                i: (np.array(to_log[i]).mean(), np.array(to_log[i]).std())
                for i in to_log
            }
        else:
            to_log = self.sampling_metrics.forward(
                samples,
                ref_metrics=self.dataset_info.ref_metrics,
                name=self.cfg.general.name,
                current_epoch=self.current_epoch,
                val_counter=-1,
                test=is_test,
                local_rank=self.local_rank,
                labels=labels if self.conditional else None,
            )

        return to_log

    def apply_noise(self, X, E, y, node_mask, t=None):
        """Sample noise and apply it to the data."""

        # Sample a timestep t.
        bs = X.size(0)
        if t is None:
            t_float = self.time_distorter.train_ft(bs, self.device)
        else:
            t_float = t

        # sample random step
        X_1_label = torch.argmax(X, dim=-1)
        E_1_label = torch.argmax(E, dim=-1)
        prob_X_t, prob_E_t = p_xt_g_x1(
            X1=X_1_label, E1=E_1_label, t=t_float, limit_dist=self.limit_dist
        )

        # step 4 - sample noised data
        sampled_t = flow_matching_utils.sample_discrete_features(
            probX=prob_X_t, probE=prob_E_t, node_mask=node_mask
        )
        noise_dims = self.noise_dist.get_noise_dims()
        X_t = F.one_hot(sampled_t.X, num_classes=noise_dims["X"])
        E_t = F.one_hot(sampled_t.E, num_classes=noise_dims["E"])

        # step 5 - create the PlaceHolder
        z_t = utils.PlaceHolder(X=X_t, E=E_t, y=y).type_as(X_t).mask(node_mask)

        noisy_data = {
            "t": t_float,
            "X_t": z_t.X,
            "E_t": z_t.E,
            "y_t": z_t.y,
            "node_mask": node_mask,
        }

        return noisy_data

    def forward(self, noisy_data, extra_data, node_mask):
        X = torch.cat((noisy_data["X_t"], extra_data.X), dim=2).float()
        E = torch.cat((noisy_data["E_t"], extra_data.E), dim=3).float()
        y = torch.hstack((noisy_data["y_t"], extra_data.y)).float()
        return self.model(X, E, y, node_mask)

    @torch.no_grad()
    def sample_batch(
        self,
        batch_id: int,
        batch_size: int,
        keep_chain: int,
        number_chain_steps: int,
        save_final: int,
        num_nodes=None,
        save_visualization: bool = True,
    ):
        """
        :param batch_id: int
        :param batch_size: int
        :param num_nodes: int, <int>tensor (batch_size) (optional) for specifying number of nodes
        :param save_final: int: number of predictions to save to file
        :param keep_chain: int: number of chains to save to file
        :param keep_chain_steps: number of timesteps to save for each chain
        :return: molecule_list. Each element of this list is a tuple (atom_types, charges, positions)
        """
        if num_nodes is None:
            n_nodes = self.node_dist.sample_n(batch_size, self.device)
        elif type(num_nodes) == int:
            n_nodes = num_nodes * torch.ones(
                batch_size, device=self.device, dtype=torch.int
            )
        else:
            assert isinstance(num_nodes, torch.Tensor)
            n_nodes = num_nodes
        n_max = torch.max(n_nodes).item()

        # Build the masks
        arange = (
            torch.arange(n_max, device=self.device).unsqueeze(0).expand(batch_size, -1)
        )
        node_mask = arange < n_nodes.unsqueeze(1)

        # Sample noise  -- z has size (n_samples, n_nodes, n_features)
        z_T = flow_matching_utils.sample_discrete_feature_noise(
            limit_dist=self.noise_dist.get_limit_dist(), node_mask=node_mask
        )
        if self.conditional:
            if "qm9" in self.cfg.dataset.name:
                y = self.test_labels
                perm = torch.randperm(y.size(0))
                idx = perm[:100]
                condition = y[idx]
                condition = condition.to(self.device)
                z_T.y = condition.repeat([10, 1])[:batch_size, :]
            elif "tls" in self.cfg.dataset.name:
                z_T.y = torch.zeros(batch_size, 1).to(self.device)
                z_T.y[: batch_size // 2] = 1
            else:
                raise NotImplementedError
        X, E, y = z_T.X, z_T.E, z_T.y

        # Init chain storing variables
        assert (E == torch.transpose(E, 1, 2)).all()
        chain_X_size = torch.Size((number_chain_steps + 1, keep_chain, X.size(1)))
        chain_E_size = torch.Size(
            (number_chain_steps + 1, keep_chain, E.size(1), E.size(2))
        )
        chain_X = torch.zeros(chain_X_size)
        chain_E = torch.zeros(chain_E_size)
        chain_times = torch.zeros((number_chain_steps + 1, keep_chain))
        chain_time_unit = 1 / number_chain_steps

        # Store initial graph
        if keep_chain > 0:
            sampled_initial = z_T.mask(node_mask, collapse=True)
            chain_X[0] = sampled_initial.X[:keep_chain]
            chain_E[0] = sampled_initial.E[:keep_chain]
            chain_times[0] = torch.zeros((keep_chain))

        for t_int in tqdm(range(0, self.cfg.sample.sample_steps)):
            # this state
            t_array = t_int * torch.ones((batch_size, 1)).type_as(y)
            t_norm = t_array / (self.cfg.sample.sample_steps + 1)
            if ("absorb" in self.cfg.model.transition) and (t_int == 0):
                # to avoid failure mode of absorbing transition, add epsilon
                t_norm = t_norm + 1e-6
            # next state
            s_array = t_array + 1
            s_norm = s_array / (self.cfg.sample.sample_steps + 1)

            # using round for precision
            write_index = int(np.ceil(np.round(s_norm[0].item() / chain_time_unit, 6)))

            # Distort time
            t_norm = self.time_distorter.sample_ft(
                t_norm, self.cfg.sample.time_distortion
            )
            s_norm = self.time_distorter.sample_ft(
                s_norm, self.cfg.sample.time_distortion
            )

            # Sample z_s
            sampled_s, discrete_sampled_s = self.sample_p_zs_given_zt(
                t_norm,
                s_norm,
                X,
                E,
                y,
                node_mask,
            )

            X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

            # Save the first keep_chain graphs
            chain_X[write_index] = discrete_sampled_s.X[:keep_chain]
            chain_E[write_index] = discrete_sampled_s.E[:keep_chain]
            chain_times[write_index] = s_norm.flatten()[:keep_chain]

        # Sample
        sampled_s = sampled_s.mask(node_mask, collapse=True)
        X, E, y = sampled_s.X, sampled_s.E, sampled_s.y

        # Prepare the chain for saving
        if keep_chain > 0:

            # Repeat last frame 10x to see final sample better
            chain_X = torch.cat([chain_X, chain_X[-1:].repeat(10, 1, 1)], dim=0)
            chain_E = torch.cat([chain_E, chain_E[-1:].repeat(10, 1, 1, 1)], dim=0)
            chain_times = torch.cat(
                [chain_times, chain_times[-1:].repeat(10, 1)], dim=0
            )
            assert chain_X.size(0) == (number_chain_steps + 1 + 10)

        X, E, y = self.noise_dist.ignore_virtual_classes(X, E, y)
        chain_X, chain_E, _ = self.noise_dist.ignore_virtual_classes(
            chain_X, chain_E, y
        )

        # Save generated graphs
        molecule_list = []
        label_list = []
        for i in range(batch_size):
            n = n_nodes[i]
            atom_types = X[i, :n].cpu()
            edge_types = E[i, :n, :n].cpu()
            molecule_list.append([atom_types, edge_types])
            label_list.append(y[i].cpu())

        if self.visualization_tools is not None and save_visualization:
            # Visualize chains
            logger.info("Visualizing chains...")
            current_path = os.getcwd()
            num_molecules = chain_X.size(1)  # number of molecules
            for i in range(num_molecules):
                result_path = os.path.join(
                    current_path,
                    f"chains/{self.cfg.general.name}/"
                    f"epoch{self.current_epoch}/"
                    f"chains/molecule_{batch_id + i}",
                )
                if not os.path.exists(result_path):
                    os.makedirs(result_path)
                    _ = self.visualization_tools.visualize_chain(
                        result_path,
                        chain_X[:, i, :].numpy(),
                        chain_E[:, i, :].numpy(),
                        chain_times[:, i].numpy(),
                    )
                logger.info(
                    "%d/%d complete", i + 1, num_molecules
                )
            logger.info("Visualizing graphs...")

            # Visualize the final molecules
            current_path = os.getcwd()
            result_path = os.path.join(
                current_path,
                f"graphs/{self.cfg.general.name}/epoch{self.current_epoch}_b{batch_id}/",
            )
            self.visualization_tools.visualize(result_path, molecule_list, save_final)
            logger.info("Done.")

        return molecule_list, label_list

    def compute_step_probs(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        """
        Original (pretrained) version of step_probs computation.
        Supports dt as scalar, (B,) or (B,1) shape; automatically broadcasts to node/edge dimensions.
        """
        # Normalize dt shape for broadcasting
        if isinstance(dt, (int, float)):
            dt_X = torch.tensor(dt, device=R_t_X.device, dtype=R_t_X.dtype)
            dt_E = dt_X
        else:
            if dt.dim() == 0:
                dt_X = dt
                dt_E = dt
            elif dt.dim() == 1:
                dt_X = dt.view(-1, 1, 1)          # (B,1,1)
                dt_E = dt.view(-1, 1, 1, 1)       # (B,1,1,1)
            elif dt.dim() == 2 and dt.shape[1] == 1:
                dt_X = dt.view(-1, 1, 1)          # (B,1,1)
                dt_E = dt.view(-1, 1, 1, 1)       # (B,1,1,1)
            else:
                dt_X = dt
                dt_E = dt

        step_probs_X = R_t_X * dt_X  # (B, N, S)
        step_probs_E = R_t_E * dt_E  # (B, N, N, S)

        # Calculate the on-diagonal step probabilities
        # 1) Zero out the diagonal entries
        step_probs_X.scatter_(-1, X_t.argmax(-1)[:, :, None], 0.0)
        step_probs_E.scatter_(-1, E_t.argmax(-1)[:, :, :, None], 0.0)

        # 2) Calculate the diagonal entries such that the probability row sums to 1
        step_probs_X.scatter_(
            -1,
            X_t.argmax(-1)[:, :, None],
            (1.0 - step_probs_X.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )
        step_probs_E.scatter_(
            -1,
            E_t.argmax(-1)[:, :, :, None],
            (1.0 - step_probs_E.sum(dim=-1, keepdim=True)).clamp(min=0.0),
        )

        # step 2 - merge to the original formulation
        prob_X = step_probs_X.clone()
        prob_E = step_probs_E.clone()

        return prob_X, prob_E

    def compute_step_probs_grpo(self, R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e):
        """
        GRPO-specific step probability computation that preserves gradient flow.
        Uses differentiable operations to replace in-place scatter_ operations.

        Implements the same logic as compute_step_probs:
        1. Compute off-diagonal probabilities: R_t * dt
        2. Set diagonal elements (current state) to 0
        3. Diagonal elements = 1 - sum(off-diagonal), ensuring rows sum to 1
        """
        # Handle dt dimensions - ensure correct broadcasting
        # dt may be scalar, (B,) or (B,1)
        if isinstance(dt, (int, float)):
            dt = torch.tensor(dt, device=R_t_X.device, dtype=R_t_X.dtype)

        # Ensure dt has correct dimensions for broadcasting
        if dt.dim() == 0:  # scalar
            dt_X = dt
            dt_E = dt
        elif dt.dim() == 1:  # (B,)
            dt_X = dt.view(-1, 1, 1)  # (B, 1, 1) for X
            dt_E = dt.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for E
        elif dt.dim() == 2 and dt.shape[1] == 1:  # (B, 1)
            dt_X = dt.view(-1, 1, 1)  # (B, 1, 1) for X
            dt_E = dt.view(-1, 1, 1, 1)  # (B, 1, 1, 1) for E
        else:
            # Default: assume dt already has the correct shape
            dt_X = dt if dt.dim() == 3 else dt.unsqueeze(-1).unsqueeze(-1)
            dt_E = dt if dt.dim() == 4 else dt.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # Step 1: Compute initial step probabilities
        step_probs_X = R_t_X * dt_X  # (B, N, S)
        step_probs_E = R_t_E * dt_E  # (B, N, N, S)

        # Step 2: Create mask to identify current state (diagonal element positions)
        # X_t and E_t are one-hot encoded; argmax yields the current state index
        X_indices = X_t.argmax(-1)  # (B, N)
        E_indices = E_t.argmax(-1)  # (B, N, N)

        # Step 3: For X - use mask to set diagonal elements to 0 (avoiding in-place operations)
        # Create diagonal element mask using one_hot (avoiding scatter_)
        bs, n, dx = step_probs_X.shape
        X_diag_mask = F.one_hot(X_indices, num_classes=dx)  # (B, N, S)

        # Set diagonal elements to 0 using mask instead of in-place ops
        # Use 1-mask instead of ~mask.float() to maintain gradient continuity
        step_probs_X_zeros = step_probs_X * (1.0 - X_diag_mask)

        # Compute off-diagonal sum per row
        off_diag_sum_X = step_probs_X_zeros.sum(dim=-1, keepdim=True)

        # Compute diagonal element values (1 - off_diagonal_sum)
        diag_values_X = (1.0 - off_diag_sum_X).clamp(min=0.0)

        # Build final probability matrix (off-diagonal + diagonal)
        prob_X = step_probs_X_zeros + diag_values_X * X_diag_mask

        # Step 4: For E - same treatment
        bs, n1, n2, de = step_probs_E.shape
        E_diag_mask = F.one_hot(E_indices, num_classes=de)  # (B, N, N, S)

        # Set diagonal elements to 0
        step_probs_E_zeros = step_probs_E * (1.0 - E_diag_mask)

        # Compute off-diagonal sum per row
        off_diag_sum_E = step_probs_E_zeros.sum(dim=-1, keepdim=True)

        # Compute diagonal element values
        diag_values_E = (1.0 - off_diag_sum_E).clamp(min=0.0)

        # Build final probability matrix
        prob_E = step_probs_E_zeros + diag_values_E * E_diag_mask

        return prob_X, prob_E

    def sample_p_zs_given_zt(
        self,
        t,
        s,
        X_t,
        E_t,
        y_t,
        node_mask,
    ):
        """Samples from zs ~ p(zs | zt). Only used during sampling."""
        bs, n, dx = X_t.shape
        _, _, _, de = E_t.shape
        dt = (s - t)[0]

        # Neural net predictions
        noisy_data = {
            "X_t": X_t,
            "E_t": E_t,
            "y_t": y_t,
            "t": t,
            "node_mask": node_mask,
        }

        extra_data = self.compute_extra_data(noisy_data)
        pred = self.forward(noisy_data, extra_data, node_mask)
        # Normalize predictions
        pred_X = F.softmax(pred.X, dim=-1)  # bs, n, d0
        pred_E = F.softmax(pred.E, dim=-1)  # bs, n, n, d0
        limit_x = self.limit_dist.X
        limit_e = self.limit_dist.E

        G_1_pred = pred_X, pred_E
        G_t = X_t, E_t

        # Use the appropriate rate matrix designer based on configuration
        rate_designer = self.get_rate_matrix_designer()
        R_t_X, R_t_E = rate_designer.compute_graph_rate_matrix(
            t,
            node_mask,
            G_t,
            G_1_pred,
        )

        prob_X, prob_E = self.compute_step_probs(
            R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
        )

        sampled_s = flow_matching_utils.sample_discrete_features(
            prob_X, prob_E, node_mask=node_mask
        )

        X_s = F.one_hot(sampled_s.X, num_classes=len(limit_x)).float()
        E_s = F.one_hot(sampled_s.E, num_classes=len(limit_e)).float()

        assert (E_s == torch.transpose(E_s, 1, 2)).all()
        assert (X_t.shape == X_s.shape) and (E_t.shape == E_s.shape)

        if self.conditional:
            y_to_save = y_t
        else:
            y_to_save = torch.zeros([y_t.shape[0], 0], device=self.device)

        out_one_hot = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)
        out_discrete = utils.PlaceHolder(X=X_s, E=E_s, y=y_to_save)

        out_one_hot = out_one_hot.mask(node_mask).type_as(y_t)
        out_discrete = out_discrete.mask(node_mask, collapse=True).type_as(y_t)

        return out_one_hot, out_discrete

    def compute_extra_data(self, noisy_data):
        """At every training step (after adding noise) and step in sampling, compute extra information and append to
        the network input."""

        # Under autocast (bf16/fp16), graph structural features are prone to precision issues;
        # use FP32 to ensure numerical stability.
        with autocast(enabled=False):
            extra_features = self.extra_features(noisy_data)

        # one additional category is added for the absorbing transition
        X, E, y = self.noise_dist.ignore_virtual_classes(
            noisy_data["X_t"], noisy_data["E_t"], noisy_data["y_t"]
        )
        noisy_data_to_mol_feat = noisy_data.copy()
        noisy_data_to_mol_feat["X_t"] = X
        noisy_data_to_mol_feat["E_t"] = E
        noisy_data_to_mol_feat["y_t"] = y
        with autocast(enabled=False):
            extra_molecular_features = self.domain_features(noisy_data_to_mol_feat)

        extra_X = torch.cat((extra_features.X, extra_molecular_features.X), dim=-1)
        extra_E = torch.cat((extra_features.E, extra_molecular_features.E), dim=-1)
        extra_y = torch.cat((extra_features.y, extra_molecular_features.y), dim=-1)

        t = noisy_data["t"]
        extra_y = torch.cat((extra_y, t), dim=1)

        return utils.PlaceHolder(X=extra_X, E=extra_E, y=extra_y)

    def search_hyperparameters(self):
        """
        Grid search for sampling hypeparameters.
        The num_step_list is tunable based on requirements.
        """

        num_step_list = [5, 10, 50, 100, 1000]
        if self.cfg.dataset.name in "qm9":
            num_step_list = [5, 10]
        if self.cfg.dataset.name == "guacamol":  # accelerate
            num_step_list = [50]
        if self.cfg.dataset.name == "moses":  # accelerate
            num_step_list = [50]

        if self.cfg.sample.search == "all":
            results_df = self.search_distortion(num_step_list)
            results_df = self.search_stochasticity(num_step_list)
            results_df = self.search_target_guidance(num_step_list)
        elif self.cfg.sample.search == "distortion":
            results_df = self.search_distortion(num_step_list)
        elif self.cfg.sample.search == "stochasticity":
            results_df = self.search_stochasticity(num_step_list)
        elif self.cfg.sample.search == "target_guidance":
            results_df = self.search_target_guidance(num_step_list)
        else:
            raise NotImplementedError(
                f"Search type {self.cfg.sample.search} not implemented."
            )

        logger.info("Finished searching. Results saved to search_hyperparameters.csv")

    def search_distortion(self, num_step_list):
        """
        Grid search for sampling distortion.
        """
        results_df = pd.DataFrame()
        distortion_list = ["identity", "polydec", "cos", "revcos", "polyinc"]

        for num_step in num_step_list:
            for distortor in distortion_list:
                self.cfg.sample.sample_steps = num_step
                self.cfg.sample.time_distortion = distortor
                logger.info(
                    "############# Testing num steps: %d, distortor: %s #############",
                    num_step, distortor,
                )
                samples, labels = self.sample(
                    is_test=True,
                    save_samples=self.cfg.general.save_samples,
                    save_visualization=False,
                )
                res = self.evaluate_samples(
                    samples=samples, labels=labels, is_test=True
                )
                mean_res = {f"{key}_mean": res[key][0] for key in res}
                std_res = {f"{key}_std": res[key][1] for key in res}
                mean_res.update(std_res)
                res_df = pd.DataFrame([mean_res])
                res_df["num_step"] = num_step
                res_df["distortor"] = distortor
                results_df = pd.concat([results_df, res_df], ignore_index=True)
                # save at each step as well
                results_df.to_csv(f"search_distortion.csv")

        # set back to default value
        self.cfg.sample.time_distortion = "identity"

        # save the final results
        results_df.reset_index(inplace=True)
        results_df.set_index(["num_step", "distortor"], inplace=True)
        results_df.to_csv(f"search_distortion.csv")

    def search_stochasticity(self, num_step_list):
        """
        Grid search for stochasticity level eta.
        DEPRECATED: RateMatrixDesigner is replaced by ContinuousRateMatrixDesigner which does not use eta.
        """
        logger.info("search_stochasticity is deprecated as eta is no longer used.")
        return pd.DataFrame()

    def search_target_guidance(self, num_step_list):
        """
        Grid search for target guidance omega.
        DEPRECATED: RateMatrixDesigner is replaced by ContinuousRateMatrixDesigner which does not use omega.
        """
        logger.info("search_target_guidance is deprecated as omega is no longer used.")
        return pd.DataFrame()
