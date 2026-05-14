"""GRPO core algorithm implementation - Flow-GRPO style refactored version.

Integrates the core design patterns of Flow-GRPO.
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import time
from grpo.trajectory_data import TrajectoryData

logger = logging.getLogger(__name__)


class PerGraphStatTracker:
    """Per-graph-configuration statistics tracker.

    Tracks reward history for different graph configurations (e.g., different
    node counts) and maintains independent reward statistics for each
    configuration. Computes configuration-specific advantages for more stable
    training.

    Graphs with different node counts have different generation difficulty, so
    independent statistics prevent interference between easy and hard graphs.
    """

    def __init__(self):
        self.stats: Dict[int, Dict[str, torch.Tensor]] = {}
        self.history_configs = set()
        self.config_to_idx: Dict[str, int] = {}
        self.device: Optional[torch.device] = None

    def update(self, graph_configs: List[str], rewards: Union[torch.Tensor, np.ndarray]) -> torch.Tensor:
        """Update reward statistics and compute intra-group normalized advantages.

        Args:
            graph_configs: List of graph configuration identifiers, e.g.,
                "nodes_10" for graphs with 10 nodes.
            rewards: Reward array or tensor, corresponding one-to-one with
                graph_configs.

        Returns:
            Intra-group normalized advantage values for PPO training.
        """
        if isinstance(rewards, torch.Tensor):
            rewards_tensor = rewards.detach()
        else:
            rewards_tensor = torch.tensor(rewards, dtype=torch.float32)

        if self.device is None:
            self.device = rewards_tensor.device
        rewards_tensor = rewards_tensor.to(self.device)

        # Intra-group normalization: compute mean/std per graph_config
        num_samples = len(graph_configs)
        advantages = torch.zeros_like(rewards_tensor, device=self.device)

        # Group indices by configuration
        from collections import defaultdict
        cfg_to_indices = defaultdict(list)
        for idx, cfg in enumerate(graph_configs):
            cfg_to_indices[cfg].append(idx)

        for cfg, indices in cfg_to_indices.items():
            idx_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
            cfg_rewards = rewards_tensor[idx_tensor]

            # Compute mean and std for this config in the current batch
            cfg_mean = cfg_rewards.mean()
            cfg_std = cfg_rewards.std(unbiased=False)

            if cfg_std < 1e-4:
                # Std near zero means samples are nearly identical; set advantage to 0
                cfg_adv = torch.zeros_like(cfg_rewards)
            else:
                cfg_adv = (cfg_rewards - cfg_mean) / (cfg_std + 1e-4)

            advantages[idx_tensor] = cfg_adv

            # Maintain statistics for summary (count only)
            cfg_id = self._ensure_config(cfg)
            self.stats[cfg_id]["count"] += float(len(indices))

        return advantages

    def _ensure_config(self, config: str) -> int:
        if config in self.config_to_idx:
            cfg_id = self.config_to_idx[config]
        else:
            cfg_id = len(self.config_to_idx)
            self.config_to_idx[config] = cfg_id
            device = self.device if self.device is not None else torch.device("cpu")
            self.stats[cfg_id] = {
                "count": torch.zeros(1, device=device),
                "mean": torch.zeros(1, device=device),
                "m2": torch.zeros(1, device=device),
            }
            self.history_configs.add(hash(config))
        return cfg_id

    def get_statistics_summary(self) -> Tuple[float, int]:
        """Return statistics summary: average group size and historical config count."""
        if not self.stats:
            return 0.0, 0
        total_counts = sum(stat["count"].item() for stat in self.stats.values())
        avg_group_size = total_counts / len(self.stats)
        history_configs = len(self.history_configs)
        return avg_group_size, history_configs

    def clear_statistics(self):
        """Clear all statistics, typically called at the start of a new epoch."""
        self.stats = {}
        self.config_to_idx = {}
        self.history_configs = set()
        self.device = None


class GRPOCore:
    """GRPO (Group Relative Policy Optimization) core algorithm implementation.

    Features:
        1. Two-phase training architecture: batch sampling followed by
           centralized training.
        2. PPO-based policy optimization with gradient clipping.
        3. Per-config statistics tracking for training stability.
        4. KL divergence regularization to prevent excessive policy drift.
    """

    def __init__(self, cfg: Dict):
        self.cfg = cfg
        grpo_config = cfg.get('grpo', cfg)

        # Core hyperparameters (PPO-style clip_range + optional asymmetric lower/upper)
        self.clip_range = grpo_config.get('clip_ratio', 0.2)
        self.clip_ratio_lower = grpo_config.get('clip_ratio_lower', self.clip_range)
        self.clip_ratio_upper = grpo_config.get('clip_ratio_upper', self.clip_range)
        self.beta = grpo_config.get('kl_penalty', 0.01)
        self.entropy_coef = grpo_config.get('entropy_coef', 0.0)
        self.timestep_fraction = grpo_config.get('timestep_fraction', 1.0)
        self.use_grpo_step_probs_for_sampling = grpo_config.get('use_grpo_step_probs_for_sampling', True)
        # Weight coefficient for positive-advantage samples (>1.0 strengthens positive samples, =1.0 no weighting)
        self.positive_advantage_weight = grpo_config.get('positive_advantage_weight', 1.0)
        # GDCR (distribution alignment + diversity) regularization coefficient
        self.gdcr_coef = grpo_config.get('gdcr_coef', 0.0)
        self.diversity_coef = grpo_config.get('diversity_coef', 0.0)
        # Extra weight for edge distribution (for long-tailed edge distributions)
        self.edge_dist_factor = grpo_config.get('edge_dist_factor', 1.0)

        # Flow-GRPO specific parameters
        self.num_inner_epochs = grpo_config.get('num_inner_epochs', 1)
        self.per_config_stat_tracking = grpo_config.get('per_config_stat_tracking', True)

        # RL mode: 'grpo' (default, PPO-style with analytic transition) or 'reinforce' (episode-level REINFORCE)
        self.rl_mode = grpo_config.get('rl_mode', 'grpo')

        # Statistics tracker
        if self.per_config_stat_tracking:
            self.stat_tracker = PerGraphStatTracker()
        else:
            self.stat_tracker = None

        # Sample buffer (key component of Flow-GRPO)
        self.sample_buffer: Optional[TrajectoryData] = None
        self._sample_batches: List[TrajectoryData] = []

    def collect_trajectory_samples(
        self,
        trajectories: Optional[List[List]],
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        graph_configs: List[str],
        node_masks: torch.Tensor,
        trajectory_tensors: Optional[Dict[str, torch.Tensor]] = None,
        dense_rewards: Optional[torch.Tensor] = None,
    ):
        """Collect generated trajectory data into the buffer for subsequent training.

        This is the first phase of GRPO two-phase training: the sampling phase.
        Collected data will be used in the second phase (training phase).

        Args:
            trajectories: List of generation trajectories, each containing a
                complete state sequence.
            rewards: Reward tensor evaluating generation quality.
            old_log_probs: Log probabilities from the sampling policy, used for
                importance sampling.
            graph_configs: Graph configuration identifiers for grouped statistics.
            node_masks: Node masks indicating valid nodes.
            trajectory_tensors: Optional precomputed trajectory tensors.
            dense_rewards: Optional dense rewards of shape [B, T]. If provided,
                overrides standard advantage calculation.
        """
        cpu_device = torch.device("cpu")
        node_masks_cpu = node_masks.detach().to(cpu_device)
        if trajectory_tensors is not None:
            vectorized = {
                key: tensor.detach().to(cpu_device)
                for key, tensor in trajectory_tensors.items()
            }
        else:
            if trajectories is None:
                raise ValueError("Either raw trajectories or precomputed tensors must be provided")
            vectorized = self._vectorize_trajectories(trajectories, node_masks_cpu)

        tensor_data = {
            "rewards": rewards.detach().to(cpu_device),
            "old_log_probs": old_log_probs.detach().to(cpu_device),
            "node_masks": node_masks_cpu,
            **vectorized,
        }

        if dense_rewards is not None:
            tensor_data["dense_rewards"] = dense_rewards.detach().to(cpu_device)

        list_data = {
            "graph_configs": list(graph_configs),
        }

        batch = TrajectoryData(tensor_data=tensor_data, list_data=list_data)
        self._sample_batches.append(batch)
        self.sample_buffer = None  # Deferred merge; concatenate once during prepare phase

    def prepare_training_batch(self) -> Optional[TrajectoryData]:
        """Combine every sampled slice into a single CPU batch and whiten advantages."""
        if not self._sample_batches:
            return None

        if self.sample_buffer is None:
            # Align all sample batches on node-related dimensions (global padding)
            # before concatenation. Only affects the training buffer, not the
            # sampling-phase graphs or node_mask.
            self._normalize_sample_batches_node_dims()
            self.sample_buffer = TrajectoryData.concatenate(self._sample_batches)

        if self.sample_buffer is None or self.sample_buffer.is_empty():
            return None

        tensor_data = dict(self.sample_buffer.tensor_data)
        list_data = dict(self.sample_buffer.list_data)

        all_rewards = tensor_data["rewards"]
        all_old_log_probs = tensor_data["old_log_probs"]
        all_node_masks = tensor_data["node_masks"]
        all_configs = list_data.get("graph_configs", [])

        num_timesteps = all_old_log_probs.shape[1]
        dense_rewards = tensor_data.get("dense_rewards")

        # Compute advantages

        # 1. Compute global advantage A^{final} via intra-group normalization
        if self.stat_tracker and self.per_config_stat_tracking:
            # stat_tracker.update performs group normalization on [B] rewards
            adv_final = self.stat_tracker.update(all_configs, all_rewards)
            adv_final = adv_final.to(all_rewards.device, dtype=all_rewards.dtype)
        else:
            # Fallback simple normalization
            std_t = all_rewards.std()
            if std_t < 1e-6:
                adv_final = torch.zeros_like(all_rewards)
            else:
                adv_final = (all_rewards - all_rewards.mean()) / (std_t + 1e-4)

        # Expand to [B, T]
        adv_final_expanded = adv_final.unsqueeze(1).expand(-1, num_timesteps)

        # 2. Compute dense advantage A^{dense} (if available)
        if dense_rewards is not None:
            # A_{i,t}^{dense} using "Next Step Value":
            # V_{next}[:, t] = R_{dense}[:, t+1] for t < T-1
            # V_{next}[:, T-1] = R_{final}

            V_next = torch.zeros_like(dense_rewards)
            if num_timesteps > 1:
                V_next[:, :-1] = dense_rewards[:, 1:]

            # Use final rewards for the last step
            # all_rewards is [B], V_next is [B, T]
            V_next[:, -1] = all_rewards

            # Group normalization per time step
            adv_dense = torch.zeros_like(V_next)

            from collections import defaultdict
            cfg_to_indices = defaultdict(list)
            for idx, cfg in enumerate(all_configs):
                cfg_to_indices[cfg].append(idx)

            for cfg, indices in cfg_to_indices.items():
                idx_tensor = torch.tensor(indices, device=V_next.device)
                # group_vals: [GroupSize, T]
                group_vals = V_next[idx_tensor]

                # Mean/Std along batch dimension (dim=0)
                g_mean = group_vals.mean(dim=0, keepdim=True)
                g_std = group_vals.std(dim=0, keepdim=True, unbiased=False)

                # Normalize; handle low-variance cases to avoid amplifying noise
                denom = g_std + 1e-4
                raw_adv = (group_vals - g_mean) / denom

                # Increase threshold to 1e-4 to avoid amplifying trivial numerical noise
                group_adv = torch.where(g_std < 1e-4, torch.zeros_like(raw_adv), raw_adv)

                adv_dense[idx_tensor] = group_adv

            # 3. Dynamic beta weighting: linear increase from 0.1 to 0.5
            beta_min = 0.1
            beta_max = 0.5
            # shape [1, T]
            t_range = torch.arange(num_timesteps, device=adv_dense.device, dtype=torch.float32)
            if num_timesteps > 1:
                beta_t = beta_min + (beta_max - beta_min) * (t_range / (num_timesteps - 1))
            else:
                beta_t = torch.tensor([beta_max], device=adv_dense.device)

            beta_t = beta_t.unsqueeze(0)  # [1, T]

            advantages = torch.max(adv_final_expanded, adv_dense)

        else:
            # Fallback to standard GRPO if no dense rewards
            advantages = adv_final_expanded

        # Clip advantage values to prevent gradient explosion from outliers
        adv_clip_max = self.cfg.grpo.get('adv_clip_max', 5.0)
        advantages = torch.clamp(advantages, -adv_clip_max, adv_clip_max)

        tensor_data["advantages"] = advantages

        return TrajectoryData(tensor_data=tensor_data, list_data=list_data)

    # ------------------------------------------------------------------
    def _normalize_sample_batches_node_dims(self) -> None:
        """Align node-related dimensions across all TrajectoryData in the sample buffer.

        Only operates on self._sample_batches (training buffer); does not affect
        sampling-phase graphs or reward computation.

        Handles the following tensors:
            - node_masks:        [B, N]
            - trajectory_X:      [B, T, N, Dx]
            - trajectory_E:      [B, T, N, N, De]
        """
        if not self._sample_batches:
            return

        # 1. Find the global maximum node count across all batches
        max_nodes = 0
        for batch in self._sample_batches:
            node_masks = batch.tensor_data.get("node_masks")
            if node_masks is None or node_masks.numel() == 0:
                continue
            max_nodes = max(max_nodes, int(node_masks.shape[1]))

        if max_nodes == 0:
            return

        # 2. Pad all batches to the same max_nodes
        for batch in self._sample_batches:
            tensor_data = batch.tensor_data

            node_masks = tensor_data.get("node_masks")
            if node_masks is None or node_masks.shape[1] == max_nodes:
                continue

            cur_nodes = int(node_masks.shape[1])
            if cur_nodes == max_nodes:
                continue

            pad_nodes = max_nodes - cur_nodes
            if pad_nodes <= 0:
                continue

            # Pad node_masks: [B, cur_nodes] -> [B, max_nodes]
            B = node_masks.shape[0]
            pad_mask = torch.zeros(B, pad_nodes, dtype=node_masks.dtype, device=node_masks.device)
            tensor_data["node_masks"] = torch.cat([node_masks, pad_mask], dim=1)

            # dense_rewards is [B, T] (independent of N), no padding needed
            if "dense_rewards" in tensor_data:
                pass

            # Pad trajectory_X: [B, T, cur_nodes, Dx] -> [B, T, max_nodes, Dx]
            traj_X = tensor_data.get("trajectory_X")
            if traj_X is not None and traj_X.shape[2] != max_nodes:
                B, T, _, Dx = traj_X.shape
                pad_X = torch.zeros(B, T, pad_nodes, Dx, dtype=traj_X.dtype, device=traj_X.device)
                tensor_data["trajectory_X"] = torch.cat([traj_X, pad_X], dim=2)

            # Pad trajectory_E: [B, T, cur_nodes, cur_nodes, De] -> [B, T, max_nodes, max_nodes, De]
            traj_E = tensor_data.get("trajectory_E")
            if traj_E is not None and traj_E.shape[2] != max_nodes:
                B, T, N1, N2, De = traj_E.shape
                if N1 == cur_nodes and N2 == cur_nodes:
                    # Pad columns to [B, T, cur_nodes, max_nodes, De]
                    pad_cols = torch.zeros(
                        B, T, cur_nodes, pad_nodes, De, dtype=traj_E.dtype, device=traj_E.device
                    )
                    E_padded = torch.cat([traj_E, pad_cols], dim=3)

                    # Pad rows to [B, T, max_nodes, max_nodes, De]
                    pad_rows = torch.zeros(
                        B, T, pad_nodes, max_nodes, De, dtype=traj_E.dtype, device=traj_E.device
                    )
                    E_padded = torch.cat([E_padded, pad_rows], dim=2)

                    tensor_data["trajectory_E"] = E_padded

    def clear_sample_buffer(self):
        """Clear the sample buffer to prepare for the next sampling round."""
        self.sample_buffer = None
        self._sample_batches = []
        if self.stat_tracker:
            self.stat_tracker.clear_statistics()

    def _vectorize_trajectories(
        self,
        trajectories: List[List],
        node_masks: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if not trajectories:
            raise ValueError("Empty trajectories cannot be vectorized")

        device = node_masks.device
        batch_size = len(trajectories)
        seq_lengths = [len(traj) for traj in trajectories]
        seq_len = min(seq_lengths)
        if seq_len < 2:
            raise ValueError("Trajectory length must be >=2 for PPO training")

        max_nodes = node_masks.shape[1]
        sample_state = trajectories[0][0]
        sample_X = self._squeeze_node_features(sample_state.X).to(device)
        sample_E = self._squeeze_edge_features(sample_state.E).to(device)
        node_feat_dim = sample_X.shape[-1]
        edge_feat_dim = sample_E.shape[-1]

        X_tensor = torch.zeros(
            batch_size, seq_len, max_nodes, node_feat_dim,
            device=device, dtype=sample_X.dtype
        )
        E_tensor = torch.zeros(
            batch_size, seq_len, max_nodes, max_nodes, edge_feat_dim,
            device=device, dtype=sample_E.dtype
        )

        has_y = sample_state.y is not None
        y_tensor = None
        if has_y:
            sample_y_vec = self._extract_condition_vector(sample_state.y, device)
            y_dim = sample_y_vec.shape[0]
            y_tensor = torch.zeros(batch_size, seq_len, y_dim, device=device, dtype=sample_y_vec.dtype)

        for b_idx, traj in enumerate(trajectories):
            for t in range(seq_len):
                state = traj[t]
                X = self._squeeze_node_features(state.X).to(device)
                E = self._squeeze_edge_features(state.E).to(device)
                n_nodes = X.shape[0]

                X_tensor[b_idx, t, :n_nodes] = X
                E_tensor[b_idx, t, :n_nodes, :n_nodes] = E

                if has_y and y_tensor is not None:
                    y_vec = self._extract_condition_vector(state.y, device)
                    if y_vec.shape[0] != y_tensor.shape[-1]:
                        raise ValueError("Conditional vector dimension mismatch across trajectories")
                    y_tensor[b_idx, t] = y_vec

        data = {
            "trajectory_X": X_tensor,
            "trajectory_E": E_tensor,
        }
        if y_tensor is not None:
            data["trajectory_y"] = y_tensor
        return data

    def compute_ppo_policy_loss(
        self,
        current_log_probs: torch.Tensor,
        old_log_probs: torch.Tensor,
        advantages: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute PPO (Proximal Policy Optimization) policy loss.

        Uses a clipped importance ratio to prevent excessively large policy
        updates. This is the core loss function for GRPO training.

        Args:
            current_log_probs: Log probabilities under the current policy,
                shape [batch_size, num_steps].
            old_log_probs: Log probabilities under the sampling policy,
                shape [batch_size, num_steps].
            advantages: Advantage values indicating relative action quality,
                shape [batch_size, num_steps].
            mask: Optional mask for valid positions.

        Returns:
            Dict containing policy_loss, approx_kl, clipfrac, and ratio
            statistics.
        """
        # Compute importance ratio
        log_ratio = current_log_probs - old_log_probs.detach()
        log_ratio = torch.clamp(log_ratio, min=-5.0, max=5.0)

        # Compute ratio directly from log_ratio (no per-decision normalization)
        ratio = torch.exp(log_ratio)

        # Build weights based on advantage sign: amplify gradient for positive advantages
        if self.positive_advantage_weight != 1.0:
            weights = torch.ones_like(advantages)
            weights = torch.where(
                advantages > 0,
                torch.as_tensor(self.positive_advantage_weight, device=advantages.device, dtype=advantages.dtype),
                weights,
            )
        else:
            weights = None

        # PPO clipped loss (supports asymmetric lower/upper) with optional weighting
        lower_bound = 1.0 - self.clip_ratio_lower
        upper_bound = 1.0 + self.clip_ratio_upper
        clipped_ratio = torch.clamp(ratio, lower_bound, upper_bound)
        if weights is not None:
            unclipped_loss = -weights * advantages * ratio
            clipped_loss = -weights * advantages * clipped_ratio
        else:
            unclipped_loss = -advantages * ratio
            clipped_loss = -advantages * clipped_ratio
        policy_loss = torch.maximum(unclipped_loss, clipped_loss)

        # Apply mask and compute mean
        if mask is not None:
            mask = mask.to(policy_loss.dtype)
            policy_loss = policy_loss * mask
            policy_loss = policy_loss.sum() / mask.sum()
        else:
            policy_loss = policy_loss.mean()

        # Compute monitoring metrics based on log_ratio
        with torch.no_grad():
            approx_kl = 0.5 * (log_ratio ** 2).mean()
            clip_mask = (ratio < lower_bound) | (ratio > upper_bound)
            clipfrac = clip_mask.float().mean()

        return {
            "policy_loss": policy_loss,
            "approx_kl": approx_kl,
            "clipfrac": clipfrac,
            "ratio_mean": ratio.mean(),
            "ratio_std": ratio.std(),
        }

    def compute_kl_regularization_loss(
        self,
        current_distribution: torch.Tensor,
        reference_distribution: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute KL divergence regularization loss.

        Prevents the current policy from drifting too far from the reference
        policy. Uses MSE as an efficient approximation of KL divergence.

        Args:
            current_distribution: Current policy distribution.
            reference_distribution: Reference policy distribution.
            mask: Optional mask for valid positions.

        Returns:
            KL divergence loss value.
        """
        # Compute log probability difference (use MSE as KL proxy, consistent with Flow-GRPO)
        log_ratio = current_distribution - reference_distribution
        per_step_kl = log_ratio.pow(2)

        if mask is not None:
            mask = mask.to(per_step_kl.dtype)
            per_step_kl = per_step_kl * mask
            denom = mask.sum().clamp(min=1.0)
            kl_loss = per_step_kl.sum() / denom
        else:
            kl_loss = per_step_kl.mean()

        return kl_loss

    def compute_losses(
        self,
        model: nn.Module,
        batch_data: Union[Dict[str, torch.Tensor], TrajectoryData],
        reference_model: Optional[nn.Module] = None,
        max_steps: Optional[int] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute all losses without performing backward pass or optimization.

        Role in GRPO architecture:
            - Core loss computation logic, called by GRPOTrainer.training_phase.
            - Handles only forward pass and loss computation.
            - Does not involve gradient computation, backward pass, or parameter
              updates.

        Call chain:
            GRPOLightningModule.training_step
            -> GRPOTrainer.run_epoch
            -> GRPOTrainer.training_phase
            -> GRPOCore.compute_losses (this method)

        Args:
            model: Current policy model.
            batch_data: Prepared batch data.
            reference_model: Reference model for KL regularization.
            max_steps: Optional maximum number of training steps.

        Returns:
            Dict of losses and metrics, containing a 'total_loss' key.
        """
        batch_dict = batch_data.as_dict() if isinstance(batch_data, TrajectoryData) else batch_data

        old_log_probs = batch_dict["old_log_probs"]
        advantages = batch_dict["advantages"]
        node_masks = batch_dict["node_masks"]

        if old_log_probs.ndim == 2:
            total_steps = old_log_probs.shape[1]
            if max_steps is not None:
                max_steps = min(int(max_steps), int(total_steps))
                # Train on the last N steps: take a tail window consistent with the
                # time window recorded during the sampling phase
                old_log_probs = old_log_probs[:, -max_steps:]
                advantages = advantages[:, -max_steps:]
            else:
                max_steps = int(total_steps)
        else:
            # Fallback: unknown time dimension; keep original behaviour.
            if max_steps is None:
                max_steps = None

        prepared_inputs = None
        if "trajectory_X" in batch_dict and "trajectory_E" in batch_dict:
            t_start = batch_dict.get("trajectory_t_start")
            total_inference_steps_tensor = batch_dict.get("trajectory_total_inference_steps")
            total_inference_steps = None
            if isinstance(total_inference_steps_tensor, torch.Tensor) and total_inference_steps_tensor.numel() > 0:
                total_inference_steps = int(total_inference_steps_tensor.flatten()[0].item())
            prepared_inputs = self._prepare_vectorized_from_tensor_cache(
                batch_dict["trajectory_X"],
                batch_dict["trajectory_E"],
                batch_dict.get("trajectory_y"),
                node_masks,
                max_steps=max_steps,
                t_start=t_start,
                total_inference_steps=total_inference_steps,
            )
        elif "trajectories" in batch_dict:
            prepared_inputs = self._prepare_vectorized_trajectory_tensors(
                batch_dict["trajectories"],
                node_masks,
                max_steps=max_steps,
            )
        else:
            raise ValueError("Batch data must include trajectory information")

        if prepared_inputs is None:
            raise ValueError("Prepared trajectory tensor batch is empty")

        # Recompute current policy log probabilities; collect distribution stats if needed
        need_dist_stats = (self.gdcr_coef > 0) or (self.diversity_coef > 0)
        dist_stats = None
        if need_dist_stats:
            current_log_probs, policy_entropy_steps, dist_stats = self.recompute_trajectory_log_probabilities(
                model,
                None,
                node_masks,
                max_steps=max_steps,
                return_entropy=True,
                return_distribution_stats=True,
                prepared_data=prepared_inputs,
            )
        else:
            current_log_probs, policy_entropy_steps = self.recompute_trajectory_log_probabilities(
                model,
                None,
                node_masks,
                max_steps=max_steps,
                return_entropy=True,
                prepared_data=prepared_inputs,
            )

        # Train on ALL steps (no partial training mask)
        time_step_mask = None

        # Compute policy loss
        if self.rl_mode == "reinforce":
            # Episode-level REINFORCE: loss = -advantage * log_prob
            # No importance ratio, no clipping.
            # Log probs are sum-reduced over graph elements (nodes + edges).
            # Divide by num_steps to normalise across trajectory length.
            # sum over steps, mean over batch, then / num_steps
            num_steps = max(current_log_probs.shape[1], 1) if current_log_probs.ndim == 2 else 1
            reinforce_loss = -(advantages * current_log_probs)  # (B, T)
            if time_step_mask is not None:
                policy_loss = (reinforce_loss * time_step_mask).sum() / time_step_mask.sum(dim=1).clamp(min=1).shape[0] / num_steps
            else:
                # sum over time → (B,), mean over batch, / T
                policy_loss = reinforce_loss.sum(dim=1).mean(dim=0) / num_steps
            loss_dict = {
                "policy_loss": policy_loss,
                "clipfrac": torch.tensor(0.0),
                "approx_kl": torch.tensor(0.0),
                "ratio_mean": torch.tensor(1.0),
                "ratio_std": torch.tensor(0.0),
            }
        else:
            loss_dict = self.compute_ppo_policy_loss(
                current_log_probs, old_log_probs, advantages, mask=time_step_mask
            )

        total_loss = loss_dict["policy_loss"]

        # Compute policy entropy and add to loss
        if policy_entropy_steps is not None and policy_entropy_steps.numel() > 0:
            policy_entropy = policy_entropy_steps.mean()
        else:
            policy_entropy = current_log_probs.new_tensor(0.0)

        loss_dict["policy_entropy"] = policy_entropy
        if self.entropy_coef > 0:
            total_loss = total_loss - self.entropy_coef * policy_entropy

        # Distribution mean matching + diversity regularization
        # Distribution loss is disabled for dense reward tasks
        dist_loss_disabled = True

        if (not dist_loss_disabled) and dist_stats is not None:
            gdcr_losses = self._compute_distribution_regularization(
                dist_stats, model
            )
            gdcr_total = current_log_probs.new_tensor(0.0)
            gdcr_applied = False

            if gdcr_losses.get("mean_match") is not None:
                gdcr_applied = True
                gdcr_total = gdcr_total + self.gdcr_coef * gdcr_losses["mean_match"]
                loss_dict["gdcr/mean_match"] = gdcr_losses["mean_match"]

            if gdcr_losses.get("diversity") is not None:
                gdcr_applied = True
                gdcr_total = gdcr_total + self.diversity_coef * gdcr_losses["diversity"]
                loss_dict["gdcr/diversity"] = gdcr_losses["diversity"]

            if gdcr_applied:
                total_loss = total_loss + gdcr_total
                loss_dict["gdcr_loss"] = gdcr_total

        # Add KL regularization if reference model is provided
        if reference_model is not None and self.beta > 0:
            with torch.no_grad():
                ref_log_probs = self.recompute_trajectory_log_probabilities(
                    reference_model,
                    None,
                    node_masks,
                    max_steps=max_steps,
                    prepared_data=prepared_inputs,
                )

            kl_loss = self.compute_kl_regularization_loss(
                current_log_probs, ref_log_probs, mask=time_step_mask
            )
            total_loss = total_loss + self.beta * kl_loss
            loss_dict["kl_loss"] = kl_loss

        loss_dict["total_loss"] = total_loss

        return loss_dict

    def _compute_distribution_regularization(
        self,
        dist_stats: Dict[str, torch.Tensor],
        model: nn.Module,
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Compute distribution mean matching (GDCR) and diversity regularization.

        Args:
            dist_stats: Dictionary of distribution statistics from the forward pass.
            model: The policy model (used to access dataset_info).

        Returns:
            Dict with optional 'mean_match' and 'diversity' loss tensors.
        """
        losses: Dict[str, Optional[torch.Tensor]] = {
            "mean_match": None,
            "diversity": None,
        }

        dataset_info = getattr(model, "dataset_info", None)
        target_node_dist = getattr(dataset_info, "node_types", None) if dataset_info is not None else None
        target_edge_dist = getattr(dataset_info, "edge_types", None) if dataset_info is not None else None

        # Mean matching
        mean_terms: List[torch.Tensor] = []
        node_pred_mean = dist_stats.get("node_pred_mean")
        if target_node_dist is not None and node_pred_mean is not None:
            target = self._normalize_target_distribution(
                target_node_dist, device=node_pred_mean.device, expected_dim=node_pred_mean.shape[-1]
            )
            if target is not None and target.shape == node_pred_mean.shape:
                mean_terms.append(F.mse_loss(node_pred_mean, target))

        edge_pred_mean = dist_stats.get("edge_pred_mean")
        if target_edge_dist is not None and edge_pred_mean is not None:
            target = self._normalize_target_distribution(
                target_edge_dist, device=edge_pred_mean.device, expected_dim=edge_pred_mean.shape[-1]
            )
            if target is not None and target.shape == edge_pred_mean.shape:
                mean_terms.append(self.edge_dist_factor * F.mse_loss(edge_pred_mean, target))

        if mean_terms:
            losses["mean_match"] = torch.stack(mean_terms).mean()

        # Diversity regularization (encourages intra-batch variance)
        diversity_terms: List[torch.Tensor] = []
        node_pred_per_step = dist_stats.get("node_pred_per_step")
        if node_pred_per_step is not None and node_pred_per_step.numel() > 0:
            node_std = node_pred_per_step.std(dim=0, unbiased=False)
            diversity_terms.append(node_std.mean())

        edge_pred_per_step = dist_stats.get("edge_pred_per_step")
        if edge_pred_per_step is not None and edge_pred_per_step.numel() > 0:
            edge_std = edge_pred_per_step.std(dim=0, unbiased=False)
            diversity_terms.append(self.edge_dist_factor * edge_std.mean())

        if diversity_terms:
            losses["diversity"] = -torch.stack(diversity_terms).mean()

        return losses

    @staticmethod
    def _normalize_target_distribution(
        dist: Union[torch.Tensor, Dict[int, float], List[float], Tuple[float, ...], np.ndarray, None],
        device: torch.device,
        expected_dim: Optional[int] = None,
    ) -> Optional[torch.Tensor]:
        """Convert a target distribution to a normalized probability tensor.

        Args:
            dist: Target distribution in various formats (tensor, dict, list,
                tuple, or ndarray).
            device: Target device for the output tensor.
            expected_dim: Expected dimensionality; pads or truncates as needed.

        Returns:
            Normalized probability tensor, or None if input is invalid.
        """
        if dist is None:
            return None

        dist_tensor: torch.Tensor
        if isinstance(dist, torch.Tensor):
            dist_tensor = dist.to(device=device, dtype=torch.float32)
        elif isinstance(dist, dict):
            if not dist:
                return None
            max_idx = max(int(k) for k in dist.keys())
            dim = expected_dim if expected_dim is not None else (max_idx + 1)
            dim = max(dim, max_idx + 1)
            values = torch.zeros(dim, device=device, dtype=torch.float32)
            for k, v in dist.items():
                idx = int(k)
                if idx < dim:
                    values[idx] = float(v)
            dist_tensor = values
        else:
            dist_tensor = torch.as_tensor(dist, device=device, dtype=torch.float32)

        if expected_dim is not None:
            if dist_tensor.numel() < expected_dim:
                pad = expected_dim - dist_tensor.numel()
                dist_tensor = F.pad(dist_tensor, (0, pad))
            elif dist_tensor.numel() > expected_dim:
                dist_tensor = dist_tensor[:expected_dim]

        total = dist_tensor.sum()
        if total <= 0:
            return None
        return dist_tensor / total

    def recompute_trajectory_log_probabilities(
        self,
        model: nn.Module,
        trajectories: Optional[List],
        node_masks: torch.Tensor,
        max_steps: Optional[int] = None,
        return_entropy: bool = False,
        return_distribution_stats: bool = False,
        prepared_data: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Recompute log probabilities per timestep, consistent with sampling.

        Uses batched implementation to avoid per-sample loops that cause CPU
        bottlenecks.

        Args:
            model: The policy model.
            trajectories: Optional list of trajectories (unused if prepared_data
                is provided).
            node_masks: Node masks tensor.
            max_steps: Optional maximum number of steps.
            return_entropy: Whether to also return entropy values.
            return_distribution_stats: Whether to also return distribution
                statistics for GDCR.
            prepared_data: Optional precomputed trajectory tensors.

        Returns:
            Log probabilities tensor of shape [B, T], optionally with entropy
            and/or distribution statistics.
        """
        if prepared_data is None:
            if trajectories is None:
                raise ValueError("Either trajectories list or prepared_data must be provided")
            batch_size = len(trajectories)
            if batch_size == 0 or len(trajectories[0]) < 2:
                empty = torch.zeros(batch_size, 0, device=node_masks.device)
                if return_entropy and return_distribution_stats:
                    return empty, empty.clone(), None
                if return_entropy:
                    return empty, empty.clone()
                if return_distribution_stats:
                    return empty, None
                return empty
            prepared = self._prepare_vectorized_trajectory_tensors(
                trajectories, node_masks, max_steps=max_steps
            )
        else:
            prepared = prepared_data
            batch_size = prepared["batch_size"]
        if prepared is None:
            empty = torch.zeros(batch_size, 0, device=node_masks.device)
            if return_entropy and return_distribution_stats:
                return empty, empty.clone(), None
            if return_entropy:
                return empty, empty.clone()
            if return_distribution_stats:
                return empty, None
            return empty

        _log_prob_args = dict(
            model=model,
            flat_X_t=prepared["flat_X_t"],
            flat_E_t=prepared["flat_E_t"],
            flat_X_next=prepared["flat_X_next"],
            flat_E_next=prepared["flat_E_next"],
            flat_y_t=prepared["flat_y_t"],
            flat_node_masks=prepared["flat_node_masks"],
            t_indices=prepared["t_indices"],
            total_inference_steps=prepared.get("total_inference_steps", prepared["num_steps"]),
            device=node_masks.device,
            return_entropy=return_entropy,
            return_distribution_stats=return_distribution_stats,
        )
        if self.rl_mode == "reinforce":
            vectorized = self._compute_reinforce_log_probs(**_log_prob_args)
        else:
            vectorized = self._compute_vectorized_log_probs(**_log_prob_args)
        num_steps = prepared["num_steps"]
        dist_stats = None
        if return_entropy and return_distribution_stats:
            flat_log_probs, flat_entropy, dist_stats = vectorized
        elif return_entropy:
            flat_log_probs, flat_entropy = vectorized
        elif return_distribution_stats:
            flat_log_probs, dist_stats = vectorized
        else:
            flat_log_probs = vectorized
        reshaped_log_probs = flat_log_probs.view(batch_size, num_steps)

        if not return_entropy and not return_distribution_stats:
            return reshaped_log_probs

        if return_entropy:
            reshaped_entropy = flat_entropy.view(batch_size, num_steps)
            if not return_distribution_stats:
                return reshaped_log_probs, reshaped_entropy
        if return_distribution_stats and not return_entropy:
            return reshaped_log_probs, dist_stats
        return reshaped_log_probs, reshaped_entropy, dist_stats

    def _prepare_vectorized_trajectory_tensors(
        self,
        trajectories: List,
        node_masks: torch.Tensor,
        max_steps: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Organize trajectory lists into batched tensors for vectorized computation.

        Args:
            trajectories: List of trajectory sequences.
            node_masks: Node masks tensor.
            max_steps: Optional maximum number of steps to include.

        Returns:
            Dict of flattened tensors ready for batch computation, or None if
            trajectories are too short.
        """
        if not trajectories:
            return None

        device = node_masks.device
        batch_size = len(trajectories)
        max_nodes = node_masks.shape[1]

        min_traj_len = min(len(traj) for traj in trajectories)
        num_steps = max(0, min_traj_len - 1)
        if num_steps == 0:
            return None
        if max_steps is not None:
            num_steps = min(num_steps, max_steps)
        if num_steps == 0:
            return None

        sample_state = trajectories[0][0]
        sample_X = self._squeeze_node_features(sample_state.X).to(device)
        sample_E = self._squeeze_edge_features(sample_state.E).to(device)

        node_feat_dim = sample_X.shape[-1]
        edge_feat_dim = sample_E.shape[-1]

        X_t_tensor = torch.zeros(
            batch_size, num_steps, max_nodes, node_feat_dim,
            device=device, dtype=sample_X.dtype
        )
        E_t_tensor = torch.zeros(
            batch_size, num_steps, max_nodes, max_nodes, edge_feat_dim,
            device=device, dtype=sample_E.dtype
        )
        X_next_tensor = torch.zeros_like(X_t_tensor)
        E_next_tensor = torch.zeros_like(E_t_tensor)

        has_y = sample_state.y is not None
        batch_y = None
        if has_y:
            sample_y_vec = self._extract_condition_vector(sample_state.y, device)
            y_dim = sample_y_vec.shape[0]
            batch_y = torch.zeros(batch_size, y_dim, device=device, dtype=sample_y_vec.dtype)

        for b_idx, traj in enumerate(trajectories):
            for t in range(num_steps):
                state_t = traj[t]
                state_next = traj[t + 1]

                X_t = self._squeeze_node_features(state_t.X).to(device)
                X_next = self._squeeze_node_features(state_next.X).to(device)
                n_nodes = X_t.shape[0]

                E_t = self._squeeze_edge_features(state_t.E).to(device)
                E_next = self._squeeze_edge_features(state_next.E).to(device)

                X_t_tensor[b_idx, t, :n_nodes] = X_t
                X_next_tensor[b_idx, t, :n_nodes] = X_next
                E_t_tensor[b_idx, t, :n_nodes, :n_nodes] = E_t
                E_next_tensor[b_idx, t, :n_nodes, :n_nodes] = E_next

            if has_y and batch_y is not None:
                y_vec = self._extract_condition_vector(traj[0].y, device)
                if y_vec.shape[0] != batch_y.shape[1]:
                    raise ValueError("Conditional vector dimension mismatch across trajectories.")
                batch_y[b_idx] = y_vec

        flat_X_t = X_t_tensor.reshape(batch_size * num_steps, max_nodes, node_feat_dim)
        flat_E_t = E_t_tensor.reshape(batch_size * num_steps, max_nodes, max_nodes, edge_feat_dim)
        flat_X_next = X_next_tensor.reshape(batch_size * num_steps, max_nodes, node_feat_dim)
        flat_E_next = E_next_tensor.reshape(batch_size * num_steps, max_nodes, max_nodes, edge_feat_dim)

        node_mask_bool = node_masks.to(device=device).bool()
        flat_node_masks = node_mask_bool.unsqueeze(1).expand(batch_size, num_steps, max_nodes)
        flat_node_masks = flat_node_masks.reshape(batch_size * num_steps, max_nodes)

        if has_y and batch_y is not None:
            flat_y = batch_y.unsqueeze(1).expand(batch_size, num_steps, batch_y.shape[-1])
            flat_y = flat_y.reshape(batch_size * num_steps, batch_y.shape[-1])
        else:
            flat_y = None

        t_indices = torch.arange(num_steps, device=device).unsqueeze(0).expand(batch_size, -1)
        t_indices = t_indices.reshape(-1)

        return {
            "flat_X_t": flat_X_t,
            "flat_E_t": flat_E_t,
            "flat_X_next": flat_X_next,
            "flat_E_next": flat_E_next,
            "flat_y_t": flat_y,
            "flat_node_masks": flat_node_masks,
            "t_indices": t_indices,
            "num_steps": num_steps,
            "batch_size": batch_size,
        }

    def _prepare_vectorized_from_tensor_cache(
        self,
        trajectory_X: torch.Tensor,
        trajectory_E: torch.Tensor,
        trajectory_y: Optional[torch.Tensor],
        node_masks: torch.Tensor,
        max_steps: Optional[int] = None,
        t_start: Optional[torch.Tensor] = None,
        total_inference_steps: Optional[int] = None,
    ) -> Optional[Dict[str, torch.Tensor]]:
        batch_size, num_states, max_nodes, node_feat_dim = trajectory_X.shape
        num_steps = max(0, num_states - 1)
        if num_steps == 0:
            return None
        if max_steps is not None:
            num_steps = min(num_steps, max_steps)
        if num_steps == 0:
            return None

        X_t = trajectory_X[:, :num_steps]
        X_next = trajectory_X[:, 1:num_steps + 1]
        E_t = trajectory_E[:, :num_steps]
        E_next = trajectory_E[:, 1:num_steps + 1]

        flat_X_t = X_t.reshape(batch_size * num_steps, max_nodes, node_feat_dim)
        flat_X_next = X_next.reshape(batch_size * num_steps, max_nodes, node_feat_dim)

        edge_feat_dim = E_t.shape[-1]
        flat_E_t = E_t.reshape(batch_size * num_steps, max_nodes, max_nodes, edge_feat_dim)
        flat_E_next = E_next.reshape(batch_size * num_steps, max_nodes, max_nodes, edge_feat_dim)

        node_mask_bool = node_masks.to(device=trajectory_X.device).bool()
        flat_node_masks = node_mask_bool.unsqueeze(1).expand(batch_size, num_steps, max_nodes)
        flat_node_masks = flat_node_masks.reshape(batch_size * num_steps, max_nodes)

        if trajectory_y is not None:
            flat_y = trajectory_y[:, :num_steps]
            flat_y = flat_y.reshape(batch_size * num_steps, trajectory_y.shape[-1])
        else:
            flat_y = None

        base_t = torch.arange(num_steps, device=trajectory_X.device).unsqueeze(0).expand(batch_size, -1)
        if t_start is not None and isinstance(t_start, torch.Tensor) and t_start.numel() > 0:
            t0 = t_start.to(device=trajectory_X.device).view(batch_size, 1).long()
            t_indices = (base_t + t0).reshape(-1)
        else:
            t_indices = base_t.reshape(-1)

        return {
            "flat_X_t": flat_X_t,
            "flat_E_t": flat_E_t,
            "flat_X_next": flat_X_next,
            "flat_E_next": flat_E_next,
            "flat_y_t": flat_y,
            "flat_node_masks": flat_node_masks,
            "t_indices": t_indices,
            "num_steps": num_steps,
            "batch_size": batch_size,
            "total_inference_steps": int(total_inference_steps) if total_inference_steps is not None else num_steps,
        }

    @staticmethod
    def _squeeze_node_features(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 3 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        return tensor

    @staticmethod
    def _squeeze_edge_features(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() == 4 and tensor.size(0) == 1:
            return tensor.squeeze(0)
        return tensor

    @staticmethod
    def _extract_condition_vector(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
        y = tensor.to(device)
        if y.dim() >= 2 and y.size(0) == 1:
            y = y.squeeze(0)
        return y.reshape(-1)

    def _compute_step_log_probs(
        self, prob_X, prob_E, X_next_list, E_next_list,
        node_masks, batch_size, max_nodes, device
    ):
        """Compute log probabilities for a single timestep.

        Args:
            prob_X: Node transition probabilities.
            prob_E: Edge transition probabilities.
            X_next_list: List of next node states per sample.
            E_next_list: List of next edge states per sample.
            node_masks: Node masks tensor.
            batch_size: Number of samples in the batch.
            max_nodes: Maximum number of nodes.
            device: Computation device.

        Returns:
            Per-sample log probabilities for this timestep.
        """
        import torch

        # Prepare next-state indices
        X_indices = torch.zeros(batch_size, max_nodes, dtype=torch.long, device=device)
        E_indices = torch.zeros(batch_size, max_nodes, max_nodes, dtype=torch.long, device=device)

        for b_idx in range(batch_size):
            X_next = X_next_list[b_idx]
            E_next = E_next_list[b_idx]

            if X_next.dim() == 3 and X_next.size(0) == 1:
                X_next = X_next.squeeze(0)
            if E_next.dim() == 4 and E_next.size(0) == 1:
                E_next = E_next.squeeze(0)

            # Get indices
            n_nodes = X_next.shape[0]
            X_idx = torch.argmax(X_next, dim=-1)
            E_idx = torch.argmax(E_next, dim=-1)

            X_indices[b_idx, :n_nodes] = X_idx
            E_indices[b_idx, :n_nodes, :n_nodes] = E_idx

        # Compute log probabilities
        X_log_probs = torch.log(prob_X.clamp(min=1e-8))
        E_log_probs = torch.log(prob_E.clamp(min=1e-8))

        # Gather log probabilities
        X_step_log_prob = torch.gather(X_log_probs, dim=-1,
                                       index=X_indices.unsqueeze(-1)).squeeze(-1)
        E_step_log_prob = torch.gather(E_log_probs, dim=-1,
                                       index=E_indices.unsqueeze(-1)).squeeze(-1)

        # Apply masks
        X_masked = (X_step_log_prob * node_masks).sum(dim=-1)

        edge_mask = node_masks.unsqueeze(1) * node_masks.unsqueeze(2)
        diag_indices = torch.arange(max_nodes, device=device)
        edge_mask[:, diag_indices, diag_indices] = 0
        E_masked = (E_step_log_prob * edge_mask).sum(dim=[-2, -1]) * 0.5

        # Combine node and edge log probabilities
        step_log_probs = X_masked + E_masked

        return step_log_probs

    def _compute_reinforce_log_probs(
        self,
        model: nn.Module,
        flat_X_t: torch.Tensor,
        flat_E_t: torch.Tensor,
        flat_X_next: torch.Tensor,
        flat_E_next: torch.Tensor,
        flat_y_t: Optional[torch.Tensor],
        flat_node_masks: torch.Tensor,
        t_indices: torch.Tensor,
        total_inference_steps: int,
        device: torch.device,
        return_entropy: bool = False,
        return_distribution_stats: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        """REINFORCE log-prob computation: uses model prediction directly, skipping rate matrix.

        Same signature and return format as _compute_vectorized_log_probs, but gathers
        log-probs directly from pred_X/pred_E (the model's softmax output for x_1 prediction)
        instead of computing rate matrices and transition probabilities.
        """
        BT = flat_X_t.shape[0]

        # Prepare time information (same as _compute_vectorized_log_probs)
        t_array = t_indices.unsqueeze(1).float()
        t_norm = t_array / (total_inference_steps + 1)

        if hasattr(model, 'time_distorter'):
            sample_cfg = getattr(self.cfg, 'sample', None)
            if sample_cfg is None:
                sample_cfg = self.cfg.get('sample', {})
            if hasattr(sample_cfg, 'get'):
                time_distortion = sample_cfg.get('time_distortion', 'polydec')
            else:
                time_distortion = getattr(sample_cfg, 'time_distortion', 'polydec')
            t_norm = model.time_distorter.sample_ft(t_norm, time_distortion)

        # Model forward pass
        noisy_data = {
            "X_t": flat_X_t,
            "E_t": flat_E_t,
            "y_t": flat_y_t,
            "t": t_norm,
            "node_mask": flat_node_masks
        }
        extra_data = model.compute_extra_data(noisy_data)
        pred = model.forward(noisy_data, extra_data, flat_node_masks)

        # Softmax with temperature
        sampling_temperature = 1.0
        try:
            grpo_cfg = getattr(self.cfg, 'grpo', None)
            if grpo_cfg is None:
                grpo_cfg = self.cfg.get('grpo', {})
            if hasattr(grpo_cfg, 'get'):
                sampling_temperature = float(grpo_cfg.get("sampling_temperature", 1.0))
            else:
                sampling_temperature = float(getattr(grpo_cfg, "sampling_temperature", 1.0))
        except Exception:
            sampling_temperature = 1.0
        if sampling_temperature <= 0:
            sampling_temperature = 1.0

        if abs(sampling_temperature - 1.0) > 1e-5:
            pred_X = F.softmax(pred.X / sampling_temperature, dim=-1)
            pred_E = F.softmax(pred.E / sampling_temperature, dim=-1)
        else:
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)

        # --- REINFORCE: skip rate matrix, use pred_X/pred_E directly as prob distribution ---

        X_indices = torch.argmax(flat_X_next, dim=-1)  # (BT, N)
        E_indices = torch.argmax(flat_E_next, dim=-1)  # (BT, N, N)

        X_log_probs = torch.log(pred_X.clamp(min=1e-8))
        X_step_log_prob = torch.gather(
            X_log_probs, dim=-1,
            index=X_indices.unsqueeze(-1)
        ).squeeze(-1)  # (BT, N)

        E_log_probs = torch.log(pred_E.clamp(min=1e-8))
        E_step_log_prob = torch.gather(
            E_log_probs, dim=-1,
            index=E_indices.unsqueeze(-1)
        ).squeeze(-1)  # (BT, N, N)

        # Apply masks and sum (standard REINFORCE log-prob aggregation).
        # Gradient magnitude is controlled via max_grad_norm clipping (set to 1.0).
        node_mask_float = flat_node_masks.float()
        X_masked = torch.sum(X_step_log_prob * node_mask_float, dim=-1)  # (BT,)

        edge_mask = (flat_node_masks.unsqueeze(1) & flat_node_masks.unsqueeze(2)).float()
        diag_indices = torch.arange(flat_node_masks.size(1), device=device)
        edge_mask[:, diag_indices, diag_indices] = 0

        E_masked = torch.sum(E_step_log_prob * edge_mask, dim=[-2, -1]) * 0.5  # (BT,)

        total_log_prob = X_masked + E_masked  # (BT,)

        # Distribution statistics (use pred_X/pred_E as the distribution)
        valid_nodes = node_mask_float.sum(dim=-1).clamp(min=1.0)
        dist_stats = None
        if return_distribution_stats:
            eps = 1e-8
            prob_X_norm = pred_X / (pred_X.sum(dim=-1, keepdim=True) + eps)
            prob_E_norm = pred_E / (pred_E.sum(dim=-1, keepdim=True) + eps)
            node_pred = (prob_X_norm * node_mask_float.unsqueeze(-1)).sum(dim=1) / valid_nodes.unsqueeze(-1)
            edge_pred = (prob_E_norm * edge_mask.unsqueeze(-1)).sum(dim=(-2, -1))
            edge_pred = 0.5 * edge_pred / valid_nodes.unsqueeze(-1)
            dist_stats = {
                "node_pred_per_step": node_pred,
                "edge_pred_per_step": edge_pred,
                "node_pred_mean": node_pred.mean(dim=0),
                "edge_pred_mean": edge_pred.mean(dim=0),
            }

        if not return_entropy and not return_distribution_stats:
            return total_log_prob
        if not return_entropy and return_distribution_stats:
            return total_log_prob, dist_stats

        # Entropy from pred_X/pred_E
        entropy_eps = 1e-8
        node_entropy = -(pred_X * torch.log(pred_X.clamp(min=entropy_eps))).sum(dim=-1)
        node_entropy = (node_entropy * node_mask_float).sum(dim=-1)
        node_entropy = node_entropy / valid_nodes

        edge_entropy = -(pred_E * torch.log(pred_E.clamp(min=entropy_eps))).sum(dim=-1)
        edge_entropy = (edge_entropy * edge_mask).sum(dim=(-2, -1))
        edge_entropy = 0.5 * edge_entropy / valid_nodes

        total_entropy = node_entropy + edge_entropy

        if return_distribution_stats:
            return total_log_prob, total_entropy, dist_stats
        return total_log_prob, total_entropy

    def _compute_vectorized_log_probs(
        self,
        model: nn.Module,
        flat_X_t: torch.Tensor,
        flat_E_t: torch.Tensor,
        flat_X_next: torch.Tensor,
        flat_E_next: torch.Tensor,
        flat_y_t: Optional[torch.Tensor],
        flat_node_masks: torch.Tensor,
        t_indices: torch.Tensor,
        total_inference_steps: int,
        device: torch.device,
        return_entropy: bool = False,
        return_distribution_stats: bool = False
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, Dict[str, torch.Tensor]],
        Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]
    ]:
        """Vectorized log probability computation for all batches and timesteps.

        Args:
            model: The policy model.
            flat_X_t: Current node features, shape (B*T, N, Dx).
            flat_E_t: Current edge features, shape (B*T, N, N, De).
            flat_X_next: Next node features, shape (B*T, N, Dx).
            flat_E_next: Next edge features, shape (B*T, N, N, De).
            flat_y_t: Optional condition vectors, shape (B*T, Dy).
            flat_node_masks: Node masks, shape (B*T, N).
            t_indices: Time step indices, shape (B*T,).
            total_inference_steps: Total number of inference steps.
            device: Computation device.
            return_entropy: Whether to return entropy values.
            return_distribution_stats: Whether to return distribution statistics.

        Returns:
            Log probabilities of shape (B*T,), optionally with entropy and/or
            distribution statistics.
        """
        BT = flat_X_t.shape[0]

        # Prepare time information (vectorized)
        t_array = t_indices.unsqueeze(1).float()  # (BT, 1)
        t_norm = t_array / (total_inference_steps + 1)
        s_array = t_array + 1
        s_norm = s_array / (total_inference_steps + 1)

        # Apply time warping if available, using config settings for consistency
        if hasattr(model, 'time_distorter'):
            sample_cfg = getattr(self.cfg, 'sample', None)
            if sample_cfg is None:
                sample_cfg = self.cfg.get('sample', {})

            if hasattr(sample_cfg, 'get'):
                time_distortion = sample_cfg.get('time_distortion', 'polydec')
            else:
                time_distortion = getattr(sample_cfg, 'time_distortion', 'polydec')
            t_norm = model.time_distorter.sample_ft(t_norm, time_distortion)
            s_norm = model.time_distorter.sample_ft(s_norm, time_distortion)

        # Single forward pass for all data (FP32)
        noisy_data = {
            "X_t": flat_X_t,
            "E_t": flat_E_t,
            "y_t": flat_y_t,
            "t": t_norm,
            "node_mask": flat_node_masks
        }

        # Compute all extra features at once
        extra_data = model.compute_extra_data(noisy_data)

        # Single forward pass
        pred = model.forward(noisy_data, extra_data, flat_node_masks)

        # Compute probability distributions (consistent with sampling: supports sampling_temperature)
        sampling_temperature = 1.0
        try:
            grpo_cfg = getattr(self.cfg, 'grpo', None)
            if grpo_cfg is None:
                grpo_cfg = self.cfg.get('grpo', {})

            if hasattr(grpo_cfg, 'get'):
                 sampling_temperature = float(grpo_cfg.get("sampling_temperature", 1.0))
            else:
                 sampling_temperature = float(getattr(grpo_cfg, "sampling_temperature", 1.0))
        except Exception:
            sampling_temperature = 1.0

        # Guard: temperature must be positive
        if sampling_temperature <= 0:
            sampling_temperature = 1.0

        if abs(sampling_temperature - 1.0) > 1e-5:
            pred_X = F.softmax(pred.X / sampling_temperature, dim=-1)
            pred_E = F.softmax(pred.E / sampling_temperature, dim=-1)
        else:
            pred_X = F.softmax(pred.X, dim=-1)
            pred_E = F.softmax(pred.E, dim=-1)

        # Batch compute rate matrices
        # Note: different timesteps have different dt values
        dt = (s_norm - t_norm)  # (BT, 1)

        rate_designer = model.get_rate_matrix_designer() if hasattr(model, 'get_rate_matrix_designer') else model.rate_matrix_designer

        # Batch compute all rate matrices
        R_t_X, R_t_E = rate_designer.compute_graph_rate_matrix(
            t_norm, flat_node_masks,
            (flat_X_t, flat_E_t),
            (pred_X, pred_E)
        )

        # Batch compute transition probabilities
        limit_x = model.limit_dist.X.to(device)
        limit_e = model.limit_dist.E.to(device)

        # Use vectorized compute_step_probs (optionally the GRPO-specific version)
        prob_X, prob_E = self._vectorized_compute_step_probs(
            model, R_t_X, R_t_E, flat_X_t, flat_E_t, dt, limit_x, limit_e,
            use_grpo_version=self.use_grpo_step_probs_for_sampling
        )

        # Batch compute log probabilities
        # Get the actual transition categories
        X_indices = torch.argmax(flat_X_next, dim=-1)  # (BT, N)
        E_indices = torch.argmax(flat_E_next, dim=-1)  # (BT, N, N)

        # Compute log probabilities
        X_log_probs = torch.log(prob_X.clamp(min=1e-8))
        X_step_log_prob = torch.gather(
            X_log_probs, dim=-1,
            index=X_indices.unsqueeze(-1)
        ).squeeze(-1)  # (BT, N)

        E_log_probs = torch.log(prob_E.clamp(min=1e-8))
        E_step_log_prob = torch.gather(
            E_log_probs, dim=-1,
            index=E_indices.unsqueeze(-1)
        ).squeeze(-1)  # (BT, N, N)

        # Apply masks and sum
        node_mask_float = flat_node_masks.float()
        X_masked = torch.sum(X_step_log_prob * node_mask_float, dim=-1)  # (BT,)

        # Edge mask
        edge_mask = (flat_node_masks.unsqueeze(1) & flat_node_masks.unsqueeze(2)).float()
        diag_indices = torch.arange(flat_node_masks.size(1), device=device)
        edge_mask[:, diag_indices, diag_indices] = 0

        E_masked = torch.sum(E_step_log_prob * edge_mask, dim=[-2, -1]) * 0.5  # (BT,)

        # Combine node and edge log probabilities
        total_log_prob = X_masked + E_masked  # (BT,)

        # Distribution statistics: batch-averaged node/edge probabilities & per-sample
        # trajectories for GDCR
        valid_nodes = node_mask_float.sum(dim=-1).clamp(min=1.0)
        dist_stats = None
        if return_distribution_stats:
            eps = 1e-8
            prob_X_norm = prob_X / (prob_X.sum(dim=-1, keepdim=True) + eps)
            prob_E_norm = prob_E / (prob_E.sum(dim=-1, keepdim=True) + eps)

            node_pred = (prob_X_norm * node_mask_float.unsqueeze(-1)).sum(dim=1) / valid_nodes.unsqueeze(-1)
            edge_pred = (prob_E_norm * edge_mask.unsqueeze(-1)).sum(dim=(-2, -1))
            edge_pred = 0.5 * edge_pred / valid_nodes.unsqueeze(-1)
            dist_stats = {
                "node_pred_per_step": node_pred,
                "edge_pred_per_step": edge_pred,
                "node_pred_mean": node_pred.mean(dim=0),
                "edge_pred_mean": edge_pred.mean(dim=0),
            }

        if not return_entropy and not return_distribution_stats:
            return total_log_prob
        if not return_entropy and return_distribution_stats:
            return total_log_prob, dist_stats

        entropy_eps = 1e-8
        # Compute entropy directly from softmax output (with temperature) for consistency
        node_entropy = -(pred_X * torch.log(pred_X.clamp(min=entropy_eps))).sum(dim=-1)
        node_entropy = (node_entropy * node_mask_float).sum(dim=-1)
        node_entropy = node_entropy / valid_nodes

        edge_entropy = -(pred_E * torch.log(pred_E.clamp(min=entropy_eps))).sum(dim=-1)
        edge_entropy = (edge_entropy * edge_mask).sum(dim=(-2, -1))
        edge_entropy = 0.5 * edge_entropy / valid_nodes

        total_entropy = node_entropy + edge_entropy

        if return_distribution_stats:
            return total_log_prob, total_entropy, dist_stats
        return total_log_prob, total_entropy

    def _vectorized_compute_step_probs(
        self,
        model: nn.Module,
        R_t_X: torch.Tensor,
        R_t_E: torch.Tensor,
        X_t: torch.Tensor,
        E_t: torch.Tensor,
        dt: torch.Tensor,
        limit_x: torch.Tensor,
        limit_e: torch.Tensor,
        use_grpo_version: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Vectorized version of compute_step_probs for batch data.

        Args:
            model: The policy model.
            R_t_X: Node rate matrices.
            R_t_E: Edge rate matrices.
            X_t: Current node features.
            E_t: Current edge features.
            dt: Time step sizes.
            limit_x: Limiting distribution for nodes.
            limit_e: Limiting distribution for edges.
            use_grpo_version: Whether to use the GRPO-specific implementation.

        Returns:
            Tuple of (prob_X, prob_E) transition probabilities.
        """
        # Use GRPO version if available on the model
        if use_grpo_version and hasattr(model, 'compute_step_probs_grpo'):
            # Handle dt dimensions to match batch
            if dt.dim() == 2 and dt.shape[1] == 1:
                dt = dt.squeeze(1)  # (BT,)

            # Batch call
            prob_X, prob_E = model.compute_step_probs_grpo(
                R_t_X, R_t_E, X_t, E_t, dt, limit_x, limit_e
            )
        else:
            # Use the original compute_step_probs (consistent with pretrain sampling)
            if dt.dim() == 2 and dt.shape[1] == 1:
                dt_scalar = dt.squeeze(1)
            else:
                dt_scalar = dt
            prob_X, prob_E = model.compute_step_probs(
                R_t_X, R_t_E, X_t, E_t, dt_scalar, limit_x, limit_e
            )

        return prob_X, prob_E
