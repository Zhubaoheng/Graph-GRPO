"""Mixin providing SwanLab logging and detailed visualization for GRPOTrainer."""

import csv
import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from grpo.trajectory_data import TrajectoryData

try:
    import swanlab
except ImportError:
    swanlab = None

logger = logging.getLogger(__name__)


class LoggingMixin:
    """SwanLab metric logging and trajectory visualization methods
    extracted from GRPOTrainer."""

    # ------------------------------------------------------------------
    # SwanLab training-metric logging
    # ------------------------------------------------------------------

    def _log_training_metrics_to_swanlab(
        self,
        epoch_losses: Dict,
        grad_norm_before_clip: float,
        grad_norm_after_clip: float,
        loss_dict: Dict,
        training_batch,
        optimizer
    ):
        """
        Unified SwanLab logging method.
        Called once per global_step to ensure metric alignment.
        """
        log_metrics = {}

        if 'policy_entropy' in loss_dict:
            policy_entropy = loss_dict['policy_entropy']
            if isinstance(policy_entropy, torch.Tensor):
                policy_entropy_value = policy_entropy.detach().item()
            else:
                policy_entropy_value = float(policy_entropy)
            log_metrics['train/policy_entropy'] = policy_entropy_value

        # Gradient-related metrics
        log_metrics['train/grad_norm_before_clip'] = grad_norm_before_clip
        log_metrics['train/grad_norm_after_clip'] = grad_norm_after_clip
        log_metrics['train/grad_clip_ratio'] = grad_norm_after_clip / (grad_norm_before_clip + 1e-8)

        batch_view = training_batch.as_dict() if isinstance(training_batch, TrajectoryData) else training_batch

        # Get reward statistics from training batch
        if 'rewards' in batch_view:
            rewards = batch_view['rewards']
            if isinstance(rewards, torch.Tensor):
                log_metrics['train/avg_reward'] = rewards.mean().item()
                log_metrics['train/std_reward'] = rewards.std().item()
                log_metrics['train/min_reward'] = rewards.min().item()
                log_metrics['train/max_reward'] = rewards.max().item()

        # Coefficient-weighted individual losses (consistent with total_loss contribution)
        if 'total_loss' in loss_dict:
            total = loss_dict['total_loss']
            log_metrics['train/loss_total'] = total.item() if isinstance(total, torch.Tensor) else float(total)
        if 'policy_loss' in loss_dict:
            pol = loss_dict['policy_loss']
            log_metrics['train/loss_policy'] = pol.item() if isinstance(pol, torch.Tensor) else float(pol)
        if 'policy_entropy' in loss_dict and getattr(self.grpo_core, 'entropy_coef', 0.0) != 0:
            ent = loss_dict['policy_entropy']
            ent_val = ent.item() if isinstance(ent, torch.Tensor) else float(ent)
            log_metrics['train/loss_entropy'] = -self.grpo_core.entropy_coef * ent_val
        if 'kl_loss' in loss_dict and getattr(self.grpo_core, 'beta', 0.0) != 0:
            kl = loss_dict['kl_loss']
            kl_val = kl.item() if isinstance(kl, torch.Tensor) else float(kl)
            log_metrics['train/loss_kl'] = self.grpo_core.beta * kl_val
        if getattr(self.grpo_core, 'gdcr_coef', 0.0) != 0:
            if 'gdcr/mean_match' in epoch_losses and epoch_losses['gdcr/mean_match']:
                recent_mm = epoch_losses['gdcr/mean_match'][-min(10, len(epoch_losses['gdcr/mean_match'])):]
                if isinstance(recent_mm[0], torch.Tensor):
                    mm_raw = torch.stack(recent_mm).mean().item()
                else:
                    mm_raw = float(np.mean(recent_mm))
                log_metrics['train/loss_gdcr_mean'] = self.grpo_core.gdcr_coef * mm_raw
            elif 'gdcr/mean_match' in loss_dict:
                mm = loss_dict['gdcr/mean_match']
                mm_val = mm.item() if isinstance(mm, torch.Tensor) else float(mm)
                log_metrics['train/loss_gdcr_mean'] = self.grpo_core.gdcr_coef * mm_val
        if getattr(self.grpo_core, 'diversity_coef', 0.0) != 0:
            if 'gdcr/diversity' in epoch_losses and epoch_losses['gdcr/diversity']:
                recent_div = epoch_losses['gdcr/diversity'][-min(10, len(epoch_losses['gdcr/diversity'])):]
                if isinstance(recent_div[0], torch.Tensor):
                    div_raw = torch.stack(recent_div).mean().item()
                else:
                    div_raw = float(np.mean(recent_div))
                log_metrics['train/loss_gdcr_div'] = self.grpo_core.diversity_coef * div_raw
            elif 'gdcr/diversity' in loss_dict:
                div = loss_dict['gdcr/diversity']
                div_val = div.item() if isinstance(div, torch.Tensor) else float(div)
                log_metrics['train/loss_gdcr_div'] = self.grpo_core.diversity_coef * div_val
        # Get additional metrics from loss dict
        if 'ratio_mean' in loss_dict:
            log_metrics['train/avg_ratio'] = loss_dict['ratio_mean'].item() if isinstance(loss_dict['ratio_mean'], torch.Tensor) else loss_dict['ratio_mean']
        if 'ratio_std' in loss_dict:
            log_metrics['train/std_ratio'] = loss_dict['ratio_std'].item() if isinstance(loss_dict['ratio_std'], torch.Tensor) else loss_dict['ratio_std']

        # Learning rate
        if optimizer is not None:
            log_metrics['train/learning_rate'] = optimizer.param_groups[0]['lr']

        # Step information
        log_metrics['train/global_step'] = self.global_step
        log_metrics['train/epoch'] = self.epoch

        # SwanLab dashboard/local logging should not make a training step fail.
        if swanlab is not None and getattr(swanlab, "run", None) is not None:
            try:
                swanlab.log(log_metrics, step=self.global_step)
            except Exception as exc:
                logger.warning("SwanLab metric logging failed; continuing training: %s", exc)

    # ------------------------------------------------------------------
    # Detailed trajectory visualization
    # ------------------------------------------------------------------

    def _log_detailed_visualization(
        self,
        trajectory_states,
        trajectory_preds,
        trajectory_probs,
        dense_rewards,
        final_rewards,
        log_dir,
        batch_indices
    ):
        """
        Log detailed visualization information (simplified version).
        Creates per-graph folders with step images and reward CSVs.
        """
        import csv
        from analysis.visualization import MolecularVisualization

        # Initialize visualization tools (requires dataset_info)
        dataset_info = getattr(self.model, "dataset_info", None)
        if dataset_info is None:
            logger.warning("Missing dataset_info, cannot perform molecular visualization")
            return

        vis_tool = MolecularVisualization(remove_h=True, dataset_infos=dataset_info)

        os.makedirs(log_dir, exist_ok=True)

        # Convert final_rewards to list format
        if torch.is_tensor(final_rewards):
            final_rewards = final_rewards.cpu().tolist()

        def _extract_state_graph(state):
            if state is None:
                return None, None
            X_state = state.X
            E_state = state.E
            if torch.is_tensor(X_state):
                X_state = X_state.detach()
            if torch.is_tensor(E_state):
                E_state = E_state.detach()
            if torch.is_tensor(X_state) and X_state.dim() == 3:
                X_state = X_state.squeeze(0)
            if torch.is_tensor(E_state) and E_state.dim() == 4:
                E_state = E_state.squeeze(0)

            if torch.is_tensor(X_state) and X_state.dim() == 2:
                node_idx = X_state.argmax(dim=-1)
                node_mask = X_state.sum(dim=-1) > 0
            else:
                node_idx = X_state
                node_mask = node_idx >= 0 if torch.is_tensor(node_idx) else None

            if torch.is_tensor(E_state) and E_state.dim() == 3:
                edge_idx = E_state.argmax(dim=-1)
            else:
                edge_idx = E_state

            if torch.is_tensor(node_mask):
                n_nodes = int(node_mask.sum().item())
            else:
                n_nodes = len(node_idx) if node_idx is not None else 0

            if n_nodes <= 0:
                return (
                    torch.empty(0, dtype=torch.long),
                    torch.empty(0, 0, dtype=torch.long),
                )

            return (
                node_idx[:n_nodes].to(torch.long).cpu(),
                edge_idx[:n_nodes, :n_nodes].to(torch.long).cpu(),
            )

        def _save_graph_image(graph_dir, nodes, adj, filename):
            mols_to_plot = [(nodes, adj,)]
            vis_tool.visualize(graph_dir, mols_to_plot, 1, log=None)

            src_img = os.path.join(graph_dir, "molecule_0.png")
            dst_img = os.path.join(graph_dir, filename)

            if os.path.exists(src_img):
                if os.path.exists(dst_img):
                    os.remove(dst_img)
                os.rename(src_img, dst_img)

        for b_idx in batch_indices:
            if b_idx >= len(trajectory_preds):
                continue

            traj_pred = trajectory_preds[b_idx]
            traj_states = None
            if trajectory_states is not None and b_idx < len(trajectory_states):
                traj_states = trajectory_states[b_idx]
            # dense_rewards: [B, T]
            traj_dense_rewards = None
            if dense_rewards is not None:
                 traj_dense_rewards = dense_rewards[b_idx].cpu().tolist()

            final_r = final_rewards[b_idx] if b_idx < len(final_rewards) else 0.0

            # Create directory for this graph
            graph_dir = os.path.join(log_dir, f"graph_{b_idx}")
            os.makedirs(graph_dir, exist_ok=True)

            # Prepare CSV recording
            csv_path = os.path.join(graph_dir, "rewards.csv")
            csv_rows = []

            num_steps = len(traj_pred)

            for t in range(num_steps):
                # 1. Visualize: current zt + predicted z1
                if traj_states is not None and t < len(traj_states):
                    try:
                        zt_nodes, zt_adj = _extract_state_graph(traj_states[t])
                        _save_graph_image(graph_dir, zt_nodes, zt_adj, f"step_{t}_zt.png")
                    except Exception as e:
                        logger.debug("Step %d zt visualization failed: %s", t, e)

                # z1 is (X_indices, E_indices)
                z1 = traj_pred[t]

                try:
                    nodes = z1[0].cpu() if torch.is_tensor(z1[0]) else z1[0]
                    adj = z1[1].cpu() if torch.is_tensor(z1[1]) else z1[1]
                    _save_graph_image(graph_dir, nodes, adj, f"step_{t}.png")

                except Exception as e:
                    logger.debug("Step %d z1 visualization failed: %s", t, e)

                # 2. Collect rewards
                reward_val = 0.0
                if traj_dense_rewards and t < len(traj_dense_rewards):
                    # dense_rewards tensor logic aligns t with step t,
                    # but check if we need to shift.
                    # Currently strict mapping: dense_rewards[:, t] corresponds to step t.
                    reward_val = traj_dense_rewards[t]

                csv_rows.append([t, reward_val, final_r])

            # Write CSV
            try:
                with open(csv_path, "w", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(["Step", "DenseReward", "FinalReward"])
                    writer.writerows(csv_rows)
            except Exception as e:
                logger.debug("CSV writing failed: %s", e)

        logger.debug("Detailed visualization saved to %s", log_dir)
