import csv
import logging
import os
import random
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

import utils
from flow_matching import flow_matching_utils

try:
    import swanlab
except ImportError:
    swanlab = None

logger = logging.getLogger(__name__)

Graph = Tuple[torch.Tensor, torch.Tensor]


class SamplingMixin:
    """Mixin providing sampling-phase methods for GRPOTrainer."""

    def sampling_phase(self):
        """
        Sampling phase: generate samples and compute rewards (synchronous multiprocessing).
        1. Generate all graphs in parallel on GPU
        2. Transfer to CPU and compute rewards using multiprocessing pool
        3. Collect all results

        Key GRPO design: all batches in one epoch share the same initial noise
        """
        self.core_model.eval()
        try:
            # Debug mode: use model native sample_batch for sampling and reward_function for scoring, skip GRPO training
            if getattr(self.cfg.grpo, "debug_use_model_sampler", False):
                logger.info("[GRPO Debug] Using GraphDiscreteFlowModel.sample_batch for sampling and reward stats (not writing to GRPO buffers)")
                device = next(self.core_model.parameters()).device
                # Sample a similar number of molecules as test_only
                debug_batch_size = self.group_size * self.concurrent_sampling_groups
                keep_chain = 0
                number_chain_steps = min(self.sample_steps, getattr(self.core_model, "sample_T", self.sample_steps))
                samples, labels = self.core_model.sample_batch(
                    batch_id=0,
                    batch_size=debug_batch_size,
                    keep_chain=keep_chain,
                    number_chain_steps=number_chain_steps,
                    save_final=0,
                    num_nodes=None,
                    save_visualization=False,
                )
                # Score directly using current reward_function
                try:
                    rewards = self.reward_function(samples)
                    if isinstance(rewards, torch.Tensor):
                        rewards_tensor = rewards
                    else:
                        rewards_tensor = torch.tensor(rewards, dtype=torch.float32)
                    if rewards_tensor.numel() > 0:
                        mean_r = rewards_tensor.mean().item()
                        std_r = rewards_tensor.std().item()
                        min_r = rewards_tensor.min().item()
                        max_r = rewards_tensor.max().item()
                        logger.info(
                            "[GRPO Debug] Molecular reward stats: mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                            mean_r, std_r, min_r, max_r,
                        )
                except Exception as e:
                    logger.warning("[GRPO Debug] reward evaluation failed: %s", e)
                return
            # Statistics
            total_rewards = []
            debug_all_rewards = []      # For aggregating all reward distributions in this round (including discarded groups)
            debug_all_node_counts = []  # Node count distribution aligned with debug_all_rewards
            debug_molecules = []        # Sampled graphs for visualization

            # [Dynamic p0 Update] Buffer for collecting samples to calculating top 10%
            p0_candidate_buffer = []
            # [Adaptive node-count] Buffer for collecting (reward, node_count)
            node_count_candidate_buffer = []

            # [Stat Tracking] Initialize counters for distribution
            from collections import Counter
            epoch_node_counts = Counter()
            epoch_edge_counts = Counter()
            sample_start_time = time.time()
            device = next(self.core_model.parameters()).device

            total_groups = self.sample_group_num
            groups_collected = 0

            while groups_collected < total_groups:
                active_groups = min(self.concurrent_sampling_groups, total_groups - groups_collected)
                group_ids = []
                for _ in range(active_groups):
                    group_id = f"group_{self._next_group_id}"
                    self._next_group_id += 1
                    group_ids.append(group_id)

                batch_size = active_groups * self.group_size
                enable_visualization = bool(self.cfg.grpo.get("enable_visualization", True))
                should_log_vis = enable_visualization and (self.epoch % 5 == 0) and (groups_collected == 0)

                graphs, node_mask, current_log_prob, ref_log_prob, trajectory_states, trajectory_preds, trajectory_probs = \
                    self.sample_graphs_with_trajectory_tracking(
                        batch_size=batch_size,
                        seed=int(time.time() * 1000) % (2**32) + groups_collected,
                        total_inference_steps=self.sample_steps,
                        force_same_start=True,
                        group_size_for_same_start=self.group_size,
                        return_probs=should_log_vis
                    )

                # Transfer data to CPU for vectorized graph conversion
                node_mask_cpu = node_mask.detach().cpu()
                graph_list = self._convert_placeholder_to_graph_list_cpu(graphs, node_mask, as_tensor=True)

                # Collect a small subset of graphs for visualization
                if enable_visualization and len(debug_molecules) < self.group_size * 4:
                    for mol in graph_list:
                        debug_molecules.append(mol)
                        if len(debug_molecules) >= self.group_size * 4:
                            break

                # [Stat Tracking] Update distribution counts
                for at, et in graph_list:
                    if torch.is_tensor(at):
                        epoch_node_counts.update(at.tolist())
                    if torch.is_tensor(et):
                        epoch_edge_counts.update(et.reshape(-1).tolist())

                # --- 1. Final Reward Calculation (First) ---
                # Compute final-step rewards first, use them to filter dense reward computation
                chunk_rewards = self._compute_rewards_multiprocess_sync(
                    graph_list,
                    timeout=1800,
                    context="sampling",
                )
                if chunk_rewards.numel() == 0:
                    logger.warning("Batch reward multiprocessing computation timeout/failed, skipping this batch")
                    groups_collected += active_groups
                    continue
                if chunk_rewards.device != device:
                    chunk_rewards = chunk_rewards.to(device, non_blocking=True)

                # If all TDC rewards in the batch are 0, check whether all molecules were judged invalid (empty SMILES / failed sanitization)
                try:
                    reward_name = getattr(self.reward_function, "name", "") or ""
                    reward_name = str(reward_name).lower()
                except Exception:
                    reward_name = ""
                if reward_name in ("tdc_oracle", "tdc_pmo", "pmo"):
                    try:
                        all_zero = float(chunk_rewards.abs().max().item()) <= 0.0
                    except Exception:
                        all_zero = False
                    if all_zero and getattr(self, "_tdc_zero_reward_debug_epoch", None) != self.epoch:
                        self._tdc_zero_reward_debug_epoch = int(self.epoch)
                        self._debug_tdc_zero_reward_batch(graph_list, max_samples=32)
                # NOTE: valsartan reward debugging is handled inside the reward itself (print in except blocks).

                rewards_np = chunk_rewards.detach().cpu().numpy()
                node_counts_batch = node_mask_cpu.sum(dim=1).detach().cpu().tolist()

                # [Dynamic p0 Update] Collect data for smooth update (and global buffer).
                if getattr(self.cfg.grpo, 'enable_dynamic_p0', False):
                    for (at, et), r in zip(graph_list, rewards_np):
                        if float(r) <= 0.001:
                            continue
                        if torch.is_tensor(at):
                            at_indices = at.argmax(-1) if at.dim() == 2 else at
                            at_list = at_indices.flatten().detach().cpu().tolist()
                        else:
                            at_list = list(at) if isinstance(at, (list, tuple)) else [int(at)]

                        if torch.is_tensor(et):
                            et_indices = et.argmax(-1) if et.dim() == 3 else et
                            et_list = et_indices.reshape(-1).detach().cpu().tolist()
                        else:
                            # Expect a flat list; fall back to flatten rows if a 2D list is provided.
                            if isinstance(et, (list, tuple)) and et and isinstance(et[0], (list, tuple)):
                                et_list = [int(x) for row in et for x in row]
                            else:
                                et_list = list(et) if isinstance(et, (list, tuple)) else [int(et)]

                        p0_candidate_buffer.append((float(r), at_list, et_list))

                        # Also update debug counters (reward>0 samples) for prints.
                        epoch_node_counts.update(at_list)
                        epoch_edge_counts.update(et_list)

                # [Dynamic node-dist] Collect (reward, node_count) pairs.
                if self.enable_dynamic_node_dist:
                    thr = float(self.dynamic_node_dist_reward_threshold)
                    for r, n in zip(rewards_np, node_counts_batch):
                        if float(r) > thr:
                            node_count_candidate_buffer.append((float(r), int(n)))

                # Record all rewards and node counts in this batch for debug statistics
                debug_all_rewards.append(chunk_rewards.detach().cpu())
                debug_all_node_counts.append(torch.as_tensor(node_counts_batch, dtype=torch.long))

                # --- 2. Dense Reward Calculation (Optimized) ---
                # Strategy: if final-step reward is 0, assume the whole trajectory is invalid, skip intermediate reward computation (set to 0)

                dense_rewards_tensor = None
                use_dense = self.cfg.grpo.get("use_dense_reward", True)
                if trajectory_preds and use_dense:
                    # Check consistency
                    steps_per_traj = [len(t) for t in trajectory_preds]
                    if not steps_per_traj:
                        T_len = 0
                    else:
                        T_len = steps_per_traj[0]

                    if T_len > 0:
                        # Compute start step (50%)
                        start_step = int(T_len * 0.0)

                        # Identify valid trajectories (Final Reward > 0)
                        # chunk_rewards is [B]
                        valid_indices = (chunk_rewards > 0).nonzero().squeeze(-1).tolist()

                        # Pre-allocate zero tensor [B, T]
                        dense_rewards_tensor = torch.zeros((batch_size, T_len), device=device, dtype=torch.float32)

                        if len(valid_indices) > 0:
                            # Only collect intermediate graphs for valid trajectories, starting from start_step
                            # This way rewards before start_step remain 0
                            valid_pred_graphs = []
                            valid_indices_flat_map = [] # Record (batch_idx, t_idx) correspondence

                            for idx in valid_indices:
                                # Only take t >= start_step
                                traj_graphs = trajectory_preds[idx]
                                for t_idx in range(start_step, T_len):
                                    if t_idx < len(traj_graphs):
                                        valid_pred_graphs.append(traj_graphs[t_idx])
                                        valid_indices_flat_map.append((idx, t_idx))

                            if valid_pred_graphs:
                                # Compute dense rewards for valid portion
                                flat_valid_rewards = self._compute_rewards_multiprocess_sync(
                                    valid_pred_graphs,
                                    timeout=1800,
                                    context="dense_sampling",
                                )

                                if flat_valid_rewards.numel() > 0:
                                    if flat_valid_rewards.device != device:
                                        flat_valid_rewards = flat_valid_rewards.to(device, non_blocking=True)

                                    # Fill back into dense_rewards_tensor
                                    # Use flat_valid_rewards and valid_indices_flat_map for assignment
                                    # This approach is safer than reshape since we only computed partial timesteps

                                    # For efficiency, convert the map to tensor indices first
                                    map_tensor = torch.tensor(valid_indices_flat_map, device=device, dtype=torch.long)
                                    batch_idxs = map_tensor[:, 0]
                                    t_idxs = map_tensor[:, 1]

                                    # Ensure lengths match
                                    count = min(flat_valid_rewards.numel(), map_tensor.shape[0])
                                    dense_rewards_tensor[batch_idxs[:count], t_idxs[:count]] = flat_valid_rewards[:count]

                # --- 3. Detailed Visualization Logging ---
                if should_log_vis and trajectory_probs is not None:
                     try:
                        self._log_detailed_visualization(
                            trajectory_states=trajectory_states,
                            trajectory_preds=trajectory_preds,
                            trajectory_probs=trajectory_probs,
                            dense_rewards=dense_rewards_tensor,
                            final_rewards=chunk_rewards,
                            log_dir=f"visualization_outputs/epoch_{self.epoch}_group_{group_ids[0]}",
                            batch_indices=[0, 1, 2, 3] # Log first 4 in the batch
                        )
                     except Exception as e:
                         logger.warning("Visualization logging failed: %s", e)

                # Prepare Data for Collection
                log_prob_cpu = current_log_prob.detach().cpu()

                vectorized_traj = self.grpo_core._vectorize_trajectories(
                    trajectory_states,
                    node_mask_cpu
                )
                # Record the global time index information for this trajectory, used to reconstruct
                # the same t_norm/s_norm during training-phase log_prob recomputation.
                # The training window is the last N steps, consistent with sample_graphs_with_trajectory_tracking.
                train_max_steps = None
                try:
                    train_max_steps = self.cfg.grpo.get("train_max_steps")
                except Exception:
                    train_max_steps = None
                if train_max_steps is None:
                    train_window_steps = int(self.sample_steps)
                else:
                    train_window_steps = max(1, min(int(train_max_steps), int(self.sample_steps)))
                train_start_step = int(self.sample_steps) - int(train_window_steps)
                vectorized_traj["trajectory_t_start"] = torch.full(
                    (batch_size,),
                    train_start_step,
                    dtype=torch.long,
                )
                vectorized_traj["trajectory_total_inference_steps"] = torch.full(
                    (batch_size,),
                    int(self.sample_steps),
                    dtype=torch.long,
                )
                for key in vectorized_traj:
                    vectorized_traj[key] = vectorized_traj[key].cpu()

                # Slice by batch, accumulate, and collect the whole batch
                new_batches = []
                for local_idx, group_id in enumerate(group_ids):
                    start = local_idx * self.group_size
                    end = start + self.group_size

                    batch_rewards = chunk_rewards[start:end]
                    group_log_probs = log_prob_cpu[start:end]
                    group_node_masks = node_mask_cpu[start:end]
                    group_vectorized = {
                        key: value[start:end]
                        for key, value in vectorized_traj.items()
                    }
                    graph_configs = [group_id for _ in range(self.group_size)]

                    batch_dense = None
                    if dense_rewards_tensor is not None:
                        batch_dense = dense_rewards_tensor[start:end].cpu()

                    new_batches.append({
                        "rewards": batch_rewards,
                        "old_log_probs": group_log_probs,
                        "graph_configs": graph_configs,
                        "node_masks": group_node_masks,
                        "trajectory_tensors": group_vectorized,
                        "dense_rewards": batch_dense,
                    })

                # Write to sample_buffer in one batch
                for batch in new_batches:
                    batch_rewards = batch["rewards"]
                    if batch_rewards.numel() == 0:
                        continue

                    # If all rewards in a group are -1 or 1, filter out the group from training
                    # Using exact comparison (reward function already clips); add atol for looser check if needed
                    all_neg_one = torch.all(batch_rewards == -1.0)
                    all_pos_one = torch.all(batch_rewards == 1.0)
                    all_zero = torch.all(
                        torch.isclose(
                            batch_rewards,
                            torch.zeros_like(batch_rewards),
                            atol=1e-8,
                            rtol=0.0,
                        )
                    )
                    if all_neg_one or all_pos_one or all_zero:
                        try:
                            group_id = batch["graph_configs"][0]
                        except Exception:
                            group_id = "unknown"
                        logger.info(
                            "Skipping group %s: all rewards equal to %.2f, excluded from GRPO training",
                            group_id, batch_rewards[0].item(),
                        )
                        continue

                    self.grpo_core.collect_trajectory_samples(
                        trajectories=None,
                        rewards=batch_rewards,
                        old_log_probs=batch["old_log_probs"],
                        graph_configs=batch["graph_configs"],
                        node_masks=batch["node_masks"],
                        trajectory_tensors=batch["trajectory_tensors"],
                        dense_rewards=batch.get("dense_rewards"),
                    )

                    # --- Debug Logging: Save one group per epoch ---
                    # User Request: Log reward and (Final - Current) for inspection
                    if getattr(self, '_last_reward_log_epoch', -1) < self.epoch:
                        dense_R = batch.get("dense_rewards")
                        final_R = batch_rewards
                        if dense_R is not None:
                            try:
                                import os
                                import csv
                                log_dir = "reward_logs"
                                os.makedirs(log_dir, exist_ok=True)

                                # Compute Diff: R_final - R_t
                                # dense_R: [B, T]
                                B, T_steps = dense_R.shape
                                final_R_cpu = final_R.cpu()
                                final_expanded = final_R_cpu.unsqueeze(1).expand(-1, T_steps)
                                diff_R = final_expanded - dense_R

                                dense_np = dense_R.detach().cpu().numpy()
                                final_np = final_R.detach().cpu().numpy()
                                diff_np = diff_R.detach().cpu().numpy()

                                save_path = f"{log_dir}/rewards_epoch_{self.epoch}.csv"
                                with open(save_path, "w", newline="") as f:
                                    writer = csv.writer(f)
                                    writer.writerow(["Epoch", "GlobalStep", "TrajID", "Step", "DenseReward", "FinalReward", "DiffReward"])

                                    for b in range(B):
                                        f_val = final_np[b]
                                        for t in range(T_steps):
                                            d_val = dense_np[b, t]
                                            diff_val = diff_np[b, t]
                                            writer.writerow([self.epoch, self.global_step, b, t, f"{d_val:.4f}", f"{f_val:.4f}", f"{diff_val:.4f}"])

                                logger.debug("Logged reward trajectories to %s", save_path)
                                self._last_reward_log_epoch = self.epoch
                            except Exception as e:
                                logger.debug("Failed to log rewards: %s", e)
                    total_rewards.append(batch_rewards)

                groups_collected += active_groups
                progress = groups_collected / total_groups
                logger.info(
                    "Sampling progress: %d/%d groups (%.1f%%)",
                    groups_collected, total_groups, progress * 100,
                )

            # Print sampling phase summary
            sample_time = time.time() - sample_start_time
            if total_rewards:
                all_rewards_tensor = torch.cat(total_rewards)
                mean_r = all_rewards_tensor.mean().item()
                std_r = all_rewards_tensor.std().item()
                min_r = all_rewards_tensor.min().item()
                max_r = all_rewards_tensor.max().item()
                gt0 = int((all_rewards_tensor > 0).sum().item())
                gt1e3 = int((all_rewards_tensor > 1e-3).sum().item())
                logger.info(
                    "Sampling phase completed: total=%d, mean=%.6f +/- %.6f, min/max=%.6f/%.6f, "
                    ">0: %d (%.2f%%), >1e-3: %d (%.2f%%), time=%.2fs",
                    len(all_rewards_tensor), mean_r, std_r, min_r, max_r,
                    gt0, gt0 / len(all_rewards_tensor) * 100,
                    gt1e3, gt1e3 / len(all_rewards_tensor) * 100,
                    sample_time,
                )

            # Extra debug: aggregate reward stats from all sampled graphs (including filtered groups)
            if debug_all_rewards:
                debug_all = torch.cat(debug_all_rewards)
                mean_r = debug_all.mean().item()
                std_r = debug_all.std().item()
                min_r = debug_all.min().item()
                max_r = debug_all.max().item()
                logger.info(
                    "[GRPO Debug] Molecular reward stats (all sampled graphs): "
                    "mean=%.4f, std=%.4f, min=%.4f, max=%.4f",
                    mean_r, std_r, min_r, max_r,
                )

            # [Dynamic p0 Update] Process buffer to filter Top 10%
            if p0_candidate_buffer:
                # Sort by reward descending
                p0_candidate_buffer.sort(key=lambda x: x[0], reverse=True)

                # Take top 10%
                top_k = max(1, int(len(p0_candidate_buffer) * 0.1))
                top_samples = p0_candidate_buffer[:top_k]

                # Populate counters
                for _, at, et in top_samples:
                    # at/et are expected to be flat lists of indices
                    if isinstance(at, torch.Tensor):
                        at = at.detach().cpu().flatten().tolist()
                    if isinstance(et, torch.Tensor):
                        et = et.detach().cpu().flatten().tolist()
                    if isinstance(at, (list, tuple)):
                        epoch_node_counts.update([int(x) for x in at])
                    if isinstance(et, (list, tuple)):
                        epoch_edge_counts.update([int(x) for x in et])

                avg_top_r = sum(s[0] for s in top_samples) / len(top_samples)
                logger.info("Dynamic p0: Selected top %d/%d samples (Avg R: %.4f)", top_k, len(p0_candidate_buffer), avg_top_r)

            # [Stat Tracking] Print Distribution Stats
            def log_dist(name, counter):
                total = sum(counter.values())
                if total == 0:
                    logger.debug("[Epoch %d] %s Dist: (Empty)", self.epoch, name)
                    return
                sorted_keys = sorted(counter.keys())
                dist_str = ", ".join([f"{k}: {v/total:.2%}" for k, v in counter.items() if k in sorted_keys])
                logger.debug("[Epoch %d] %s Dist (N=%d): %s", self.epoch, name, total, dist_str)

            log_dist("Node", epoch_node_counts)
            log_dist("Edge", epoch_edge_counts)

            # [Dynamic p0 Update] Global Top-K Strategy
            if getattr(self.cfg.grpo, 'enable_dynamic_p0', False) and p0_candidate_buffer:
                # 1. Merge new candidates into global buffer
                self.global_p0_buffer.extend(p0_candidate_buffer)

                # 2. Sort and Keep Top-K (Global "Golden Gate")
                # Sort descending by reward
                self.global_p0_buffer.sort(key=lambda x: x[0], reverse=True)

                # Truncate
                if len(self.global_p0_buffer) > self.p0_buffer_size:
                    self.global_p0_buffer = self.global_p0_buffer[:self.p0_buffer_size]

                logger.info(
                    "[Dynamic p0] Global Buffer Size: %d. Best Reward: %.4f, Worst in Buffer: %.4f",
                    len(self.global_p0_buffer), self.global_p0_buffer[0][0], self.global_p0_buffer[-1][0],
                )

                # 3. Aggregate statistics from Global Buffer (NOT local batch)
                # Re-calculate counts from the global buffer
                global_node_counts = Counter()
                global_edge_counts = Counter()

                # Use all samples in the buffer (they are all high quality)
                for _, at, et in self.global_p0_buffer:
                    if isinstance(at, torch.Tensor): at = at.tolist()
                    if isinstance(et, torch.Tensor): et = et.tolist()

                    # Flatten if necessary
                    if isinstance(at, list):
                        for a in at: global_node_counts[a] += 1
                    else:
                         global_node_counts[at] += 1

                    if isinstance(et, list):
                        for e in et: global_edge_counts[e] += 1
                    else:
                        global_edge_counts[et] += 1

                logger.debug("Check Dynamic p0 Update... (Valid atoms in Global Buffer: %d)", sum(global_node_counts.values()))

                # 4. Fetch current p0 to determine correct shape
                curr_node_dist = self.core_model.limit_dist.X.to(device)
                curr_edge_dist = self.core_model.limit_dist.E.to(device)

                dx = curr_node_dist.shape[0]
                de = curr_edge_dist.shape[0]

                # 5. Compute new distributions from Global Buffer
                new_node_dist = torch.zeros(dx, device=device)
                for k, v in global_node_counts.items():
                     if k < dx: new_node_dist[k] = v

                new_edge_dist = torch.zeros(de, device=device)
                for k, v in global_edge_counts.items():
                     if k < de: new_edge_dist[k] = v

                # Normalize
                if new_node_dist.sum() > 0:
                     new_node_dist = new_node_dist / new_node_dist.sum()
                else:
                     new_node_dist = curr_node_dist

                if new_edge_dist.sum() > 0:
                     new_edge_dist = new_edge_dist / new_edge_dist.sum()
                else:
                     new_edge_dist = curr_edge_dist

                # 6. Smooth Update
                alpha = self.cfg.grpo.get('dynamic_p0_alpha', 0.05)

                updated_node_dist = (1 - alpha) * curr_node_dist + alpha * new_node_dist
                updated_edge_dist = (1 - alpha) * curr_edge_dist + alpha * new_edge_dist

                # 7. Apply to model -> DELAYED to end of epoch
                self._pending_p0_update = (updated_node_dist, updated_edge_dist)

                # Log changes (Preview)
                safe_curr = curr_node_dist + 1e-9
                safe_updated = updated_node_dist + 1e-9
                kl_node = (safe_updated * (safe_updated.log() - safe_curr.log())).sum().item()
                logger.info("Pending Global p0 Update (alpha=%s). KL(new||old): %.6f", alpha, kl_node)

            # [Dynamic node-dist] Global Top-K strategy (p0-like), persisted in model buffers.
            if self.enable_dynamic_node_dist and node_count_candidate_buffer:
                if self.target_node_count is not None:
                    logger.debug("[Dynamic node-dist] target_node_count is set to a fixed value, skipping online node_dist update.")
                elif not hasattr(self.core_model, "node_count_prob"):
                    logger.debug("[Dynamic node-dist] core_model lacks node_count_prob buffer, skipping online node_dist update.")
                else:
                    rewards_new = torch.tensor([r for r, _ in node_count_candidate_buffer], dtype=torch.float32)
                    nodes_new = torch.tensor([n for _, n in node_count_candidate_buffer], dtype=torch.long)

                    # Merge into persistent Top-K buffers.
                    buf_rewards = self.core_model.node_count_buffer_rewards.detach().cpu()
                    buf_nodes = self.core_model.node_count_buffer_nodes.detach().cpu()
                    filled = int(self.core_model.node_count_buffer_filled.detach().cpu().item())
                    filled = max(0, min(filled, int(buf_rewards.numel())))

                    if filled > 0:
                        rewards_all = torch.cat([buf_rewards[:filled], rewards_new], dim=0)
                        nodes_all = torch.cat([buf_nodes[:filled], nodes_new], dim=0)
                    else:
                        rewards_all = rewards_new
                        nodes_all = nodes_new

                    k = min(int(buf_rewards.numel()), int(rewards_all.numel()))
                    if k <= 0:
                        logger.debug("[Dynamic node-dist] empty buffer after merge; skip update.")
                        k = 0
                    if k == 0:
                        # Nothing to update.
                        pass
                    else:
                        top_idx = torch.topk(rewards_all, k=k, largest=True).indices
                        top_rewards = rewards_all[top_idx]
                        top_nodes = nodes_all[top_idx]

                        # Write back buffers (keep on model device).
                        self.core_model.node_count_buffer_rewards.fill_(-1e9)
                        self.core_model.node_count_buffer_nodes.zero_()
                        self.core_model.node_count_buffer_rewards[:k].copy_(top_rewards.to(self.core_model.node_count_buffer_rewards.device))
                        self.core_model.node_count_buffer_nodes[:k].copy_(top_nodes.to(self.core_model.node_count_buffer_nodes.device))
                        self.core_model.node_count_buffer_filled.fill_(int(k))

                        # Build new node-count distribution from Top-K node counts.
                        curr_prob = self.core_model.node_count_prob.detach().cpu().to(dtype=torch.float32)
                        new_prob = torch.zeros_like(curr_prob)

                        min_n = int(self.node_count_min) if self.node_count_min is not None else None
                        max_n = int(self.node_count_max) if self.node_count_max is not None else None
                        for n in top_nodes.tolist():
                            n_int = int(n)
                            if n_int < 0 or n_int >= int(new_prob.numel()):
                                continue
                            if min_n is not None and n_int < min_n:
                                continue
                            if max_n is not None and n_int > max_n:
                                continue
                            new_prob[n_int] += 1.0

                        if float(new_prob.sum().item()) <= 0.0:
                            logger.debug("[Dynamic node-dist] Top-K node_count histogram is empty after bounds; skip update.")
                        else:
                            new_prob = new_prob / new_prob.sum()
                            alpha = max(0.0, min(1.0, float(self.dynamic_node_dist_alpha)))
                            updated = (1.0 - alpha) * curr_prob + alpha * new_prob
                            # Enforce bounds again and renormalize.
                            if min_n is not None:
                                updated[:min_n] = 0.0
                            if max_n is not None and max_n + 1 < int(updated.numel()):
                                updated[max_n + 1 :] = 0.0
                            if float(updated.sum().item()) > 0.0:
                                updated = updated / updated.sum()
                                self._pending_node_count_prob_update = updated
                                logger.info(
                                    "Pending node_dist update (alpha=%s): support=[%s,%s], filled=%d, best_reward=%.4f",
                                    alpha, min_n, max_n, k, float(top_rewards.max().item()),
                                )



            # Use the same visualization tools as test.only, save a small subset of GRPO-sampled graphs
            if debug_molecules and bool(self.cfg.grpo.get("enable_visualization", True)):
                try:
                    viz = getattr(self.core_model, "visualization_tools", None)
                    if viz is not None:
                        import os
                        result_path = os.path.join(
                            os.getcwd(),
                            f"graphs/grpo_sampling_debug/epoch{self.epoch}",
                        )
                        num_to_vis = 5
                        logger.debug(
                            "[GRPO Debug] Visualizing %d sampled graphs to %s",
                            num_to_vis, result_path,
                        )
                        viz.visualize(result_path, debug_molecules, num_to_vis)
                except Exception as e:
                    logger.warning("[GRPO Debug] visualization failed: %s", e)
        finally:
            self.core_model.eval()

    def sample_graphs_with_trajectory_tracking(
        self,
        batch_size: int,
        seed: Optional[int] = None,
        total_inference_steps: int = 50,
        force_same_start: bool = True,
        group_size_for_same_start: Optional[int] = None,
        return_probs: bool = False,
    ) -> Tuple:
        """
        Sample graphs and retain trajectory information for subsequent training.
        Caller must ensure model is in eval mode (sampling_phase handles this automatically).
        """
        import random
        from flow_matching import flow_matching_utils

        try:
            if seed is not None:
                torch.manual_seed(seed)
                np.random.seed(seed % (2**31))
                random.seed(seed)

            device = next(self.core_model.parameters()).device
            use_sampling_autocast = True
            try:
                use_sampling_autocast = (
                    device.type == "cuda"
                    and bool(self.cfg.grpo.get("sampling_autocast", False))
                )
            except Exception:
                use_sampling_autocast = False
            sampling_autocast_ctx = (
                autocast(enabled=True, dtype=torch.bfloat16)
                if use_sampling_autocast
                else nullcontext()
            )

            sampling_start_step = 0
            if force_same_start and group_size_for_same_start is not None:
                if batch_size % group_size_for_same_start != 0:
                    raise ValueError("batch_size must be divisible by group_size_for_same_start when force_same_start=True")
                num_groups = batch_size // group_size_for_same_start

                group_nodes = torch.zeros((num_groups,), device=device, dtype=torch.long)
                for g in range(num_groups):
                    if self.target_node_count is not None:
                        group_nodes[g] = int(self.target_node_count)
                    else:
                        node_count = self.core_model.node_dist.sample_n(1, device=device).long()[0]
                        if self.node_count_min is not None:
                            node_count = torch.clamp(node_count, min=int(self.node_count_min))
                        if self.node_count_max is not None:
                            node_count = torch.clamp(node_count, max=int(self.node_count_max))
                        group_nodes[g] = int(node_count)

                n_max = int(group_nodes.max().item())
                arange = torch.arange(n_max, device=device).unsqueeze(0).expand(num_groups, -1)
                base_masks = arange < group_nodes.unsqueeze(1)
                node_mask = base_masks.repeat_interleave(group_size_for_same_start, dim=0)

                X_blocks = []
                E_blocks = []
                y_blocks = [] if self.core_model.conditional else None
                noise_dist = self.core_model.noise_dist.get_limit_dist()
                for g in range(num_groups):
                    single_mask = base_masks[g:g+1]
                    z_single = flow_matching_utils.sample_discrete_feature_noise(
                        limit_dist=noise_dist,
                        node_mask=single_mask
                    )
                    X_single = z_single.X
                    E_single = z_single.E
                    if self.core_model.conditional:
                        y_single = torch.zeros(1, 1, device=device)
                    else:
                        y_single = torch.zeros(1, 0, device=device)

                    X_blocks.append(X_single.repeat(group_size_for_same_start, 1, 1))
                    E_blocks.append(E_single.repeat(group_size_for_same_start, 1, 1, 1))
                    if self.core_model.conditional:
                        y_blocks.append(y_single.repeat(group_size_for_same_start, 1))

                X = torch.cat(X_blocks, dim=0)
                E = torch.cat(E_blocks, dim=0)
                if self.core_model.conditional:
                    y = torch.cat(y_blocks, dim=0)
                else:
                    y = torch.zeros(batch_size, 0, device=device)


            else:
                # When not forcing shared start within group, determine node count independently per sample:
                # Use fixed target_node_count if set, otherwise sample per-graph from distribution
                if self.target_node_count is not None:
                    n_nodes = torch.full(
                        (batch_size,),
                        int(self.target_node_count),
                        device=device,
                        dtype=torch.long,
                    )
                else:
                    n_nodes = self.core_model.node_dist.sample_n(
                        batch_size, device=device
                    ).long()
                    if self.node_count_min is not None:
                        n_nodes = torch.clamp(n_nodes, min=int(self.node_count_min))
                    if self.node_count_max is not None:
                        n_nodes = torch.clamp(n_nodes, max=int(self.node_count_max))
                n_max = int(n_nodes.max().item())
                arange = torch.arange(n_max, device=device).unsqueeze(0).expand(batch_size, -1)
                node_mask = arange < n_nodes.unsqueeze(1)
                z_T = flow_matching_utils.sample_discrete_feature_noise(
                    limit_dist=self.core_model.noise_dist.get_limit_dist(),
                    node_mask=node_mask
                )
                if self.core_model.conditional:
                    z_T.y = torch.zeros(batch_size, 1).to(device)
                X, E, y = z_T.X, z_T.E, z_T.y
                if not self.core_model.conditional and y is None:
                    y = torch.zeros(batch_size, 0, device=device)

            # Store trajectories - organized by sample: [batch_sizetrajectories, each trajectory contains time_steps states]
            trajectory_states = [[] for _ in range(batch_size)]
            # Store actual log probabilities from sampling - per batch
            trajectory_log_probs = []

            # Whether to collect per-step predicted clean graphs (for dense reward or visualization)
            use_dense_reward = False
            try:
                use_dense_reward = bool(self.cfg.grpo.get("use_dense_reward", False))
            except Exception:
                use_dense_reward = False

            # Visualization only needs a few samples; avoid excessive CPU/GPU memory and sync overhead for large batch_size
            vis_num_samples = 1
            try:
                vis_num_samples = int(self.cfg.grpo.get("vis_num_samples", 1))
            except Exception:
                vis_num_samples = 1
            vis_num_samples = max(1, min(int(batch_size), int(vis_num_samples)))

            trajectory_preds = None
            if use_dense_reward:
                trajectory_preds = [[] for _ in range(batch_size)]
            elif return_probs:
                trajectory_preds = [[] for _ in range(vis_num_samples)]

            # Store predicted probability distributions (visualization only; keep only a few samples)
            trajectory_probs = [[] for _ in range(vis_num_samples)] if return_probs else None

            # Training window (optional): only train on the last N steps, to reduce memory / speed up.
            # Note: "steps" here refers to the number of log_prob time steps, so states length = steps + 1.
            train_max_steps = None
            try:
                train_max_steps = self.cfg.grpo.get("train_max_steps")
            except Exception:
                train_max_steps = None

            if train_max_steps is None:
                train_window_steps = int(total_inference_steps) - int(sampling_start_step)
            else:
                train_window_steps = max(
                    1,
                    min(
                        int(train_max_steps),
                        int(total_inference_steps) - int(sampling_start_step),
                    ),
                )

            train_start_step = int(total_inference_steps) - int(train_window_steps)
            if train_start_step < sampling_start_step:
                train_start_step = int(sampling_start_step)

            # Inference loop
            for t_int in range(int(sampling_start_step), total_inference_steps):
                t_array = t_int * torch.ones((batch_size, 1)).type_as(X)
                t_norm = t_array / (total_inference_steps + 1)
                s_array = t_array + 1
                s_norm = s_array / (total_inference_steps + 1)

                # Apply time distortion
                t_norm = self.core_model.time_distorter.sample_ft(
                    t_norm, self.cfg.sample.time_distortion
                )
                s_norm = self.core_model.time_distorter.sample_ft(
                    s_norm, self.cfg.sample.time_distortion
                )

                # Save current state (before transition)
                # Only record the last-N-steps window: t_int in [train_start_step, total_inference_steps-1]
                if t_int >= train_start_step:
                    for i in range(batch_size):
                        sample_state = utils.PlaceHolder(
                            X=X[i:i+1].clone(),
                            E=E[i:i+1].clone(),
                            y=y[i:i+1] if y is not None else None,
                        )
                        trajectory_states[i].append(sample_state)

                with torch.inference_mode(), sampling_autocast_ctx:
                    # Forward pass (FP32 or BF16 autocast, controlled by sampling_autocast)
                    noisy_data = {
                        "X_t": X, "E_t": E, "y_t": y,
                        "t": t_norm, "node_mask": node_mask
                    }

                    extra_data = self.core_model.compute_extra_data(noisy_data)
                    pred = self.core_model.forward(noisy_data, extra_data, node_mask)

                    # Compute probabilities (Apply Temperature Here)
                    sampling_temperature = self.cfg.grpo.get('sampling_temperature', 1.0)

                    # pred.X/pred.E are logits
                    if abs(sampling_temperature - 1.0) > 1e-5:
                        pred_X = F.softmax(pred.X / sampling_temperature, dim=-1)
                        pred_E = F.softmax(pred.E / sampling_temperature, dim=-1)
                    else:
                        pred_X = F.softmax(pred.X, dim=-1)
                        pred_E = F.softmax(pred.E, dim=-1)

                    if return_probs and t_int >= train_start_step:
                        for i in range(vis_num_samples):
                            n_nodes_i = int(node_mask[i].sum().item())
                            if n_nodes_i > 0:
                                trajectory_probs[i].append(
                                    (
                                        pred_X[i, :n_nodes_i].detach().cpu(),
                                        pred_E[i, :n_nodes_i, :n_nodes_i].detach().cpu(),
                                    )
                                )
                            else:
                                trajectory_probs[i].append(None)

                        # --- Dense Reward: sample from p(z1 | z_t) ---
                    if trajectory_preds is not None and t_int >= train_start_step:
                        # Sample discrete graphs for Dense Reward (and visualization)
                        # Use multinomial sampling (Categorical) based on temperature-scaled probs
                        sampled_indices_X = torch.distributions.Categorical(probs=pred_X).sample()
                        pred_X_discrete = sampled_indices_X

                        sampled_indices_E = torch.distributions.Categorical(probs=pred_E).sample()
                        pred_E_discrete = sampled_indices_E  # [Batch, N, N]

                        # Apply mask
                        pred_X_discrete = pred_X_discrete * node_mask

                        # E mask (including diagonal)
                        edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                        diag_indices = torch.arange(node_mask.size(1), device=node_mask.device)
                        edge_mask[:, diag_indices, diag_indices] = 0

                        # Symmetrize E (ensure undirected graph property for sampled edges)
                        pred_E_discrete = torch.triu(pred_E_discrete, diagonal=1)
                        pred_E_discrete = pred_E_discrete + pred_E_discrete.transpose(1, 2)
                        pred_E_discrete = pred_E_discrete * edge_mask

                        if use_dense_reward:
                            target_samples = batch_size
                        else:
                            target_samples = vis_num_samples

                        for i in range(target_samples):
                            n_nodes_i = int(node_mask[i].sum().item())
                            if n_nodes_i <= 0:
                                trajectory_preds[i].append(
                                    (
                                        torch.empty(0, dtype=torch.long),
                                        torch.empty(0, 0, dtype=torch.long),
                                    )
                                )
                                continue
                            trajectory_preds[i].append(
                                (
                                    pred_X_discrete[i, :n_nodes_i].detach().cpu(),
                                    pred_E_discrete[i, :n_nodes_i, :n_nodes_i].detach().cpu(),
                                )
                            )

                    # --- Transition sampling ---
                    dt = (s_norm - t_norm)[:, 0]
                    rate_designer = self.core_model.get_rate_matrix_designer() if hasattr(self.core_model, "get_rate_matrix_designer") else self.core_model.rate_matrix_designer
                    R_t_X, R_t_E = rate_designer.compute_graph_rate_matrix(
                        t_norm, node_mask, (X, E), (pred_X, pred_E)
                    )

                    limit_x = self.core_model.limit_dist.X
                    limit_e = self.core_model.limit_dist.E
                    if self.use_grpo_step_probs_for_sampling:
                        prob_X, prob_E = self.core_model.compute_step_probs_grpo(
                            R_t_X, R_t_E, X, E, dt, limit_x, limit_e
                        )
                    else:
                        prob_X, prob_E = self.core_model.compute_step_probs(
                            R_t_X, R_t_E, X, E, dt, limit_x, limit_e
                        )

                    sampled_s = flow_matching_utils.sample_discrete_features(
                        prob_X, prob_E, node_mask=node_mask
                    )

                    # Compute log probability for this step immediately after sampling
                    X_next = F.one_hot(sampled_s.X, num_classes=X.size(-1)).float()
                    E_next = F.one_hot(sampled_s.E, num_classes=E.size(-1)).float()
                    # Align with sample_batch path: zero out padding positions before next step to prevent feature/rate contamination from padding noise
                    X_next = X_next * node_mask.unsqueeze(-1)
                    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                    E_next = E_next * edge_mask.unsqueeze(-1)

                    # E is already symmetrized in sample_discrete_features, no further processing needed
                    # Directly use sampled indices
                    X_indices = sampled_s.X  # (batch_size, N)
                    E_indices = sampled_s.E  # (batch_size, N, N)

                    # Node log probability
                    X_log_probs = torch.log(prob_X.clamp(min=1e-8))
                    X_step_log_prob = torch.gather(X_log_probs, dim=-1,
                                                    index=X_indices.unsqueeze(-1)).squeeze(-1)
                    X_masked = (X_step_log_prob * node_mask).sum(dim=-1)

                    # Edge log probability
                    E_log_probs = torch.log(prob_E.clamp(min=1e-8))
                    E_step_log_prob = torch.gather(E_log_probs, dim=-1,
                                                    index=E_indices.unsqueeze(-1)).squeeze(-1)
                    edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
                    diag_indices = torch.arange(node_mask.size(1), device=node_mask.device)
                    edge_mask[:, diag_indices, diag_indices] = 0
                    E_masked = (E_step_log_prob * edge_mask).sum(dim=[-2, -1]) * 0.5

                    # Merge log probabilities - only keep the last-N-steps window for training
                    step_log_prob = X_masked + E_masked  # (batch_size,)
                    if t_int >= train_start_step:
                        trajectory_log_probs.append(step_log_prob)


                    # Update state
                    X = X_next
                    E = E_next

                    # Symmetry is already handled above, no need to process again

            # Append final state to ensure trajectory length reaches train_window_steps + 1 (corresponding to train_window_steps transitions)
            for i in range(batch_size):
                sample_state = utils.PlaceHolder(
                    X=X[i:i+1].clone(),
                    E=E[i:i+1].clone(),
                    y=y[i:i+1] if y is not None else None,
                )
                trajectory_states[i].append(sample_state)

            # Trajectory states handled in loop and supplement logic
            # Verify trajectory length
            if len(trajectory_states) > 0 and len(trajectory_states[0]) > 0:
                actual_traj_len = len(trajectory_states[0])
                expected_traj_len = int(train_window_steps) + 1
                if actual_traj_len != expected_traj_len:
                    logger.warning("Trajectory length mismatch. Actual: %d, Expected: %d", actual_traj_len, expected_traj_len)

            # Clean up virtual categories
            X, E, y = self.core_model.noise_dist.ignore_virtual_classes(X, E, y)
            clean_graphs = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)

            # Stack recorded log_probs into tensor [batch_size, num_steps]
            if trajectory_log_probs:
                current_log_probs = torch.stack(trajectory_log_probs, dim=1)  # [batch_size, num_steps]
            else:
                current_log_probs = torch.zeros(batch_size, 0, device=device)

            ref_log_probs = None

            return clean_graphs, node_mask, current_log_probs, ref_log_probs, trajectory_states, trajectory_preds, trajectory_probs

        finally:
            pass

    @torch.no_grad()
    def refine_candidate_via_denoising(
        self,
        init_X: torch.Tensor,
        init_E: torch.Tensor,
        num_variations: int,
        noise_fraction: "Union[float, torch.Tensor]",
        total_inference_steps: Optional[int] = None,
        seed: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Refine a candidate graph by re-noising it to an intermediate time and running the
        same denoising loop as `sample_graphs_with_trajectory_tracking`.

        This keeps refinement aligned with GRPO's sampling/denoising behavior:
        - same time grid and time distortion
        - same temperature scaling

        Args:
            init_X/init_E: single-graph tensors (indices or one-hot), shapes:
                - X: (N,) or (N, Dx)
                - E: (N, N) or (N, N, De)
            num_variations: number of refined samples to generate from this candidate
            noise_fraction: how much noise to add before denoising.
                - float in [0, 1]: same noise level for every variation
                - Tensor shape (B,) or (B,1): per-variation noise levels in [0, 1]
            total_inference_steps: number of denoising steps; defaults to `self.sample_steps`
            seed: optional RNG seed for reproducibility

        Returns:
            refined_X/refined_E as discrete indices:
                - refined_X: (B, N)
                - refined_E: (B, N, N)
        """
        import random
        from flow_matching import flow_matching_utils

        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed % (2**31))
            random.seed(seed)

        device = next(self.core_model.parameters()).device
        steps = int(total_inference_steps if total_inference_steps is not None else self.sample_steps)
        if steps <= 0:
            raise ValueError(f"total_inference_steps must be > 0, got {steps}")
        if num_variations <= 0:
            raise ValueError(f"num_variations must be > 0, got {num_variations}")

        if torch.is_tensor(noise_fraction):
            noise_frac = noise_fraction.detach().to(device=device, dtype=torch.float32)
            if noise_frac.dim() == 2 and noise_frac.shape[1] == 1:
                noise_frac = noise_frac[:, 0]
            if noise_frac.dim() != 1 or noise_frac.shape[0] != num_variations:
                raise ValueError(
                    f"noise_fraction tensor must have shape ({num_variations},) or ({num_variations},1), "
                    f"got {tuple(noise_frac.shape)}"
                )
            noise_frac = noise_frac.clamp(0.0, 1.0)
        else:
            nf = max(0.0, min(1.0, float(noise_fraction)))
            noise_frac = torch.full((num_variations,), nf, device=device, dtype=torch.float32)

        x = init_X.to(device)
        e = init_E.to(device)

        # Build a single-graph node mask (support both trimmed and padded graphs).
        if x.dim() == 1:
            node_mask_1 = (x >= 0)
            x_idx_1 = x.long()
        elif x.dim() == 2:
            node_mask_1 = (x.sum(dim=-1) > 0)
            x_idx_1 = x.argmax(dim=-1).long()
        else:
            raise ValueError(f"Unsupported init_X shape: {tuple(x.shape)}")

        if e.dim() == 2:
            e_idx_1 = e.long()
        elif e.dim() == 3:
            e_idx_1 = e.argmax(dim=-1).long()
        else:
            raise ValueError(f"Unsupported init_E shape: {tuple(e.shape)}")

        # Replicate candidate `num_variations` times and convert to one-hot for `apply_noise`.
        input_dims = getattr(self.core_model, "input_dims", None)
        if not isinstance(input_dims, dict) or "X" not in input_dims or "E" not in input_dims:
            raise RuntimeError("core_model.input_dims is missing; cannot one-hot encode candidate graph")

        x_idx = x_idx_1.unsqueeze(0).expand(num_variations, -1).contiguous()
        e_idx = e_idx_1.unsqueeze(0).expand(num_variations, -1, -1).contiguous()
        node_mask = node_mask_1.unsqueeze(0).expand(num_variations, -1).contiguous().bool()
        x_idx = x_idx.masked_fill(~node_mask, -1)
        edge_mask_idx = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        e_idx = e_idx.masked_fill(~edge_mask_idx, -1)

        X = F.one_hot(x_idx.clamp(min=0), num_classes=int(input_dims["X"])).float()
        E = F.one_hot(e_idx.clamp(min=0), num_classes=int(input_dims["E"])).float()

        X = X * node_mask.unsqueeze(-1)
        E = E * edge_mask_idx.unsqueeze(-1)

        if self.core_model.conditional:
            y = torch.zeros(num_variations, 1, device=device)
        else:
            y = torch.zeros(num_variations, 0, device=device)

        if float(noise_frac.max().item()) <= 0.0:
            return x_idx, e_idx

        # Per-variation starting steps based on desired noise levels.
        # Start at t_start = 1 - noise_fraction, mapped onto the sampler's grid t=i/(steps+1).
        t_start = 1.0 - noise_frac  # (B,)
        start_steps = torch.round(t_start * float(steps + 1)).to(dtype=torch.long)  # (B,)
        start_steps = start_steps.clamp(0, steps - 1)
        min_start_step = int(start_steps.min().item())

        t_raw = (start_steps.to(dtype=torch.float32) / float(steps + 1)).view(num_variations, 1)
        t_noisy = self.core_model.time_distorter.sample_ft(t_raw, self.cfg.sample.time_distortion)

        noisy_data = self.core_model.apply_noise(X, E, y, node_mask, t=t_noisy)
        X = noisy_data["X_t"]
        E = noisy_data["E_t"]
        y = noisy_data["y_t"]

        sampling_temperature = float(self.cfg.grpo.get("sampling_temperature", 1.0))

        for t_int in range(min_start_step, steps):
            t_array = torch.full((num_variations, 1), float(t_int), device=device, dtype=X.dtype)
            t_norm = t_array / (steps + 1)
            s_array = t_array + 1
            s_norm = s_array / (steps + 1)

            t_norm = self.core_model.time_distorter.sample_ft(t_norm, self.cfg.sample.time_distortion)
            s_norm = self.core_model.time_distorter.sample_ft(s_norm, self.cfg.sample.time_distortion)

            noisy_data = {"X_t": X, "E_t": E, "y_t": y, "t": t_norm, "node_mask": node_mask}
            extra_data = self.core_model.compute_extra_data(noisy_data)
            pred = self.core_model.forward(noisy_data, extra_data, node_mask)

            if abs(sampling_temperature - 1.0) > 1e-5:
                pred_X = F.softmax(pred.X / sampling_temperature, dim=-1)
                pred_E = F.softmax(pred.E / sampling_temperature, dim=-1)
            else:
                pred_X = F.softmax(pred.X, dim=-1)
                pred_E = F.softmax(pred.E, dim=-1)

            dt = (s_norm - t_norm)[:, 0]
            rate_designer = (
                self.core_model.get_rate_matrix_designer()
                if hasattr(self.core_model, "get_rate_matrix_designer")
                else self.core_model.rate_matrix_designer
            )
            R_t_X, R_t_E = rate_designer.compute_graph_rate_matrix(
                t_norm, node_mask, (X, E), (pred_X, pred_E)
            )

            limit_x = self.core_model.limit_dist.X
            limit_e = self.core_model.limit_dist.E
            if self.use_grpo_step_probs_for_sampling:
                prob_X, prob_E = self.core_model.compute_step_probs_grpo(
                    R_t_X, R_t_E, X, E, dt, limit_x, limit_e
                )
            else:
                prob_X, prob_E = self.core_model.compute_step_probs(
                    R_t_X, R_t_E, X, E, dt, limit_x, limit_e
                )

            sampled_s = flow_matching_utils.sample_discrete_features(prob_X, prob_E, node_mask=node_mask)

            X_next = F.one_hot(sampled_s.X, num_classes=X.size(-1)).float()
            E_next = F.one_hot(sampled_s.E, num_classes=E.size(-1)).float()

            X_next = X_next * node_mask.unsqueeze(-1)
            edge_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
            E_next = E_next * edge_mask.unsqueeze(-1)

            # Only advance samples whose denoising has started.
            active = (t_int >= start_steps).view(num_variations, 1, 1)
            active_E = active.view(num_variations, 1, 1, 1)
            X = torch.where(active, X_next, X)
            E = torch.where(active_E, E_next, E)

        X, E, _ = self.core_model.noise_dist.ignore_virtual_classes(X, E, y)
        clean_graphs = utils.PlaceHolder(X=X, E=E, y=y).mask(node_mask, collapse=True)
        return clean_graphs.X, clean_graphs.E
