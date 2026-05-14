"""Mixin for creating, loading, and updating the GRPO reference model."""

import logging
import os

import torch

from graph_discrete_flow_model import GraphDiscreteFlowModel

logger = logging.getLogger(__name__)


class ReferenceModelMixin:
    """Methods for managing the frozen reference model used in KL
    regularization during GRPO training."""

    def _initialize_training_components(self):
        """Initialize training components (reference model, etc.)."""
        # Get the base model
        self.core_model = self.model

        # Ensure all main model parameters require gradients
        for param in self.core_model.parameters():
            param.requires_grad = True
        # Create reference model
        self.reference_model = None
        self._create_reference_model()

    def _create_reference_model(self):
        """Create reference model (for KL regularization)."""
        if self.beta == 0:
            return

        device = next(self.model.parameters()).device

        # Create new model instance
        self.reference_model = GraphDiscreteFlowModel(cfg=self.cfg, **self.model_kwargs).to(device)

        # Preferentially load reference model from pretrained checkpoint
        loaded_from_pretrained = False
        ref_ckpt_path = self.cfg.grpo.get('pretrained_checkpoint')
        if ref_ckpt_path:
            ref_ckpt_path = os.path.expanduser(ref_ckpt_path)
            if os.path.exists(ref_ckpt_path):
                loaded_from_pretrained = self._load_reference_model_from_checkpoint(ref_ckpt_path)
            else:
                logger.warning("Reference model checkpoint not found: %s", ref_ckpt_path)

        if not loaded_from_pretrained:
            with torch.no_grad():
                state_dict = self.core_model.state_dict()
                self.reference_model.load_state_dict(state_dict)

        # If online training updates p0/node-count distributions (buffers) during training,
        # keep reference model's distributions aligned with current model. Otherwise KL may
        # jump on resume purely due to distribution buffers being restored while the reference
        # still uses the default dataset initialization (common when pretrained ckpt lacks these buffers).
        try:
            sync_p0 = bool(self.cfg.grpo.get("reference_sync_p0", True))
        except Exception:
            sync_p0 = True

        if sync_p0:
            try:
                if (
                    hasattr(self.core_model, "p0_node_dist")
                    and hasattr(self.core_model, "p0_edge_dist")
                    and hasattr(self.reference_model, "update_limit_dist")
                ):
                    self.reference_model.update_limit_dist(
                        self.core_model.p0_node_dist,
                        self.core_model.p0_edge_dist,
                    )
                if (
                    hasattr(self.core_model, "node_count_prob")
                    and hasattr(self.reference_model, "update_node_count_dist")
                ):
                    self.reference_model.update_node_count_dist(self.core_model.node_count_prob)
            except Exception as e:
                logger.warning("Failed to sync reference model distributions (p0/node_count): %s", e)

        # Freeze parameters (only freeze reference model, not the main model)
        for param in self.reference_model.parameters():
            param.requires_grad = False

        self.reference_model.eval()

    def _load_reference_model_from_checkpoint(self, ckpt_path: str) -> bool:
        """Load reference model weights from checkpoint."""
        try:
            checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            state_dict = checkpoint.get('state_dict', checkpoint)
            if not isinstance(state_dict, dict):
                logger.warning("Reference model checkpoint format unexpected (not dict state_dict): %s", type(state_dict))
                return False

            # Auto-remap common Lightning/DDP prefixes:
            # - GraphDiscreteFlowModel ckpt keys: `model.*`, `_device_buffer`, ...
            # - GRPOLightningModule ckpt keys: `model.model.*` (outer `model.` + inner GraphDiscreteFlowModel `model.*`)
            # - DDP: may include `module.*`
            model_state = self.reference_model.state_dict()
            model_keys = set(model_state.keys())

            def _score_strip_prefix(prefix: str) -> int:
                if not prefix:
                    return sum(1 for k in state_dict.keys() if k in model_keys)
                n = 0
                for k in state_dict.keys():
                    if isinstance(k, str) and k.startswith(prefix):
                        if k[len(prefix):] in model_keys:
                            n += 1
                return n

            prefixes = [
                "",
                "model.",
                "module.",
                "module.model.",
                "model.module.",
                "module.model.model.",
                "model.model.",
            ]
            best_prefix = max(prefixes, key=_score_strip_prefix)
            best_match = _score_strip_prefix(best_prefix)

            # Only strip when it meaningfully improves match.
            if best_prefix and best_match > 0:
                remapped = {}
                for k, v in state_dict.items():
                    new_k = k[len(best_prefix):] if isinstance(k, str) and k.startswith(best_prefix) else k
                    if new_k not in remapped:
                        remapped[new_k] = v
                state_dict = remapped

            # Compatibility: resize/normalize distribution buffers; skip other shape mismatches
            filtered_state = {}
            resized_keys = []
            dropped_keys = []

            def _resize_1d_distribution(src_tensor, target_tensor):
                """Resize to target shape and normalize, for distribution vectors."""
                device = target_tensor.device
                dtype = target_tensor.dtype
                tgt_len = target_tensor.numel()
                out = torch.zeros(tgt_len, device=device, dtype=dtype)
                copy_len = min(src_tensor.numel(), tgt_len)
                out[:copy_len] = src_tensor.reshape(-1)[:copy_len].to(device=device, dtype=dtype)
                total = out.sum()
                if total > 0:
                    out = out / total
                return out

            for k, v in state_dict.items():
                if k in model_state and hasattr(model_state[k], "shape") and hasattr(v, "shape"):
                    if model_state[k].shape == v.shape:
                        filtered_state[k] = v
                    else:
                        # Compatibility resize for distribution vectors in sampling_metrics
                        if (
                            len(model_state[k].shape) == 1
                            and len(v.shape) == 1
                            and ("sampling_metrics" in k or k in ("p0_node_dist", "p0_edge_dist", "node_count_prob"))
                        ):
                            resized = _resize_1d_distribution(v, model_state[k])
                            filtered_state[k] = resized
                            resized_keys.append(k)
                        else:
                            dropped_keys.append(k)
                else:
                    filtered_state[k] = v

            if resized_keys:
                logger.info("Distribution vectors resized for compatibility: %s", resized_keys)
            if dropped_keys:
                logger.info("Skipping mismatched weights (shape mismatch): %s", dropped_keys)

            incompatible = self.reference_model.load_state_dict(filtered_state, strict=False)
            matched_keys = sum(1 for k in filtered_state.keys() if k in model_state)
            ratio = (matched_keys / max(1, len(model_state)))
            # If almost nothing matches, treat as failed load (otherwise reference model is effectively random).
            if ratio < 0.3:
                logger.warning(
                    "Reference model checkpoint matched too few keys; model may be barely loaded (KL will be abnormal). "
                    "ckpt=%s matched=%d/%d (%.1f%%). "
                    "Falling back to reference=current model (equivalent to disabling KL regularization).",
                    ckpt_path, matched_keys, len(model_state), ratio * 100,
                )
                return False

            logger.info("Reference model loaded from pretrained checkpoint: %s (matched=%d/%d; %.1f%%)", ckpt_path, matched_keys, len(model_state), ratio * 100)
            # Manually apply loaded buffers to internal distributions (Lightning hook won't run here).
            try:
                if hasattr(self.reference_model, "p0_node_dist") and hasattr(self.reference_model, "p0_edge_dist"):
                    self.reference_model.update_limit_dist(self.reference_model.p0_node_dist, self.reference_model.p0_edge_dist)
            except Exception as e:
                logger.warning("Failed to apply p0 buffers to reference model: %s", e)
            try:
                if hasattr(self.reference_model, "node_count_prob"):
                    self.reference_model.update_node_count_dist(self.reference_model.node_count_prob)
            except Exception as e:
                logger.warning("Failed to apply node_count_prob to reference model: %s", e)
            if getattr(incompatible, 'missing_keys', None):
                mk = list(incompatible.missing_keys)
                logger.info("Missing params: %d (showing up to 20): %s", len(mk), mk[:20])
            if getattr(incompatible, 'unexpected_keys', None):
                uk = list(incompatible.unexpected_keys)
                logger.info("Unused params: %d (showing up to 20): %s", len(uk), uk[:20])
            return True
        except Exception as e:
            logger.warning("Failed to load reference model checkpoint (%s): %s", ckpt_path, e)
            return False

    def _update_reference_model(self):
        """Update reference model (soft update)."""
        if self.reference_model is None or self.beta == 0:
            return

        if self.global_step > 0 and self.global_step % self.ref_model_update_freq == 0:
            tau = self.cfg.grpo.get('ref_model_update_tau', 0.01)
            with torch.no_grad():
                for online_param, target_param in zip(
                    self.core_model.parameters(),
                    self.reference_model.parameters()
                ):
                    target_param.data.copy_(
                        tau * online_param.data + (1.0 - tau) * target_param.data
                    )
