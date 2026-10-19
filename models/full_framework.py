"""
full_framework.py
=================
End-to-end two-phase Swin-FANE + BiLSTM-ResNet framework.

Integrates Phase I (spatial emotion recognition) and Phase II (temporal
stress estimation) into a single nn.Module with a unified forward pass.

Two inference modes are supported:
  A) FRAME mode:    Single-frame input → emotion label + probabilities
  B) SEQUENCE mode: Temporal sequence input (T frames) → emotions + stress score

Training is done in two stages:
  Stage 1: Train only the Swin-FANE encoder (frame-level cross-entropy)
  Stage 2: Freeze encoder, train BiLSTM temporal module

Paper reference: Section 4, Figures 4–5
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple

from models.swin_fane import SwinFANEEncoder
from models.bilstm_resnet import BiLSTMTemporalModule, TemporalMLP, GRUTemporalModule


class SwinFANEFramework(nn.Module):
    """
    Complete two-phase framework for emotion recognition and stress estimation.

    Phase I — Swin-FANE Encoder:
        Input: facial frames (B, C, H, W) or sequence (B, T, C, H, W)
        Output: emotion probabilities p_t ∈ R^7 per frame

    Phase II — BiLSTM Temporal Module:
        Input: sequence of embeddings Z = {z_1, ..., z_T}
        Output: stress score s ∈ [0, 1]

    Args:
        swin_model_name:    timm model name for Swin backbone.
        pretrained:         Load ImageNet pretrained Swin weights.
        embedding_dim:      Frame embedding dimensionality.
        num_classes:        Emotion categories (7 for FANE).
        bilstm_hidden_dim:  BiLSTM hidden state size.
        bilstm_layers:      Number of stacked BiLSTM layers.
        fane_hidden_dim:    FANE bottleneck projection size.
        use_mask_guidance:  Use expressive-region masks for FANE supervision.
        dropout:            Dropout rate across all modules.
        temporal_model:     Temporal module type: 'bilstm', 'mlp', 'gru'.
                            Used for ablation study.
    """

    def __init__(
        self,
        swin_model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        embedding_dim: int = 1024,
        num_classes: int = 7,
        bilstm_hidden_dim: int = 512,
        bilstm_layers: int = 2,
        fane_hidden_dim: int = 256,
        use_mask_guidance: bool = False,
        dropout: float = 0.3,
        temporal_model: str = 'bilstm',
        image_size: int = 224,
        region_attention_enabled: bool = False,
        region_attention_hidden_dim: int = 128,
        region_attention_fusion_weight: float = 0.3,
        region_attention_fallback_to_global: bool = True,
        region_attention_mode: Optional[str] = None,
    ):
        super().__init__()

        self.embedding_dim   = embedding_dim
        self.num_classes     = num_classes
        self.temporal_model  = temporal_model

        # ---- Phase I: Spatial Encoder ----
        self.spatial_encoder = SwinFANEEncoder(
            swin_model_name=swin_model_name,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            fane_hidden_dim=fane_hidden_dim,
            use_mask_guidance=use_mask_guidance,
            classifier_dropout=dropout,
            image_size=image_size,
            region_attention_enabled=region_attention_enabled,
            region_attention_hidden_dim=region_attention_hidden_dim,
            region_attention_fusion_weight=region_attention_fusion_weight,
            region_attention_fallback_to_global=region_attention_fallback_to_global,
            region_attention_mode=region_attention_mode,
        )

        # ---- Phase II: Temporal Module (ablation-switchable) ----
        if temporal_model == 'bilstm':
            self.temporal_encoder = BiLSTMTemporalModule(
                input_dim=embedding_dim,
                hidden_dim=bilstm_hidden_dim,
                num_layers=bilstm_layers,
                dropout=dropout,
                residual=True,
            )
        elif temporal_model == 'bilstm_no_residual':
            self.temporal_encoder = BiLSTMTemporalModule(
                input_dim=embedding_dim,
                hidden_dim=bilstm_hidden_dim,
                num_layers=bilstm_layers,
                dropout=dropout,
                residual=False,         # Ablation: remove residual refinement
            )
        elif temporal_model == 'gru':
            self.temporal_encoder = GRUTemporalModule(
                input_dim=embedding_dim,
                hidden_dim=bilstm_hidden_dim,
            )
        elif temporal_model == 'mlp':
            self.temporal_encoder = TemporalMLP(
                input_dim=embedding_dim,
                hidden_dim=bilstm_hidden_dim,
            )
        else:
            raise ValueError(f"Unknown temporal model: {temporal_model}")

    # ------------------------------------------------------------------
    # Forward passes
    # ------------------------------------------------------------------

    def forward_frame(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Frame-level forward pass (Phase I only).

        Args:
            x:    Single-frame or mini-batch. Shape: (B, C, H, W).
            mask: Optional binary mask. Shape: (B, 1, H, W).

        Returns:
            Dict with 'embedding', 'logits', 'probs', 'pred_label', 'attention'.
        """
        return self.spatial_encoder(x, mask=mask)

    def forward_sequence(
        self,
        x_seq: torch.Tensor,
        mask_seq: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sequence-level forward pass (Phase I + Phase II).

        Args:
            x_seq:    Temporal sequence of frames. Shape: (B, T, C, H, W).
            mask_seq: Optional masks. Shape: (B, T, 1, H, W).

        Returns:
            Dict with per-frame emotion outputs AND sequence-level stress score:
              'frame_probs':  Emotion probability vectors  (B, T, num_classes)
              'frame_labels': Predicted emotion labels     (B, T)
              'embeddings':   Frame embeddings             (B, T, embedding_dim)
              'stress_score': Sequence stress score        (B, 1)
              'stress_traj':  Per-timestep stress          (B, T, 1)
              'refined':      Refined temporal hidden      (B, T, hidden×2)
        """
        B, T, C, H, W = x_seq.shape

        # ---- Phase I: Process all T frames ----
        # Reshape (B, T, C, H, W) → (B*T, C, H, W) for batch processing
        x_flat    = x_seq.view(B * T, C, H, W)
        mask_flat = None
        if mask_seq is not None:
            _, _, Cm, Hm, Wm = mask_seq.shape
            mask_flat = mask_seq.view(B * T, Cm, Hm, Wm)

        # Get frame-level outputs for all frames simultaneously
        frame_output = self.spatial_encoder(x_flat, mask=mask_flat)

        # Reshape back to (B, T, ...) format
        frame_probs  = frame_output['probs'].view(B, T, self.num_classes)    # (B, T, K)
        frame_labels = frame_output['pred_label'].view(B, T)                  # (B, T)
        embeddings   = frame_output['embedding'].view(B, T, self.embedding_dim)  # (B, T, D)

        # ---- Phase II: Temporal stress estimation ----
        temporal_output = self.temporal_encoder(embeddings, return_hidden=True)
        stress_score    = temporal_output['stress_score']    # (B, 1)
        refined         = temporal_output['refined']         # (B, T, hidden×2)

        # Per-timestep stress trajectory (for visualisation)
        if hasattr(self.temporal_encoder, 'forward_sequence_stress'):
            stress_traj = self.temporal_encoder.forward_sequence_stress(embeddings)
        else:
            stress_traj = stress_score.unsqueeze(1).expand(B, T, 1)

        return {
            'frame_probs':  frame_probs,
            'frame_labels': frame_labels,
            'embeddings':   embeddings,
            'stress_score': stress_score,
            'stress_traj':  stress_traj,
            'refined':      refined,
            'attention':    frame_output.get('attention'),
        }

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Unified forward pass — auto-detects frame vs sequence input.

        Args:
            x: Either:
               - Single frame (B, C, H, W)       → frame mode
               - Sequence     (B, T, C, H, W)    → sequence mode

        Returns:
            See forward_frame / forward_sequence docs.
        """
        if x.dim() == 4:
            return self.forward_frame(x, mask=mask)
        elif x.dim() == 5:
            return self.forward_sequence(x, mask_seq=mask)
        else:
            raise ValueError(f"Unexpected input shape: {x.shape}. "
                             f"Expected 4D (B,C,H,W) or 5D (B,T,C,H,W).")

    # ------------------------------------------------------------------
    # Parameter freeze/unfreeze helpers for two-stage training
    # ------------------------------------------------------------------

    def freeze_spatial_encoder(self):
        """Freezes all Phase I parameters (spatial encoder)."""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = False
        print("[SwinFANEFramework] Spatial encoder FROZEN.")

    def unfreeze_spatial_encoder(self):
        """Unfreezes all Phase I parameters."""
        for param in self.spatial_encoder.parameters():
            param.requires_grad = True
        print("[SwinFANEFramework] Spatial encoder UNFROZEN.")

    def freeze_backbone(self):
        """Freezes only the Swin backbone (keeps FANE and classifier trainable)."""
        for param in self.spatial_encoder.backbone.parameters():
            param.requires_grad = False
        print("[SwinFANEFramework] Swin backbone FROZEN.")

    def get_param_count(self) -> Dict[str, int]:
        """Returns total, trainable, and frozen parameter counts."""
        total     = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            'total':     total,
            'trainable': trainable,
            'frozen':    total - trainable,
        }

    # ------------------------------------------------------------------
    # Model summary
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        counts = self.get_param_count()
        return (
            f"SwinFANEFramework(\n"
            f"  Phase I  — Swin-FANE Encoder:  {counts['total'] // 1_000_000:.1f}M params\n"
            f"  Phase II — {self.temporal_model.upper()} Temporal Module\n"
            f"  Trainable: {counts['trainable'] // 1_000_000:.1f}M  |  "
            f"Total: {counts['total'] // 1_000_000:.1f}M\n"
            f")"
        )


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def build_framework(config: dict) -> SwinFANEFramework:
    """
    Builds the full framework from a config dictionary.

    Args:
        config: Configuration dictionary (from configs/config.yaml).

    Returns:
        Instantiated SwinFANEFramework.
    """
    m_cfg = config.get('model', {})

    region_cfg = m_cfg.get('region_attention', {})
    region_mode = region_cfg.get('mode')
    if region_mode is None:
        region_mode = (
            'adaptive' if region_cfg.get('enabled', False) else 'off'
        )

    framework = SwinFANEFramework(
        swin_model_name=m_cfg.get('swin', {}).get(
            'variant', 'swin_base_patch4_window7_224'),
        pretrained=m_cfg.get('swin', {}).get('pretrained', True),
        embedding_dim=m_cfg.get('classifier', {}).get('embedding_dim', 1024),
        num_classes=m_cfg.get('classifier', {}).get('num_classes', 7),
        bilstm_hidden_dim=m_cfg.get('bilstm', {}).get('hidden_dim', 512),
        bilstm_layers=m_cfg.get('bilstm', {}).get('num_layers', 2),
        fane_hidden_dim=m_cfg.get('fane', {}).get('hidden_dim', 256),
        use_mask_guidance=m_cfg.get('fane', {}).get('use_mask_guidance', False),
        dropout=m_cfg.get('classifier', {}).get('dropout', 0.3),
        temporal_model='bilstm',
        image_size=m_cfg.get('swin', {}).get(
            'input_size', config.get('dataset', {}).get('image_size', 224)
        ),
        region_attention_enabled=region_mode != 'off',
        region_attention_hidden_dim=region_cfg.get('hidden_dim', 128),
        region_attention_fusion_weight=region_cfg.get('fusion_weight', 0.3),
        region_attention_fallback_to_global=region_cfg.get('fallback_to_global', True),
        region_attention_mode=region_mode,
    )

    return framework
