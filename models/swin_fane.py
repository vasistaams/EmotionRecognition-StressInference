"""
swin_fane.py
============
Swin-FANE Spatial Encoder — Phase I of the two-phase framework.

Combines:
  1. SwinTransformerBackbone — hierarchical feature extraction
  2. FANEModule              — region-aware attention over facial areas
  3. EmotionClassifierHead   — maps embedding to emotion probability distribution

Algorithm 1 and Algorithm 2 from the paper are implemented here.

Forward pass:
    x_t (B, C, H, W)
      ↓  Swin Transformer
    F   (B, N_tokens, feature_dim)     ← intermediate feature map
      ↓  FANE Module
    z_t (B, embedding_dim)             ← frame-level embedding
      ↓  Linear + Softmax
    p_t (B, num_classes)               ← emotion probability vector

Paper reference: Sections 4.1–4.2, Algorithms 1–2, Equations 6–10
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from models.swin_transformer import build_swin_backbone
from models.fane_module import FANEModule


class AdaptiveRegionWeightingModule(nn.Module):
    """
    Adaptive Region Weighting Module (ARWM).

    Uses deterministic geometric priors to create soft region masks for the
    eyes, eyebrows, and mouth on the 160x160 face crop. These masks are mapped
    to the Swin token grid and converted into region-specific features. A small
    learnable network predicts softmax weights across the three regions, and the
    weighted region representation is fused with the global Swin/FANE embedding.

    The module is intentionally optional and should be disabled by default to
    preserve the exact baseline behavior.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 128,
        enabled: bool = False,
        fusion_weight: float = 0.3,
        fallback_to_global: bool = True,
        mode: Optional[str] = None,
    ):
        super().__init__()
        self.mode = mode or ("adaptive" if enabled else "off")
        if self.mode not in {"off", "fixed", "adaptive"}:
            raise ValueError(
                f"Unknown region-attention mode: {self.mode!r}; "
                "expected 'off', 'fixed', or 'adaptive'"
            )
        self.enabled = self.mode != "off"
        self.fallback_to_global = fallback_to_global
        self.fusion_weight = float(fusion_weight)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        if self.mode != "off":
            if self.mode == "adaptive":
                self.region_mlp = nn.Sequential(
                    nn.Linear(feature_dim * 3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 3),
                )

            self.region_proj = nn.Sequential(
                nn.LayerNorm(feature_dim),
                nn.Linear(feature_dim, feature_dim),
                nn.GELU(),
            )

    def _make_region_mask(self, tokens_per_side: int, region: str) -> torch.Tensor:
        """
        Create a deterministic geometric soft mask on the token grid.

        Swin token grid mapping for 160x160 input:
        - patch size = 4
        - 4 downsampling stages => 160/32 = 5
        - final token map is approximately 5x5
        - therefore each token corresponds to a 32x32 patch of the 160x160 image
        """
        coords = torch.linspace(0.0, 1.0, steps=tokens_per_side, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing='ij')
        yy = yy.reshape(-1)
        xx = xx.reshape(-1)

        if region == 'eye':
            # Left and right eye regions in the upper-middle face area.
            eye_mask = (
                (torch.exp(-((xx - 0.30) ** 2) / (2.0 * 0.12 ** 2)) * torch.exp(-((yy - 0.35) ** 2) / (2.0 * 0.18 ** 2))) +
                (torch.exp(-((xx - 0.70) ** 2) / (2.0 * 0.12 ** 2)) * torch.exp(-((yy - 0.35) ** 2) / (2.0 * 0.18 ** 2)))
            )
            return eye_mask

        if region == 'eyebrow':
            # Slightly above the eye line; covers eyebrow region.
            brow_mask = (
                (torch.exp(-((xx - 0.30) ** 2) / (2.0 * 0.12 ** 2)) * torch.exp(-((yy - 0.22) ** 2) / (2.0 * 0.18 ** 2))) +
                (torch.exp(-((xx - 0.70) ** 2) / (2.0 * 0.12 ** 2)) * torch.exp(-((yy - 0.22) ** 2) / (2.0 * 0.18 ** 2)))
            )
            return brow_mask

        if region == 'mouth':
            # Central lower face region.
            mouth_mask = torch.exp(-((xx - 0.50) ** 2) / (2.0 * 0.18 ** 2)) * torch.exp(-((yy - 0.72) ** 2) / (2.0 * 0.20 ** 2))
            return mouth_mask

        raise ValueError(f"Unknown region: {region}")

    def _mask_pool(self, F: torch.Tensor, region_mask: torch.Tensor) -> torch.Tensor:
        """Pool token features using a soft mask: (B, N, C) -> (B, C)."""
        B, N, C = F.shape
        mask = region_mask.to(F.device, dtype=F.dtype)
        if mask.numel() != N:
            raise ValueError(
                f"Region mask length mismatch: expected {N}, got {mask.numel()}"
            )
        mask = mask.view(1, N, 1).expand(B, N, 1)
        weight_sum = mask.sum(dim=1).clamp_min(1e-6)  # (B, 1)
        pooled = (F * mask).sum(dim=1) / weight_sum  # (B, C)
        return pooled

    def _compute_region_masks(self, F: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens_per_side = int(math.sqrt(F.size(1)))
        masks = {
            'eye': self._make_region_mask(tokens_per_side, 'eye'),
            'eyebrow': self._make_region_mask(tokens_per_side, 'eyebrow'),
            'mouth': self._make_region_mask(tokens_per_side, 'mouth'),
        }
        return masks

    def forward(self, F: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            F: Swin feature map. Shape: (B, N_tokens, C)

        Returns:
            z_region: Weighted region representation. Shape: (B, C)
            weights: Learned region weights. Shape: (B, 3)
            meta: Region-specific metadata including pooled features.
        """
        if not self.enabled:
            zeros = torch.zeros(
                F.size(0), F.size(-1), device=F.device, dtype=F.dtype
            )
            return zeros, None, {
                'enabled': False,
                'mode': 'off',
            }

        masks = self._compute_region_masks(F)
        region_features = {
            'eye': self._mask_pool(F, masks['eye']),
            'eyebrow': self._mask_pool(F, masks['eyebrow']),
            'mouth': self._mask_pool(F, masks['mouth']),
        }

        stacked = torch.stack([
            region_features['eye'],
            region_features['eyebrow'],
            region_features['mouth'],
        ], dim=1)  # (B, 3, C)

        if self.mode == "fixed":
            weights = F.new_full((F.size(0), 3), 1.0 / 3.0)
        else:
            logits = self.region_mlp(stacked.reshape(F.size(0), -1))  # (B, 3)
            weights = torch.softmax(logits, dim=-1)
        enhanced = torch.sum(weights.unsqueeze(-1) * stacked, dim=1)  # (B, C)
        z_region = self.region_proj(enhanced)

        return z_region, weights, {
            'enabled': True,
            'mode': self.mode,
            'weights': weights,
            'region_features': region_features,
            'masks': masks,
        }

    def get_weight_summary(self, weights: torch.Tensor) -> Dict[str, float]:
        """Returns mean region weights for logging and visualisation."""
        if weights.dim() != 2:
            raise ValueError(f"Expected weights shape (B, 3), got {tuple(weights.shape)}")
        return {
            'avg_eye_weight': float(weights[:, 0].mean().item()),
            'avg_eyebrow_weight': float(weights[:, 1].mean().item()),
            'avg_mouth_weight': float(weights[:, 2].mean().item()),
        }


class EmotionClassifierHead(nn.Module):
    """
    Linear classification head for frame-level emotion prediction.

    Maps embedding z_t → logits → emotion probabilities p_t.
    (Paper Equations 8–10)

    Args:
        embedding_dim: Dimension of the input embedding z_t.
        num_classes:   Number of emotion categories (7 for FANE dataset).
        dropout:       Dropout probability before the linear layer.
    """

    def __init__(
        self,
        embedding_dim: int = 1024,
        num_classes: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.dropout    = nn.Dropout(p=dropout)
        self.layer_norm = nn.LayerNorm(embedding_dim)

        # Linear classifier W_k, b_k (paper Equation 8)
        self.fc = nn.Linear(embedding_dim, num_classes)

        # Weight initialisation
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            z: Embedding tensor (B, embedding_dim).

        Returns:
            logits: Raw class logits (B, num_classes).
            probs:  Softmax probabilities (B, num_classes) — paper Equation 9.
        """
        z      = self.layer_norm(z)
        z      = self.dropout(z)
        logits = self.fc(z)                      # Equation 8: ℓ_k = W_k z + b_k
        probs  = F.softmax(logits, dim=-1)       # Equation 9: softmax normalisation
        return logits, probs


class SwinFANEEncoder(nn.Module):
    """
    Phase I spatial encoder: Swin Transformer + FANE + Emotion Classifier.

    Implements Algorithm 1 (Swin-FANE Spatial Encoding) and
    Algorithm 2 (Frame-Level Emotion Classification) from the paper.

    Args:
        swin_model_name:   timm model identifier for the Swin backbone.
        pretrained:        Load ImageNet pretrained Swin weights.
        embedding_dim:     Target embedding dimension (z_t dimension).
        num_classes:       Number of emotion categories.
        fane_hidden_dim:   FANE bottleneck projection size.
        use_mask_guidance: Enable mask-supervised FANE attention.
        classifier_dropout: Dropout in the classification head.
        drop_path_rate:    Swin stochastic depth rate.
    """

    def __init__(
        self,
        swin_model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        embedding_dim: int = 1024,
        num_classes: int = 7,
        fane_hidden_dim: Optional[int] = 256,
        use_mask_guidance: bool = False,
        classifier_dropout: float = 0.3,
        drop_path_rate: float = 0.2,
        image_size: int = 224,
        region_attention_enabled: bool = False,
        region_attention_hidden_dim: int = 128,
        region_attention_fusion_weight: float = 0.3,
        region_attention_fallback_to_global: bool = True,
        region_attention_mode: Optional[str] = None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_classes   = num_classes

        # ---- 1. Swin Transformer backbone ----
        self.backbone = build_swin_backbone(
            model_name=swin_model_name,
            pretrained=pretrained,
            output_dim=embedding_dim,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
        )

        # Native feature dim of the backbone (before our projection)
        native_dim = self.backbone.native_dim if hasattr(self.backbone, 'native_dim') \
                     else embedding_dim

        # ---- 2. FANE attention module ----
        self.fane = FANEModule(
            feature_dim=native_dim,
            hidden_dim=fane_hidden_dim,
            use_mask_guidance=use_mask_guidance,
        )

        # ---- 2.5 Adaptive Region Weighting Module (optional) ----
        resolved_region_mode = region_attention_mode or (
            "adaptive" if region_attention_enabled else "off"
        )
        if resolved_region_mode not in {"off", "fixed", "adaptive"}:
            raise ValueError(
                f"Unknown region-attention mode: {resolved_region_mode!r}; "
                "expected 'off', 'fixed', or 'adaptive'"
            )
        self.region_attention_mode = resolved_region_mode
        self.region_attention = (
            AdaptiveRegionWeightingModule(
                feature_dim=native_dim,
                hidden_dim=region_attention_hidden_dim,
                enabled=region_attention_enabled,
                fusion_weight=region_attention_fusion_weight,
                fallback_to_global=region_attention_fallback_to_global,
                mode=resolved_region_mode,
            )
            if resolved_region_mode != "off" else None
        )

        # ---- Projection from FANE output (native_dim) → embedding_dim ----
        # Only needed if FANE output dim != embedding_dim
        if native_dim != embedding_dim:
            self.fane_proj = nn.Sequential(
                nn.LayerNorm(native_dim),
                nn.Linear(native_dim, embedding_dim),
                nn.GELU(),
            )
        else:
            self.fane_proj = nn.Identity()

        # ---- 3. Emotion classifier head ----
        self.classifier = EmotionClassifierHead(
            embedding_dim=embedding_dim,
            num_classes=num_classes,
            dropout=classifier_dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass of the Swin-FANE spatial encoder.

        Implements Algorithm 1 + Algorithm 2 from the paper.

        Args:
            x:    Input facial frame batch. Shape: (B, C, H, W).
            mask: Optional binary expressive-region mask.
                  Shape: (B, 1, H, W). Used for FANE mask guidance.

        Returns:
            Dict containing:
              'embedding': Frame embedding z_t    (B, embedding_dim)
              'logits':    Class logits            (B, num_classes)
              'probs':     Emotion probabilities   (B, num_classes)
              'attention': FANE attention map      (B, N_tokens, feature_dim)
              'pred_label': Predicted emotion class (B,)  — Equation 10
        """
        # Step 1: Swin Transformer forward
        # Returns final embedding AND intermediate feature map
        z_swin, F = self.backbone(x)   # z_swin: (B, embedding_dim), F: (B, N, C)

        # Step 2: FANE module — region-aware attention refinement
        # z_fane is GlobalAvgPool(F'), A is the attention mask
        z_fane, A = self.fane(F, mask=mask)   # (B, native_dim), (B, N, C)

        # Optional region-aware adaptive weighting branch
        z_region = torch.zeros_like(z_swin)
        region_weights = None
        region_meta = {'enabled': False, 'mode': 'off'}
        if self.region_attention is not None:
            z_region, region_weights, region_meta = self.region_attention(F)
            z_region = self.fane_proj(z_region)

        # Project FANE embedding to target dimension
        z_fane = self.fane_proj(z_fane)        # (B, embedding_dim)

        # Combine backbone embedding with FANE embedding (additive fusion)
        # Keep baseline behavior exactly when ARWM is disabled.
        z = z_swin + z_fane
        if self.region_attention is not None:
            z = z + self.region_attention.fusion_weight * z_region

        # Step 3: Emotion classification (Equations 8–10)
        logits, probs = self.classifier(z)     # (B, num_classes) each

        # Predicted class label (Equation 10): ŷ_t = argmax_k p_t^k
        pred_label = torch.argmax(probs, dim=-1)   # (B,)

        return {
            'embedding':  z,
            'logits':     logits,
            'probs':      probs,
            'attention':  A,
            'pred_label': pred_label,
            'region_weights': region_weights,
            'region_meta': region_meta,
        }

    def extract_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extracts frame-level embeddings without computing classification output.
        Used for building the temporal embedding sequence Z in Phase II.

        Args:
            x: Input frames (B, C, H, W).

        Returns:
            Embeddings z_t (B, embedding_dim).
        """
        with torch.no_grad():
            output = self.forward(x)
        return output['embedding']

    def get_trainable_params(self) -> int:
        """Returns the number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
