"""
fane_module.py
==============
Facial Attention Network Embedding (FANE) module.

The FANE module takes intermediate feature maps from the Swin Transformer
backbone and computes a spatial attention mask that amplifies features
corresponding to physiologically expressive facial regions:
  - Eyes and periocular region
  - Eyebrows
  - Mouth and perioral region
  - Nose (secondary)

This is implemented as a lightweight two-layer bottleneck network
followed by sigmoid activation (paper Section 4.1, Equation 6):

    A = σ( W₂ · δ( W₁ · F ) )
    F' = F + A ⊙ F        (residual refinement)
    z  = GlobalAvgPool(F')

Where:
  - F  : Intermediate feature map from Swin backbone  (B, N, C)
  - A  : Spatial attention mask                        (B, N, C)
  - δ  : ReLU activation
  - σ  : Sigmoid activation
  - ⊙  : Element-wise multiplication

Paper reference: Section 4.1, Equations 6–7
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FANEModule(nn.Module):
    """
    Facial Attention Network Embedding.

    Computes a channel-and-spatial attention mask over the Swin Transformer
    feature map to emphasise emotion-discriminative facial regions.

    Design choices:
    - Two-layer bottleneck (W1, W2) with reduction ratio r=4 for efficiency
    - Residual connection: F' = F + A ⊙ F (preserves global context)
    - Global average pooling to produce the final compact embedding z_t
    - Optional mask guidance: if dataset provides expressive-region masks,
      they can be used to supervise the attention at train time

    Args:
        feature_dim:    Dimensionality of backbone feature map (C dimension).
        hidden_dim:     Bottleneck projection dimension (W1 output size).
                        Default: feature_dim // 4.
        residual_weight: Scalar multiplier for the attention branch.
                         1.0 = standard residual; >1.0 amplifies attention.
        use_mask_guidance: Whether to apply mask supervision loss during training.
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: Optional[int] = None,
        residual_weight: float = 1.0,
        use_mask_guidance: bool = False,
    ):
        super().__init__()

        self.feature_dim     = feature_dim
        self.hidden_dim      = hidden_dim or max(feature_dim // 4, 64)
        self.residual_weight = residual_weight
        self.use_mask_guidance = use_mask_guidance

        # ----------------------------------------------------------------
        # Attention bottleneck: W1 (feature_dim → hidden_dim)
        # Paper: δ(W1 · F) — ReLU-activated projection
        # ----------------------------------------------------------------
        self.W1 = nn.Linear(feature_dim, self.hidden_dim, bias=True)

        # ----------------------------------------------------------------
        # Attention gate: W2 (hidden_dim → feature_dim)
        # Paper: σ(W2 · ...)  — Sigmoid-activated gate
        # ----------------------------------------------------------------
        self.W2 = nn.Linear(self.hidden_dim, feature_dim, bias=True)

        # Layer norm for stable training with deep features
        self.norm = nn.LayerNorm(feature_dim)

        # ----------------------------------------------------------------
        # Optional: mask projection for supervised attention guidance
        # Maps spatial mask to token-level attention target
        # ----------------------------------------------------------------
        if use_mask_guidance:
            self.mask_proj = nn.Linear(1, feature_dim)

        # Weight initialisation for better training stability
        self._init_weights()

    def _init_weights(self):
        """Initialises weights using small values for stable attention start."""
        nn.init.xavier_uniform_(self.W1.weight, gain=0.5)
        nn.init.zeros_(self.W1.bias)
        nn.init.xavier_uniform_(self.W2.weight, gain=0.5)
        # Initialise W2 bias to 0 → sigmoid(0) = 0.5 → mild initial attention
        nn.init.zeros_(self.W2.bias)

    def forward(
        self,
        F: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the FANE module.

        Args:
            F:    Backbone feature map. Shape: (B, N_tokens, C)
                  where N_tokens = H' × W' (e.g., 49 for 224×224 input)
            mask: Optional binary expressive-region mask.
                  Shape: (B, 1, H_orig, W_orig).
                  If provided and use_mask_guidance=True, computes mask loss.

        Returns:
            z:    Global average-pooled embedding. Shape: (B, C)
            A:    Attention mask for visualisation/debugging. Shape: (B, N, C)
        """
        # ---- Normalise feature map ----
        F_norm = self.norm(F)    # (B, N, C)

        # ---- Compute attention mask (Equation 6) ----
        # Step 1: Linear reduction W1 + ReLU
        hidden = torch.relu(self.W1(F_norm))   # (B, N, hidden_dim)

        # Step 2: Linear expansion W2 + Sigmoid → attention gate A ∈ [0,1]
        A = torch.sigmoid(self.W2(hidden))  # (B, N, C)

        # ---- Residual refinement: F' = F + A ⊙ F (Equation 6 residual) ----
        # The residual ensures that original global context is preserved
        F_prime = F + self.residual_weight * (A * F)   # (B, N, C)

        # ---- Global Average Pooling over token dimension ----
        # Reduces (B, N, C) → (B, C), paper Equation 7
        z = F_prime.mean(dim=1)   # (B, C)

        return z, A

    def compute_mask_loss(
        self,
        A: torch.Tensor,
        mask: torch.Tensor,
        tokens_per_side: int,
    ) -> torch.Tensor:
        """
        Computes a supervision loss that encourages the attention map
        to align with the ground-truth expressive-region mask.

        This is an auxiliary loss; the primary loss is cross-entropy over emotions.

        Loss = MSE( mean(A, dim=-1), downsampled_mask )

        Args:
            A:               Attention tensor (B, N_tokens, C).
            mask:            Binary mask (B, 1, H, W).
            tokens_per_side: Square root of N_tokens (e.g., 7 for 224px).

        Returns:
            Scalar auxiliary mask loss.
        """
        B = A.size(0)
        N = tokens_per_side * tokens_per_side

        # Average attention across channels: (B, N)
        A_spatial = A.mean(dim=-1)   # (B, N)

        # Downsample mask to match token resolution: (B, 1, 7, 7)
        mask_ds = F.interpolate(
            mask.float(),
            size=(tokens_per_side, tokens_per_side),
            mode='bilinear',
            align_corners=False,
        )
        mask_flat = mask_ds.view(B, N)   # (B, N)

        # MSE between predicted attention and binary mask
        loss = F.mse_loss(A_spatial, mask_flat)
        return loss


class FANEModuleLite(nn.Module):
    """
    Lightweight FANE variant with fewer parameters.

    Uses a single projection layer instead of two, suitable for
    resource-constrained deployment (Appendix D).

    Args:
        feature_dim: Feature map channel dimension.
    """

    def __init__(self, feature_dim: int):
        super().__init__()
        self.gate = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )

    def forward(
        self,
        F: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        A      = self.gate(F)             # (B, N, C)
        F_prime = F + A * F
        z      = F_prime.mean(dim=1)     # (B, C)
        return z, A
