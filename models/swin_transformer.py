"""
swin_transformer.py
===================
Swin Transformer backbone for the spatial encoding phase.

Uses the `timm` library to load pre-trained Swin Transformer weights.
The forward pass returns both the final embedding and the intermediate
feature map F required by the FANE attention module.

Architecture highlights (paper Section 4.1):
  - Hierarchical feature extraction via 4 stages
  - Shifted window self-attention: O(HW·M²) complexity (Equation 16)
    vs O(HW)² for standard ViT
  - Window size M=7 for 224×224 input
  - Pre-trained on ImageNet-21k for strong visual initialisation

Paper reference: Section 4.1, Equation 16
"""

import torch
import torch.nn as nn
from typing import Tuple, Optional

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("[WARNING] timm not installed. Install with: pip install timm")


class SwinTransformerBackbone(nn.Module):
    """
    Wrapper around the timm Swin Transformer that exposes both the
    penultimate feature map F and the final pooled embedding z.

    The intermediate feature map F is used by the FANE module to compute
    spatial attention masks (paper Section 4.1, Equation 6).

    Args:
        model_name:      timm model identifier (e.g., 'swin_base_patch4_window7_224').
        pretrained:      Whether to load ImageNet-pretrained weights.
        output_dim:      Target embedding dimensionality after optional projection.
                         Set to None to use the backbone's native output dim.
        drop_path_rate:  Stochastic depth drop path rate.
    """

    def __init__(
        self,
        model_name: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        output_dim: Optional[int] = None,
        drop_path_rate: float = 0.2,
        image_size: int = 224,
    ):
        super().__init__()

        if not TIMM_AVAILABLE:
            raise ImportError("timm is required. Install with: pip install timm>=0.9.2")

        # ----------------------------------------------------------------
        # Load the Swin Transformer from timm
        # We set num_classes=0 to get the feature extractor (no classifier)
        # ----------------------------------------------------------------
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,          # Remove classification head
            drop_path_rate=drop_path_rate,
            img_size=image_size,
        )

        # Native feature dimension of the backbone (before projection)
        self.native_dim = self.backbone.num_features
        # e.g., swin_base → 1024; swin_small → 768; swin_tiny → 768

        # ----------------------------------------------------------------
        # Optional linear projection to a fixed output_dim
        # Useful for aligning different backbone sizes to a common embedding space
        # ----------------------------------------------------------------
        self.output_dim = output_dim or self.native_dim
        if output_dim is not None and output_dim != self.native_dim:
            self.proj = nn.Sequential(
                nn.LayerNorm(self.native_dim),
                nn.Linear(self.native_dim, output_dim),
                nn.GELU(),
            )
        else:
            self.proj = nn.Identity()

        # ----------------------------------------------------------------
        # Hook to capture the intermediate feature map before final pooling
        # We hook into the last norm layer of stage 3 (the deepest stage)
        # ----------------------------------------------------------------
        self._feature_map: Optional[torch.Tensor] = None
        self._register_feature_hook()

    def _register_feature_hook(self):
        """
        Registers a forward hook on the last normalization layer of the backbone
        to capture the pre-pooling feature tensor F.

        For Swin Transformer, the last norm produces features of shape
        (B, H', W', C) where H'=W'=7 for 224×224 input.
        """
        def hook_fn(module, input, output):
            # timm Swin versions may expose either (B, H', W', C) or
            # (B, H'*W', C); FANE consumes the flattened token layout.
            if output.dim() == 4:
                B, H, W, C = output.shape
                output = output.reshape(B, H * W, C)
            self._feature_map = output

        # Access the last norm layer (architecture-dependent)
        # For timm Swin: backbone.norm is the final LayerNorm before pooling
        if hasattr(self.backbone, 'norm'):
            self.backbone.norm.register_forward_hook(hook_fn)
        elif hasattr(self.backbone, 'layers'):
            # Fallback: hook the last transformer stage
            last_stage = list(self.backbone.layers.children())[-1]
            last_stage.register_forward_hook(hook_fn)

    def get_feature_map(self) -> Optional[torch.Tensor]:
        """
        Returns the intermediate feature map F captured by the forward hook.

        Shape: (B, N_tokens, C) where N_tokens = H'*W'

        This is used by the FANE module to compute spatial attention.
        """
        return self._feature_map

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the Swin Transformer backbone.

        Args:
            x: Input tensor of shape (B, C, H, W).

        Returns:
            z: Global average-pooled embedding of shape (B, output_dim).
            F: Intermediate feature map of shape (B, N_tokens, native_dim).
               Used by FANE for spatial attention computation.
        """
        # Full backbone forward (hook captures F along the way)
        z_native = self.backbone(x)       # (B, native_dim)

        # Retrieve feature map captured by hook
        F = self._feature_map             # (B, N_tokens, native_dim)

        # Optional projection
        z = self.proj(z_native)           # (B, output_dim)

        return z, F


class SwinTransformerBackboneMock(nn.Module):
    """
    Mock Swin Transformer backbone for testing without timm installed.

    Produces tensors of the correct shapes using random projections.
    Replace with SwinTransformerBackbone for real training.
    """

    def __init__(
        self,
        output_dim: int = 1024,
        native_dim: int = 1024,
        image_size: int = 224,
        patch_size: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.native_dim = native_dim
        # After 4 stages of 2× downsampling: 224/32 = 7
        self.tokens_per_side = image_size // (patch_size * 8)

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((self.tokens_per_side, self.tokens_per_side)),
        )
        self.token_proj = nn.Linear(64, native_dim)
        self.pool       = nn.AdaptiveAvgPool1d(1)
        self.proj       = nn.Linear(native_dim, output_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        feat = self.conv(x)                           # (B, 64, 7, 7)
        feat = feat.flatten(2).transpose(1, 2)        # (B, 49, 64)
        F    = self.token_proj(feat)                  # (B, 49, native_dim)
        z    = self.pool(F.transpose(1, 2)).squeeze(-1)  # (B, native_dim)
        z    = self.proj(z)                           # (B, output_dim)
        return z, F


def build_swin_backbone(
    model_name: str = "swin_base_patch4_window7_224",
    pretrained: bool = True,
    output_dim: Optional[int] = 1024,
    drop_path_rate: float = 0.2,
    image_size: int = 224,
) -> nn.Module:
    """
    Factory function to build a Swin Transformer backbone.

    Falls back to the mock implementation if timm is not available.

    Args:
        model_name:      timm model identifier.
        pretrained:      Load ImageNet pretrained weights.
        output_dim:      Embedding dimensionality.
        drop_path_rate:  Stochastic depth rate.

    Returns:
        SwinTransformerBackbone (or mock if timm unavailable).
    """
    if TIMM_AVAILABLE:
        return SwinTransformerBackbone(
            model_name=model_name,
            pretrained=pretrained,
            output_dim=output_dim,
            drop_path_rate=drop_path_rate,
            image_size=image_size,
        )
    else:
        print("[WARNING] Using SwinTransformerBackboneMock (timm not installed).")
        return SwinTransformerBackboneMock(output_dim=output_dim or 1024)
