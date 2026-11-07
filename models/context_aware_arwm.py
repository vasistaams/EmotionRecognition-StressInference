"""Global-context-conditioned weighting for the three existing face regions."""

from typing import Dict, Tuple

import torch
from torch import nn
import math


class ContextAwareRegionWeightingModule(nn.Module):
    """Score eye, eyebrow, and mouth features conditioned on global context.

    Input tokens have shape ``(B, N, C)``. The three local region features and
    global feature each have shape ``(B, C)``. Independent region-specific
    parameters implement

        u_r = tanh(W_r F_r + U_r F_global + b_r)
        score_r = v_r^T u_r
        alpha = softmax(score)

    The deterministic masks match legacy ARWM and ``region_proj`` retains its
    state-dictionary name so compatible parameters can be transferred.
    """

    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 128,
        fusion_weight: float = 0.3,
        fallback_to_global: bool = True,
    ):
        super().__init__()
        self.mode = "context_aware"
        self.enabled = True
        self.fallback_to_global = fallback_to_global
        self.fusion_weight = float(fusion_weight)
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.region_proj = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
        )
        self.local_weight = nn.Parameter(torch.empty(3, hidden_dim, feature_dim))
        self.context_weight = nn.Parameter(torch.empty(3, hidden_dim, feature_dim))
        self.attention_bias = nn.Parameter(torch.zeros(3, hidden_dim))
        self.score_vector = nn.Parameter(torch.empty(3, hidden_dim))
        self.reset_context_parameters()

    def reset_context_parameters(self) -> None:
        for region in range(3):
            nn.init.xavier_uniform_(self.local_weight[region])
            nn.init.xavier_uniform_(self.context_weight[region])
            nn.init.xavier_uniform_(self.score_vector[region].unsqueeze(0))
        nn.init.zeros_(self.attention_bias)

    @staticmethod
    def _make_region_mask(tokens_per_side: int, region: str) -> torch.Tensor:
        coords = torch.linspace(0.0, 1.0, steps=tokens_per_side)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        if region in {"eye", "eyebrow"}:
            y = 0.35 if region == "eye" else 0.22
            mask = (
                torch.exp(-((xx - 0.30) ** 2) / (2.0 * 0.12**2))
                + torch.exp(-((xx - 0.70) ** 2) / (2.0 * 0.12**2))
            ) * torch.exp(-((yy - y) ** 2) / (2.0 * 0.18**2))
            return mask.reshape(-1)
        if region == "mouth":
            return (
                torch.exp(-((xx - 0.50) ** 2) / (2.0 * 0.18**2))
                * torch.exp(-((yy - 0.72) ** 2) / (2.0 * 0.20**2))
            ).reshape(-1)
        raise ValueError(f"Unknown facial region: {region}")

    def _compute_region_masks(self, features: torch.Tensor):
        side = math.isqrt(features.shape[1])
        if side * side != features.shape[1]:
            raise ValueError("Swin token count must form a square feature grid")
        return {
            name: self._make_region_mask(side, name)
            for name in ("eye", "eyebrow", "mouth")
        }

    def _mask_pool(self, features: torch.Tensor, mask: torch.Tensor):
        batch, tokens, _ = features.shape
        if mask.numel() != tokens:
            raise ValueError(f"Region mask has {mask.numel()} tokens; expected {tokens}")
        mask = mask.to(device=features.device, dtype=features.dtype).view(1, tokens, 1)
        return (features * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-6)

    def score_regions(
        self,
        region_features: torch.Tensor,
        global_context: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if region_features.ndim != 3 or region_features.shape[1:] != (
            3, self.feature_dim
        ):
            raise ValueError(
                "region_features must have shape "
                f"(B, 3, {self.feature_dim}); got {tuple(region_features.shape)}"
            )
        if global_context.shape != (region_features.shape[0], self.feature_dim):
            raise ValueError(
                "global_context must have shape "
                f"(B, {self.feature_dim}); got {tuple(global_context.shape)}"
            )
        if global_context.device != region_features.device:
            raise ValueError("global_context and region_features must share a device")

        local = torch.einsum(
            "brc,rhc->brh", region_features, self.local_weight
        )
        context = torch.einsum(
            "bc,rhc->brh", global_context, self.context_weight
        )
        hidden = torch.tanh(
            local + context + self.attention_bias.unsqueeze(0)
        )
        scores = torch.einsum("brh,rh->br", hidden, self.score_vector)
        weights = torch.softmax(scores.float(), dim=-1).to(scores.dtype)
        return scores, weights

    def forward(
        self, features: torch.Tensor, global_context: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        if features.ndim != 3 or features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (B, N, {self.feature_dim}); "
                f"got {tuple(features.shape)}"
            )
        if global_context is None:
            global_context = features.mean(dim=1)

        masks = self._compute_region_masks(features)
        ordered_masks = (masks["eye"], masks["eyebrow"], masks["mouth"])
        region_features = torch.stack(
            [self._mask_pool(features, mask) for mask in ordered_masks], dim=1
        )
        scores, weights = self.score_regions(region_features, global_context)
        combined = torch.sum(weights.unsqueeze(-1) * region_features, dim=1)
        projected = self.region_proj(combined)
        named_features = dict(zip(("eye", "eyebrow", "mouth"), region_features.unbind(1)))
        return projected, weights, {
            "enabled": True,
            "mode": self.mode,
            "weights": weights,
            "scores": scores,
            "global_context": global_context,
            "region_features": named_features,
            "masks": masks,
        }
