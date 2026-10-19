"""
bilstm_resnet.py
================
Phase II: BiLSTM + Residual Temporal Module for Stress Estimation.

This module takes a sequence of frame-level embeddings Z = {z1, ..., zT}
produced by the Swin-FANE encoder and models temporal emotional dynamics
to derive a continuous stress-related score.

Architecture (paper Section 4.3, Algorithm 3):
  1. BiLSTM:             Captures bidirectional temporal dependencies
                         h_t = BiLSTM(z_t)
  2. Residual 1D-Conv:   Smooths transient micro-expression noise
                         u_t = h_t + Conv1D(h_t)
  3. Sigmoid Projection: Maps final temporal representation to stress score
                         s = σ(W_s · u_T + b_s)

Computational complexity:
  - BiLSTM: O(2Td²)  — paper Equation 17
  - Conv1D refinement: O(Td)  — negligible overhead

Paper reference: Sections 4.3–4.4, Equations 11–15, Algorithm 3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class ResidualTemporalBlock(nn.Module):
    """
    Lightweight residual 1D-convolution block for smoothing temporal
    emotion trajectories (paper Section 4.3, Equation 12).

    u_t = h_t + Conv1D(h_t)

    Uses depthwise-separable convolution for efficiency.

    Args:
        channels:    Number of feature channels (matches BiLSTM hidden×2 for BiDir).
        kernel_size: Convolution kernel size (default 3 → ±1 frame context).
    """

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2     # 'same' padding to preserve sequence length

        # Depthwise convolution (efficient residual smoothing)
        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,           # Depthwise: one filter per channel
            bias=False,
        )
        # Pointwise 1×1 convolution to mix channels after depthwise
        self.pointwise = nn.Conv1d(channels, channels, kernel_size=1, bias=False)

        self.norm = nn.LayerNorm(channels)
        self.act  = nn.GELU()

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        Args:
            h: Hidden states (B, T, channels).

        Returns:
            u: Refined hidden states (B, T, channels).
        """
        # Conv1d expects (B, C, T)
        h_t = h.transpose(1, 2)              # (B, channels, T)
        h_t = self.act(self.pointwise(self.conv(h_t)))
        h_t = h_t.transpose(1, 2)            # (B, T, channels)

        # Residual connection: u_t = h_t + Conv1D(h_t)
        u = self.norm(h + h_t)
        return u


class BiLSTMTemporalModule(nn.Module):
    """
    Bidirectional LSTM temporal module with residual refinement.

    Full implementation of the Phase II temporal stress estimation
    described in Section 4.3 of the paper.

    Args:
        input_dim:    Dimension of frame embeddings (must match Swin-FANE output).
        hidden_dim:   BiLSTM hidden state size per direction.
        num_layers:   Number of stacked BiLSTM layers.
        dropout:      Dropout rate between BiLSTM layers.
        residual:     Whether to apply residual 1D-conv refinement.
        stress_head_dim: Hidden dim in the stress score projection head.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,
        num_layers: int = 2,
        dropout: float = 0.3,
        residual: bool = True,
        stress_head_dim: int = 256,
    ):
        super().__init__()

        self.input_dim  = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # ----------------------------------------------------------------
        # BiLSTM (paper Equation 11)
        # hidden_size × 2 because bidirectional
        # ----------------------------------------------------------------
        self.bilstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,            # Expects (B, T, input_dim)
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Output dim = hidden_dim × 2 (forward + backward)
        self.bilstm_out_dim = hidden_dim * 2

        # ----------------------------------------------------------------
        # Residual 1D-conv temporal refinement (paper Equation 12)
        # ----------------------------------------------------------------
        self.residual_block = ResidualTemporalBlock(
            channels=self.bilstm_out_dim,
            kernel_size=3,
        ) if residual else nn.Identity()

        # ----------------------------------------------------------------
        # Stress score projection head (paper Equation 13)
        # s = σ(W_s · u_T + b_s)
        # ----------------------------------------------------------------
        self.stress_head = nn.Sequential(
            nn.Linear(self.bilstm_out_dim, stress_head_dim),
            nn.LayerNorm(stress_head_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(stress_head_dim, 1),
            nn.Sigmoid(),                # Bounded output in [0, 1]
        )

        # Dropout for the final temporal representation
        self.dropout = nn.Dropout(dropout)

        self._init_lstm_weights()

    def _init_lstm_weights(self):
        """
        Initialises LSTM weights using orthogonal initialisation
        for more stable gradient flow in deep recurrent networks.
        """
        for name, param in self.bilstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                nn.init.zeros_(param.data)
                # Set forget gate bias to 1 (common LSTM trick)
                n = param.size(0)
                param.data[n // 4:n // 2].fill_(1.0)

    def forward(
        self,
        Z: torch.Tensor,
        return_hidden: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the temporal stress estimation module.

        Implements Algorithm 3 from the paper.

        Args:
            Z: Sequence of frame embeddings.
               Shape: (B, T, input_dim) where T = sequence length.
            return_hidden: If True, also return the full hidden state sequence.

        Returns:
            Dict with:
              'stress_score': Scalar stress estimate per sequence  (B, 1)
              'hidden':       BiLSTM hidden states                 (B, T, hidden×2)
              'refined':      Residual-refined hidden states       (B, T, hidden×2)
        """
        B, T, _ = Z.shape

        # ---- Step 1: BiLSTM — bidirectional temporal modelling ----
        # h_t captures context from both past and future frames
        H, (h_n, c_n) = self.bilstm(Z)   # H: (B, T, hidden_dim×2)

        # ---- Step 2: Residual 1D-conv refinement ----
        # u_t = h_t + Conv1D(h_t)  — smooths transient noise
        U = self.residual_block(H)         # (B, T, hidden_dim×2)

        # ---- Step 3: Stress score projection ----
        # Use the last time step u_T as the sequence summary representation
        u_T = self.dropout(U[:, -1, :])   # (B, hidden_dim×2)
        s   = self.stress_head(u_T)        # (B, 1)

        output = {
            'stress_score': s,    # Primary output — continuous stress score
            'refined':      U,    # Full refined trajectory (for visualisation)
        }
        if return_hidden:
            output['hidden'] = H  # Raw BiLSTM output (useful for analysis)

        return output

    def forward_sequence_stress(
        self,
        Z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Computes per-timestep stress estimates over a sequence.

        This is used to produce the stress trajectory plot (Figure 9 in paper).

        Args:
            Z: Embedding sequence (B, T, input_dim).

        Returns:
            Per-timestep stress scores (B, T, 1).
        """
        B, T, _ = Z.shape
        H, _ = self.bilstm(Z)       # (B, T, hidden×2)
        U    = self.residual_block(H)

        # Apply stress head to every time step independently
        U_flat  = U.reshape(B * T, -1)      # (B*T, hidden×2)
        s_flat  = self.stress_head(U_flat)  # (B*T, 1)
        s_seq   = s_flat.reshape(B, T, 1)   # (B, T, 1)
        return s_seq


class TemporalMLP(nn.Module):
    """
    Ablation baseline: simple MLP temporal aggregation (no sequential modelling).

    Used in ablation study (paper Section 6.7) to compare against BiLSTM.
    Replaces the BiLSTM with global average pooling + MLP.

    Args:
        input_dim:  Frame embedding dimension.
        hidden_dim: MLP hidden size.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, Z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Args:
            Z: (B, T, input_dim)

        Returns:
            Dict with 'stress_score': (B, 1)
        """
        # Pool over time dimension (ignores temporal order)
        z_pooled = Z.mean(dim=1)          # (B, input_dim)
        s        = self.mlp(z_pooled)     # (B, 1)
        return {'stress_score': s, 'refined': Z}


class GRUTemporalModule(nn.Module):
    """
    Ablation baseline: GRU-based temporal module.

    Used in ablation study (Appendix A.2) to compare GRU vs BiLSTM.

    Args:
        input_dim:  Frame embedding dimension.
        hidden_dim: GRU hidden state size.
    """

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 512):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.3,
        )
        self.stress_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, Z: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        H, _ = self.gru(Z)
        s    = self.stress_head(H[:, -1, :])
        return {'stress_score': s, 'refined': H}
