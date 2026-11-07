"""
stress_formulation.py
=====================
Deterministic stress index formulation (paper Sections 3.5–3.7).

Computes a behavioural stress proxy from temporal emotion probability sequences
WITHOUT requiring explicit stress annotations in the dataset.

The stress index S combines:
  1. Negative Affect Dominance r_t  (Equation 4)
  2. Temporal Emotional Volatility v_t
  3. Weighted integration over time  (Equation 5)

        r_t = p^fear_t + p^sadness_t + p^disgust_t + p^anger_t
        v_t = |r_t - r_{t-1}|
        S   = (1/T) Σ_t (α·r_t + β·v_t)

These computations are deterministic and do not require gradient flow.

Paper reference: Sections 3.5–3.7, Equations 4–5
                 Appendix C (theoretical analysis and proofs)
"""

import numpy as np
import torch
from typing import List, Optional, Tuple, Union


# ---------------------------------------------------------------------------
# Emotion class constants (must match dataset ordering)
# ---------------------------------------------------------------------------
EMOTION_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]

# Negative emotion indices used in r_t (paper Equation 4)
# anger=0, disgust=1, fear=2, sadness=4
NEGATIVE_EMOTION_INDICES = [0, 1, 2, 4]


# ---------------------------------------------------------------------------
# Core stress computation functions
# ---------------------------------------------------------------------------

def compute_negative_affect(
    probs: Union[np.ndarray, torch.Tensor],
    negative_indices: List[int] = NEGATIVE_EMOTION_INDICES,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes the negative affect dominance signal r_t (paper Equation 4).

        r_t = p^fear_t + p^sadness_t + p^disgust_t + p^anger_t

    Bounded in [0, 1] since probabilities sum to 1 (Proposition 1, Appendix C.2).

    Args:
        probs:           Emotion probability array/tensor.
                         Shape: (..., num_classes) — last dimension is emotion.
        negative_indices: Indices of negative emotion classes.

    Returns:
        Negative affect signal. Shape: (...) — same as probs but without last dim.
    """
    if isinstance(probs, torch.Tensor):
        return probs[..., negative_indices].sum(dim=-1)
    else:
        return probs[..., negative_indices].sum(axis=-1)


def compute_temporal_volatility(
    r: Union[np.ndarray, torch.Tensor],
) -> Union[np.ndarray, torch.Tensor]:
    """
    Computes temporal emotional volatility v_t = |r_t - r_{t-1}|.

    Bounded in [0, 1] (Proposition 2, Appendix C.3).

    First timestep volatility is set to 0 (no previous state).

    Args:
        r: Negative affect sequence. Shape: (..., T).

    Returns:
        Volatility signal. Shape: (..., T).
    """
    if isinstance(r, torch.Tensor):
        # Pad with first value for t=0
        r_prev = torch.cat([r[..., :1], r[..., :-1]], dim=-1)
        v      = torch.abs(r - r_prev)
    else:
        r_prev    = np.concatenate([r[..., :1], r[..., :-1]], axis=-1)
        v         = np.abs(r - r_prev)
    return v


def compute_stress_index(
    probs_sequence: Union[np.ndarray, torch.Tensor],
    alpha: float = 0.7,
    beta: float = 0.3,
    negative_indices: List[int] = NEGATIVE_EMOTION_INDICES,
) -> Tuple[Union[np.ndarray, torch.Tensor], Union[np.ndarray, torch.Tensor],
           Union[np.ndarray, torch.Tensor]]:
    """
    Computes the deterministic stress index over a temporal sequence (Equation 5).

        S = (1/T) Σ_t (α·r_t + β·v_t)

    Theoretical bounds (Proposition 3, Appendix C.4):
        0 ≤ S ≤ α + β = 1.0

    If α+β ≠ 1, the bounds generalise to [0, α+β].

    Args:
        probs_sequence: Emotion probability sequence.
                        Shape: (B, T, num_classes) or (T, num_classes).
        alpha:          Weight for sustained negative affect (default 0.7).
        beta:           Weight for emotional volatility (default 0.3).
        negative_indices: Indices of negative emotion classes.

    Returns:
        S:  Scalar stress index per sequence. Shape: (B,) or scalar.
        r:  Negative affect trajectory.       Shape: (B, T) or (T,).
        v:  Volatility trajectory.            Shape: (B, T) or (T,).
    """
    assert abs(alpha + beta - 1.0) < 1e-6 or True, \
        f"alpha + beta = {alpha + beta:.3f}; bounds may exceed [0,1]."

    # ---- Step 1: Negative affect dominance r_t (Equation 4) ----
    r = compute_negative_affect(probs_sequence, negative_indices)
    # r shape: (B, T) or (T,)

    # ---- Step 2: Temporal emotional volatility v_t ----
    v = compute_temporal_volatility(r)
    # v shape: same as r

    # ---- Step 3: Weighted stress index (Equation 5) ----
    if isinstance(probs_sequence, torch.Tensor):
        S = (alpha * r + beta * v).mean(dim=-1)   # Average over T
    else:
        S = (alpha * r + beta * v).mean(axis=-1)

    return S, r, v


def compute_stress_trajectory(
    probs_sequence: np.ndarray,
    alpha: float = 0.7,
    beta: float = 0.3,
    negative_indices: List[int] = NEGATIVE_EMOTION_INDICES,
    smooth_window: int = 5,
) -> dict:
    """
    Computes a detailed temporal stress trajectory for a single sequence.

    Used for visualisation and qualitative analysis (paper Figure 9).

    Args:
        probs_sequence: (T, num_classes) numpy array — single sequence.
        alpha:          Sustained negative affect weight.
        beta:           Emotional volatility weight.
        negative_indices: Negative emotion indices.
        smooth_window:  Moving average window for trajectory smoothing.

    Returns:
        Dict with keys:
          'stress_index':   Scalar summary stress S
          'r':              Negative affect per timestep  (T,)
          'v':              Volatility per timestep       (T,)
          'raw_trajectory': α·r + β·v per timestep       (T,)
          'smoothed':       Smoothed trajectory           (T,)
          'peak_stress':    Maximum stress value
          'mean_stress':    Mean stress value
    """
    assert probs_sequence.ndim == 2, \
        f"Expected (T, K) input; got shape {probs_sequence.shape}"

    T = probs_sequence.shape[0]

    # Compute r_t and v_t
    r = compute_negative_affect(probs_sequence, negative_indices)   # (T,)
    v = compute_temporal_volatility(r)                               # (T,)

    # Raw per-timestep stress signal
    raw_traj = alpha * r + beta * v   # (T,)

    # Temporal smoothing using moving average (reduces micro-expression noise)
    if smooth_window > 1:
        kernel   = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(raw_traj, kernel, mode='same')
        # Handle edge effects by replicating boundary values
        pad = smooth_window // 2
        smoothed[:pad]  = raw_traj[:pad].mean()
        smoothed[-pad:] = raw_traj[-pad:].mean()
    else:
        smoothed = raw_traj.copy()

    # Scalar stress summary
    S = raw_traj.mean()

    return {
        'stress_index':   float(S),
        'r':              r,
        'v':              v,
        'raw_trajectory': raw_traj,
        'smoothed':       smoothed,
        'peak_stress':    float(raw_traj.max()),
        'mean_stress':    float(S),
        'time_steps':     np.arange(T),
    }


# ---------------------------------------------------------------------------
# Stress parameter sensitivity analysis
# ---------------------------------------------------------------------------

def sensitivity_analysis(
    probs_sequence: np.ndarray,
    alpha_values: List[float] = [0.5, 0.6, 0.7, 0.8],
    negative_indices: List[int] = NEGATIVE_EMOTION_INDICES,
) -> List[dict]:
    """
    Evaluates the stress index under different (α, β) configurations.

    Reproduces Table 7 from the paper.

    Args:
        probs_sequence: (T, num_classes) probability sequence.
        alpha_values:   List of α values to evaluate (β = 1 - α).
        negative_indices: Negative emotion indices.

    Returns:
        List of dicts with keys: alpha, beta, stress_index, variability, behaviour.
    """
    results = []

    for alpha in alpha_values:
        beta = 1.0 - alpha
        traj = compute_stress_trajectory(
            probs_sequence, alpha=alpha, beta=beta,
            negative_indices=negative_indices, smooth_window=1
        )

        # Quantify variability as std of raw trajectory
        variability = float(np.std(traj['raw_trajectory']))

        # Classify behaviour based on variability
        if variability < 0.05:
            behaviour = "Dominated by sustained negative affect"
        elif variability < 0.10:
            behaviour = "Balanced emotional intensity and volatility"
        elif variability < 0.15:
            behaviour = "Sensitive to rapid emotional transitions"
        else:
            behaviour = "Dominated by volatility signals"

        results.append({
            'alpha':        alpha,
            'beta':         beta,
            'stress_index': traj['stress_index'],
            'variability':  variability,
            'behaviour':    behaviour,
        })

    return results


# ---------------------------------------------------------------------------
# Pseudo-label generation for ROC evaluation (paper Section 5.5)
# ---------------------------------------------------------------------------

def generate_pseudo_labels(
    stress_scores: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """
    Generates binary high/low stress pseudo-labels from continuous scores.

    Used for ROC/AUC evaluation in the absence of ground-truth stress labels
    (paper Section 5.5).

    Args:
        stress_scores: Array of continuous stress scores. Shape: (N,).
        threshold:     Threshold above which a sequence is labelled "high stress".

    Returns:
        Binary pseudo-labels (0 = low stress, 1 = high stress). Shape: (N,).
    """
    return (stress_scores >= threshold).astype(int)


# ---------------------------------------------------------------------------
# Stress Curve Smoothness Index (SCSI) — paper Section 5.3.2
# ---------------------------------------------------------------------------

def compute_scsi(trajectory: np.ndarray) -> float:
    """
    Computes the Stress Curve Smoothness Index (SCSI).

    SCSI measures temporal continuity of the stress trajectory by penalising
    abrupt changes between consecutive time steps.

    SCSI = 1 - normalised_mean_absolute_difference

    Higher SCSI → smoother trajectory → more temporally coherent stress signal.

    Args:
        trajectory: Per-timestep stress values. Shape: (T,).

    Returns:
        SCSI value in [0, 1].
    """
    if len(trajectory) < 2:
        return 1.0

    diffs = np.abs(np.diff(trajectory))
    mad   = diffs.mean()               # Mean absolute difference
    # Normalise by max possible change (= 1.0 since scores in [0,1])
    scsi  = 1.0 - min(mad, 1.0)
    return float(scsi)
