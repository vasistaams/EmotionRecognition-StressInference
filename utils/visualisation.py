"""
visualisation.py
================
Visualisation utilities for the Swin-FANE framework.

Produces all figures referenced in the paper:
  - Figure 1:  Emotion class distribution (bar chart)
  - Figure 6:  Training convergence curves
  - Figure 7:  Multi-class ROC and Precision-Recall curves
  - Figure 8:  Temporal prediction entropy
  - Figure 9:  Emotion volatility vs. stress score correlation
  - Figure 10: Temporal prediction confidence stability
  - Figure 11: Markov transition probability matrix (heatmap)
  - Figure 12: Markov transition network (directed graph)
  - Figure 13: Ablation study bar chart
  - Grad-CAM:  Saliency maps on facial images

Paper reference: Section 6 (all figures)
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from typing import Dict, List, Optional, Tuple

from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.preprocessing import label_binarize

# ---- Emotion class labels ----
EMOTION_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise", "neutral"]
EMOTION_COLORS  = ['#e74c3c', '#8e44ad', '#3498db', '#f1c40f',
                   '#2ecc71', '#e67e22', '#95a5a6']


# ---------------------------------------------------------------------------
# Figure 1: Emotion class distribution
# ---------------------------------------------------------------------------

def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: str = "results/class_distribution.png",
):
    """Plots the emotion class distribution bar chart (Figure 1 in paper)."""
    fig, ax = plt.subplots(figsize=(10, 6))

    classes = list(class_counts.keys())
    counts  = list(class_counts.values())
    colors  = EMOTION_COLORS[:len(classes)]

    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.8)

    # Annotate bars with counts
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f'{count:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_title('Emotion Class Distribution', fontsize=14, fontweight='bold')
    ax.set_xlabel('Emotion Category', fontsize=12)
    ax.set_ylabel('Number of Samples', fontsize=12)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved class distribution → {save_path}")


# ---------------------------------------------------------------------------
# Figure 6: Training convergence
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_accs: List[float],
    val_accs: List[float],
    train_losses: Optional[List[float]] = None,
    val_losses: Optional[List[float]] = None,
    save_path: str = "results/training_convergence.png",
):
    """Plots training and validation accuracy/loss convergence (Figure 6)."""
    n_plots = 2 if (train_losses and val_losses) else 1
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(train_accs) + 1)

    # Accuracy plot
    ax = axes[0]
    ax.plot(epochs, train_accs, 'o-', color='#2196F3', linewidth=2,
            markersize=5, label='Training accuracy')
    ax.plot(epochs, val_accs,   's--', color='#FF5722', linewidth=2,
            markersize=5, label='Validation accuracy')

    # Annotate final accuracy
    final_val = val_accs[-1]
    ax.axhline(y=final_val, color='#FF5722', linestyle=':', alpha=0.5)
    ax.text(len(epochs) * 0.7, final_val + 0.5, f'{final_val:.1f}%',
            color='#FF5722', fontsize=10)

    ax.set_title('Training and Validation Accuracy Convergence\nover 50 Epochs',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('Training Epoch', fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(55, 100)
    ax.grid(True, alpha=0.3)

    # Loss plot (optional)
    if n_plots > 1 and train_losses and val_losses:
        ax2 = axes[1]
        ax2.plot(epochs, train_losses, 'o-', color='#2196F3', linewidth=2,
                 markersize=5, label='Training loss')
        ax2.plot(epochs, val_losses,   's--', color='#FF5722', linewidth=2,
                 markersize=5, label='Validation loss')
        ax2.set_title('Loss Convergence', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Training Epoch', fontsize=11)
        ax2.set_ylabel('Loss', fontsize=11)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved training curves → {save_path}")


# ---------------------------------------------------------------------------
# Figure 7: ROC and Precision-Recall curves
# ---------------------------------------------------------------------------

def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str] = EMOTION_CLASSES,
    save_path: str = "results/roc_curves.png",
):
    """Plots multi-class ROC curves (Figure 7a)."""
    n_classes = len(class_names)
    y_bin     = label_binarize(y_true, classes=list(range(n_classes)))

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- ROC curves ---
    ax = axes[0]
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.']
    colors      = EMOTION_COLORS

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, color=colors[i], linestyle=line_styles[i],
                label=f'Class {i} (AUC = {roc_auc_val:.2f})')

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Chance')
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('Multi-Class ROC Curves', fontsize=12, fontweight='bold')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # --- Precision-Recall curves ---
    ax2 = axes[1]
    for i, cls_name in enumerate(class_names):
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_prob[:, i])
        ap           = average_precision_score(y_bin[:, i], y_prob[:, i])
        ax2.plot(rec, prec, lw=2, color=colors[i], linestyle=line_styles[i],
                 label=f'{cls_name.capitalize()} (AP={ap:.2f})')

    # Chance level
    ax2.axhline(y=1.0 / n_classes, color='k', linestyle='--', lw=1,
                label=f'Chance ({1.0/n_classes:.2f})')
    ax2.set_xlabel('Recall', fontsize=11)
    ax2.set_ylabel('Precision', fontsize=11)
    ax2.set_title('Multi-Class Precision–Recall Curves', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved ROC/PR curves → {save_path}")


# ---------------------------------------------------------------------------
# Figure 8: Temporal prediction entropy
# ---------------------------------------------------------------------------

def plot_temporal_entropy(
    entropies: np.ndarray,
    save_path: str = "results/temporal_entropy.png",
    window: int = 7,
):
    """Plots temporal variation in prediction entropy (Figure 8)."""
    T = len(entropies)

    # Moving average
    kernel  = np.ones(window) / window
    smoothed = np.convolve(entropies, kernel, mode='same')

    # Max entropy (uniform over K=7 classes)
    max_entropy = np.log(7)   # Natural log of number of classes

    fig, ax = plt.subplots(figsize=(12, 5))
    frames = np.arange(T)

    ax.fill_between(frames, entropies, alpha=0.35, color='#90CAF9', label='Instantaneous entropy H(t)')
    ax.plot(frames, smoothed, color='#c0392b', linewidth=2,
            label=f'Moving average (w={window} frames)')
    ax.axhline(y=max_entropy, color='#455A64', linestyle='--', lw=1.5,
               label='Max H (uniform)')

    ax.set_xlim(0, T)
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Shannon Entropy H(eₜ) (nats)', fontsize=11)
    ax.set_title('Temporal Variation in Prediction Entropy\nHighlights Ambiguous vs. Confident Frames',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved temporal entropy → {save_path}")


# ---------------------------------------------------------------------------
# Figure 9: Emotion volatility vs. stress score
# ---------------------------------------------------------------------------

def plot_volatility_stress_correlation(
    volatility: np.ndarray,
    stress_scores: np.ndarray,
    save_path: str = "results/volatility_stress_correlation.png",
):
    """Plots emotion volatility vs. stress score scatter (Figure 9)."""
    r = np.corrcoef(volatility, stress_scores)[0, 1]

    # Linear regression line
    m, b  = np.polyfit(volatility, stress_scores, 1)
    x_fit = np.linspace(volatility.min(), volatility.max(), 100)
    y_fit = m * x_fit + b

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(volatility, stress_scores, alpha=0.5, s=40, color='#5C6BC0',
               edgecolors='none', label='Sample frames')
    ax.plot(x_fit, y_fit, color='#e74c3c', linewidth=2,
            label=f'Linear fit (r = {r:.3f})')

    ax.set_xlabel('Emotion Volatility Index V(t)', fontsize=11)
    ax.set_ylabel('Derived Stress Score S(t)', fontsize=11)
    ax.set_title('Relationship Between Predicted Emotion Volatility\n'
                 'and Estimated Psychological Stress Score', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved volatility-stress plot → {save_path} (r={r:.3f})")


# ---------------------------------------------------------------------------
# Figure 10: Temporal prediction confidence
# ---------------------------------------------------------------------------

def plot_prediction_confidence(
    confidences: np.ndarray,
    save_path: str = "results/prediction_confidence.png",
    window: int = 9,
):
    """Plots temporal stability of prediction confidence (Figure 10)."""
    T        = len(confidences)
    kernel   = np.ones(window) / window
    smoothed = np.convolve(confidences, kernel, mode='same')

    fig, ax = plt.subplots(figsize=(10, 5))
    frames  = np.arange(T)

    ax.fill_between(frames, confidences, alpha=0.3, color='#90CAF9',
                    label='Frame-level confidence c(t)')
    ax.plot(frames, smoothed, color='#c0392b', linewidth=2,
            label=f'Moving average (w={window} frames)')

    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Softmax Prediction Confidence c(t)', fontsize=11)
    ax.set_title('Temporal Stability of Prediction Confidence\nAcross Consecutive Video Frames',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved confidence plot → {save_path}")


# ---------------------------------------------------------------------------
# Figure 11: Markov transition matrix
# ---------------------------------------------------------------------------

def plot_markov_transition_matrix(
    pred_labels: np.ndarray,
    class_names: List[str] = EMOTION_CLASSES,
    save_path: str = "results/markov_matrix.png",
):
    """Plots the first-order Markov transition probability matrix (Figure 11)."""
    K = len(class_names)

    # Build transition count matrix
    counts = np.zeros((K, K), dtype=int)
    for t in range(len(pred_labels) - 1):
        i = pred_labels[t]
        j = pred_labels[t + 1]
        if 0 <= i < K and 0 <= j < K:
            counts[i, j] += 1

    # Normalise rows to get transition probabilities
    row_sums = counts.sum(axis=1, keepdims=True) + 1e-8
    trans_prob = counts / row_sums

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(trans_prob, cmap='Blues', vmin=0, vmax=0.20)

    # Annotate cells with values
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f'{trans_prob[i, j]:.2f}',
                    ha='center', va='center', fontsize=9,
                    color='white' if trans_prob[i, j] > 0.12 else 'black')

    plt.colorbar(im, ax=ax, label='Transition probability p(eₜ₊₁|eₜ)')
    names_display = [c.capitalize() for c in class_names]
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(names_display, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(names_display, fontsize=9)
    ax.set_xlabel('Next Emotion State eₜ₊₁', fontsize=11)
    ax.set_ylabel('Current Emotion State eₜ', fontsize=11)
    ax.set_title('First-Order Markov Transition Probability Matrix\n'
                 'Estimated from Predicted Emotion Sequences', fontsize=12, fontweight='bold')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved Markov matrix → {save_path}")


# ---------------------------------------------------------------------------
# Figure 13: Ablation study
# ---------------------------------------------------------------------------

def plot_ablation_study(
    component_names: List[str],
    full_accuracies: List[float],
    ablated_accuracies: List[float],
    save_path: str = "results/ablation_study.png",
):
    """Plots ablation study results (Figure 13)."""
    x    = np.arange(len(component_names))
    w    = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    b1 = ax.bar(x - w/2, full_accuracies,    w, label='Full model',
                color='#2196F3', edgecolor='white')
    b2 = ax.bar(x + w/2, ablated_accuracies, w, label='Variant (component removed)',
                color='#c0392b', edgecolor='white')

    # Annotate bars
    for bar in b1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)
    for bar in b2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                f'{bar.get_height():.1f}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Architectural Configuration', fontsize=11)
    ax.set_ylabel('Recognition Accuracy (%)', fontsize=11)
    ax.set_title('Ablation Study: Impact of Each Architectural\nComponent on Validation Accuracy',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(component_names, fontsize=9)
    ax.legend(fontsize=10)
    ax.set_ylim(min(ablated_accuracies) - 2, max(full_accuracies) + 2)
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved ablation study → {save_path}")


# ---------------------------------------------------------------------------
# Confusion matrix heatmap
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = EMOTION_CLASSES,
    normalise: bool = True,
    save_path: str = "results/confusion_matrix.png",
):
    """Plots the confusion matrix as a heatmap."""
    if normalise:
        cm_plot  = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)
        fmt      = '.2f'
        title    = 'Normalised Confusion Matrix'
    else:
        cm_plot  = cm
        fmt      = 'd'
        title    = 'Confusion Matrix'

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cm_plot, annot=True, fmt=fmt,
        cmap='Blues', ax=ax,
        xticklabels=[c.capitalize() for c in class_names],
        yticklabels=[c.capitalize() for c in class_names],
        linewidths=0.5, linecolor='white',
    )
    ax.set_xlabel('Predicted Label', fontsize=11)
    ax.set_ylabel('True Label', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved confusion matrix → {save_path}")


def plot_region_weights_by_emotion(
    per_emotion_mean: Dict[str, Dict[str, float]],
    save_path: str = "results/region_weights_by_emotion.png",
):
    """Plots mean ARWM region weights grouped by emotion."""
    class_names = list(per_emotion_mean.keys())
    region_names = ['eye', 'eyebrow', 'mouth']
    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(11, 6))
    for index, region in enumerate(region_names):
        values = [per_emotion_mean[name].get(region, 0.0) for name in class_names]
        ax.bar(x + (index - 1) * width, values, width, label=region.capitalize())

    ax.set_xlabel('Emotion Category', fontsize=11)
    ax.set_ylabel('Mean ARWM Region Weight', fontsize=11)
    ax.set_title('Adaptive Region Weights by Emotion', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([name.capitalize() for name in class_names], rotation=35, ha='right')
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.2, axis='y')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualisation] Saved region weights plot → {save_path}")
