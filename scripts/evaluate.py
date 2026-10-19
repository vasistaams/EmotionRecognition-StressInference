"""
evaluate.py
===========
Comprehensive evaluation script for the trained Swin-FANE framework.

Produces:
  - Frame-level emotion recognition metrics (Table 5, 6 in paper)
  - Temporal consistency metrics (Section 5.3.2)
  - Robustness evaluation under perturbations (Section 5.6)
  - Stress trajectory analysis (Section 5.5)
  - All visualisation figures (Section 6)

Usage:
    python scripts/evaluate.py \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/best_model.pth \\
        --output results/

Paper reference: Section 5 (experimental evaluation), Section 6 (results)
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from models.full_framework import build_framework
from data.fane_dataset import build_fane_dataloaders, FANEDataset
from data.fer2013_dataset import build_fer2013_dataloaders
from data.augmentations import get_robustness_transform
from utils.metrics import (
    EmotionMetrics,
    summarize_region_weights,
    compute_etsi, compute_tve, compute_tce, compute_all_temporal_metrics,
    pearson_correlation,
)
from utils.stress_formulation import (
    compute_stress_index,
    compute_stress_trajectory,
    compute_scsi,
    generate_pseudo_labels,
    sensitivity_analysis,
    EMOTION_CLASSES,
)
from utils.visualisation import (
    plot_class_distribution,
    plot_training_curves,
    plot_roc_curves,
    plot_temporal_entropy,
    plot_volatility_stress_correlation,
    plot_prediction_confidence,
    plot_markov_transition_matrix,
    plot_ablation_study,
    plot_confusion_matrix,
    plot_region_weights_by_emotion,
)
from sklearn.metrics import roc_auc_score
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config, set_region_attention_mode


# ---------------------------------------------------------------------------
# Frame-level evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_frame_level(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    num_classes: int = 7,
) -> dict:
    """
    Evaluates frame-level emotion recognition on a DataLoader.

    Returns full metrics dict + raw predictions for visualisation.
    """
    model.eval()
    metrics = EmotionMetrics(num_classes=num_classes)
    region_weights = []

    for batch in loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        output = model.forward_frame(images)
        metrics.update(output['probs'], labels)
        if output.get('region_weights') is not None:
            region_weights.append(output['region_weights'].detach().cpu().numpy())

    results      = metrics.compute()
    cm           = metrics.get_confusion_matrix()
    per_class    = metrics.get_per_class_metrics()
    all_probs    = np.array(metrics.all_probs)
    all_labels   = np.array(metrics.all_labels)
    all_region_weights = (
        np.concatenate(region_weights, axis=0) if region_weights else None
    )

    return {
        'metrics':      results,
        'confusion_matrix': cm,
        'per_class':    per_class,
        'all_probs':    all_probs,
        'all_labels':   all_labels,
        'region_weights': all_region_weights,
        'region_summary': summarize_region_weights(
            all_region_weights, all_labels, num_classes=num_classes
        ) if all_region_weights is not None else None,
    }


# ---------------------------------------------------------------------------
# Temporal evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_temporal(
    model: torch.nn.Module,
    seq_loader,
    device: torch.device,
    alpha: float = 0.7,
    beta: float  = 0.3,
) -> dict:
    """
    Evaluates temporal emotion dynamics and stress estimation.

    Returns temporal metrics, stress trajectories, and emotion sequences.
    """
    model.eval()

    all_stress_scores   = []
    all_volatilities    = []
    all_pred_labels_seq = []   # Per-sequence label lists
    all_probs_seq       = []   # Per-sequence prob arrays
    all_embeddings_seq  = []

    for batch in seq_loader:
        x_seq  = batch['image'].to(device)    # (B, T, C, H, W)

        # Check dimensions — some loaders return (B, C, H, W) for seq_len=1
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)

        output = model.forward_sequence(x_seq)

        B, T = output['frame_probs'].shape[:2]
        for b in range(B):
            probs_np     = output['frame_probs'][b].cpu().numpy()    # (T, K)
            labels_np    = output['frame_labels'][b].cpu().numpy()   # (T,)
            embeds_np    = output['embeddings'][b].cpu().numpy()      # (T, D)

            S, r, v = compute_stress_index(probs_np, alpha=alpha, beta=beta)

            all_stress_scores.append(float(S))
            all_volatilities.append(float(v.mean()))
            all_pred_labels_seq.append(labels_np)
            all_probs_seq.append(probs_np)
            all_embeddings_seq.append(embeds_np)

    # ---- Temporal metrics ----
    all_etsi  = []
    all_tve   = []
    all_tce   = []
    all_scsi  = []

    for i in range(len(all_pred_labels_seq)):
        probs_i    = all_probs_seq[i]
        labels_i   = all_pred_labels_seq[i]
        embeds_i   = all_embeddings_seq[i]
        _, r_i, v_i = compute_stress_index(probs_i, alpha, beta)
        stress_traj = alpha * r_i + beta * v_i

        all_etsi.append(compute_etsi(labels_i))
        all_tve.append(compute_tve(embeds_i))
        all_tce.append(compute_tce(probs_i))
        all_scsi.append(compute_scsi(stress_traj))

    # ---- Stress–volatility Pearson correlation ----
    stress_arr = np.array(all_stress_scores)
    volat_arr  = np.array(all_volatilities)
    pearson_r  = pearson_correlation(volat_arr, stress_arr)

    # ---- Pseudo-label ROC for stress (Section 5.5) ----
    pseudo_labels = generate_pseudo_labels(stress_arr, threshold=0.5)
    # For ROC we need two-class scores — use stress score itself
    try:
        stress_roc_auc = roc_auc_score(pseudo_labels, stress_arr)
    except ValueError:
        stress_roc_auc = float('nan')

    return {
        'etsi':             float(np.mean(all_etsi)),
        'tve':              float(np.mean(all_tve)),
        'tce':              float(np.mean(all_tce)),
        'scsi':             float(np.mean(all_scsi)),
        'mean_stress':      float(stress_arr.mean()),
        'stress_std':       float(stress_arr.std()),
        'pearson_r':        float(pearson_r),
        'stress_roc_auc':   float(stress_roc_auc),
        # Raw arrays for plotting
        'stress_scores':    stress_arr,
        'volatilities':     volat_arr,
        'pred_labels_seqs': all_pred_labels_seq,
        'probs_seqs':       all_probs_seq,
        'embeddings_seqs':  all_embeddings_seq,
    }


# ---------------------------------------------------------------------------
# Robustness evaluation (Section 5.6)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_robustness(
    model: torch.nn.Module,
    dataset: FANEDataset,
    device: torch.device,
    perturbations: dict,
    num_classes: int = 7,
) -> dict:
    """
    Evaluates the model under various visual perturbations.

    Args:
        model:         Trained model.
        dataset:       Base test dataset (samples will be re-transformed).
        device:        Computation device.
        perturbations: Dict mapping perturbation type → list of levels.
                       e.g., {'occlusion': [0.1, 0.2], 'low_light': [0.3, 0.5]}
        num_classes:   Number of emotion classes.

    Returns:
        Dict mapping (perturbation, level) → accuracy.
    """
    robustness_results = {}

    for perturb_type, levels in perturbations.items():
        robustness_results[perturb_type] = {}

        for level in levels:
            transform = get_robustness_transform(
                perturbation=perturb_type,
                level=level,
                image_size=dataset.image_size,
            )

            # Re-run inference with perturbed transform
            metrics = EmotionMetrics(num_classes=num_classes)

            for sample in dataset.samples[:500]:   # Use subset for speed
                import cv2
                img    = cv2.imread(sample['path'])
                if img is None:
                    continue
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                tensor  = transform(img_rgb).unsqueeze(0).to(device)
                label   = torch.tensor([sample['label']]).to(device)

                output  = model.forward_frame(tensor)
                metrics.update(output['probs'], label)

            result = metrics.compute()
            robustness_results[perturb_type][level] = result['accuracy']

            print(f"  Robustness [{perturb_type} @ {level:.2f}]: "
                  f"Acc = {result['accuracy']:.2f}%")

    return robustness_results


# ---------------------------------------------------------------------------
# Generate all paper figures
# ---------------------------------------------------------------------------

def generate_figures(eval_results: dict, output_dir: str = "results"):
    """
    Generates all visualisation figures from evaluation results.

    Args:
        eval_results: Dict containing all evaluation outputs.
        output_dir:   Directory to save figures.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Figure 7: ROC + PR curves
    if 'all_probs' in eval_results.get('frame', {}):
        plot_roc_curves(
            y_true=eval_results['frame']['all_labels'],
            y_prob=eval_results['frame']['all_probs'],
            save_path=os.path.join(output_dir, "roc_pr_curves.png"),
        )

    # Confusion matrix
    if 'confusion_matrix' in eval_results.get('frame', {}):
        plot_confusion_matrix(
            cm=eval_results['frame']['confusion_matrix'],
            save_path=os.path.join(output_dir, "confusion_matrix.png"),
        )

    region_summary = eval_results.get('frame', {}).get('region_summary')
    if region_summary is not None and region_summary.get('n_samples', 0) > 0:
        plot_region_weights_by_emotion(
            per_emotion_mean=region_summary['per_emotion_mean'],
            save_path=os.path.join(output_dir, "region_weights_by_emotion.png"),
        )

    # Figure 8: Temporal entropy (use first available sequence)
    if 'probs_seqs' in eval_results.get('temporal', {}):
        probs_seq_0 = eval_results['temporal']['probs_seqs'][0]
        from scipy.stats import entropy as sp_entropy
        entropies = np.array([sp_entropy(p) for p in probs_seq_0])
        plot_temporal_entropy(
            entropies=entropies,
            save_path=os.path.join(output_dir, "temporal_entropy.png"),
        )

    # Figure 9: Volatility vs stress
    if 'volatilities' in eval_results.get('temporal', {}):
        plot_volatility_stress_correlation(
            volatility=eval_results['temporal']['volatilities'],
            stress_scores=eval_results['temporal']['stress_scores'],
            save_path=os.path.join(output_dir, "volatility_stress.png"),
        )

    # Figure 11: Markov transition matrix
    if 'pred_labels_seqs' in eval_results.get('temporal', {}):
        all_labels_flat = np.concatenate(eval_results['temporal']['pred_labels_seqs'])
        plot_markov_transition_matrix(
            pred_labels=all_labels_flat,
            save_path=os.path.join(output_dir, "markov_matrix.png"),
        )

    print(f"[Evaluate] All figures saved to: {output_dir}")


# ---------------------------------------------------------------------------
# Main evaluation entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Evaluate Swin-FANE Framework')
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, default=None)
    parser.add_argument('--output',      type=str, default=None)
    parser.add_argument('--arwm', action='store_true',
                        help='Legacy alias for --arwm-mode adaptive')
    parser.add_argument('--arwm-mode', choices=['off', 'fixed', 'adaptive'],
                        default=None,
                        help='Require a checkpoint trained for the selected mode')
    parser.add_argument('--robustness',  action='store_true',
                        help='Run robustness evaluation')
    parser.add_argument('--temporal',    action='store_true',
                        help='Run temporal metrics evaluation')
    args = parser.parse_args()

    # ---- Load config ----
    config = load_config(args.config)
    if args.arwm_mode:
        set_region_attention_mode(config, args.arwm_mode)
    elif args.arwm:
        set_region_attention_mode(config, 'adaptive')
    region_mode = get_region_attention_mode(config)
    arwm_enabled = region_mode != 'off'
    if args.checkpoint is None:
        if region_mode == 'fixed':
            args.checkpoint = 'checkpoints/arwm_variants/fixed/best_model.pth'
        elif region_mode == 'adaptive':
            args.checkpoint = os.path.join(
                config.get('paths', {}).get('arwm_checkpoint_dir', 'checkpoints/fer2013_arwm'),
                'best_model.pth',
            )
        else:
            args.checkpoint = config.get('paths', {}).get('best_model', 'checkpoints/best_model.pth')
    if args.output is None:
        if args.arwm_mode:
            args.output = os.path.join(
                'results', 'arwm', 'variants', region_mode, 'evaluation'
            )
        else:
            args.output = (
                'results/fer2013_arwm'
                if arwm_enabled
                else config.get('paths', {}).get('results_dir', 'results')
            )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Evaluate] Device: {device}")

    # ---- Build model and load weights ----
    config.setdefault('model', {}).setdefault('swin', {})['pretrained'] = False
    model = build_framework(config).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    validate_checkpoint_region_mode(checkpoint, region_mode, args.checkpoint)
    model.load_state_dict(checkpoint['model'])
    print(f"[Evaluate] Loaded checkpoint from: {args.checkpoint}")
    print(f"           (trained to epoch {checkpoint.get('epoch', '?')})")

    # ---- Build test dataloader ----
    dataset_name = config['dataset']['name'].upper()
    num_classes = config['dataset']['num_classes']
    if dataset_name == 'FER2013':
        _, _, test_loader = build_fer2013_dataloaders(
            root_dir=config['dataset']['root'],
            image_size=config['dataset']['image_size'],
            batch_size=config['training']['batch_size'],
            num_workers=0,
            pin_memory=False,
        )
    else:
        test_loader = build_fane_dataloaders(
            root_dir=config['dataset']['root'],
            image_size=config['dataset']['image_size'],
            sequence_length=1,
            batch_size=config['training']['batch_size'],
            num_workers=config['training']['num_workers'],
            subject_independent=config['dataset']['subject_independent'],
        )[2]

    # ---- Frame-level evaluation ----
    print("\n[Evaluate] Running frame-level evaluation...")
    frame_results = evaluate_frame_level(
        model, test_loader, device, num_classes=num_classes
    )

    print("\n=== Frame-Level Results ===")
    for k, v in frame_results['metrics'].items():
        if isinstance(v, float):
            print(f"  {k:<25}: {v:.4f}")

    print("\n=== Per-Class Metrics ===")
    for class_name, class_metrics in frame_results['per_class'].items():
        print(
            f"  {class_name:<12}: "
            f"Precision {class_metrics['precision']:.4f} | "
            f"Recall {class_metrics['recall']:.4f} | "
            f"F1 {class_metrics['f1']:.4f}"
        )

    eval_results = {'frame': frame_results}

    if arwm_enabled and frame_results['region_summary'] is not None:
        os.makedirs(args.output, exist_ok=True)
        region_file = os.path.join(args.output, 'region_weights.json')
        with open(region_file, 'w') as f:
            json.dump(frame_results['region_summary'], f, indent=2)
        print(f"[Evaluate] Region weights saved to: {region_file}")

    # ---- Temporal evaluation ----
    if args.temporal:
        print("\n[Evaluate] Running temporal evaluation (seq_len=16)...")
        _, _, seq_loader = build_fane_dataloaders(
            root_dir=config['dataset']['root'],
            image_size=config['dataset']['image_size'],
            sequence_length=config['dataset']['sequence_length'],
            batch_size=8,
            num_workers=config['training']['num_workers'],
        )
        alpha = config['stress']['alpha']
        beta  = config['stress']['beta']
        temp_results = evaluate_temporal(model, seq_loader, device, alpha, beta)
        eval_results['temporal'] = temp_results

        print("\n=== Temporal Metrics ===")
        for k, v in temp_results.items():
            if isinstance(v, float):
                print(f"  {k:<25}: {v:.4f}")

    # ---- Generate figures ----
    generate_figures(eval_results, output_dir=args.output)

    # ---- Save numerical results ----
    results_file = os.path.join(args.output, "evaluation_results.json")
    save_dict = {
        'frame': {k: (v.tolist() if hasattr(v, 'tolist') else v)
                  for k, v in frame_results['metrics'].items()},
        'per_class': frame_results['per_class'],
    }
    if arwm_enabled and frame_results['region_summary'] is not None:
        save_dict['arwm'] = frame_results['region_summary']
    if 'temporal' in eval_results:
        save_dict['temporal'] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                v.tolist() if hasattr(v, 'tolist') else v)
            for k, v in eval_results['temporal'].items()
            if k not in {'pred_labels_seqs', 'probs_seqs', 'embeddings_seqs'}
        }

    with open(results_file, 'w') as f:
        json.dump(save_dict, f, indent=2)
    print(f"\n[Evaluate] Results saved to: {results_file}")


if __name__ == '__main__':
    main()
