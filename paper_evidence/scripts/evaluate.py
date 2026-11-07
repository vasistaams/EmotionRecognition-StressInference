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
import csv
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.full_framework import build_framework
from data.fane_dataset import build_fane_dataloaders, FANEDataset
from data.fer2013_dataset import (
    FER2013_CLASSES,
    FER2013FolderDataset,
    build_fer2013_dataloaders,
)
from data.augmentations import get_robustness_transform, get_val_transforms
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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
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


@torch.inference_mode()
def evaluate_fer2013_official_test(
    config: dict,
    config_path: str,
    checkpoint_path: str,
    output_dir: str = None,
) -> Path:
    """Evaluate only the official FER2013 test folders and save isolated outputs."""
    dataset_config = config["dataset"]
    if dataset_config["name"].upper() != "FER2013":
        raise ValueError("--fer2013-test-only requires dataset.name=FER2013")
    if dataset_config["num_classes"] != len(FER2013_CLASSES):
        raise ValueError(
            f"Expected {len(FER2013_CLASSES)} FER2013 classes, "
            f"got {dataset_config['num_classes']}"
        )
    configured_names = dataset_config.get("class_names")
    if configured_names is not None and configured_names != FER2013_CLASSES:
        raise ValueError(
            "Configured class_names do not match the FER2013 model label order: "
            f"{FER2013_CLASSES}"
        )

    dataset_root = Path(dataset_config["root"])
    test_root = dataset_root / "test"
    if not test_root.is_dir():
        raise FileNotFoundError(f"Official FER2013 test split not found: {test_root}")
    if output_dir is not None and Path(output_dir).exists():
        raise FileExistsError(
            f"Output directory already exists; refusing to overwrite: {output_dir}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Invalid model checkpoint: {checkpoint_path}")
    region_mode = get_region_attention_mode(config)
    validate_checkpoint_region_mode(checkpoint, region_mode, checkpoint_path)

    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(config).to(device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    test_dataset = FER2013FolderDataset(
        root_dir=str(dataset_root),
        split="test",
        image_size=dataset_config["image_size"],
        transform=get_val_transforms(dataset_config["image_size"]),
    )
    if not len(test_dataset):
        raise RuntimeError(f"Official FER2013 test split is empty: {test_root}")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
    )

    all_labels = []
    all_predictions = []
    all_probabilities = []
    loss_sum = 0.0
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    for batch in test_loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device)
        logits = model.forward_frame(images)["logits"]
        loss_sum += float(criterion(logits, labels).item())
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        all_probabilities.append(probabilities)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(probabilities.argmax(axis=1).tolist())

    labels = np.asarray(all_labels, dtype=np.int64)
    predictions = np.asarray(all_predictions, dtype=np.int64)
    probabilities = np.concatenate(all_probabilities, axis=0)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(FER2013_CLASSES))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(FER2013_CLASSES))),
        average="macro",
        zero_division=0,
    )
    try:
        roc_auc = float(roc_auc_score(
            labels,
            probabilities,
            labels=list(range(len(FER2013_CLASSES))),
            multi_class="ovr",
            average="macro",
        ))
        roc_auc_note = None
    except ValueError as error:
        roc_auc = None
        roc_auc_note = str(error)

    metrics = {
        "accuracy": float(accuracy_score(labels, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "roc_auc_ovr_macro": roc_auc,
        "roc_auc_unavailable_reason": roc_auc_note,
        "test_loss": loss_sum / len(labels),
    }
    per_class = [
        {
            "label": index,
            "class_name": class_name,
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, class_name in enumerate(FER2013_CLASSES)
    ]
    confusion = confusion_matrix(
        labels,
        predictions,
        labels=list(range(len(FER2013_CLASSES))),
    )
    evaluated_at = datetime.now(timezone.utc).astimezone().isoformat()
    result = {
        "result_type": "newly evaluated official test-split results",
        "evaluated_at": evaluated_at,
        "checkpoint": {
            "filename": Path(checkpoint_path).name,
            "path": str(Path(checkpoint_path).resolve()),
            "epoch": checkpoint.get("epoch"),
            "stored_metrics": checkpoint.get("metrics"),
            "stored_metrics_note": (
                "Previously recorded checkpoint metrics; not recomputed by this "
                "test-set evaluation."
            ),
        },
        "configuration": {
            "path": str(Path(config_path).resolve()),
            "dataset": dataset_config["name"],
            "dataset_root": str(dataset_root.resolve()),
            "dataset_split": "official FER2013 test folder",
            "model_region_attention_mode": region_mode,
            "image_size": dataset_config["image_size"],
            "class_names_in_label_order": FER2013_CLASSES,
            "effective_config": config,
        },
        "evaluation": {
            "device": str(device),
            "sample_count": len(test_dataset),
            "confusion_matrix_label_order": FER2013_CLASSES,
            "metric_definitions": {
                "accuracy": "Fraction of all test predictions that are correct.",
                "balanced_accuracy": "Mean per-class recall, as a fraction.",
                "macro_precision": "Unweighted mean of per-class precision.",
                "macro_recall": "Unweighted mean of per-class recall.",
                "macro_f1": "Unweighted mean of per-class F1.",
                "roc_auc_ovr_macro": (
                    "Macro one-vs-rest multiclass ROC-AUC; null if unsupported "
                    "by the evaluated labels."
                ),
                "test_loss": (
                    "Unweighted cross-entropy summed over test samples then "
                    "divided by sample count; no label smoothing."
                ),
            },
            "metrics": metrics,
            "per_class": per_class,
            "confusion_matrix": confusion.tolist(),
        },
    }

    if output_dir is None:
        timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
        output_path = Path("results/experiments") / (
            f"evaluation_{Path(checkpoint_path).stem}_{timestamp}"
        )
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=False)
    (output_path / "evaluation_metrics.json").write_text(
        json.dumps(result, indent=2, allow_nan=False), encoding="utf-8"
    )

    with (output_path / "classification_report.csv").open(
        "w", newline="", encoding="utf-8"
    ) as report_file:
        writer = csv.DictWriter(
            report_file,
            fieldnames=["label", "class_name", "precision", "recall", "f1", "support"],
        )
        writer.writeheader()
        writer.writerows(per_class)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(confusion, interpolation="nearest", cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(
        xticks=np.arange(len(FER2013_CLASSES)),
        yticks=np.arange(len(FER2013_CLASSES)),
        xticklabels=FER2013_CLASSES,
        yticklabels=FER2013_CLASSES,
        xlabel="Predicted label",
        ylabel="True label",
        title="FER2013 official test confusion matrix",
    )
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    threshold = confusion.max() / 2.0
    for row in range(confusion.shape[0]):
        for column in range(confusion.shape[1]):
            axis.text(
                column,
                row,
                str(confusion[row, column]),
                ha="center",
                va="center",
                color="white" if confusion[row, column] > threshold else "black",
            )
    figure.tight_layout()
    figure.savefig(output_path / "confusion_matrix.png", dpi=160)
    plt.close(figure)

    summary_lines = [
        "# FER2013 official test evaluation",
        "",
        f"- Evaluation date: {evaluated_at}",
        f"- Checkpoint: `{Path(checkpoint_path).name}`",
        f"- Checkpoint epoch: {checkpoint.get('epoch')}",
        f"- Configuration: `{Path(config_path).resolve()}`",
        f"- Dataset split: official FER2013 test folder (`{test_root.resolve()}`)",
        f"- Evaluated samples: {len(test_dataset)}",
        f"- Device: {device}",
        "- Stored checkpoint metrics are historical checkpoint metadata, not "
        "the newly evaluated test metrics below.",
        "",
        "## Newly evaluated metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    summary_lines.extend(
        f"| {name} | {value if value is not None else 'unavailable'} |"
        for name, value in metrics.items()
        if name != "roc_auc_unavailable_reason"
    )
    summary_lines.extend([
        "",
        "Accuracy, balanced accuracy, precision, recall, F1, and ROC-AUC are "
        "fractions in [0, 1]. Test loss is unweighted mean cross-entropy without "
        "label smoothing. Precision/recall/F1 in the CSV report are per-class "
        "fractions; support is a sample count.",
        "",
    ])
    (output_path / "experiment_summary.md").write_text(
        "\n".join(summary_lines), encoding="utf-8"
    )
    print(f"[Evaluate] Newly evaluated official test metrics saved to {output_path}")
    print(json.dumps(metrics, indent=2, allow_nan=False))
    return output_path


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
    parser.add_argument(
        '--fer2013-test-only',
        action='store_true',
        help='Evaluate only the official FER2013 test folders and save a unique report directory',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Optional new output directory for --fer2013-test-only; must not exist',
    )
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
    if args.fer2013_test_only:
        output_dir = evaluate_fer2013_official_test(
            config=config,
            config_path=args.config,
            checkpoint_path=args.checkpoint,
            output_dir=args.output_dir,
        )
        print(f"[Evaluate] Output directory: {output_dir}")
        return
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
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )
    validate_checkpoint_region_mode(checkpoint, region_mode, args.checkpoint)
    model.load_state_dict(checkpoint['model'], strict=True)
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
