"""
run_ablation.py
===============
Ablation study runner for the Swin-FANE framework.

Reproduces Table 7 and Figure 13 from the paper by systematically
disabling individual architectural components and measuring impact.

Ablation variants (paper Section 5.7):
  1. Full model (baseline)
  2. w/o FANE module          → accuracy drop ~4.5pp
  3. w/o Shifted window attn  → accuracy drop ~3.1pp
  4. MLP instead of BiLSTM   → lower stress smoothness
  5. w/o Residual refinement  → lower SCSI
  6. Reduced sequence length  → lower temporal metrics

Temporal architecture comparison (Appendix A.2):
  MLP → GRU → BiLSTM → BiLSTM+Residual  (SCSI: 0.71 → 0.82 → 0.86 → 0.91)

Usage:
    python scripts/run_ablation.py \\
        --config configs/config.yaml \\
        --checkpoint checkpoints/best_model.pth

Paper reference: Section 5.7, Section 6.7, Appendix A.2
"""

import os
import sys
import argparse
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import yaml

from models.full_framework import SwinFANEFramework, build_framework
from data.fane_dataset import build_fane_dataloaders
from utils.metrics import EmotionMetrics
from utils.stress_formulation import compute_stress_index, compute_scsi, EMOTION_CLASSES
from utils.visualisation import plot_ablation_study


# ---------------------------------------------------------------------------
# Individual ablation configurations
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "Full Model": {
        "temporal_model": "bilstm",
        "fane_enabled":   True,
        "shifted_window": True,
        "residual":       True,
        "seq_len":        16,
    },
    "w/o FANE": {
        "temporal_model": "bilstm",
        "fane_enabled":   False,   # Disable FANE attention
        "shifted_window": True,
        "residual":       True,
        "seq_len":        16,
    },
    "w/o Shifted Window": {
        "temporal_model": "bilstm",
        "fane_enabled":   True,
        "shifted_window": False,   # Disable shifted window attention
        "residual":       True,
        "seq_len":        16,
    },
    "w/o Temporal": {
        "temporal_model": "mlp",   # MLP instead of BiLSTM
        "fane_enabled":   True,
        "shifted_window": True,
        "residual":       True,
        "seq_len":        16,
    },
    "MLP Temporal Baseline": {
        "temporal_model": "mlp",
        "fane_enabled":   True,
        "shifted_window": True,
        "residual":       False,
        "seq_len":        16,
    },
    "w/o Residual Refinement": {
        "temporal_model": "bilstm_no_residual",
        "fane_enabled":   True,
        "shifted_window": True,
        "residual":       False,   # No residual conv smoothing
        "seq_len":        16,
    },
}

TEMPORAL_ARCH_CONFIGS = {
    "MLP":             "mlp",
    "GRU":             "gru",
    "BiLSTM":          "bilstm",
    "BiLSTM+Residual": "bilstm",
}


# ---------------------------------------------------------------------------
# Run a single ablation variant
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_ablation_variant(
    config: dict,
    variant_config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Evaluates one ablation variant on the test set.

    Args:
        config:           Full training config.
        variant_config:   Ablation variant overrides.
        checkpoint_path:  Path to trained model checkpoint.
        device:           Computation device.

    Returns:
        Dict with accuracy, SCSI, and other metrics.
    """
    # Build model with ablation modifications
    temporal_model = variant_config.get("temporal_model", "bilstm")
    seq_len        = variant_config.get("seq_len", 16)

    model = SwinFANEFramework(
        swin_model_name=config['model']['swin']['variant'],
        pretrained=False,           # Don't re-download for ablation
        embedding_dim=config['model']['classifier']['embedding_dim'],
        num_classes=config['dataset']['num_classes'],
        bilstm_hidden_dim=config['model']['bilstm']['hidden_dim'],
        fane_hidden_dim=config['model']['fane']['hidden_dim'],
        temporal_model=temporal_model,
    ).to(device)

    # Load weights (strict=False to allow architecture differences)
    ckpt = torch.load(checkpoint_path, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt['model'], strict=False)

    # ---- Frame-level evaluation ----
    _, _, test_loader = build_fane_dataloaders(
        root_dir=config['dataset']['root'],
        image_size=config['dataset']['image_size'],
        sequence_length=1,
        batch_size=config['training']['batch_size'],
        num_workers=2,
    )

    metrics = EmotionMetrics(num_classes=config['dataset']['num_classes'])
    model.eval()

    for batch in test_loader:
        images = batch['image'].to(device)
        labels = batch['label'].to(device)
        output = model.forward_frame(images)
        metrics.update(output['probs'], labels)

    frame_results = metrics.compute()

    # ---- Temporal (SCSI) evaluation ----
    _, _, seq_loader = build_fane_dataloaders(
        root_dir=config['dataset']['root'],
        image_size=config['dataset']['image_size'],
        sequence_length=seq_len,
        batch_size=4,
        num_workers=2,
    )

    all_scsi = []
    alpha    = config['stress']['alpha']
    beta     = config['stress']['beta']

    for batch in seq_loader:
        x_seq = batch['image'].to(device)
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)
        output = model.forward_sequence(x_seq)

        B, T = output['frame_probs'].shape[:2]
        for b in range(B):
            probs_b = output['frame_probs'][b].cpu().numpy()
            S, r, v = compute_stress_index(probs_b, alpha, beta)
            traj    = alpha * r + beta * v
            all_scsi.append(compute_scsi(traj))

    return {
        'accuracy': frame_results['accuracy'],
        'f1':       frame_results['f1'],
        'scsi':     float(np.mean(all_scsi)) if all_scsi else 0.0,
    }


# ---------------------------------------------------------------------------
# Temporal architecture comparison
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_temporal_arch_comparison(
    config: dict,
    checkpoint_path: str,
    device: torch.device,
) -> dict:
    """
    Compares SCSI across temporal architecture variants (Appendix A.2).

    Returns:
        Dict mapping architecture name → SCSI.
    """
    results = {}

    for arch_name, temporal_model in TEMPORAL_ARCH_CONFIGS.items():
        model = SwinFANEFramework(
            swin_model_name=config['model']['swin']['variant'],
            pretrained=False,
            embedding_dim=config['model']['classifier']['embedding_dim'],
            num_classes=config['dataset']['num_classes'],
            temporal_model=temporal_model,
        ).to(device)

        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'], strict=False)
        model.eval()

        _, _, seq_loader = build_fane_dataloaders(
            root_dir=config['dataset']['root'],
            image_size=config['dataset']['image_size'],
            sequence_length=16,
            batch_size=4,
            num_workers=2,
        )

        all_scsi = []
        for batch in seq_loader:
            x_seq = batch['image'].to(device)
            if x_seq.dim() == 4:
                x_seq = x_seq.unsqueeze(1)

            output = model.forward_sequence(x_seq)
            B, T   = output['frame_probs'].shape[:2]

            for b in range(B):
                probs_b = output['frame_probs'][b].cpu().numpy()
                alpha   = config['stress']['alpha']
                beta    = config['stress']['beta']
                S, r, v = compute_stress_index(probs_b, alpha, beta)
                traj    = alpha * r + beta * v
                all_scsi.append(compute_scsi(traj))

        results[arch_name] = float(np.mean(all_scsi)) if all_scsi else 0.0
        print(f"  {arch_name:<25}: SCSI = {results[arch_name]:.3f}")

    return results


# ---------------------------------------------------------------------------
# Main ablation runner
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Run ablation studies')
    parser.add_argument('--config',     type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--output',     type=str, default='results')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output, exist_ok=True)

    # ---- Main ablation study ----
    print("\n" + "="*60)
    print("ABLATION STUDY")
    print("="*60)

    ablation_results = {}
    full_model_acc   = None

    for variant_name, variant_cfg in ABLATION_CONFIGS.items():
        print(f"\nVariant: {variant_name}")
        result = run_ablation_variant(
            config, variant_cfg, args.checkpoint, device
        )
        ablation_results[variant_name] = result

        if variant_name == "Full Model":
            full_model_acc = result['accuracy']

        acc_drop = (full_model_acc - result['accuracy']) if full_model_acc else 0
        print(f"  Accuracy: {result['accuracy']:.2f}% "
              f"(drop: {acc_drop:.2f}pp) | "
              f"F1: {result['f1']:.2f}% | "
              f"SCSI: {result['scsi']:.3f}")

    # ---- Temporal architecture comparison ----
    print("\n" + "="*60)
    print("TEMPORAL ARCHITECTURE COMPARISON (Appendix A.2)")
    print("="*60)
    temporal_results = run_temporal_arch_comparison(config, args.checkpoint, device)

    # ---- Save results ----
    all_results = {
        'ablation':  ablation_results,
        'temporal':  temporal_results,
    }
    results_path = os.path.join(args.output, "ablation_results.json")
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)

    # ---- Plot ablation figure ----
    variant_names = list(ABLATION_CONFIGS.keys())
    full_accs     = [ablation_results['Full Model']['accuracy']] * len(variant_names)
    ablated_accs  = [ablation_results[v]['accuracy'] for v in variant_names]

    plot_ablation_study(
        component_names=variant_names,
        full_accuracies=full_accs,
        ablated_accuracies=ablated_accs,
        save_path=os.path.join(args.output, "ablation_study.png"),
    )

    print(f"\nAblation results saved to: {results_path}")


if __name__ == '__main__':
    main()
