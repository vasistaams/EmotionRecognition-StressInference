"""
train.py
========
Main training script for the Swin-FANE framework.

Two-stage training procedure (paper Section 5.1):
  Stage 1: Train the Swin-FANE spatial encoder (frame-level cross-entropy)
  Stage 2: Freeze encoder, train BiLSTM temporal module

Usage:
    python scripts/train.py --config configs/config.yaml

    # Resume training
    python scripts/train.py --config configs/config.yaml \\
                            --resume checkpoints/checkpoint_epoch020.pth

    # Use FER2013 dataset instead of FANE
    python scripts/train.py --config configs/config.yaml --dataset fer2013

Paper reference: Section 5.1 (experimental setup), Section 4.4 (optimisation)
"""

import os
import sys
import argparse
import random
import shutil
import time
from pathlib import Path

# Add project root to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

from models.full_framework import build_framework
from data.fane_dataset import build_fane_dataloaders
from data.fer2013_dataset import build_fer2013_dataloaders
from utils.metrics import EmotionMetrics, RegionWeightMetrics
from utils.logger import TrainingLogger
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import (
    get_region_attention_mode,
    load_config,
    set_region_attention_mode,
)


# ---------------------------------------------------------------------------
# Reproducibility: Fix all random seeds (paper Section 5.1)
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    """Sets all random seeds for reproducibility across runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def compute_inverse_frequency_weights(dataset, num_classes: int):
    """Returns normalized inverse-frequency weights for a dataset split."""
    labels = [
        sample['label'] if isinstance(sample, dict) else sample[1]
        for sample in dataset.samples
    ]
    counts = torch.bincount(
        torch.tensor(labels, dtype=torch.long), minlength=num_classes
    ).float()
    weights = counts.sum() / (num_classes * counts.clamp_min(1.0))
    return weights / weights.mean(), counts


# ---------------------------------------------------------------------------
# Training epoch
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    metrics: EmotionMetrics,
    epoch: int,
    log_interval: int = 10,
    logger: TrainingLogger = None,
) -> dict:
    """
    Runs one training epoch over the dataset.

    Args:
        model:        The framework model.
        loader:       Training DataLoader.
        criterion:    Loss function (cross-entropy with label smoothing).
        optimizer:    AdamW optimizer.
        device:       Computation device (CPU/GPU).
        metrics:      EmotionMetrics accumulator.
        epoch:        Current epoch number (for logging).
        log_interval: Print progress every N batches.
        logger:       Optional TrainingLogger instance.

    Returns:
        Training metrics dict.
    """
    model.train()
    metrics.reset()
    total_loss = 0.0
    n_batches  = 0

    for batch_idx, batch in enumerate(loader):
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)
        masks  = batch.get('mask')
        if masks is not None:
            masks = masks.to(device, non_blocking=True)

        # ---- Forward pass (frame mode: images shape B,C,H,W) ----
        optimizer.zero_grad()
        output = model.forward_frame(images, mask=masks)

        # ---- Cross-entropy loss (paper Equation 14) ----
        loss = criterion(output['logits'], labels)

        # ---- Backward + gradient clip ----
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # ---- Accumulate metrics ----
        with torch.no_grad():
            metrics.update(output['probs'], labels, loss=loss.item())
            total_loss += loss.item()
            n_batches  += 1

        # ---- Progress logging ----
        if (batch_idx + 1) % log_interval == 0 and logger:
            logger.info(
                f"  Epoch {epoch} [{batch_idx + 1}/{len(loader)}] "
                f"Loss: {total_loss / n_batches:.4f}"
            )

    return metrics.compute()


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    metrics: EmotionMetrics,
    region_metrics: RegionWeightMetrics = None,
) -> dict:
    """
    Evaluates the model on the validation set.

    Args:
        model:     The framework model.
        loader:    Validation DataLoader.
        criterion: Loss function.
        device:    Computation device.
        metrics:   EmotionMetrics accumulator.

    Returns:
        Validation metrics dict.
    """
    model.eval()
    metrics.reset()
    if region_metrics is not None:
        region_metrics.reset()

    for batch in loader:
        images = batch['image'].to(device, non_blocking=True)
        labels = batch['label'].to(device, non_blocking=True)

        output = model.forward_frame(images)
        loss   = criterion(output['logits'], labels)

        metrics.update(output['probs'], labels, loss=loss.item())
        if region_metrics is not None:
            region_metrics.update(output.get('region_weights'), labels)

    results = metrics.compute()
    if region_metrics is not None:
        region_summary = region_metrics.compute()
        results.update({
            'arwm_eye_weight_mean': region_summary['mean']['eye'],
            'arwm_eyebrow_weight_mean': region_summary['mean']['eyebrow'],
            'arwm_mouth_weight_mean': region_summary['mean']['mouth'],
            'arwm_eye_weight_std': region_summary['std']['eye'],
            'arwm_eyebrow_weight_std': region_summary['std']['eyebrow'],
            'arwm_mouth_weight_std': region_summary['std']['mouth'],
        })
    return results


# ---------------------------------------------------------------------------
# Early stopping helper
# ---------------------------------------------------------------------------

class EarlyStopping:
    """Stops training if validation loss doesn't improve for `patience` epochs."""

    def __init__(self, patience: int = 8, mode: str = 'max'):
        self.patience    = patience
        self.mode        = mode
        self.best_value  = float('-inf') if mode == 'max' else float('inf')
        self.counter     = 0
        self.should_stop = False

    def step(self, value: float) -> bool:
        """
        Updates early stopping state.

        Returns:
            True if training should stop.
        """
        improved = (
            (self.mode == 'max' and value > self.best_value) or
            (self.mode == 'min' and value < self.best_value)
        )

        if improved:
            self.best_value = value
            self.counter    = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(config: dict, args):
    """
    Full training pipeline for the Swin-FANE framework.

    Args:
        config: Configuration dictionary loaded from YAML.
        args:   Parsed command-line arguments.
    """
    if args.arwm_mode:
        set_region_attention_mode(config, args.arwm_mode)
        config.setdefault('paths', {})['checkpoint_dir'] = (
            f"checkpoints/arwm_variants/{args.arwm_mode}"
        )
        config['paths']['log_dir'] = f"logs/arwm_variants/{args.arwm_mode}"
    elif args.arwm:
        set_region_attention_mode(config, 'adaptive')
        config.setdefault('paths', {})['checkpoint_dir'] = (
            "checkpoints/arwm_variants/adaptive"
        )
        config['paths']['log_dir'] = "logs/arwm_variants/adaptive"
    else:
        set_region_attention_mode(config, get_region_attention_mode(config))
    region_mode = get_region_attention_mode(config)
    checkpoint_dir = Path(config['paths']['checkpoint_dir']).resolve()
    output_checkpoints = list(checkpoint_dir.glob('*.pth'))
    if output_checkpoints and (
        not args.resume
        or Path(args.resume).resolve().parent != checkpoint_dir
    ):
        raise FileExistsError(
            f"Checkpoint output directory is not empty: {checkpoint_dir}. "
            "Resume from a checkpoint in this variant directory or choose a "
            "fresh output directory."
        )
    if args.arwm_mode or args.arwm:
        if args.multi_run:
            raise ValueError(
                "ARWM variant multi-run output isolation is not configured. "
                "Run each seed with a fresh mode-specific output config."
            )
    print(f"[Train] Region-attention mode: {region_mode}")
    print(f"[Train] Checkpoints: {config['paths']['checkpoint_dir']}")

    # ---- Device setup ----
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )
    print(f"[Train] Using device: {device}")

    # ---- Multiple seeds (paper Section 5.1: 3 independent runs) ----
    n_runs = config['training'].get('num_runs', 3) if args.multi_run else 1

    all_run_results = []

    for run_idx in range(n_runs):
        seed = config['training'].get('seed', 42) + run_idx
        set_seed(seed)
        print(f"\n{'='*60}")
        print(f"  Run {run_idx + 1}/{n_runs}  |  Seed: {seed}")
        print(f"{'='*60}")

        # ---- Logger ----
        logger = TrainingLogger(
            log_dir=config['paths']['log_dir'],
            experiment_name=f"swin_fane_run{run_idx + 1}",
            use_tensorboard=config['logging']['use_tensorboard'],
            use_wandb=config['logging']['use_wandb'],
        )

        # ---- Data loaders ----
        dataset_name = args.dataset or config['dataset']['name']
        logger.info(f"Loading dataset: {dataset_name}")

        if dataset_name.upper() == 'FER2013':
            train_loader, val_loader, _ = build_fer2013_dataloaders(
                root_dir=config['dataset']['root'],
                image_size=config['dataset']['image_size'],
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                pin_memory=config['training'].get('pin_memory', False),
                seed=config['training'].get('seed', 42),
            )
        else:
            train_loader, val_loader, _ = build_fane_dataloaders(
                root_dir=config['dataset']['root'],
                image_size=config['dataset']['image_size'],
                sequence_length=1,                         # Frame-level training
                batch_size=config['training']['batch_size'],
                num_workers=config['training']['num_workers'],
                subject_independent=config['dataset']['subject_independent'],
                seed=seed,
                split_file=os.path.join(config['paths']['results_dir'],
                                        'data_split.json'),
            )

        logger.info(f"Train batches: {len(train_loader)} | "
                    f"Val batches:   {len(val_loader)}")

        # ---- Model ----
        model = build_framework(config).to(device)
        freeze_epochs = config['training'].get('freeze_backbone_epochs', 0)
        if freeze_epochs > 0:
            model.freeze_backbone()
            logger.info(
                f"Swin backbone frozen for the first {freeze_epochs} epochs; "
                "FANE and classifier remain trainable."
            )
        logger.info(f"Model parameters: {model.get_param_count()}")

        # ---- Optional FER2013 class weighting ----
        class_weights = None
        if (
            dataset_name.upper() == 'FER2013'
            and config['training'].get('class_weighted_loss', False)
        ):
            class_weights, class_counts = compute_inverse_frequency_weights(
                train_loader.dataset,
                config['dataset']['num_classes'],
            )
            logger.info(
                f"FER2013 class counts: {class_counts.tolist()} | "
                f"loss weights: {[round(w, 4) for w in class_weights.tolist()]}"
            )

        # ---- Loss function: cross-entropy with label smoothing ----
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(device) if class_weights is not None else None,
            label_smoothing=config['training'].get('label_smoothing', 0.1)
        )

        # ---- Optimiser (AdamW, paper Section 5.1) ----
        optimizer = AdamW(
            model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
            betas=tuple(config['training']['betas']),
        )

        # ---- LR Scheduler ----
        scheduler_name = config['training'].get('scheduler', 'StepLR')
        if scheduler_name == 'CosineAnnealingLR':
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=config['training']['num_epochs'],
            )
        else:
            scheduler = StepLR(
                optimizer,
                step_size=config['training']['step_size'],
                gamma=config['training']['gamma'],
            )

        # ---- Resume from checkpoint ----
        start_epoch = 1
        resume_early_stopping_state = None
        if args.resume and not os.path.isfile(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")
        if args.resume:
            ckpt = torch.load(
                args.resume, map_location=device, weights_only=False
            )
            validate_checkpoint_region_mode(ckpt, region_mode, args.resume)
            experiment_metadata = ckpt.get('experiment', {})
            saved_training_config = experiment_metadata.get('training_config')
            if saved_training_config is not None and saved_training_config != config:
                raise ValueError(
                    f"Resume checkpoint training configuration does not match "
                    f"the current config: {args.resume}"
                )
            saved_seed = experiment_metadata.get('seed')
            configured_seed = config['training'].get('seed', 42)
            if saved_seed is not None and saved_seed != configured_seed:
                raise ValueError(
                    f"Resume checkpoint seed {saved_seed} does not match "
                    f"configured seed {configured_seed}: {args.resume}"
                )
            model.load_state_dict(ckpt['model'])
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt['epoch'] + 1
            if 'scheduler' in ckpt:
                scheduler.load_state_dict(ckpt['scheduler'])
            elif scheduler_name == 'StepLR':
                scheduler.last_epoch = ckpt['epoch']
            best_checkpoint_path = os.path.join(
                config['paths']['checkpoint_dir'], 'best_model.pth'
            )
            best_checkpoint = (
                torch.load(
                    best_checkpoint_path,
                    map_location='cpu',
                    weights_only=False,
                )
                if os.path.isfile(best_checkpoint_path)
                else ckpt
            )
            best_metrics = best_checkpoint.get('metrics', {})
            if 'accuracy' not in best_metrics:
                raise ValueError(
                    f"Cannot resume without validation accuracy in "
                    f"{best_checkpoint_path if os.path.isfile(best_checkpoint_path) else args.resume}"
                )
            latest_metrics = ckpt.get('metrics', {})
            if 'accuracy' not in latest_metrics:
                raise ValueError(
                    f"Cannot resume without validation accuracy in {args.resume}"
                )
            if (
                not os.path.isfile(best_checkpoint_path)
                or float(latest_metrics['accuracy']) > float(best_metrics['accuracy'])
            ):
                temporary_best_path = f"{best_checkpoint_path}.tmp"
                shutil.copyfile(args.resume, temporary_best_path)
                os.replace(temporary_best_path, best_checkpoint_path)
                best_checkpoint = ckpt
                best_metrics = latest_metrics
                logger.info(
                    "Recovered best validation checkpoint from the latest "
                    f"completed epoch ({ckpt['epoch']})."
                )
            logger.best_val_acc = float(best_metrics['accuracy'])
            logger.best_epoch = int(best_checkpoint.get('epoch', ckpt['epoch']))
            resume_early_stopping_state = ckpt.get(
                'experiment', {}
            ).get('early_stopping')
            rng_state = ckpt.get('rng_state')
            if rng_state is not None:
                random.setstate(rng_state['python'])
                np.random.set_state(rng_state['numpy'])
                torch.set_rng_state(rng_state['torch'].cpu())
                if device.type == 'cuda' and rng_state.get('cuda') is not None:
                    torch.cuda.set_rng_state_all(rng_state['cuda'])
            logger.info(f"Resumed from checkpoint: {args.resume} "
                        f"(epoch {ckpt['epoch']})")

        if freeze_epochs > 0 and start_epoch > freeze_epochs:
            for param in model.spatial_encoder.backbone.parameters():
                param.requires_grad = True

        # ---- Metrics + early stopping ----
        train_metrics_acc = EmotionMetrics(num_classes=config['dataset']['num_classes'])
        val_metrics_acc   = EmotionMetrics(num_classes=config['dataset']['num_classes'])
        val_region_metrics = (
            RegionWeightMetrics(num_classes=config['dataset']['num_classes'])
            if region_mode != 'off'
            else None
        )
        early_stopping    = EarlyStopping(
            patience=config['training']['early_stopping_patience'],
            mode='max',
        )
        if resume_early_stopping_state is not None:
            early_stopping.best_value = resume_early_stopping_state['best_value']
            early_stopping.counter = resume_early_stopping_state['counter']
            early_stopping.should_stop = resume_early_stopping_state['should_stop']
            if early_stopping.should_stop:
                start_epoch = config['training']['num_epochs'] + 1
                logger.info(
                    "Checkpoint already met the early-stopping criterion; "
                    "preserving the completed run without additional training."
                )

        # ---- Training loop ----
        num_epochs = config['training']['num_epochs']
        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(start_epoch, num_epochs + 1):
            t0 = time.time()

            # Unfreeze backbone after N epochs if it was frozen
            if freeze_epochs > 0 and epoch == freeze_epochs + 1:
                for param in model.spatial_encoder.backbone.parameters():
                    param.requires_grad = True
                logger.info(f"Epoch {epoch}: Swin backbone UNFROZEN.")

            # Train one epoch
            train_mets = train_epoch(
                model, train_loader, criterion, optimizer, device,
                train_metrics_acc, epoch,
                log_interval=config['logging']['log_interval'],
                logger=logger,
            )

            # Validate
            val_mets = validate_epoch(
                model, val_loader, criterion, device, val_metrics_acc,
                region_metrics=val_region_metrics,
            )

            # LR step
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']

            is_best = val_mets['accuracy'] > logger.best_val_acc

            # Log epoch results
            logger.log_epoch(epoch, train_mets, val_mets, lr=current_lr)

            should_stop = early_stopping.step(val_mets['accuracy'])

            # Save best checkpoint and the latest resumable state.
            logger.save_checkpoint(
                model, optimizer, epoch, val_mets,
                checkpoint_dir=config['paths']['checkpoint_dir'],
                is_best=is_best,
                scheduler=scheduler,
                save_epoch_checkpoint=config['training'].get(
                    'save_epoch_checkpoints', True
                ),
                metadata={
                    'region_attention_mode': region_mode,
                    'seed': seed,
                    'training_config': config,
                    'early_stopping': {
                        'best_value': early_stopping.best_value,
                        'counter': early_stopping.counter,
                        'should_stop': early_stopping.should_stop,
                    },
                },
            )

            # Early stopping check
            if should_stop:
                logger.info(f"Early stopping triggered at epoch {epoch}. "
                            f"Best val acc: {early_stopping.best_value:.2f}%")
                break

            elapsed = time.time() - t0
            logger.info(f"  Epoch {epoch} completed in {elapsed:.1f}s")

        # ---- Record run results ----
        all_run_results.append({
            'run':          run_idx + 1,
            'seed':         seed,
            'best_val_acc': logger.best_val_acc,
            'best_epoch':   logger.best_epoch,
        })
        logger.close()

    # ---- Multi-run summary ----
    if n_runs > 1:
        accs  = [r['best_val_acc'] for r in all_run_results]
        mean  = np.mean(accs)
        std   = np.std(accs)
        print(f"\n{'='*60}")
        print(f"Multi-run summary ({n_runs} runs):")
        print(f"  Val accuracy: {mean:.2f}% ± {std:.2f}%")
        for r in all_run_results:
            print(f"  Run {r['run']} (seed={r['seed']}): "
                  f"{r['best_val_acc']:.2f}% @ epoch {r['best_epoch']}")
        print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Train Swin-FANE Emotion Recognition Framework'
    )
    parser.add_argument('--config',    type=str, required=True,
                        help='Path to config YAML file')
    parser.add_argument('--resume',    type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--dataset',   type=str, default=None,
                        choices=['FANE', 'FER2013'],
                        help='Override dataset choice from config')
    parser.add_argument('--multi-run', action='store_true',
                        help='Run N independent seeds (paper: 3 runs)')
    parser.add_argument('--arwm', action='store_true',
                        help='Legacy alias for adaptive ARWM; writes to a new variant directory')
    parser.add_argument('--arwm-mode', choices=['off', 'fixed', 'adaptive'],
                        default=None,
                        help='Select controlled ARWM mode and use separate outputs')
    parser.add_argument('--gpu',       type=int, default=0,
                        help='GPU index to use (default: 0)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # ---- Set CUDA device ----
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu)

    # ---- Load config ----
    config = load_config(args.config)

    # ---- Create output directories ----
    for dir_key in ['checkpoint_dir', 'results_dir', 'log_dir']:
        os.makedirs(config['paths'][dir_key], exist_ok=True)

    train(config, args)
