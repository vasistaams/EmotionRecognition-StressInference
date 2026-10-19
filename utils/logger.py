"""
logger.py
=========
Training logger with TensorBoard and console output support.

Provides a unified logging interface for:
  - Training loss and accuracy per epoch
  - Validation metrics
  - Learning rate schedule
  - Temporal stress metrics
  - Model checkpoint management

Paper reference: Section 5.1 (training setup)
"""

import os
import json
import time
import logging
import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch


class TrainingLogger:
    """
    Unified logger for training the Swin-FANE framework.

    Supports:
      - Console output (Python logging)
      - JSON metric file (for reproducibility)
      - TensorBoard (optional)
      - W&B (optional)

    Args:
        log_dir:         Directory to save logs and checkpoints.
        experiment_name: Name for this training run.
        use_tensorboard: Whether to log to TensorBoard.
        use_wandb:       Whether to log to Weights & Biases.
        wandb_project:   W&B project name.
    """

    def __init__(
        self,
        log_dir: str = "logs",
        experiment_name: str = "swin_fane",
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: str = "swin-fane-stress",
    ):
        self.log_dir         = Path(log_dir)
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb       = use_wandb

        # Timestamp-based run name for uniqueness
        self.run_name = f"{experiment_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.run_dir  = self.log_dir / self.run_name
        self.run_dir.mkdir(parents=True, exist_ok=True)

        # ---- Console logger ----
        self.logger = logging.getLogger(self.run_name)
        self.logger.setLevel(logging.INFO)

        # File handler — saves all log messages to a .log file
        fh = logging.FileHandler(self.run_dir / "train.log")
        fh.setLevel(logging.INFO)

        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        fmt = logging.Formatter('[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
        fh.setFormatter(fmt)
        ch.setFormatter(fmt)

        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

        # ---- TensorBoard ----
        self.tb_writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=str(self.run_dir / 'tensorboard'))
                self.info("TensorBoard enabled.")
            except ImportError:
                self.info("TensorBoard not available (install: pip install tensorboard).")

        # ---- W&B ----
        self.wandb_run = None
        if use_wandb:
            try:
                import wandb
                self.wandb_run = wandb.init(
                    project=wandb_project,
                    name=self.run_name,
                    dir=str(self.run_dir),
                )
                self.info("Weights & Biases logging enabled.")
            except ImportError:
                self.info("W&B not available (install: pip install wandb).")

        # ---- Metric history (persisted to JSON) ----
        self.metric_history: Dict[str, list] = {}
        self.best_val_acc    = 0.0
        self.best_epoch      = 0

        self.info(f"Logger initialised. Run directory: {self.run_dir}")

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def info(self, msg: str):
        """Logs an info-level message to console and file."""
        self.logger.info(msg)

    def log_epoch(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
        lr: float,
    ):
        """
        Logs metrics for one training epoch.

        Args:
            epoch:         Current epoch number (1-indexed).
            train_metrics: Training metrics dict (e.g., {'loss': 0.4, 'accuracy': 82.0}).
            val_metrics:   Validation metrics dict.
            lr:            Current learning rate.
        """
        # Console output
        self.info(
            f"Epoch {epoch:3d} | "
            f"Train Loss: {train_metrics.get('mean_loss', 0):.4f} | "
            f"Train Acc:  {train_metrics.get('accuracy', 0):.2f}% | "
            f"Val Acc:    {val_metrics.get('accuracy', 0):.2f}% | "
            f"Val F1:     {val_metrics.get('f1', 0):.2f}% | "
            f"LR: {lr:.2e}"
        )

        # Track best model
        val_acc = val_metrics.get('accuracy', 0.0)
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.best_epoch   = epoch
            self.info(f"  New best validation accuracy: {val_acc:.2f}% at epoch {epoch}")

        # TensorBoard logging
        if self.tb_writer:
            for key, val in train_metrics.items():
                self.tb_writer.add_scalar(f'Train/{key}', val, epoch)
            for key, val in val_metrics.items():
                self.tb_writer.add_scalar(f'Val/{key}', val, epoch)
            self.tb_writer.add_scalar('LR/lr', lr, epoch)

        # W&B logging
        if self.wandb_run:
            log_dict = {f'train_{k}': v for k, v in train_metrics.items()}
            log_dict.update({f'val_{k}': v for k, v in val_metrics.items()})
            log_dict['lr'] = lr
            log_dict['epoch'] = epoch
            self.wandb_run.log(log_dict)

        # Persist to history
        for key, val in train_metrics.items():
            self.metric_history.setdefault(f'train_{key}', []).append(val)
        for key, val in val_metrics.items():
            self.metric_history.setdefault(f'val_{key}', []).append(val)
        self.metric_history.setdefault('lr', []).append(lr)

    def log_test_results(self, test_metrics: Dict[str, Any]):
        """
        Logs final test set evaluation results.

        Args:
            test_metrics: Complete metrics dict from evaluation.
        """
        self.info("=" * 60)
        self.info("FINAL TEST SET EVALUATION RESULTS")
        self.info("=" * 60)
        for key, val in test_metrics.items():
            if isinstance(val, float):
                self.info(f"  {key:<25}: {val:.4f}")
            else:
                self.info(f"  {key:<25}: {val}")
        self.info(f"  Best epoch                : {self.best_epoch}")
        self.info(f"  Best val accuracy         : {self.best_val_acc:.2f}%")
        self.info("=" * 60)

        # Save results to JSON
        results_path = self.run_dir / "test_results.json"
        with open(results_path, 'w') as f:
            json.dump(test_metrics, f, indent=2, default=str)

    def save_metric_history(self):
        """Saves the full metric history to a JSON file."""
        history_path = self.run_dir / "metric_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.metric_history, f, indent=2)

    # ------------------------------------------------------------------
    # Checkpoint management
    # ------------------------------------------------------------------

    def save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        checkpoint_dir: str = "checkpoints",
        is_best: bool = False,
        metadata: Optional[Dict[str, object]] = None,
        scheduler=None,
        save_epoch_checkpoint: bool = True,
    ):
        """
        Saves a model checkpoint.

        Args:
            model:          The model to save.
            optimizer:      The optimizer (for resuming training).
            epoch:          Current epoch.
            metrics:        Metrics dict to embed in checkpoint.
            checkpoint_dir: Directory to save checkpoints.
            is_best:        Whether to also save as 'best_model.pth'.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)

        state = {
            'epoch':     epoch,
            'model':     model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'metrics':   metrics,
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        if scheduler is not None:
            state['scheduler'] = scheduler.state_dict()
        if metadata is not None:
            state['experiment'] = metadata

        def save_atomically(path):
            temporary_path = f'{path}.tmp'
            torch.save(state, temporary_path)
            os.replace(temporary_path, path)

        latest_path = os.path.join(checkpoint_dir, 'latest_model.pth')
        save_atomically(latest_path)

        if save_epoch_checkpoint:
            ckpt_path = os.path.join(
                checkpoint_dir, f'checkpoint_epoch{epoch:03d}.pth'
            )
            save_atomically(ckpt_path)

        # Save best model
        if is_best:
            best_path = os.path.join(checkpoint_dir, 'best_model.pth')
            save_atomically(best_path)
            self.info(f"  Saved best model: {best_path}")

    def close(self):
        """Closes all logging handles."""
        self.save_metric_history()
        if self.tb_writer:
            self.tb_writer.close()
        if self.wandb_run:
            self.wandb_run.finish()
        self.info("Logger closed.")
        for handler in self.logger.handlers[:]:
            handler.flush()
            handler.close()
            self.logger.removeHandler(handler)
