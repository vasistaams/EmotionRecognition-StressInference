"""Evaluate available off/fixed/adaptive checkpoints on FER2013 test."""

import argparse
import csv
import copy
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from data.fer2013_dataset import build_fer2013_dataloaders, FER2013_CLASSES
from models.full_framework import build_framework
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config, set_region_attention_mode


DEFAULT_CHECKPOINTS = {
    "off": "checkpoints/fer2013_weighted_freeze3_lr1e4/best_model.pth",
    "fixed": "checkpoints/arwm_variants/fixed/best_model.pth",
    "adaptive": "checkpoints/fer2013_arwm/best_model.pth",
}


def sync_device(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_checkpoint(
    mode,
    checkpoint_path,
    base_config,
    test_loader,
    device,
    warmup_iterations,
    max_samples=None,
):
    config = json.loads(json.dumps(base_config))
    set_region_attention_mode(config, mode)
    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    sync_device(device)
    model_construction_start = time.perf_counter()
    model = build_framework(config).to(device)
    sync_device(device)
    model_construction_seconds = time.perf_counter() - model_construction_start

    sync_device(device)
    checkpoint_read_start = time.perf_counter()
    checkpoint = torch.load(
        checkpoint_path, map_location=device, weights_only=False
    )
    sync_device(device)
    checkpoint_read_seconds = time.perf_counter() - checkpoint_read_start
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("model"), dict):
        raise ValueError(f"Invalid model checkpoint: {checkpoint_path}")
    validate_checkpoint_region_mode(checkpoint, mode, checkpoint_path)
    sync_device(device)
    state_load_start = time.perf_counter()
    model.load_state_dict(checkpoint["model"], strict=True)
    sync_device(device)
    state_load_seconds = time.perf_counter() - state_load_start
    model.eval()

    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    all_labels = []
    all_predictions = []
    all_probabilities = []
    batch_times = []
    total_model_seconds = 0.0
    samples_seen = 0
    warmed = 0
    if warmup_iterations:
        try:
            warmup_batch = next(iter(test_loader))
        except StopIteration as error:
            raise RuntimeError("FER2013 test loader is empty") from error
        warmup_images = warmup_batch["image"].to(device, non_blocking=False)
        while warmed < warmup_iterations:
            with torch.inference_mode():
                model.forward_frame(warmup_images)
            sync_device(device)
            warmed += 1

    for batch in test_loader:
        images = batch["image"].to(device, non_blocking=False)
        labels = batch["label"].to(device, non_blocking=False)

        remaining = None if max_samples is None else max_samples - samples_seen
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and images.shape[0] > remaining:
            images = images[:remaining]
            labels = labels[:remaining]

        sync_device(device)
        start = time.perf_counter()
        with torch.inference_mode():
            output = model.forward_frame(images)
        sync_device(device)
        elapsed = time.perf_counter() - start
        batch_times.append(elapsed)
        total_model_seconds += elapsed
        probabilities = output["probs"].detach().cpu().numpy()
        predictions = probabilities.argmax(axis=1)
        all_labels.extend(labels.cpu().numpy().tolist())
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())
        samples_seen += len(labels)
        if samples_seen % 512 < len(labels):
            print(f"[{mode}] Evaluated {samples_seen} test images")
        if max_samples is not None and samples_seen >= max_samples:
            break

    del model
    if not all_labels:
        raise RuntimeError(f"No test samples evaluated for mode {mode}")

    labels = np.asarray(all_labels)
    predictions = np.asarray(all_predictions)
    probabilities = np.asarray(all_probabilities)
    per_precision, per_recall, per_f1, per_support = precision_recall_fscore_support(
        labels,
        predictions,
        labels=list(range(len(FER2013_CLASSES))),
        average=None,
        zero_division=0,
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        labels, predictions, average="weighted", zero_division=0
    )
    confusion = confusion_matrix(
        labels, predictions, labels=list(range(len(FER2013_CLASSES)))
    )
    mean_batch_seconds = float(np.mean(batch_times))
    result = {
        "status": (
            "complete_test"
            if max_samples is None or samples_seen == len(test_loader.dataset)
            else "partial_smoke"
        ),
        "mode": mode,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": checkpoint.get("metrics"),
        "checkpoint_metrics_source": (
            "validation metrics saved by the existing training loop"
            if checkpoint.get("metrics") else "not recorded"
        ),
        "experiment_metadata": checkpoint.get("experiment"),
        "model_mode_from_config": get_region_attention_mode(config),
        "dataset_split": "FER2013 official test folder",
        "number_of_evaluated_samples": int(samples_seen),
        "trainable_parameters": int(trainable_parameters),
        "total_parameters": int(total_parameters),
        "metrics": {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "per_class": {
                class_name: {
                    "precision": float(per_precision[index]),
                    "recall": float(per_recall[index]),
                    "f1": float(per_f1[index]),
                    "support": int(per_support[index]),
                }
                for index, class_name in enumerate(FER2013_CLASSES)
            },
        },
        "confusion_matrix": confusion.tolist(),
        "timing": {
            "method": (
                "torch.inference_mode; perf_counter around forward_frame only; "
                "input loading and host-to-device transfer excluded; CUDA synchronized "
                "before and after timed forward pass"
            ),
            "warmup_batches": int(warmed),
            "timed_batches": len(batch_times),
            "mean_batch_latency_ms": mean_batch_seconds * 1000.0,
            "mean_latency_ms_per_image": (
                total_model_seconds * 1000.0 / samples_seen
            ),
            "p50_batch_latency_ms": float(np.percentile(batch_times, 50) * 1000.0),
            "p95_batch_latency_ms": float(np.percentile(batch_times, 95) * 1000.0),
            "throughput_images_per_second": (
                samples_seen / total_model_seconds if total_model_seconds else None
            ),
            "peak_gpu_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda" else None
            ),
            "model_construction_seconds": model_construction_seconds,
            "checkpoint_read_seconds": checkpoint_read_seconds,
            "state_dict_load_seconds": state_load_seconds,
            "model_and_checkpoint_load_seconds": (
                model_construction_seconds + checkpoint_read_seconds + state_load_seconds
            ),
        },
    }
    return result


def write_confusion_plot(results, output_path):
    modes = list(DEFAULT_CHECKPOINTS)
    figure, axes = plt.subplots(1, len(modes), figsize=(6 * len(modes), 5))
    for axis, mode in zip(axes, modes):
        result = results.get(mode)
        if result and result.get("confusion_matrix"):
            matrix = np.asarray(result["confusion_matrix"])
            axis.imshow(matrix, cmap="Blues")
            axis.set_xticks(range(len(FER2013_CLASSES)), FER2013_CLASSES, rotation=45, ha="right")
            axis.set_yticks(range(len(FER2013_CLASSES)), FER2013_CLASSES)
            for row in range(matrix.shape[0]):
                for column in range(matrix.shape[1]):
                    axis.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=7)
            axis.set_title(f"{mode}: {result['number_of_evaluated_samples']} samples")
            axis.set_xlabel("Predicted")
            axis.set_ylabel("True")
        else:
            axis.axis("off")
            axis.set_title(f"{mode}: unavailable")
            axis.text(0.5, 0.5, results.get(mode, {}).get("limitation", "Not evaluated"), ha="center", va="center", wrap=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def comparable_training_conditions(experiment_metadata):
    """Normalize only expected mode/output-path differences between variants."""
    if not isinstance(experiment_metadata, dict):
        return None
    training_config = experiment_metadata.get("training_config")
    if not isinstance(training_config, dict):
        return None
    normalized = copy.deepcopy(training_config)
    region_config = normalized.get("model", {}).get("region_attention", {})
    region_config.pop("mode", None)
    region_config.pop("enabled", None)
    paths = normalized.get("paths", {})
    for key in ("checkpoint_dir", "log_dir", "results_dir"):
        paths.pop(key, None)
    return experiment_metadata.get("seed"), normalized


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/config_arwm_50.yaml")
    parser.add_argument("--off-checkpoint", default=DEFAULT_CHECKPOINTS["off"])
    parser.add_argument("--fixed-checkpoint", default=DEFAULT_CHECKPOINTS["fixed"])
    parser.add_argument("--adaptive-checkpoint", default=DEFAULT_CHECKPOINTS["adaptive"])
    parser.add_argument("--output", default="results/arwm")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--warmup-batches", type=int, default=3)
    parser.add_argument(
        "--max-samples", type=int, default=None,
        help="Optional smoke-test limit; results are explicitly marked partial",
    )
    args = parser.parse_args()
    if args.warmup_batches < 0:
        raise ValueError("--warmup-batches must be non-negative")
    if args.max_samples is not None and args.max_samples < 1:
        raise ValueError("--max-samples must be positive")

    config = load_config(args.config)
    if config["dataset"]["name"].upper() != "FER2013":
        raise ValueError("This evaluator currently expects FER2013.")
    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    root = Path(config["dataset"]["root"])
    if not (root / "test").is_dir():
        raise FileNotFoundError(
            f"Official FER2013 test split not found: {root / 'test'}"
        )
    batch_size = args.batch_size or config["training"]["batch_size"]
    _, _, test_loader = build_fer2013_dataloaders(
        root_dir=str(root),
        image_size=config["dataset"]["image_size"],
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
        seed=config["training"].get("seed", 42),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = {
        "off": args.off_checkpoint,
        "fixed": args.fixed_checkpoint,
        "adaptive": args.adaptive_checkpoint,
    }
    results = {}
    for mode, checkpoint_path in checkpoints.items():
        if not os.path.isfile(checkpoint_path):
            results[mode] = {
                "status": "unavailable",
                "mode": mode,
                "checkpoint": checkpoint_path,
                "limitation": (
                    "No trained checkpoint is available for this mode. "
                    "This variant was not evaluated."
                ),
            }
            print(f"[{mode}] SKIPPED: checkpoint not found: {checkpoint_path}")
            continue
        results[mode] = evaluate_checkpoint(
            mode=mode,
            checkpoint_path=checkpoint_path,
            base_config=config,
            test_loader=test_loader,
            device=device,
            warmup_iterations=args.warmup_batches,
            max_samples=args.max_samples,
        )

    complete_modes = [
        mode for mode, result in results.items()
        if result.get("status") == "complete_test"
    ]
    training_conditions = [
        comparable_training_conditions(results[mode].get("experiment_metadata"))
        for mode in complete_modes
    ]
    fully_comparable = (
        len(complete_modes) == 3
        and all(condition is not None for condition in training_conditions)
        and all(
            condition == training_conditions[0]
            for condition in training_conditions[1:]
        )
    )
    report = {
        "evaluation": {
            "dataset": config["dataset"]["name"],
            "dataset_root": str(root),
            "split": "official test",
            "sample_count_in_split": len(test_loader.dataset),
            "seed_for_train_val_split_reproducibility": config["training"].get("seed"),
            "device": str(device),
            "hardware": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else "CPU; CUDA unavailable"
            ),
            "image_size": config["dataset"]["image_size"],
            "batch_size": batch_size,
            "config_path": args.config,
            "configuration": config,
            "model_selection": (
                "Select checkpoints using validation data only; evaluate best_model "
                "once on the official FER2013 test split. Existing legacy checkpoint "
                "metrics are recorded as validation metrics and are not substituted "
                "for these test metrics."
            ),
            "comparison_status": (
                "controlled_three_way_comparison"
                if fully_comparable else
                "incomplete_or_legacy_checkpoints_not_a_controlled_ablation"
            ),
            "comparison_limitation": (
                "A causal three-way comparison requires trained checkpoints from "
                "the separate off/fixed/adaptive configs, same seed and hyperparameters. "
                "Legacy checkpoint metadata is insufficient to certify that."
            ),
            "timing_method": (
                "Model-only forward latency excludes data loading and input transfer; "
                "CUDA synchronization is used when available."
            ),
        },
        "variants": results,
    }
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "metrics.json")
    csv_path = os.path.join(args.output, "metrics.csv")
    plot_path = os.path.join(args.output, "confusion_matrix.png")
    markdown_path = os.path.join(args.output, "evaluation_summary.md")
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)
    csv_columns = [
        "mode", "status", "checkpoint", "number_of_evaluated_samples",
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "class_name", "class_precision",
        "class_recall", "class_f1", "class_support",
        "trainable_parameters", "total_parameters", "mean_latency_ms_per_image",
        "throughput_images_per_second", "model_construction_seconds",
        "checkpoint_read_seconds", "state_dict_load_seconds",
        "model_and_checkpoint_load_seconds", "device",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_columns)
        writer.writeheader()
        for mode, result in results.items():
            metrics = result.get("metrics", {})
            base_row = {
                "mode": mode,
                "status": result.get("status"),
                "checkpoint": result.get("checkpoint"),
                "number_of_evaluated_samples": result.get("number_of_evaluated_samples"),
                "accuracy": metrics.get("accuracy"),
                "balanced_accuracy": metrics.get("balanced_accuracy"),
                "macro_precision": metrics.get("macro_precision"),
                "macro_recall": metrics.get("macro_recall"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "trainable_parameters": result.get("trainable_parameters"),
                "total_parameters": result.get("total_parameters"),
                "mean_latency_ms_per_image": result.get("timing", {}).get("mean_latency_ms_per_image"),
                "throughput_images_per_second": result.get("timing", {}).get("throughput_images_per_second"),
                "model_construction_seconds": result.get("timing", {}).get("model_construction_seconds"),
                "checkpoint_read_seconds": result.get("timing", {}).get("checkpoint_read_seconds"),
                "state_dict_load_seconds": result.get("timing", {}).get("state_dict_load_seconds"),
                "model_and_checkpoint_load_seconds": result.get("timing", {}).get("model_and_checkpoint_load_seconds"),
                "device": str(device),
            }
            writer.writerow(base_row)
            for class_name, class_metrics in metrics.get("per_class", {}).items():
                writer.writerow({
                    **base_row,
                    "class_name": class_name,
                    "class_precision": class_metrics["precision"],
                    "class_recall": class_metrics["recall"],
                    "class_f1": class_metrics["f1"],
                    "class_support": class_metrics["support"],
                })
    write_confusion_plot(results, plot_path)

    lines = [
        "# ARWM Variant Evaluation",
        "",
        f"- Dataset/split: {config['dataset']['name']} official test",
        f"- Samples in split: {len(test_loader.dataset)}",
        f"- Device: {device}",
        f"- Image size / batch size: {config['dataset']['image_size']} / {batch_size}",
        f"- Configuration: `{args.config}`",
        f"- Comparison status: **{report['evaluation']['comparison_status']}**",
        "- Existing checkpoint metrics embedded in checkpoints are validation metrics, not test results.",
        "",
        "| Mode | Status | N | Accuracy | Balanced acc. | Macro F1 | Weighted F1 | ms/image | images/s | Load seconds |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, result in results.items():
        metrics = result.get("metrics", {})
        timing = result.get("timing", {})
        def fmt(value):
            return f"{value:.4f}" if isinstance(value, (int, float)) else "—"
        lines.append(
            f"| {mode} | {result.get('status')} | "
            f"{result.get('number_of_evaluated_samples', '—')} | "
            f"{fmt(metrics.get('accuracy'))} | {fmt(metrics.get('balanced_accuracy'))} | "
            f"{fmt(metrics.get('macro_f1'))} | {fmt(metrics.get('weighted_f1'))} | "
            f"{fmt(timing.get('mean_latency_ms_per_image'))} | "
            f"{fmt(timing.get('throughput_images_per_second'))} | "
            f"{fmt(timing.get('model_and_checkpoint_load_seconds'))} |"
        )
        if result.get("limitation"):
            lines.append(f"\n{mode} limitation: {result['limitation']}")
    lines.extend([
        "",
        "A result is not a controlled ARWM ablation unless the variants were trained with the same data split, seed, model-selection rule, and training hyperparameters. Unavailable checkpoints are not substituted with validation scores.",
        "",
    ])
    with open(markdown_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))
    print(f"Saved: {json_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {plot_path}")
    print(f"Saved: {markdown_path}")


if __name__ == "__main__":
    main()
