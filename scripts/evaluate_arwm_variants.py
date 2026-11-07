"""Evaluate available off/fixed/adaptive/context-aware checkpoints on FER2013 test."""

import argparse
import csv
import copy
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
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
    "context_aware": "checkpoints/context_aware_arwm/best_model.pth",
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
        checkpoint_path, map_location="cpu", weights_only=True
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
    all_region_weights = []
    raw_predictions = []
    total_test_loss = 0.0
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
        batch_labels = labels.cpu().numpy().tolist()
        paths = batch["path"]
        if isinstance(paths, str):
            paths = [paths]
        all_labels.extend(batch_labels)
        all_predictions.extend(predictions.tolist())
        all_probabilities.extend(probabilities.tolist())
        total_test_loss += float(
            F.cross_entropy(output["logits"], labels, reduction="sum").item()
        )
        raw_predictions.extend(
            {
                "sample_index": samples_seen + index,
                "path": str(paths[index]),
                "true_label": FER2013_CLASSES[int(batch_labels[index])],
                "predicted_label": FER2013_CLASSES[int(predictions[index])],
                "probabilities": probabilities[index].tolist(),
            }
            for index in range(len(batch_labels))
        )
        region_weights = output.get("region_weights")
        if region_weights is not None:
            weights = region_weights.detach().cpu().numpy()
            if weights.shape != (len(labels), 3) or not np.isfinite(weights).all():
                raise ValueError(
                    f"Invalid ARWM region-weight output for mode {mode}: "
                    f"shape={weights.shape}"
                )
            all_region_weights.extend(weights.tolist())
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
    region_weight_values = np.asarray(all_region_weights, dtype=np.float64)
    region_weight_analysis = None
    if len(region_weight_values):
        region_weight_analysis = {
            "source": "held-out FER2013 test inputs; internal model weights, not explanations",
            "number_of_samples": int(len(region_weight_values)),
            "region_order": ["eye", "eyebrow", "mouth"],
            "mean": region_weight_values.mean(axis=0).tolist(),
            "std": region_weight_values.std(axis=0).tolist(),
            "min": region_weight_values.min(axis=0).tolist(),
            "max": region_weight_values.max(axis=0).tolist(),
            "maximum_absolute_normalization_error": float(
                np.max(np.abs(region_weight_values.sum(axis=1) - 1.0))
            ),
            "representative_examples": [
                {
                    "test_sample_index": index,
                    "true_label": FER2013_CLASSES[int(labels[index])],
                    "predicted_label": FER2013_CLASSES[int(predictions[index])],
                    "weights": region_weight_values[index].tolist(),
                }
                for index in range(min(32, len(region_weight_values)))
            ],
            "interpretation": (
                "Weights are input-conditioned model signals for adaptive mode; "
                "their variation does not establish causal explanations or improved accuracy."
            ),
        }
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
        "test_loss": float(total_test_loss / samples_seen),
        "region_weight_analysis": region_weight_analysis,
        "_region_weight_values": region_weight_values.tolist(),
        "_raw_predictions": raw_predictions,
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


def write_metric_comparison_plot(results, output_path):
    modes = [
        mode for mode in ("off", "fixed", "adaptive", "context_aware")
        if results.get(mode, {}).get("status") == "complete_test"
    ]
    if len(modes) < 2:
        return False
    metric_names = ("accuracy", "macro_f1", "weighted_f1", "balanced_accuracy")
    labels = ("Accuracy", "Macro F1", "Weighted F1", "Balanced accuracy")
    x = np.arange(len(metric_names))
    width = 0.8 / len(modes)
    figure, axis = plt.subplots(figsize=(9, 5))
    for index, mode in enumerate(modes):
        values = [
            results[mode]["metrics"][metric] * 100.0
            for metric in metric_names
        ]
        axis.bar(
            x - 0.4 + width / 2 + index * width,
            values,
            width,
            label=mode,
        )
    axis.set_xticks(x, labels)
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.set_title("FER2013 official-test comparison")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
    return True


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
    parser.add_argument(
        "--split-manifest",
        default=None,
        help="Directory containing train/validation/test CSV manifests",
    )
    parser.add_argument(
        "--test-split",
        choices=("test_combined", "test_private", "test_public"),
        default="test_combined",
    )
    parser.add_argument("--off-checkpoint", default=DEFAULT_CHECKPOINTS["off"])
    parser.add_argument("--fixed-checkpoint", default=DEFAULT_CHECKPOINTS["fixed"])
    parser.add_argument("--adaptive-checkpoint", default=DEFAULT_CHECKPOINTS["adaptive"])
    parser.add_argument(
        "--context-aware-checkpoint",
        default=DEFAULT_CHECKPOINTS["context_aware"],
    )
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
    output_path = Path(args.output)
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise FileExistsError(
            f"Evaluation output directory already contains files: {args.output}. "
            "Choose a new directory to preserve prior results."
        )

    config = load_config(args.config)
    if config["dataset"]["name"].upper() != "FER2013":
        raise ValueError("This evaluator currently expects FER2013.")
    if config["dataset"].get("class_names") != FER2013_CLASSES:
        raise ValueError(
            "Configured class order does not match the FER2013 loader: "
            f"{config['dataset'].get('class_names')!r} vs {FER2013_CLASSES!r}"
        )
    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    root = Path(config["dataset"]["root"])
    if not (root / "test").is_dir():
        raise FileNotFoundError(
            f"Official FER2013 test split not found: {root / 'test'}"
        )
    batch_size = args.batch_size or config["training"]["batch_size"]
    _, _, test_loader = build_fer2013_dataloaders(
        root_dir=str(root),
        split_manifest=(
            args.split_manifest or config["dataset"].get("split_manifest")
        ),
        test_split=args.test_split,
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
        "context_aware": args.context_aware_checkpoint,
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

    region_weight_rows = []
    prediction_files = {}
    for mode, result in results.items():
        values = result.pop("_region_weight_values", [])
        for index, weights in enumerate(values):
            region_weight_rows.append({
                "mode": mode,
                "test_sample_index": index,
                "eye_weight": weights[0],
                "eyebrow_weight": weights[1],
                "mouth_weight": weights[2],
            })
        predictions = result.pop("_raw_predictions", [])
        if predictions:
            prediction_files[mode] = predictions

    complete_modes = [
        mode for mode, result in results.items()
        if result.get("status") == "complete_test"
    ]
    training_conditions = [
        comparable_training_conditions(results[mode].get("experiment_metadata"))
        for mode in complete_modes
    ]
    fully_comparable = (
        len(complete_modes) == 4
        and all(condition is not None for condition in training_conditions)
        and all(
            condition == training_conditions[0]
            for condition in training_conditions[1:]
        )
    )
    off_result = results.get("off", {})
    adaptive_result = results.get("adaptive", {})
    same_test_samples = (
        off_result.get("status") == "complete_test"
        and adaptive_result.get("status") == "complete_test"
        and off_result.get("number_of_evaluated_samples")
        == adaptive_result.get("number_of_evaluated_samples")
    )
    off_conditions = comparable_training_conditions(
        off_result.get("experiment_metadata")
    )
    adaptive_conditions = comparable_training_conditions(
        adaptive_result.get("experiment_metadata")
    )
    controlled_two_way = (
        same_test_samples
        and off_conditions is not None
        and adaptive_conditions is not None
        and off_conditions == adaptive_conditions
    )
    comparison = {
        "status": (
            "controlled_off_vs_adaptive"
            if controlled_two_way else
            "unavailable_or_training_conditions_unverified"
        ),
        "same_complete_test_sample_count": bool(same_test_samples),
        "training_conditions_verified_equal": bool(controlled_two_way),
        "adaptive_minus_off_percentage_points": None,
    }
    if same_test_samples:
        comparison["adaptive_minus_off_percentage_points"] = {
            metric: (
                adaptive_result["metrics"][metric]
                - off_result["metrics"][metric]
            ) * 100.0
            for metric in ("accuracy", "macro_f1")
        }
    report = {
        "evaluation": {
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
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
                "controlled_four_way_comparison"
                if fully_comparable else
                "incomplete_or_legacy_checkpoints_not_a_controlled_ablation"
            ),
            "comparison_limitation": (
                "A controlled four-way comparison requires trained checkpoints from "
                "the separate off/fixed/adaptive/context-aware configs, same seed and hyperparameters. "
                "Legacy checkpoint metadata is insufficient to certify that."
            ),
            "timing_method": (
                "Model-only forward latency excludes data loading and input transfer; "
                "CUDA synchronization is used when available."
            ),
        },
        "variants": results,
        "off_vs_adaptive_comparison": comparison,
    }
    os.makedirs(args.output, exist_ok=True)
    json_path = os.path.join(args.output, "metrics.json")
    csv_path = os.path.join(args.output, "metrics.csv")
    plot_path = os.path.join(args.output, "confusion_matrix.png")
    markdown_path = os.path.join(args.output, "evaluation_summary.md")
    if region_weight_rows:
        with open(
            os.path.join(args.output, "region_weight_samples.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(file, fieldnames=list(region_weight_rows[0]))
            writer.writeheader()
            writer.writerows(region_weight_rows)
        available_modes = [
            mode for mode in ("off", "fixed", "adaptive", "context_aware")
            if any(row["mode"] == mode for row in region_weight_rows)
        ]
        figure, axes = plt.subplots(
            1, len(available_modes),
            figsize=(5 * len(available_modes), 5),
            squeeze=False,
        )
        for axis, mode in zip(axes.flat, available_modes):
            mode_rows = [row for row in region_weight_rows if row["mode"] == mode]
            axis.boxplot(
                [[row[f"{region}_weight"] for row in mode_rows]
                 for region in ("eye", "eyebrow", "mouth")],
                showmeans=True,
            )
            axis.set_xticks([1, 2, 3], ("Eye", "Eyebrow", "Mouth"))
            axis.set_ylim(0, 1)
            axis.set_ylabel("Region weight")
            axis.set_title(f"{mode}: {len(mode_rows)} test samples")
            axis.grid(axis="y", alpha=0.25)
        figure.suptitle(
            "ARWM region weights on held-out inputs (internal signals, not explanations)"
        )
        figure.tight_layout()
        figure.savefig(
            os.path.join(args.output, "arwm_region_weights.png"),
            dpi=160,
            bbox_inches="tight",
        )
        plt.close(figure)

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=str)
    csv_columns = [
        "mode", "status", "checkpoint", "number_of_evaluated_samples",
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "test_loss", "class_name", "class_precision",
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
                "test_loss": result.get("test_loss"),
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
    probability_columns = [
        f"probability_{class_name}" for class_name in FER2013_CLASSES
    ]
    for mode, rows in prediction_files.items():
        predictions_path = os.path.join(
            args.output, f"predictions_{mode}.csv"
        )
        with open(
            predictions_path,
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            columns = [
                "sample_index",
                "path",
                "true_label",
                "predicted_label",
                *probability_columns,
            ]
            writer = csv.DictWriter(file, fieldnames=columns)
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "sample_index": row["sample_index"],
                    "path": row["path"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    **dict(zip(probability_columns, row["probabilities"])),
                })
        confusion_path = os.path.join(
            args.output, f"confusion_matrix_{mode}.csv"
        )
        with open(confusion_path, "w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["true\\predicted", *FER2013_CLASSES])
            for class_name, row in zip(
                FER2013_CLASSES,
                results[mode]["confusion_matrix"],
            ):
                writer.writerow([class_name, *row])
    write_confusion_plot(results, plot_path)
    comparison_plot_path = os.path.join(args.output, "arwm_comparison.png")
    comparison_plot_written = write_metric_comparison_plot(
        results, comparison_plot_path
    )
    if comparison_plot_written:
        print(f"Saved: {comparison_plot_path}")

    lines = [
        "# ARWM Variant Evaluation",
        "",
        f"- Evaluated at (UTC): {report['evaluation']['evaluated_at_utc']}",
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
        weight_summary = result.get("region_weight_analysis")
        if weight_summary is not None:
            lines.append(
                f"\n{mode} region weights: mean="
                f"{[round(value, 4) for value in weight_summary['mean']]}, "
                f"std={[round(value, 4) for value in weight_summary['std']]}, "
                f"normalization error="
                f"{weight_summary['maximum_absolute_normalization_error']:.2e}. "
                "These are internal model signals, not explanations."
            )
    lines.extend([
        "",
        f"- Off vs Adaptive comparison: **{comparison['status']}**",
    ])
    if comparison["adaptive_minus_off_percentage_points"] is not None:
        differences = comparison["adaptive_minus_off_percentage_points"]
        lines.append(
            "- Adaptive minus Off: "
            f"accuracy {differences['accuracy']:+.3f} pp; "
            f"macro-F1 {differences['macro_f1']:+.3f} pp."
        )
    lines.extend([
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
