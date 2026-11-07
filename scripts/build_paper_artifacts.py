"""Build a validated paper-artifact package from saved experiment outputs."""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import sys
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from utils.video_analysis import reconstruct_temporal_analysis, write_temporal_plots


CLASS_NAMES = (
    "anger",
    "disgust",
    "fear",
    "happiness",
    "sadness",
    "surprise",
    "neutral",
)
MODES = ("off", "fixed", "adaptive")
REGIONS = ("eye", "eyebrow", "mouth")
COLORS = {"off": "#4c78a8", "fixed": "#f2a541", "adaptive": "#59a14f"}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def installed_version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def normalized_training_conditions(metadata: dict):
    if not isinstance(metadata, dict):
        return None
    config = metadata.get("training_config")
    if not isinstance(config, dict):
        return None
    config = json.loads(json.dumps(config))
    region = config.get("model", {}).get("region_attention", {})
    region.pop("mode", None)
    region.pop("enabled", None)
    paths = config.get("paths", {})
    for key in (
        "checkpoint_dir",
        "log_dir",
        "results_dir",
        "arwm_checkpoint_dir",
        "arwm_log_dir",
        "best_model",
    ):
        paths.pop(key, None)
    return metadata.get("seed"), config


def save_figure(figure, path: Path, title: str, description: str) -> None:
    figure.savefig(
        path,
        dpi=300,
        bbox_inches="tight",
        metadata={"Title": title, "Description": description},
    )
    plt.close(figure)
    with Image.open(path) as image:
        image.verify()
    if path.stat().st_size == 0:
        raise RuntimeError(f"Generated figure is empty: {path}")


def read_predictions(source_dir: Path, report: dict) -> dict:
    output = {}
    identities = {}
    expected_probability_columns = [
        f"probability_{name}" for name in CLASS_NAMES
    ]

    for mode in MODES:
        result = report["variants"].get(mode, {})
        if result.get("status") != "complete_test":
            continue
        path = source_dir / f"predictions_{mode}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing saved per-sample predictions: {path}")
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
        if not rows:
            raise ValueError(f"Prediction file is empty: {path}")

        required_columns = {
            "sample_index",
            "path",
            "true_label",
            "predicted_label",
            *expected_probability_columns,
        }
        if not required_columns.issubset(rows[0]):
            raise ValueError(f"Unexpected prediction columns in {path}")
        sample_ids = [(row["path"], row["true_label"]) for row in rows]
        identities[mode] = sample_ids
        labels = np.asarray(
            [CLASS_NAMES.index(row["true_label"]) for row in rows],
            dtype=np.int64,
        )
        probabilities = np.asarray(
            [
                [float(row[column]) for column in expected_probability_columns]
                for row in rows
            ],
            dtype=np.float64,
        )
        if probabilities.shape != (len(rows), len(CLASS_NAMES)):
            raise ValueError(f"Invalid probability matrix shape for {mode}")
        if not np.isfinite(probabilities).all():
            raise ValueError(f"Non-finite probability score in {path}")
        if np.any(probabilities < 0.0) or np.any(probabilities > 1.0):
            raise ValueError(f"Probability score outside [0, 1] in {path}")
        if not np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-4):
            raise ValueError(f"Class probabilities do not sum to 1 in {path}")
        predictions = probabilities.argmax(axis=1)
        saved_predictions = np.asarray(
            [CLASS_NAMES.index(row["predicted_label"]) for row in rows],
            dtype=np.int64,
        )
        if not np.array_equal(predictions, saved_predictions):
            raise ValueError(f"Saved hard predictions disagree with probabilities: {path}")

        values = report["variants"][mode]["metrics"]
        precision, recall, f1, support = precision_recall_fscore_support(
            labels,
            predictions,
            labels=np.arange(len(CLASS_NAMES)),
            average=None,
            zero_division=0,
        )
        macro_precision, macro_recall, macro_f1, _ = (
            precision_recall_fscore_support(
                labels,
                predictions,
                labels=np.arange(len(CLASS_NAMES)),
                average="macro",
                zero_division=0,
            )
        )
        weighted_f1 = precision_recall_fscore_support(
            labels, predictions, average="weighted", zero_division=0
        )[2]
        metrics = {
            "accuracy": float(accuracy_score(labels, predictions)),
            "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
            "macro_precision": float(macro_precision),
            "macro_recall": float(macro_recall),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
            "roc_auc_ovr_macro": float(
                roc_auc_score(
                    labels,
                    probabilities,
                    labels=np.arange(len(CLASS_NAMES)),
                    multi_class="ovr",
                    average="macro",
                )
            ),
            "per_class_roc_auc_ovr": {
                name: float(
                    roc_auc_score((labels == index).astype(int), probabilities[:, index])
                )
                for index, name in enumerate(CLASS_NAMES)
            },
            "per_class": {
                name: {
                    "precision": float(precision[index]),
                    "recall": float(recall[index]),
                    "f1": float(f1[index]),
                    "support": int(support[index]),
                }
                for index, name in enumerate(CLASS_NAMES)
            },
        }
        for key in (
            "accuracy",
            "balanced_accuracy",
            "macro_precision",
            "macro_recall",
            "macro_f1",
        ):
            if not math.isclose(
                metrics[key], float(values[key]), rel_tol=0.0, abs_tol=1e-10
            ):
                raise ValueError(
                    f"Recomputed {mode} {key}={metrics[key]} does not match "
                    f"the saved evaluation report value {values[key]}"
                )

        output[mode] = {
            "rows": rows,
            "labels": labels,
            "probabilities": probabilities,
            "predictions": predictions,
            "metrics": metrics,
        }

    if set(identities) != set(MODES):
        raise ValueError(
            "A valid three-mode comparison requires complete results for Off, "
            "Fixed, and Adaptive."
        )
    if not all(identities[mode] == identities[MODES[0]] for mode in MODES[1:]):
        raise ValueError("ARWM variants do not contain the same ordered samples and labels")
    if len(identities[MODES[0]]) != report["evaluation"]["sample_count_in_split"]:
        raise ValueError("Saved prediction count differs from the recorded test split")
    if not np.array_equal(
        output["off"]["labels"], output["fixed"]["labels"]
    ) or not np.array_equal(output["off"]["labels"], output["adaptive"]["labels"]):
        raise ValueError("Ground-truth labels differ across ARWM variants")
    return output


def write_classification_and_confusion_reports(
    output_dir: Path, predictions: dict, report: dict
) -> list:
    metric_rows = []
    for mode in MODES:
        data = predictions[mode]
        result = report["variants"][mode]
        labels = data["labels"]
        hard_predictions = data["predictions"]
        matrix = confusion_matrix(
            labels, hard_predictions, labels=np.arange(len(CLASS_NAMES))
        )
        saved_matrix = np.asarray(result.get("confusion_matrix"), dtype=np.int64)
        if not np.array_equal(matrix, saved_matrix):
            raise ValueError(
                f"Recomputed confusion matrix disagrees with saved {mode} evaluation"
            )
        saved_per_class = result["metrics"]["per_class"]
        for index, name in enumerate(CLASS_NAMES):
            for metric_name, value in (
                ("precision", data["metrics"]["per_class"][name]["precision"]),
                ("recall", data["metrics"]["per_class"][name]["recall"]),
                ("f1", data["metrics"]["per_class"][name]["f1"]),
            ):
                if not math.isclose(
                    value,
                    float(saved_per_class[name][metric_name]),
                    rel_tol=0.0,
                    abs_tol=1e-10,
                ):
                    raise ValueError(
                        f"Recomputed {mode} {name} {metric_name} differs from the "
                        "saved evaluation report"
                    )
            if (
                data["metrics"]["per_class"][name]["support"]
                != int(saved_per_class[name]["support"])
            ):
                raise ValueError(
                    f"Recomputed {mode} {name} support differs from the saved report"
                )
        with (output_dir / f"confusion_matrix_{mode}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.writer(file)
            writer.writerow(["true\\predicted", *CLASS_NAMES])
            for name, row in zip(CLASS_NAMES, matrix):
                writer.writerow([name, *row.tolist()])

        report_dict = classification_report(
            labels,
            hard_predictions,
            labels=np.arange(len(CLASS_NAMES)),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )
        with (output_dir / f"classification_report_{mode}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=("label", "precision", "recall", "f1", "support")
            )
            writer.writeheader()
            for name, values in report_dict.items():
                if isinstance(values, dict):
                    row = {
                        "label": name,
                        "precision": values.get("precision"),
                        "recall": values.get("recall"),
                        "f1": values.get("f1-score"),
                        "support": values.get("support"),
                    }
                else:
                    row = {
                        "label": name,
                        "precision": "",
                        "recall": "",
                        "f1": values,
                        "support": len(labels),
                    }
                writer.writerow(
                    row
                )

        metrics = data["metrics"]
        timing = result.get("timing", {})
        metric_rows.append(
            {
                "mode": mode,
                "status": result["status"],
                "samples": len(labels),
                "checkpoint": result["checkpoint"],
                "checkpoint_epoch": result.get("checkpoint_epoch"),
                "seed": (result.get("experiment_metadata") or {}).get("seed"),
                "accuracy": metrics["accuracy"],
                "balanced_accuracy": metrics["balanced_accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"],
                "weighted_f1": metrics["weighted_f1"],
                "roc_auc_ovr_macro": metrics["roc_auc_ovr_macro"],
                "parameter_count": result.get("total_parameters"),
                "device": report["evaluation"]["device"],
                "batch_size": report["evaluation"]["batch_size"],
                "warmup_batches": timing.get("warmup_batches"),
                "mean_forward_latency_ms_per_image": timing.get(
                    "mean_latency_ms_per_image"
                ),
                "mean_forward_latency_ms_per_batch": timing.get(
                    "mean_batch_latency_ms"
                ),
                "p50_forward_latency_ms_per_batch": timing.get(
                    "p50_batch_latency_ms"
                ),
                "p95_forward_latency_ms_per_batch": timing.get(
                    "p95_batch_latency_ms"
                ),
                **{
                    f"roc_auc_ovr_{name}": metrics["per_class_roc_auc_ovr"][name]
                    for name in CLASS_NAMES
                },
            }
        )

        with (output_dir / f"predictions_{mode}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as file:
            fieldnames = (
                "sample_index",
                "path",
                "true_label",
                "predicted_label",
                *[f"probability_{name}" for name in CLASS_NAMES],
            )
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data["rows"])

    with (output_dir / "metrics_by_variant.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(metric_rows[0]))
        writer.writeheader()
        writer.writerows(metric_rows)
    write_json(
        output_dir / "variant_evaluation.json",
        {
            "source_evaluation_report": str(
                (ROOT / "results" / "arwm" / "final" / "off_fixed_20"
                 / "evaluation" / "metrics.json").resolve()
            ),
            "class_order": list(CLASS_NAMES),
            "comparison_assessment": (
                "All three variants have complete predictions for the same ordered "
                "official test samples; comparison is descriptive, not a controlled "
                "three-way training ablation."
            ),
            "metrics_recomputed_from_saved_probabilities": True,
            "roc_auc_method": "macro one-vs-rest; raw saved class probabilities",
            "variants": {
                mode: {
                    **{
                        key: value
                        for key, value in predictions[mode]["metrics"].items()
                        if key != "per_class"
                    },
                    "checkpoint": report["variants"][mode]["checkpoint"],
                    "checkpoint_epoch": report["variants"][mode].get(
                        "checkpoint_epoch"
                    ),
                    "training_metadata": report["variants"][mode].get(
                        "experiment_metadata"
                    ),
                }
                for mode in MODES
            },
            "comparison_status": (
                "complete_same_samples_descriptive_not_controlled"
            ),
            "source_report_comparison_status": report["evaluation"][
                "comparison_status"
            ],
            "comparison_note": (
                "All three modes use the same official test samples, preprocessing, "
                "image size, class order, and inference protocol. The Adaptive "
                "checkpoint's training metadata is absent, so this is not a controlled "
                "three-way training ablation."
            ),
        },
    )
    return metric_rows


def plot_comparison(path: Path, metric_rows: list, sample_count: int) -> None:
    metrics = (
        ("accuracy", "Accuracy"),
        ("balanced_accuracy", "Balanced accuracy"),
        ("macro_precision", "Macro precision"),
        ("macro_recall", "Macro recall"),
        ("macro_f1", "Macro F1"),
        ("roc_auc_ovr_macro", "Macro OVR ROC-AUC"),
    )
    figure, axis = plt.subplots(figsize=(12, 6.5))
    positions = np.arange(len(metrics))
    width = 0.23
    for index, mode in enumerate(MODES):
        value_by_name = {
            key: next(item for item in metric_rows if item["mode"] == mode)[key]
            for key, _ in metrics
        }
        bars = axis.bar(
            positions + (index - 1) * width,
            [value_by_name[key] * 100.0 for key, _ in metrics],
            width,
            color=COLORS[mode],
            label=mode.capitalize(),
        )
        axis.bar_label(bars, fmt="%.1f", fontsize=8, padding=2, rotation=90)
    axis.set_xticks(positions, [label for _, label in metrics])
    axis.set_ylabel("Score (%)")
    axis.set_ylim(0, 100)
    axis.set_title(
        f"FER2013 official test split ({sample_count:,} identical samples)\n"
        "Descriptive checkpoint comparison — Adaptive training provenance is incomplete"
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=3, frameon=False, loc="upper left")
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "ARWM variant comparison",
        "Six metrics recomputed from the same official FER2013 test images. "
        "Not a controlled three-way training ablation.",
    )


def plot_confusion(path: Path, predictions: dict, report: dict) -> None:
    data = predictions["adaptive"]
    matrix = confusion_matrix(
        data["labels"],
        data["predictions"],
        labels=np.arange(len(CLASS_NAMES)),
    )
    figure, axis = plt.subplots(figsize=(9, 8))
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis, label="Number of images")
    axis.set_xticks(range(len(CLASS_NAMES)), [x.capitalize() for x in CLASS_NAMES],
                    rotation=35, ha="right")
    axis.set_yticks(range(len(CLASS_NAMES)), [x.capitalize() for x in CLASS_NAMES])
    axis.set_xlabel("Predicted emotion")
    axis.set_ylabel("True emotion")
    axis.set_title(
        "Adaptive checkpoint — FER2013 official test\n"
        f"{len(data['labels']):,} samples; class order matches FER2013"
    )
    threshold = matrix.max() / 2.0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                color="white" if matrix[row, column] > threshold else "#263238",
                fontsize=9,
            )
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "Adaptive checkpoint confusion matrix",
        f"Checkpoint: {report['variants']['adaptive']['checkpoint']}. "
        "Counts were generated from saved probabilities and labels.",
    )


def plot_roc_and_pr(path: Path, predictions: dict) -> None:
    data = predictions["adaptive"]
    labels = data["labels"]
    probabilities = data["probabilities"]
    binary = (labels[:, None] == np.arange(len(CLASS_NAMES))).astype(int)
    macro_auc = roc_auc_score(
        labels,
        probabilities,
        labels=np.arange(len(CLASS_NAMES)),
        multi_class="ovr",
        average="macro",
    )
    figure, (roc_axis, pr_axis) = plt.subplots(1, 2, figsize=(15, 6.5))
    tpr_values = []
    for index, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        auc_value = roc_auc_score(binary[:, index], probabilities[:, index])
        tpr_values.append(np.interp(np.linspace(0, 1, 201), fpr, tpr))
        roc_axis.plot(fpr, tpr, linewidth=1.6, label=f"{name.capitalize()} (AUC {auc_value:.3f})")
        precision, recall, _ = precision_recall_curve(
            binary[:, index], probabilities[:, index]
        )
        ap = average_precision_score(binary[:, index], probabilities[:, index])
        pr_axis.plot(
            recall,
            precision,
            linewidth=1.5,
            label=f"{name.capitalize()} (AP {ap:.3f})",
        )

    macro_tpr = np.mean(tpr_values, axis=0)
    roc_axis.plot(
        np.linspace(0, 1, 201),
        macro_tpr,
        color="#111111",
        linestyle="--",
        linewidth=2.0,
        label=f"Macro ROC (AUC {macro_auc:.3f})",
    )
    micro_fpr, micro_tpr, _ = roc_curve(binary.ravel(), probabilities.ravel())
    roc_axis.plot(
        micro_fpr,
        micro_tpr,
        color="#777777",
        linestyle=":",
        linewidth=1.8,
        label=f"Micro ROC (AUC {roc_auc_score(binary, probabilities, average='micro'):.3f})",
    )
    precision, recall, _ = precision_recall_curve(
        binary.ravel(), probabilities.ravel()
    )
    micro_ap = average_precision_score(binary, probabilities, average="micro")
    pr_axis.plot(
        recall,
        precision,
        color="#111111",
        linestyle="--",
        linewidth=2.0,
        label=f"Micro PR (AP {micro_ap:.3f})",
    )

    roc_axis.plot([0, 1], [0, 1], color="#999999", linestyle=":", linewidth=1)
    roc_axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title=f"One-vs-rest ROC (macro AUC {macro_auc:.3f})",
    )
    pr_axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title=(
            "One-vs-rest precision–recall "
            f"(macro AP {average_precision_score(binary, probabilities, average='macro'):.3f})"
        ),
    )
    for axis in (roc_axis, pr_axis):
        axis.grid(alpha=0.25)
        axis.legend(fontsize=7.5, loc="lower left", frameon=True)
    figure.suptitle(
        "Adaptive checkpoint — official FER2013 test; scores from saved class probabilities"
    )
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "Adaptive checkpoint ROC and precision-recall curves",
        "One-vs-rest class curves, plus macro/micro summaries. Curves use saved "
        "class probability scores, not hard labels.",
    )


def plot_roc_only(path: Path, predictions: dict) -> None:
    data = predictions["adaptive"]
    labels = data["labels"]
    probabilities = data["probabilities"]
    figure, axis = plt.subplots(figsize=(9, 7))
    aucs = {}
    binary = (labels[:, None] == np.arange(len(CLASS_NAMES))).astype(int)
    tpr_values = []
    for index, name in enumerate(CLASS_NAMES):
        fpr, tpr, _ = roc_curve(binary[:, index], probabilities[:, index])
        auc_value = roc_auc_score(binary[:, index], probabilities[:, index])
        aucs[name] = float(auc_value)
        tpr_values.append(np.interp(np.linspace(0, 1, 201), fpr, tpr))
        axis.plot(fpr, tpr, linewidth=1.7, label=f"{name.capitalize()} (AUC {auc_value:.3f})")
    macro_auc = float(
        roc_auc_score(
            labels,
            probabilities,
            labels=np.arange(len(CLASS_NAMES)),
            multi_class="ovr",
            average="macro",
        )
    )
    axis.plot(
        np.linspace(0, 1, 201),
        np.mean(tpr_values, axis=0),
        color="#111111",
        linestyle="--",
        linewidth=2.2,
        label=f"Macro average (AUC {macro_auc:.3f})",
    )
    axis.plot([0, 1], [0, 1], color="#888888", linestyle=":", linewidth=1)
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="One-vs-rest ROC — Adaptive checkpoint, FER2013 official test",
    )
    axis.grid(alpha=0.25)
    axis.legend(loc="lower right", fontsize=8, frameon=True)
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "Adaptive checkpoint one-vs-rest ROC curves",
        json.dumps({"class_auc": aucs, "macro_auc": macro_auc}),
    )


def plot_region_weights(
    path: Path,
    weight_csv: Path,
    output_dir: Path,
    checkpoint: str,
) -> dict:
    with weight_csv.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    values = {
        mode: np.asarray(
            [
                [float(row[f"{region}_weight"]) for region in REGIONS]
                for row in rows
                if row["mode"] == mode
            ],
            dtype=np.float64,
        )
        for mode in ("fixed", "adaptive")
    }
    for mode, data in values.items():
        if data.ndim != 2 or data.shape[1] != 3 or not len(data):
            raise ValueError(f"Missing or malformed {mode} ARWM region weights")
        if not np.isfinite(data).all() or not np.allclose(data.sum(axis=1), 1, atol=1e-5):
            raise ValueError(f"Invalid {mode} ARWM region weights")
    adaptive_rows = [row for row in rows if row["mode"] == "adaptive"]
    if len(adaptive_rows) != 7178:
        raise ValueError(
            "Adaptive region-weight output does not cover the complete 7,178-sample test split"
        )
    with (output_dir / "adaptive_region_weights.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fieldnames = (
            "mode",
            "test_sample_index",
            "eye_weight",
            "eyebrow_weight",
            "mouth_weight",
        )
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(adaptive_rows)

    figure, axis = plt.subplots(figsize=(9, 6))
    positions = np.arange(3)
    width = 0.32
    for index, mode in enumerate(("fixed", "adaptive")):
        offset = (index - 0.5) * width
        box = axis.boxplot(
            [values[mode][:, i] for i in range(3)],
            positions=positions + offset,
            widths=width * 0.8,
            patch_artist=True,
            showfliers=False,
            showmeans=True,
        )
        for patch in box["boxes"]:
            patch.set_facecolor(COLORS[mode])
            patch.set_alpha(0.65)
        for mean in box["means"]:
            mean.set_marker("D")
            mean.set_markerfacecolor("#222222")
            mean.set_markersize(4)
        axis.plot([], [], color=COLORS[mode], linewidth=7, label=mode.capitalize())
    axis.set_xticks(positions, [region.capitalize() for region in REGIONS])
    axis.set_ylim(0, 1)
    axis.set_ylabel("ARWM weight")
    axis.set_title(
        "Region weights on official FER2013 test samples\n"
        "Internal model signals; not causal explanations"
    )
    axis.text(
        0.01,
        -0.20,
        f"Adaptive checkpoint: {checkpoint}\n"
        "Fixed mode uses equal 1/3 weights; Off has no region weights.",
        transform=axis.transAxes,
        fontsize=8,
        va="top",
    )
    axis.grid(axis="y", alpha=0.25)
    axis.legend(frameon=False)
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "ARWM region weights",
        f"Adaptive source checkpoint: {checkpoint}. Held-out model weights, "
        "not explanations or causal attributions.",
    )
    return {
        mode: {
            "n": int(len(data)),
            "mean": data.mean(axis=0).tolist(),
            "std": data.std(axis=0).tolist(),
            "source_checkpoint": (
                checkpoint if mode == "adaptive" else
                "checkpoints/arwm_variants/fixed_20/best_model.pth"
            ),
        }
        for mode, data in values.items()
    }


def plot_architecture(path: Path) -> None:
    figure, axis = plt.subplots(figsize=(16, 9))
    axis.set(xlim=(0, 1.03), ylim=(0, 1))
    axis.axis("off")
    objective1 = "#fff2cf"
    objective2 = "#e6f2ff"

    def box(x, y, width, height, text, color, fontsize=10, linewidth=1.4, style="round"):
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle=f"{style},pad=0.012",
            facecolor=color,
            edgecolor="#34495e",
            linewidth=linewidth,
        )
        axis.add_patch(patch)
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            wrap=True,
        )

    def arrow(start, end, color="#40566f", style="-", connection="arc3,rad=0"):
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=13,
                linewidth=1.6,
                color=color,
                linestyle=style,
                connectionstyle=connection,
            )
        )

    boxes = [
        (0.02, 0.68, 0.105, 0.14, "Video /\nwebcam", "#edf2f7"),
        (0.16, 0.68, 0.12, 0.14, "MediaPipe\nface detection", "#edf2f7"),
        (0.315, 0.68, 0.14, 0.14, "Face crop +\npreprocessing\n160 × 160", "#edf2f7"),
        (0.49, 0.68, 0.12, 0.14, "Swin\nTransformer\nfeature tokens", "#e8eaf6"),
        (0.645, 0.68, 0.12, 0.14, "FANE\nattention", "#e8eaf6"),
        (0.81, 0.68, 0.12, 0.14, "Fused emotion\nembedding +\nclassifier", "#e8eaf6"),
        (0.945, 0.43, 0.075, 0.14, "Emotion\nprobabilities\n(7 classes)", "#edf2f7"),
    ]
    for values in boxes:
        box(*values)
    for left, right in (
        ((0.125, 0.75), (0.16, 0.75)),
        ((0.28, 0.75), (0.315, 0.75)),
        ((0.455, 0.75), (0.49, 0.75)),
        ((0.61, 0.75), (0.645, 0.75)),
        ((0.765, 0.75), (0.81, 0.75)),
        ((0.87, 0.68), (0.945, 0.57)),
    ):
        arrow(left, right)

    box(
        0.34,
        0.39,
        0.26,
        0.19,
        "Geometric soft masks\nEyes · Eyebrows · Mouth\n(region-pooled Swin tokens)",
        objective1,
        fontsize=9,
    )
    box(
        0.64,
        0.39,
        0.23,
        0.19,
        "Fixed: equal weights\nAdaptive: input-conditioned\nsoftmax weights\nWeighted region representation\n(additive fusion, weight 0.3)",
        objective1,
        fontsize=8.5,
    )
    arrow((0.545, 0.68), (0.48, 0.58), color="#bc7a00")
    arrow((0.60, 0.485), (0.64, 0.485), color="#bc7a00")
    arrow((0.755, 0.58), (0.84, 0.68), color="#bc7a00")
    axis.text(
        0.35,
        0.62,
        "Objective 1 — ARWM branch",
        color="#9a6400",
        fontsize=9,
        weight="bold",
        ha="left",
    )

    box(
        0.63,
        0.10,
        0.21,
        0.15,
        "Objective 2 — Temporal signal tracker\nTransitions · Persistence\nProbability volatility\nNegative-affect volatility",
        objective2,
        fontsize=8.5,
    )
    box(
        0.86,
        0.10,
        0.12,
        0.15,
        "Deterministic\nstress-related\nproxy",
        objective2,
        fontsize=8.5,
    )
    arrow((0.982, 0.43), (0.79, 0.26), color="#2673a8", connection="arc3,rad=0.08")
    arrow((0.84, 0.135), (0.86, 0.135), color="#2673a8")
    axis.text(
        0.02,
        0.035,
        "Recorded-video inference uses per-frame probabilities for the descriptive proxy. "
        "The implemented BiLSTM sequence path is not called by this video inference flow "
        "and does not feed this proxy.",
        fontsize=9,
        color="#34495e",
        wrap=True,
    )
    axis.set_title(
        "Region-aware Swin–FANE emotion analysis and implemented video temporal path",
        fontsize=15,
        pad=14,
        weight="bold",
    )
    figure.tight_layout()
    save_figure(
        figure,
        path,
        "Framework architecture and implemented video inference path",
        "ARWM pools geometric eye, eyebrow, and mouth priors from Swin tokens; "
        "the recorded-video deterministic proxy consumes frame emotion probabilities, "
        "not BiLSTM outputs.",
    )


def write_training_history(output_dir: Path, figures_dir: Path) -> dict:
    sources = {
        "off": ROOT / "logs" / "arwm_variants" / "off_20"
        / "swin_fane_run1_20261020_132509" / "metric_history.json",
        "fixed": ROOT / "logs" / "arwm_variants" / "fixed_20"
        / "swin_fane_run1_20261020_180118" / "metric_history.json",
    }
    histories = {}
    rows = []
    for mode, source in sources.items():
        if not source.is_file():
            continue
        history = read_json(source)
        required = ("train_mean_loss", "val_mean_loss", "train_accuracy", "val_accuracy", "val_f1")
        if any(key not in history for key in required):
            raise ValueError(f"Incomplete training history: {source}")
        count = len(history["train_mean_loss"])
        if any(len(history[key]) != count for key in required):
            raise ValueError(f"Training history metrics have different lengths: {source}")
        histories[mode] = history
        for index in range(count):
            rows.append(
                {
                    "mode": mode,
                    "epoch": index + 1,
                    "train_loss": history["train_mean_loss"][index],
                    "validation_loss": history["val_mean_loss"][index],
                    "train_accuracy_percent": history["train_accuracy"][index],
                    "validation_accuracy_percent": history["val_accuracy"][index],
                    "validation_macro_f1_percent": history["val_f1"][index],
                }
            )
    if not histories:
        raise FileNotFoundError("No training metric histories were found")
    with (output_dir / "training_history.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = (
        ("train_mean_loss", "val_mean_loss", "Loss", "Loss"),
        ("train_accuracy", "val_accuracy", "Accuracy (%)", "Accuracy (%)"),
        (None, "val_f1", "Validation macro-F1 (%)", "Macro-F1 (%)"),
    )
    for axis, (train_key, val_key, title, ylabel) in zip(axes.flat, plots):
        for mode, history in histories.items():
            epochs = np.arange(1, len(history[val_key]) + 1)
            if train_key:
                axis.plot(
                    epochs,
                    history[train_key],
                    color=COLORS[mode],
                    linestyle="--",
                    alpha=0.65,
                    label=f"{mode.capitalize()} train",
                )
            axis.plot(
                epochs,
                history[val_key],
                color=COLORS[mode],
                linewidth=1.8,
                label=f"{mode.capitalize()} validation",
            )
        axis.set_title(title)
        axis.set_xlabel("Epoch")
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8, frameon=False)
    axes.flat[3].axis("off")
    axes.flat[3].text(
        0.02,
        0.8,
        "Available training histories:\n"
        + "\n".join(
            f"{mode.capitalize()}: {len(history['train_mean_loss'])} epochs"
            for mode, history in histories.items()
        )
        + "\nAdaptive history: unavailable",
        va="top",
        fontsize=11,
    )
    figure.suptitle(
        "Recorded FER2013 training curves — one seed per available mode",
        fontsize=13,
    )
    figure.tight_layout()
    save_figure(
        figure,
        figures_dir / "training_curves.png",
        "Training and validation curves",
        "Existing training logs for Off and Fixed modes. Adaptive training history "
        "was not found.",
    )
    return {
        mode: {
            "source": str(sources[mode].resolve()),
            "epochs": len(history["train_mean_loss"]),
        }
        for mode, history in histories.items()
    }


def write_video_artifacts(output_dir: Path, figures_dir: Path) -> dict:
    source_dir = ROOT / "results" / "testvideo1"
    inference_log = read_json(source_dir / "inference_log.json")
    result = read_json(source_dir / "results.json")
    checkpoint_report = read_json(source_dir / "checkpoint_load.json")
    if checkpoint_report.get("strict_load") is not True:
        raise ValueError("Video checkpoint report does not confirm a strict load")
    if checkpoint_report.get("region_attention_mode") != "adaptive":
        raise ValueError("Video checkpoint report does not confirm adaptive mode")
    if checkpoint_report.get("emotion_class_order") != list(CLASS_NAMES):
        raise ValueError("Video checkpoint report has an unexpected emotion class order")
    if result.get("temporal_model", {}).get("learned_temporal_model_used") is not False:
        raise ValueError("Video result does not confirm that the learned temporal model was unused")

    config = inference_log.get("configuration", {})
    stress_config = config.get("stress", {})
    alpha = float(stress_config.get("alpha", 0.7))
    beta = float(stress_config.get("beta", 0.3))
    analysis, primary = reconstruct_temporal_analysis(
        inference_log, alpha=alpha, beta=beta
    )
    write_temporal_plots(
        primary,
        analysis,
        str(figures_dir),
        include_temporal_plot=True,
        dpi=300,
    )

    frames = inference_log.get("frames", [])
    sampled = [
        frame for frame in frames
        if frame.get("detection_status") != "skipped_by_stride"
    ]
    detections = sum(
        len(frame.get("detections", [])) for frame in sampled
    )
    valid_predictions = sum(
        1
        for frame in frames
        for face in frame.get("faces", [])
        if face.get("probabilities")
    )
    timestamps = np.asarray(
        [float(frame["timestamp_seconds"]) for frame in sampled],
        dtype=np.float64,
    )
    intervals = np.diff(timestamps)
    positive_intervals = intervals[intervals > 0]
    input_info = inference_log.get("input", {})
    inference_info = inference_log.get("inference", {})
    interval = (
        float(np.median(positive_intervals))
        if positive_intervals.size
        else None
    )
    primary_id = str(analysis["primary_plot_track_id"])
    primary_summary = analysis["tracks"][primary_id]
    primary_predictions = primary["probabilities"]
    emotion_counts = np.bincount(
        primary_predictions.argmax(axis=1), minlength=len(CLASS_NAMES)
    )
    dominant_index = int(emotion_counts.argmax())
    dominant_frequency = int(emotion_counts[dominant_index])
    runs = primary_summary["persistence_runs"]
    summary = {
        "source_video": str((ROOT / inference_log["video"]).resolve()),
        "source_inference_directory": str(source_dir.resolve()),
        "checkpoint": checkpoint_report["checkpoint"],
        "checkpoint_epoch": checkpoint_report.get("checkpoint_epoch"),
        "region_attention_mode": checkpoint_report["region_attention_mode"],
        "class_order": list(CLASS_NAMES),
        "number_of_sampled_frames": int(len(sampled)),
        "number_of_source_frames": int(input_info.get("source_frame_count", len(frames))),
        "number_of_detected_faces": int(detections),
        "number_of_valid_predictions": int(valid_predictions),
        "video_duration_seconds": float(input_info["duration_seconds"]),
        "video_fps": float(input_info["fps"]),
        "frame_sampling_stride": int(inference_info["frame_stride"]),
        "frame_sampling_interval_seconds_median": interval,
        "frame_sampling_interval_seconds_nominal": (
            float(inference_info["frame_stride"])
            / float(inference_info["effective_fps_used_for_timestamps"])
        ),
        "primary_track_id": primary_id,
        "primary_track_prediction_count": primary_summary["prediction_count"],
        "emotion_transition_count": primary_summary["transition_count"],
        "emotion_persistence_durations_seconds": [
            run["duration_seconds"] for run in runs
        ],
        "longest_emotion_persistence_seconds": max(
            (run["duration_seconds"] for run in runs), default=0.0
        ),
        "mean_probability_volatility_half_l1": (
            primary_summary["probability_volatility_mean_half_l1"]
        ),
        "mean_negative_affect_volatility": (
            primary_summary["negative_affect_volatility_mean"]
        ),
        "mean_stress_related_proxy": primary_summary["stress_proxy_mean"],
        "dominant_predicted_emotion": CLASS_NAMES[dominant_index],
        "dominant_emotion_frequency": dominant_frequency,
        "dominant_emotion_fraction": float(
            dominant_frequency / primary_summary["prediction_count"]
        ),
        "scope": (
            "Temporal statistics and dominant emotion refer to the primary face "
            "track (the track with the most valid predictions); frame, detection, "
            "and prediction counts cover the entire recorded video."
        ),
        "formula": analysis["method"],
        "all_tracks": analysis["tracks"],
        "learned_bilstm_used_for_video_proxy": False,
        "stress_interpretation": (
            "Descriptive deterministic proxy from emotion probabilities; not a "
            "clinically validated stress measurement."
        ),
    }
    write_json(output_dir / "video_temporal_summary.json", summary)
    summary_fields = [
        key for key, value in summary.items()
        if not isinstance(value, (dict, list))
    ]
    summary_fields.append("emotion_persistence_durations_seconds_json")
    summary_row = {
        key: value for key, value in summary.items()
        if not isinstance(value, (dict, list))
    }
    summary_row["emotion_persistence_durations_seconds_json"] = json.dumps(
        summary["emotion_persistence_durations_seconds"],
        allow_nan=False,
    )
    with (output_dir / "video_temporal_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerow(summary_row)
    with (output_dir / "video_persistence_runs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = (
            "track_id",
            "emotion",
            "start_seconds",
            "end_seconds",
            "duration_seconds",
            "frame_count",
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for track_id, track in analysis["tracks"].items():
            for run in track["persistence_runs"]:
                writer.writerow({"track_id": track_id, **run})
    with (output_dir / "video_temporal_records.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        fields = (
            "track_id",
            "timestamp_seconds",
            "dominant_emotion",
            "transition",
            "persistence_seconds",
            "probability_volatility",
            "negative_affect",
            "negative_affect_volatility",
            "stress_proxy",
            *[f"probability_{name}" for name in CLASS_NAMES],
        )
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for track_id, track in analysis["tracks"].items():
            for record in track["time_steps"]:
                writer.writerow(
                    {
                        "track_id": track_id,
                        "timestamp_seconds": record["timestamp_seconds"],
                        "dominant_emotion": record["dominant_emotion"],
                        "transition": record["transition"],
                        "persistence_seconds": record["persistence_seconds"],
                        "probability_volatility": record["probability_volatility"],
                        "negative_affect": record["negative_affect"],
                        "negative_affect_volatility": record[
                            "negative_affect_volatility"
                        ],
                        "stress_proxy": record["stress_proxy"],
                        **{
                            f"probability_{name}": record["emotion_probabilities"][name]
                            for name in CLASS_NAMES
                        },
                    }
                )
    return summary


def write_architecture(path: Path) -> None:
    plot_architecture(path)


def write_readme(path: Path, metric_rows: list, video_summary: dict) -> None:
    lines = [
        "# Reproducibility artifacts",
        "",
        "All FER2013 metrics and curves use saved predictions from the official "
        "test folder. No model inference or retraining was run while assembling "
        "this package. Figures are rendered at 300 DPI.",
        "",
        "## Artifacts",
        "",
        "| Artifact | Contents and provenance |",
        "|---|---|",
        "| `figures/architecture.png` | Diagram checked against the Swin/FANE/ARWM implementation and recorded-video flow. |",
        "| `figures/arwm_comparison.png` | Six test metrics for Off, Fixed, and Adaptive on the same 7,178 images; descriptive, not a controlled three-way ablation. |",
        "| `figures/confusion_matrix.png` | Adaptive checkpoint confusion matrix, all seven FER2013 classes. |",
        "| `figures/roc_curves.png` | Adaptive one-vs-rest ROC curves from saved class probabilities. |",
        "| `figures/roc_pr_curves.png` | ROC and precision-recall curves; the repository plotting workflow supports both. |",
        "| `figures/region_weights.png` | Evaluated Fixed and Adaptive region-weight distributions; internal signals, not explanations. |",
        "| `figures/emotion_timeline.png`, `figures/stress_trajectory.png`, `figures/temporal_analysis.png` | Rebuilt from `results/testvideo1/inference_log.json` using repository temporal-analysis formulas. |",
        "| `figures/training_curves.png` | Existing 20-epoch Off and Fixed training histories; Adaptive training history was unavailable. |",
        "| `metrics_by_variant.csv`, `variant_evaluation.json` | Recomputed metrics, macro OVR ROC-AUC, per-class AUC, model size, forward latency, and provenance. |",
        "| `predictions_<mode>.csv` | Per-sample labels and seven saved class probabilities for each complete mode. |",
        "| `classification_report_<mode>.csv`, `confusion_matrix_<mode>.csv` | Per-variant class reports and counts in FER2013 class order. |",
        "| `adaptive_region_weights.csv` | Actual adaptive weights from 7,178 held-out test inputs. |",
        "| `video_temporal_summary.csv`, `video_temporal_summary.json` | Counts and primary-track temporal statistics from the recorded video. |",
        "| `video_temporal_records.csv`, `video_persistence_runs.csv` | Prediction-level temporal signals and persistence intervals. |",
        "| `training_history.csv` | Recorded Off and Fixed training/validation history from run logs. |",
        "| `experiment_manifest.json`, `NOTES.md` | Commands, software/device provenance, limitations, and unresolved issues. |",
        "| `preexisting_figures/` | Copies of root figures that were present before refreshing `figures/`; retained as backups. |",
        "",
        "## Evaluation results",
        "",
        "| Mode | N | Accuracy | Balanced accuracy | Macro precision | Macro recall | Macro F1 | Macro OVR AUC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metric_rows:
        lines.append(
            f"| {row['mode']} | {row['samples']} | {row['accuracy']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['macro_precision']:.4f} | "
            f"{row['macro_recall']:.4f} | {row['macro_f1']:.4f} | "
            f"{row['roc_auc_ovr_macro']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Recorded video",
            "",
            f"- Sampled frames: {video_summary['number_of_sampled_frames']}; "
            f"detected faces: {video_summary['number_of_detected_faces']}; "
            f"valid predictions: {video_summary['number_of_valid_predictions']}.",
            f"- Duration: {video_summary['video_duration_seconds']:.3f}s; "
            f"median sampling interval: "
            f"{video_summary['frame_sampling_interval_seconds_median']:.4f}s.",
            f"- Temporal summaries use primary face track {video_summary['primary_track_id']}; "
            f"dominant emotion: {video_summary['dominant_predicted_emotion']} "
            f"({video_summary['dominant_emotion_frequency']} predictions).",
            "- The stress-related signal is deterministic and descriptive, not a "
            "clinically validated stress measurement.",
            "",
            "## Rebuilding this package",
            "",
            "Use `python scripts/build_paper_artifacts.py --output results/paper_artifacts_rebuilt` "
            "to regenerate from the saved source results without overwriting this package.",
            "",
            "The root `figures/` copies are refreshed from this package. Existing root "
            "images were preserved unchanged under `preexisting_figures/` before replacement.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_package(output_dir: Path) -> Path:
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing artifacts: {output_dir}"
        )
    source_dir = ROOT / "results" / "arwm" / "final" / "off_fixed_20" / "evaluation"
    report = read_json(source_dir / "metrics.json")
    if report["evaluation"]["dataset"] != "FER2013":
        raise ValueError("Saved ARWM evaluation is not FER2013")
    if report["evaluation"]["split"] != "official test":
        raise ValueError("Saved ARWM evaluation is not the official test split")
    if report["evaluation"]["configuration"]["dataset"]["class_names"] != list(CLASS_NAMES):
        raise ValueError("Saved evaluation class mapping differs from FER2013")
    predictions = read_predictions(source_dir, report)

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".paper_artifacts_build_", dir=output_dir.parent))
    try:
        figures_dir = stage / "figures"
        figures_dir.mkdir()
        output_tables = stage / "evaluation"
        output_tables.mkdir()

        metric_rows = write_classification_and_confusion_reports(
            output_tables, predictions, report
        )
        plot_comparison(
            figures_dir / "arwm_comparison.png",
            metric_rows,
            report["evaluation"]["sample_count_in_split"],
        )
        plot_confusion(figures_dir / "confusion_matrix.png", predictions, report)
        plot_roc_only(figures_dir / "roc_curves.png", predictions)
        plot_roc_and_pr(figures_dir / "roc_pr_curves.png", predictions)
        write_architecture(figures_dir / "architecture.png")
        region_stats = plot_region_weights(
            figures_dir / "region_weights.png",
            source_dir / "region_weight_samples.csv",
            output_tables,
            report["variants"]["adaptive"]["checkpoint"],
        )
        video_summary = write_video_artifacts(output_tables, figures_dir)
        training_sources = write_training_history(output_tables, figures_dir)

        # Store the original per-sample outputs unmodified for audit and reuse.
        for mode in MODES:
            shutil.copy2(
                source_dir / f"predictions_{mode}.csv",
                output_tables / f"predictions_{mode}.csv",
            )
        if not (output_tables / "adaptive_region_weights.csv").is_file():
            raise RuntimeError("Adaptive region-weight CSV was not generated")

        evaluation = report["evaluation"]
        variants = report["variants"]
        mode_configs = {
            "off": ROOT / "configs" / "arwm" / "off_20.yaml",
            "fixed": ROOT / "configs" / "arwm" / "fixed_20.yaml",
            "adaptive": ROOT / "configs" / "arwm" / "adaptive.yaml",
        }
        off_conditions = normalized_training_conditions(
            variants["off"].get("experiment_metadata")
        )
        fixed_conditions = normalized_training_conditions(
            variants["fixed"].get("experiment_metadata")
        )
        off_fixed_conditions_match = (
            off_conditions is not None and off_conditions == fixed_conditions
        )
        command = (
            ".\\.venv\\Scripts\\python.exe scripts\\evaluate_arwm_variants.py "
            "--config configs\\arwm\\off_20.yaml "
            "--off-checkpoint checkpoints\\arwm_variants\\off_20\\best_model.pth "
            "--fixed-checkpoint checkpoints\\arwm_variants\\fixed_20\\best_model.pth "
            "--adaptive-checkpoint checkpoints\\best_model.pth "
            "--batch-size 8 --warmup-batches 3 "
            "--output results\\arwm\\paper_reproduction"
        )
        command_video = (
            ".\\.venv\\Scripts\\python.exe scripts\\infer_video.py "
            "--video data\\videos\\testvideo1.mp4 "
            "--checkpoint checkpoints\\best_model.pth "
            "--config configs\\arwm\\adaptive.yaml --arwm-mode adaptive "
            "--face-mode all --frame-stride 3 "
            "--output results\\testvideo1_reproduced"
        )
        experiment_manifest = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifact_source": {
                "evaluation_report": str((source_dir / "metrics.json").resolve()),
                "saved_per_sample_predictions": {
                    mode: str((source_dir / f"predictions_{mode}.csv").resolve())
                    for mode in MODES
                },
                "video_inference_log": str(
                    (ROOT / "results" / "testvideo1" / "inference_log.json").resolve()
                ),
                "training_histories": {
                    mode: source for mode, source in training_sources.items()
                },
            },
            "evaluation_protocol": {
                key: evaluation.get(key)
                for key in (
                    "dataset",
                    "dataset_root",
                    "split",
                    "evaluated_at_utc",
                    "sample_count_in_split",
                    "device",
                    "hardware",
                    "image_size",
                    "batch_size",
                    "config_path",
                    "comparison_status",
                )
            },
            "class_order": list(CLASS_NAMES),
            "sample_identity_validation": {
                "ordered_sample_paths_and_labels_match_across_modes": True,
                "sample_count": report["evaluation"]["sample_count_in_split"],
            },
            "figure_dpi": 300,
            "variants": {
                mode: {
                    "checkpoint_path": str(
                        (ROOT / variants[mode]["checkpoint"]).resolve()
                    ),
                    "checkpoint_epoch": variants[mode].get("checkpoint_epoch"),
                    "mode_compatible_configuration_path": str(
                        mode_configs[mode].resolve()
                    ),
                    "training_configuration_provenance": (
                        "Checkpoint experiment metadata records the training "
                        "configuration."
                        if variants[mode].get("experiment_metadata")
                        else (
                            "Unavailable. This configuration file describes the "
                            "current mode but is not verified as the checkpoint's "
                            "training configuration."
                        )
                    ),
                    "evaluation_base_config_path": str(
                        (ROOT / evaluation["config_path"]).resolve()
                    ),
                    "effective_mode": variants[mode].get("model_mode_from_config"),
                    "experiment_metadata": variants[mode].get(
                        "experiment_metadata"
                    ),
                    "parameter_count": variants[mode].get("total_parameters"),
                    "latency_protocol": variants[mode].get("timing"),
                    "metric_provenance": (
                        "Recomputed from saved per-sample labels and probabilities; "
                        "official FER2013 test split."
                    ),
                }
                for mode in MODES
            },
            "off_fixed_training_conditions_match": off_fixed_conditions_match,
            "region_weight_provenance": {
                "source": str((source_dir / "region_weight_samples.csv").resolve()),
                "adaptive_checkpoint": variants["adaptive"]["checkpoint"],
                "adaptive_mode": "adaptive",
                "statistics": region_stats,
                "interpretation": "Internal model weights, not causal explanations.",
            },
            "video_temporal_provenance": {
                "source_log": str(
                    (ROOT / "results" / "testvideo1" / "inference_log.json").resolve()
                ),
                "checkpoint": video_summary["checkpoint"],
                "formula": video_summary["formula"],
                "learned_bilstm_used": False,
                "stress_proxy_validated": False,
            },
            "commands": {
                "supported_arwm_evaluation_reproduction": command,
                "arwm_command_status": (
                    "Equivalent supported command for a new output directory; the "
                    "original argv was not stored in the result report. Not rerun "
                    "during package generation."
                ),
                "supported_video_inference_reproduction": command_video,
                "video_command_status": (
                    "Supported inference command; not rerun because complete "
                    "per-frame predictions and timestamps already exist."
                ),
                "artifact_package_rebuild": (
                    ".\\.venv\\Scripts\\python.exe scripts\\build_paper_artifacts.py "
                    "--output results\\paper_artifacts_rebuilt"
                ),
            },
            "software": {
                "python_version_at_artifact_generation": platform.python_version(),
                "package_versions_at_artifact_generation": {
                    name: installed_version(name)
                    for name in (
                        "torch",
                        "torchvision",
                        "timm",
                        "numpy",
                        "scikit-learn",
                        "matplotlib",
                        "Pillow",
                        "opencv-python",
                        "PyYAML",
                    )
                },
                "training_environment_recorded_for_off_fixed": {
                    "python_version": "3.11.9",
                    "torch_version": "2.14.1+cu126",
                    "cuda_runtime": "12.6",
                    "device": "NVIDIA GeForce RTX 3050 6GB Laptop GPU",
                },
                "evaluation_runtime_note": (
                    "The current environment's torch imports, but importing timm "
                    "raises RuntimeError: operator torchvision::nms does not exist. "
                    "The temporal-analysis unittest import also fails because "
                    "Application Control blocks the stringzilla DLL. "
                    "The saved evaluations are reused; this environment issue was "
                    "not repaired or hidden."
                ),
            },
            "metric_method": {
                "accuracy_metrics": "Recomputed from hard predictions and true labels.",
                "roc_auc": (
                    "Seven-class macro one-vs-rest ROC-AUC from saved class "
                    "probability vectors; each per-class AUC uses its OVR probability."
                ),
                "classification_report": "Scikit-learn class precision, recall, F1 and support.",
                "latency": (
                    "Recorded model-forward timing only; inference_mode, perf_counter, "
                    "CUDA synchronization, three warm-up batches; input loading and "
                    "host-to-device transfer excluded."
                ),
            },
            "statistical_scope": (
                "One recorded seed per mode; no confidence intervals or statistical "
                "significance claim."
            ),
        }
        write_json(stage / "experiment_manifest.json", experiment_manifest)
        write_readme(stage / "README.md", metric_rows, video_summary)
        training_durations = {
            "Off": sum(
                read_json(
                    ROOT / "logs" / "arwm_variants" / "off_20"
                    / "swin_fane_run1_20261020_132509" / "metric_history.json"
                ).get("epoch_duration_seconds", [])
            ),
            "Fixed": sum(
                read_json(
                    ROOT / "logs" / "arwm_variants" / "fixed_20"
                    / "swin_fane_run1_20261020_180118" / "metric_history.json"
                ).get("epoch_duration_seconds", [])
            ),
        }
        estimated_adaptive_hours = (
            sum(training_durations.values()) / len(training_durations) / 3600.0
        )
        notes = [
            "# Limitations and unresolved issues",
            "",
            "- All three modes have full saved FER2013 official-test predictions for "
            "the same 7,178 images, with the same labels, class order, 160-pixel "
            "preprocessing, and evaluation protocol.",
            "- This is not a controlled three-way ablation: the Adaptive checkpoint "
            "is `checkpoints/best_model.pth`, epoch 18, and has no experiment/training "
            "metadata or corresponding Adaptive training log in the repository. Its "
            "mode compatibility was confirmed by the recorded strict evaluation.",
            "- Off and Fixed each have one recorded seed (42). Their recorded "
            "training settings match after excluding the requested ARWM mode and "
            "separate output paths: "
            f"{off_fixed_conditions_match}. One seed per condition does not support "
            "statistical significance claims.",
            "- To obtain a controlled three-way result, train Adaptive with the same "
            "dataset split, seed, preprocessing, model-selection rule, epoch cap, and "
            "hyperparameters as the Off/Fixed 20-epoch configs, then evaluate all "
            "three on the identical official test set. Do not overwrite the existing "
            "checkpoints. Approval is required before starting this long run.",
            "- Existing 20-epoch logs took approximately "
            f"{training_durations['Off'] / 3600:.1f} GPU hours (Off) and "
            f"{training_durations['Fixed'] / 3600:.1f} GPU hours (Fixed) on the "
            "RTX 3050 6GB. A matching 20-epoch Adaptive run is estimated at about "
            f"{estimated_adaptive_hours:.1f} GPU hours and must be approved first.",
            "- The current environment's installed `timm`/`torchvision` combination "
            "fails import with `RuntimeError: operator torchvision::nms does not exist`; "
            "the completed saved evaluations were reused. No dependency changes or "
            "fresh FER2013 evaluation were made.",
            "- `tests/test_video_analysis.py` could not be imported by unittest "
            "because the Application Control policy blocked the `stringzilla` DLL "
            "through the video inference preprocessing dependencies.",
            "- The test video has no independent emotion/stress annotations. Its "
            "stress-related output is a deterministic descriptive proxy, not a "
            "clinically validated measurement.",
            "- Video inference consumes frame-level probabilities and does not invoke "
            "the BiLSTM. The available training loop optimizes `forward_frame`; "
            "learned temporal outputs were not used as video stress predictions.",
            "- The region branch uses fixed geometric soft masks over Swin tokens "
            "for eyes, eyebrows, and mouth. The observed adaptive weights are model "
            "signals, not causal explanations; no global-context causal claim is made.",
            "- ROC/PR curves and AUC were recomputed from saved softmax probabilities. "
            "The original saved evaluation report did not include test ROC-AUC.",
            "- The evaluation report does not preserve the exact original argv. The "
            "manifest supplies an equivalent supported command that writes to a new "
            "output directory; it was not rerun in the current environment.",
            "",
        ]
        (stage / "NOTES.md").write_text("\n".join(notes), encoding="utf-8")

        required_pngs = (
            "architecture.png",
            "arwm_comparison.png",
            "confusion_matrix.png",
            "roc_curves.png",
            "roc_pr_curves.png",
            "region_weights.png",
            "emotion_timeline.png",
            "stress_trajectory.png",
            "temporal_analysis.png",
            "training_curves.png",
        )
        for filename in required_pngs:
            image_path = figures_dir / filename
            if not image_path.is_file() or image_path.stat().st_size == 0:
                raise RuntimeError(f"Required figure was not created: {image_path}")
            with Image.open(image_path) as image:
                image.verify()

        for file_path in stage.rglob("*.csv"):
            with file_path.open(encoding="utf-8", newline="") as file:
                if not next(csv.reader(file), None):
                    raise RuntimeError(f"CSV is empty: {file_path}")
        for file_path in stage.rglob("*.json"):
            read_json(file_path)
            if file_path.stat().st_size == 0:
                raise RuntimeError(f"JSON is empty: {file_path}")

        os.replace(stage, output_dir)
        return output_dir
    except Exception:
        if stage.exists():
            shutil.rmtree(stage)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default="results/paper_artifacts",
        help="New artifact directory; existing directories are never overwritten",
    )
    args = parser.parse_args()
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    result = build_package(output.resolve())
    print(f"Validated paper artifacts: {result}")
    for path in sorted(result.rglob("*")):
        if path.is_file():
            print(f"{path.relative_to(result)} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
