"""Export metrics already stored in checkpoints without evaluating the model."""

import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path

import torch


METRIC_FIELDS = (
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "mean_loss",
    "arwm_eye_weight_mean",
    "arwm_eye_weight_std",
    "arwm_eyebrow_weight_mean",
    "arwm_eyebrow_weight_std",
    "arwm_mouth_weight_mean",
    "arwm_mouth_weight_std",
)

METRIC_LABELS = {
    "accuracy": "accuracy_percent",
    "balanced_accuracy": "balanced_accuracy_fraction",
    "precision": "macro_precision_percent",
    "recall": "macro_recall_percent",
    "f1": "macro_f1_percent",
    "roc_auc": "macro_ovr_roc_auc_fraction",
    "mean_loss": "mean_loss",
}


def unique_output_dir(output_root: Path, prefix: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    base = output_root / f"{prefix}_{timestamp}"
    candidate = base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base.name}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def read_checkpoint_metrics(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Checkpoint must contain a dictionary: {checkpoint_path}")
    metrics = checkpoint.get("metrics", {})
    if not isinstance(metrics, dict):
        raise ValueError(f"Checkpoint metrics must be a dictionary: {checkpoint_path}")

    return {
        "checkpoint_filename": checkpoint_path.name,
        "checkpoint_path": str(checkpoint_path.resolve()),
        "epoch": checkpoint.get("epoch"),
        "metric_source": "previously recorded checkpoint metrics; not newly evaluated",
        "metrics": {key: metrics.get(key) for key in METRIC_FIELDS},
        "metric_units": METRIC_LABELS,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoints",
        nargs="+",
        default=["best_model.pth", "best_model_before_50epochs.pth"],
        help="Checkpoint files to read safely (default: both project checkpoints)",
    )
    parser.add_argument(
        "--output-root",
        default="results/experiments",
        help="Parent directory for a new, non-overwriting export directory",
    )
    args = parser.parse_args()

    checkpoint_paths = [Path(path) for path in args.checkpoints]
    missing = [str(path) for path in checkpoint_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Checkpoint file(s) not found: {', '.join(missing)}")

    records = [read_checkpoint_metrics(path) for path in checkpoint_paths]
    output_dir = unique_output_dir(
        Path(args.output_root), "checkpoint_metrics"
    )
    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    payload = {
        "generated_at": generated_at,
        "result_type": "checkpoint-stored metrics; no model evaluation was run",
        "metric_definitions_and_units": {
            "accuracy": "Percent, as stored in the training checkpoint.",
            "balanced_accuracy": "Fraction in [0, 1], as stored.",
            "precision": "Macro precision in percent, as stored.",
            "recall": "Macro recall in percent, as stored.",
            "f1": "Macro F1 in percent, as stored.",
            "roc_auc": "Macro one-vs-rest ROC-AUC fraction, as stored.",
            "mean_loss": "Mean validation loss, as stored; scale depends on training loss.",
            "arwm_region_statistics": (
                "Stored region-weight means and standard deviations; dimensionless."
            ),
        },
        "checkpoints": records,
    }
    json_path = output_dir / "checkpoint_metrics.json"
    csv_path = output_dir / "checkpoint_metrics.csv"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")

    fieldnames = ["checkpoint_filename", "checkpoint_path", "epoch", "metric_source"]
    fieldnames.extend(METRIC_FIELDS)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                key: record[key]
                for key in ("checkpoint_filename", "checkpoint_path", "epoch", "metric_source")
            }
            row.update(record["metrics"])
            writer.writerow(row)

    summary = [
        "# Previously recorded checkpoint metrics",
        "",
        f"Exported: {generated_at}",
        "",
        "These values were read from checkpoint metadata using safe loading. "
        "No model inference or metric recomputation was performed.",
        "",
        "Units: accuracy, precision, recall, and macro F1 are percentages; "
        "balanced accuracy and ROC-AUC are fractions; loss is the stored "
        "validation mean loss. ARWM region statistics are dimensionless.",
        "",
    ]
    for record in records:
        summary.extend([
            f"## {record['checkpoint_filename']}",
            "",
            f"- Epoch: {record['epoch']}",
            f"- Metric source: {record['metric_source']}",
            f"- Stored metrics: `{json.dumps(record['metrics'], sort_keys=True)}`",
            "",
        ])
    (output_dir / "experiment_summary.md").write_text(
        "\n".join(summary), encoding="utf-8"
    )
    print(f"[CheckpointMetrics] Exported {len(records)} record(s) to {output_dir}")
    print(f"[CheckpointMetrics] CSV: {csv_path}")
    print(f"[CheckpointMetrics] JSON: {json_path}")


if __name__ == "__main__":
    main()
