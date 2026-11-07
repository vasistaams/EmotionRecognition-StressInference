"""Rebuild plots and diagnostics from an existing face-inference JSON log."""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import yaml

from scripts.visualize_video_diagnostics import (
    make_contact_sheet,
    make_mask_visualization,
    select_examples,
)
from utils.configuration import get_region_attention_mode, load_config
from utils.video_analysis import reconstruct_temporal_analysis, write_temporal_plots


COPY_FILES = (
    "annotated_video.mp4",
    "checkpoint_load.json",
    "diagnostic_report.json",
    "inference_log.json",
    "prediction_probabilities.csv",
)


def _read_json(path: Path) -> dict:
    with path.open(encoding="utf-8") as file:
        value = json.load(file)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value


def _create_visual_diagnostics(
    video_path: Path,
    inference_log: dict,
    output_dir: Path,
    image_size: int,
    example_limit: int,
) -> None:
    examples = select_examples(inference_log["frames"], example_limit)
    if not examples:
        raise RuntimeError("No face predictions are available for diagnostic images")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Cannot open video for diagnostic images: {video_path}")
    try:
        make_contact_sheet(
            capture,
            examples,
            str(output_dir / "face_contact_sheet.jpg"),
            video_name=video_path.name,
            image_size=image_size,
        )
        weight_ranked = sorted(
            (
                item for item in examples
                if item["face"].get("arwm_region_weights")
            ),
            key=lambda item: item["face"]["arwm_region_weights"]["mouth"],
            reverse=True,
        )
        low_confidence = min(examples, key=lambda item: item["face"]["confidence"])
        multi_face = next(
            (item for item in examples if item["detection_count"] > 1),
            examples[0],
        )
        mask_examples = []
        for item in [low_confidence, *weight_ranked[:1], multi_face]:
            key = (item["frame_index"], item["face"]["track_id"])
            if all(
                (entry["frame_index"], entry["face"]["track_id"]) != key
                for entry in mask_examples
            ):
                mask_examples.append(item)
        make_mask_visualization(
            capture,
            mask_examples,
            str(output_dir / "arwm_token_masks.jpg"),
            image_size=image_size,
            video_name=video_path.name,
        )
    finally:
        capture.release()


def rebuild_results(
    video_path: Path,
    source_dir: Path,
    config_path: Path,
    output_dir: Path,
    example_limit: int = 16,
) -> Path:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not source_dir.is_dir():
        raise FileNotFoundError(f"Source inference directory not found: {source_dir}")
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration not found: {config_path}")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing result directory: {output_dir}"
        )
    if example_limit < 1:
        raise ValueError("--example-limit must be positive")

    source_results_path = source_dir / "results.json"
    source_log_path = source_dir / "inference_log.json"
    source_checkpoint_path = source_dir / "checkpoint_load.json"
    source_diagnostic_path = source_dir / "diagnostic_report.json"
    required = [
        source_results_path,
        source_log_path,
        source_checkpoint_path,
        source_diagnostic_path,
        source_dir / "prediction_probabilities.csv",
        source_dir / "annotated_video.mp4",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Source inference output is incomplete: " + ", ".join(missing)
        )

    results = _read_json(source_results_path)
    inference_log = _read_json(source_log_path)
    checkpoint_report = _read_json(source_checkpoint_path)
    diagnostic_report = _read_json(source_diagnostic_path)
    if inference_log.get("frames") is None or not isinstance(inference_log["frames"], list):
        raise ValueError("Inference log must contain a frames list")
    config = load_config(str(config_path))
    region_mode = get_region_attention_mode(config)
    if checkpoint_report.get("strict_load") is not True:
        raise ValueError("Checkpoint report does not confirm a strict state-dict load")
    if checkpoint_report.get("region_attention_mode") != region_mode:
        raise ValueError(
            "Configuration/checkpoint ARWM mode mismatch: "
            f"{region_mode!r} vs {checkpoint_report.get('region_attention_mode')!r}"
        )
    if checkpoint_report.get("emotion_class_order") != config["dataset"]["class_names"]:
        raise ValueError("Checkpoint report class order does not match configuration")

    alpha = float(config["stress"]["alpha"])
    beta = float(config["stress"]["beta"])
    temporal_analysis, primary_track = reconstruct_temporal_analysis(
        inference_log,
        alpha=alpha,
        beta=beta,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    plot_names = write_temporal_plots(
        primary_track,
        temporal_analysis,
        str(output_dir),
    )

    for filename in COPY_FILES:
        shutil.copy2(source_dir / filename, output_dir / filename)
    samples_dir = source_dir / "annotated_samples"
    if samples_dir.is_dir():
        shutil.copytree(samples_dir, output_dir / "annotated_samples")

    generated_at = datetime.now(timezone.utc).astimezone().isoformat()
    results["reconstruction"] = {
        "generated_at": generated_at,
        "source_inference_directory": str(source_dir.resolve()),
        "source_predictions_reused": True,
        "model_inference_rerun": False,
        "configuration_path": str(config_path.resolve()),
        "checkpoint_report": checkpoint_report,
        "temporal_analysis": temporal_analysis,
        "generated_artifacts": [
            *COPY_FILES,
            "results.json",
            "temporal_analysis.json",
            "emotion_timeline.png",
            "stress_trajectory.png",
            "temporal_analysis.png",
            "face_contact_sheet.jpg",
            "arwm_token_masks.jpg",
            "annotated_samples/",
        ],
    }
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "temporal_analysis.json").write_text(
        json.dumps(temporal_analysis, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    (output_dir / "configuration_snapshot.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    _create_visual_diagnostics(
        video_path,
        inference_log,
        output_dir,
        image_size=int(config["dataset"]["image_size"]),
        example_limit=example_limit,
    )
    diagnostic_report["reconstruction"] = {
        "generated_at": generated_at,
        "source_predictions_reused": True,
        "model_inference_rerun": False,
        "temporal_analysis": temporal_analysis["method"],
        "generated_artifacts": [
            *plot_names,
            "temporal_analysis.json",
            "face_contact_sheet.jpg",
            "arwm_token_masks.jpg",
        ],
        "arwm_visualization_note": (
            "ARWM image shows configured geometric token-mask priors over "
            "saved face crops, not model activation maps."
        ),
    }
    (output_dir / "diagnostic_report.json").write_text(
        json.dumps(diagnostic_report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(
        f"[RebuildVideoResults] Reused "
        f"{temporal_analysis['tracks'][temporal_analysis['primary_plot_track_id']]['prediction_count']} "
        "saved face predictions; model inference was not rerun."
    )
    print(f"[RebuildVideoResults] Output: {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--source-dir", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--example-limit", type=int, default=16)
    args = parser.parse_args()
    rebuild_results(
        video_path=Path(args.video),
        source_dir=Path(args.source_dir),
        config_path=Path(args.config),
        output_dir=Path(args.output_dir),
        example_limit=args.example_limit,
    )


if __name__ == "__main__":
    main()
