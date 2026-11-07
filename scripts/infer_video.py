"""
infer_video.py
==============
Face-aware emotion inference with per-frame probabilities and diagnostics.

Usage:
    python scripts/infer_video.py \\
        --video path/to/video.mp4 \\
        --checkpoint checkpoints/best_model.pth \\
        --config configs/config.yaml \\
        --output results/video_output/

Output files:
    - inference_log.json             : Per-frame detections and probabilities
    - prediction_probabilities.csv   : Long-form class probability table
    - diagnostic_report.json         : Detection and prediction stability summary
    - checkpoint_load.json           : Strict checkpoint-load details
    - annotated_samples/             : Sampled frames with face labels
    - annotated_video.mp4            : Optional annotated video

The prediction scores are not accuracy measurements without video labels.
"""

import os
import sys
import argparse
import json
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from models.full_framework import build_framework
from data import video_dataset
from data.video_dataset import VideoFrameDataset, get_video_info
from data.augmentations import get_val_transforms, IMAGENET_MEAN, IMAGENET_STD
from utils.stress_formulation import compute_stress_trajectory, EMOTION_CLASSES
from utils.temporal_analysis import TemporalSignalTracker, summarize_temporal_records
from utils.face_inference import (
    FaceTracker,
    detect_faces,
    preprocess_face,
    region_weight_log_record,
    select_faces,
)
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config, set_region_attention_mode
from utils.video_analysis import (
    reconstruct_temporal_analysis,
    write_empty_temporal_plots,
    write_temporal_plots,
)


# Emotion label colours for annotation overlay
EMOTION_COLOURS_BGR = {
    'anger':    (0,  50,  200),
    'disgust':  (140, 40, 150),
    'fear':     (200, 50,  50),
    'happiness':(50, 200,  50),
    'sadness':  (200, 100, 50),
    'surprise': (50, 150, 200),
    'neutral':  (100,100, 100),
}

DASHBOARD_HEIGHT = 150
VIDEO_FOOTER_HEIGHT = 42


def draw_video_dashboard(
    frame_bgr: np.ndarray,
    snapshot: dict = None,
    elapsed_seconds: float = 0.0,
    fps: float = 0.0,
) -> np.ndarray:
    """Add a data-backed black dashboard and stress-proxy bar without resizing video."""
    height, width = frame_bgr.shape[:2]
    canvas = np.zeros(
        (height + DASHBOARD_HEIGHT + VIDEO_FOOTER_HEIGHT, width, 3),
        dtype=np.uint8,
    )
    canvas[DASHBOARD_HEIGHT:DASHBOARD_HEIGHT + height] = frame_bgr

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.45, min(0.72, width / 1500.0))
    line = max(20, int(29 * scale / 0.72))
    left = max(12, int(width * 0.018))
    color = (210, 210, 210)
    if snapshot is None:
        title = "Waiting for a face prediction"
        cv2.putText(canvas, title, (left, 32), font, scale * 1.1, color, 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"Video time: {elapsed_seconds:.1f}s | {fps:.2f} FPS",
            (left, 32 + line),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        stress_proxy = None
    else:
        emotion = snapshot["emotion"]
        emotion_color = EMOTION_COLOURS_BGR.get(emotion, color)
        cv2.putText(
            canvas,
            emotion.upper(),
            (left, 34),
            font,
            scale * 1.15,
            emotion_color,
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Confidence: {snapshot['confidence']:.1%}   Track: {snapshot['track_id']}",
            (left, 34 + line),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            f"Transition: {snapshot['last_transition']}",
            (left, 34 + 2 * line),
            font,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"Transitions: {snapshot['transition_count']}   "
                f"Persistence: {snapshot['persistence_seconds']:.1f}s   "
                f"Volatility (last 16 predictions): "
                f"{snapshot['probability_volatility_16']:.3f}"
            ),
            (left, 34 + 3 * line),
            font,
            scale * 0.9,
            color,
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            (
                f"Negative affect: {snapshot['negative_affect']:.3f}   "
                f"Stress proxy: {snapshot['stress_proxy']:.3f} / 1.000   "
                f"Video time: {elapsed_seconds:.1f}s"
            ),
            (left, 34 + 4 * line),
            font,
            scale * 0.9,
            color,
            1,
            cv2.LINE_AA,
        )
        stress_proxy = snapshot["stress_proxy"]

    footer_y = DASHBOARD_HEIGHT + height
    cv2.putText(
        canvas,
        "Calculated stress proxy (not validated stress)",
        (left, footer_y + 27),
        font,
        scale * 0.72,
        color,
        1,
        cv2.LINE_AA,
    )
    bar_left = min(width - 24, left + max(235, int(width * 0.37)))
    bar_right = max(bar_left + 1, width - left)
    bar_top = footer_y + 12
    bar_bottom = footer_y + 29
    cv2.rectangle(canvas, (bar_left, bar_top), (bar_right, bar_bottom), (90, 90, 90), 1)
    if stress_proxy is not None:
        fill_right = bar_left + int((bar_right - bar_left) * np.clip(stress_proxy, 0.0, 1.0))
        cv2.rectangle(
            canvas,
            (bar_left + 1, bar_top + 1),
            (max(bar_left + 1, fill_right - 1), bar_bottom - 1),
            (40, 150, 230),
            cv2.FILLED,
        )
    return canvas


def validate_required_video_artifacts(
    output_dir: Path,
    expected_width: int,
    expected_height: int,
) -> dict:
    """Verify required image/video files decode and results JSON is consistent."""
    required = (
        "annotated_video.mp4",
        "emotion_timeline.png",
        "stress_trajectory.png",
        "temporal_analysis.png",
        "results.json",
    )
    for filename in required:
        path = output_dir / filename
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Required video inference artifact is missing or empty: {path}"
            )

    for filename in required[1:4]:
        path = output_dir / filename
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise RuntimeError(f"Required plot is unreadable: {path}")

    video_path = output_dir / "annotated_video.mp4"
    video = cv2.VideoCapture(str(video_path))
    try:
        if not video.isOpened():
            raise RuntimeError(f"Required annotated video is unreadable: {video_path}")
        ok, first_frame = video.read()
        if not ok or first_frame is None:
            raise RuntimeError(
                f"Required annotated video has no readable frames: {video_path}"
            )
        actual_dimensions = (first_frame.shape[1], first_frame.shape[0])
        required_dimensions = (
            expected_width,
            expected_height + DASHBOARD_HEIGHT + VIDEO_FOOTER_HEIGHT,
        )
        if actual_dimensions != required_dimensions:
            raise RuntimeError(
                f"Annotated video dimensions are {actual_dimensions}; expected "
                f"{required_dimensions}: {video_path}"
            )
    finally:
        video.release()

    results_path = output_dir / "results.json"
    try:
        with results_path.open(encoding="utf-8") as result_file:
            results = json.load(result_file)
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"Required results JSON is unreadable: {results_path}: {error}"
        ) from error
    frames = results.get("frames")
    if not isinstance(frames, list) or len(frames) != results["input"]["source_frame_count"]:
        raise RuntimeError(
            "results.json frame records do not match the source frame count."
        )
    timestamps = [float(frame["timestamp_seconds"]) for frame in frames]
    if any(
        not np.isfinite(timestamp)
        for timestamp in timestamps
    ) or any(right < left for left, right in zip(timestamps, timestamps[1:])):
        raise RuntimeError("results.json frame timestamps are invalid or unordered.")
    prediction_count = sum(len(frame.get("faces", [])) for frame in frames)
    if prediction_count != results["prediction_diagnostics"]["predicted_face_count"]:
        raise RuntimeError(
            "results.json prediction count does not match its frame records."
        )
    for value in results.get("stress_trajectory", {}).get("stress_proxy", []):
        if not np.isfinite(float(value)):
            raise RuntimeError("results.json contains an invalid stress-proxy value.")
    return results


@torch.inference_mode()
def infer_video(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    image_size: int,
    output_dir: str,
    face_mode: str = "largest",
    face_index: int = 0,
    frame_stride: int = 3,
    detection_max_side: int = 1280,
    sample_every: int = 30,
    annotate_video: bool = True,
    write_diagnostics: bool = True,
    show_region_weights: bool = False,
    temporal_analysis: bool = False,
    alpha: float = 0.7,
    beta: float = 0.3,
    config_path: str = None,
    checkpoint_report: dict = None,
    timing_collector: dict = None,
) -> dict:
    """Run face-crop inference and save a per-frame diagnostic package."""
    if frame_stride < 1:
        raise ValueError("frame_stride must be at least 1")
    if sample_every < 1:
        raise ValueError("sample_every must be at least 1")
    if face_mode not in {"largest", "selected", "all"}:
        raise ValueError(f"Unknown face mode: {face_mode}")
    if face_index < 0:
        raise ValueError("face_index must be non-negative")
    if not annotate_video:
        raise ValueError(
            "A complete video run must write annotated_video.mp4; "
            "--no-annotate is not supported."
        )
    if not video_dataset.MTCNN_AVAILABLE:
        raise ImportError(
            "facenet-pytorch is required for video face detection. "
            "Install it with: pip install facenet-pytorch"
        )

    video_file = Path(video_path)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    output_path = Path(output_dir)
    if output_path.exists():
        if not output_path.is_dir():
            raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
        existing_files = {
            item.name for item in output_path.iterdir()
        } - {"checkpoint_load.json"}
        if existing_files:
            raise FileExistsError(
                f"Output directory contains existing results: {output_dir}. "
                "Choose a new output directory to preserve them."
            )
    output_path.mkdir(parents=True, exist_ok=True)
    sample_dir = os.path.join(output_dir, "annotated_samples")
    if write_diagnostics:
        os.makedirs(sample_dir, exist_ok=True)
    transform = get_val_transforms(image_size)
    detector = video_dataset.MTCNN(
        image_size=image_size,
        keep_all=True,
        min_face_size=40,
        device="cpu",
    )
    tracker = FaceTracker(max_gap=max(3, frame_stride * 3))
    model.eval()

    cap = cv2.VideoCapture(str(video_file))
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        cap.release()
        raise ValueError(f"Video contains no readable frames: {video_path}")
    if not np.isfinite(fps) or fps < 0:
        cap.release()
        raise ValueError(f"Video reports an invalid frame rate: {fps}")
    effective_fps = fps if fps > 0 else 25.0
    output_video = None
    output_video = cv2.VideoWriter(
        os.path.join(output_dir, "annotated_video.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps > 0 else 25.0,
        (width, height + DASHBOARD_HEIGHT + VIDEO_FOOTER_HEIGHT),
    )
    if not output_video.isOpened():
        cap.release()
        raise IOError(
            f"Cannot create required annotated video: "
            f"{output_path / 'annotated_video.mp4'}"
        )

    records = []
    csv_rows = []
    labels_by_track = defaultdict(list)
    confidences_by_track = defaultdict(list)
    probabilities_by_track = defaultdict(list)
    region_weights_by_track = defaultdict(list)
    temporal_trackers = {}
    temporal_records_by_track = defaultdict(list)
    recent_probabilities_by_track = defaultdict(list)
    dashboard_snapshot = None
    detected_frames = 0
    no_face_frames = 0
    processed_frames = 0
    frame_index = 0
    sample_saved = False
    if timing_collector is not None:
        timing_collector.clear()
        timing_collector["per_frame"] = []
        timing_collector["cuda_synchronized"] = device.type == "cuda"

    print(
        f"[InferVideo] {Path(video_path).name}: {width}x{height}, "
        f"{fps:.2f} FPS, {total_frames} frames; "
        f"face_mode={face_mode}, stride={frame_stride}"
    )
    try:
        while True:
            frame_start = (
                time.perf_counter() if timing_collector is not None else None
            )
            decode_start = (
                time.perf_counter() if timing_collector is not None else None
            )
            ok, frame_bgr = cap.read()
            if not ok:
                break
            frame_timing = None
            if timing_collector is not None:
                frame_timing = {
                    "frame_index": frame_index,
                    "processed": frame_index % frame_stride == 0,
                    "detected_face_count": None,
                    "selected_face_count": None,
                    "decode_time_ms": (
                        time.perf_counter() - decode_start
                    ) * 1000.0,
                    "face_detection_time_ms": None,
                    "preprocessing_time_ms": None,
                    "model_forward_time_ms": None,
                    "prediction_postprocessing_time_ms": None,
                    "annotation_dashboard_time_ms": None,
                    "video_write_time_ms": None,
                    "sample_write_time_ms": None,
                }
            timestamp = frame_index / effective_fps
            annotation_start = (
                time.perf_counter() if timing_collector is not None else None
            )
            annotated = frame_bgr.copy()

            if frame_index % frame_stride:
                frame_record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "detection_status": "skipped_by_stride",
                    "faces": [],
                }
            else:
                processed_frames += 1
                detection_start = (
                    time.perf_counter() if timing_collector is not None else None
                )
                faces = detect_faces(
                    frame_bgr,
                    detector,
                    max_side=detection_max_side,
                )
                if frame_timing is not None:
                    frame_timing["face_detection_time_ms"] = (
                        time.perf_counter() - detection_start
                    ) * 1000.0
                    frame_timing["detected_face_count"] = len(faces)
                tracker.update(faces, frame_index)
                selected = select_faces(faces, face_mode, face_index)
                if frame_timing is not None:
                    frame_timing["selected_face_count"] = len(selected)
                if faces:
                    detected_frames += 1
                    status = "detected"
                else:
                    no_face_frames += 1
                    status = "no_face"

                predictions = []
                if selected:
                    preprocessing_start = (
                        time.perf_counter()
                        if timing_collector is not None else None
                    )
                    face_tensors = torch.stack([
                        preprocess_face(
                            frame_bgr, face["bbox"], transform, image_size
                        )
                        for face in selected
                    ]).to(device)
                    if timing_collector is not None and device.type == "cuda":
                        torch.cuda.synchronize(device)
                    if frame_timing is not None:
                        frame_timing["preprocessing_time_ms"] = (
                            time.perf_counter() - preprocessing_start
                        ) * 1000.0
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    model_start = time.perf_counter()
                    output = model.forward_frame(face_tensors)
                    if device.type == "cuda":
                        torch.cuda.synchronize(device)
                    model_batch_time_ms = (
                        time.perf_counter() - model_start
                    ) * 1000.0
                    model_time_per_face_ms = (
                        model_batch_time_ms / len(selected)
                    )
                    if frame_timing is not None:
                        frame_timing["model_forward_time_ms"] = (
                            model_batch_time_ms
                        )
                    postprocessing_start = (
                        time.perf_counter()
                        if timing_collector is not None else None
                    )
                    probs_batch = output["probs"].detach().cpu().numpy()
                    region_weights = output.get("region_weights")
                    if region_weights is not None:
                        region_weights = region_weights.detach().cpu().numpy()

                    for face_number, (face, probabilities) in enumerate(
                        zip(selected, probs_batch)
                    ):
                        top_indices = np.argsort(probabilities)[-3:][::-1]
                        emotion_index = int(top_indices[0])
                        emotion = EMOTION_CLASSES[emotion_index]
                        confidence = float(probabilities[emotion_index])
                        track_id = int(face["track_id"])
                        temporal_tracker = temporal_trackers.setdefault(
                            track_id,
                            TemporalSignalTracker(alpha=alpha, beta=beta),
                        )
                        temporal_record = temporal_tracker.update(
                            probabilities, timestamp
                        )
                        temporal_records_by_track[track_id].append(temporal_record)
                        recent_probabilities = recent_probabilities_by_track[track_id]
                        recent_probabilities.append(
                            probabilities.astype(float).tolist()
                        )
                        del recent_probabilities[:-16]
                        recent_changes = [
                            0.5 * np.abs(
                                np.asarray(current) - np.asarray(previous)
                            ).sum()
                            for previous, current in zip(
                                recent_probabilities,
                                recent_probabilities[1:],
                            )
                        ]
                        probability_volatility_16 = (
                            float(np.mean(recent_changes))
                            if recent_changes else 0.0
                        )
                        transition_count = sum(
                            item["transition"] is not None
                            for item in temporal_records_by_track[track_id]
                        )
                        region_log = region_weight_log_record(
                            model.spatial_encoder.region_attention_mode,
                            (
                                region_weights[face_number]
                                if region_weights is not None else None
                            ),
                        )
                        face_record = {
                            "track_id": track_id,
                            "bbox_xyxy": [int(value) for value in face["bbox"]],
                            "frame_index": frame_index,
                            "timestamp_seconds": timestamp,
                            "detector_confidence": float(
                                face["detection_confidence"]
                            ),
                            "predicted_emotion": emotion,
                            "confidence": confidence,
                            "model_inference_time_ms": model_time_per_face_ms,
                            "model_batch_inference_time_ms": model_batch_time_ms,
                            **region_log,
                            "top_three": [
                                {
                                    "emotion": EMOTION_CLASSES[int(index)],
                                    "probability": float(probabilities[index]),
                                }
                                for index in top_indices
                            ],
                            "probabilities": {
                                class_name: float(probabilities[class_index])
                                for class_index, class_name in enumerate(
                                    EMOTION_CLASSES
                                )
                            },
                            "temporal_metrics": {
                                "transition": temporal_record["transition"],
                                "last_transition": temporal_record["last_transition"],
                                "transition_count": transition_count,
                                "persistence_seconds": temporal_record[
                                    "persistence_seconds"
                                ],
                                "probability_volatility_16": (
                                    probability_volatility_16
                                ),
                                "negative_affect": temporal_record[
                                    "negative_affect"
                                ],
                                "stress_proxy": temporal_record["stress_proxy"],
                            },
                        }
                        predictions.append(face_record)
                        labels_by_track[face["track_id"]].append(emotion)
                        confidences_by_track[face["track_id"]].append(confidence)
                        probabilities_by_track[face["track_id"]].append(
                            probabilities.astype(float).tolist()
                        )
                        if region_weights is not None:
                            region_weights_by_track[face["track_id"]].append(
                                region_weights[face_number].astype(float).tolist()
                            )
                        if face_number == 0:
                            dashboard_snapshot = {
                                "track_id": track_id,
                                "emotion": emotion,
                                "confidence": confidence,
                                "last_transition": temporal_record[
                                    "last_transition"
                                ],
                                "transition_count": transition_count,
                                "persistence_seconds": temporal_record[
                                    "persistence_seconds"
                                ],
                                "probability_volatility_16": (
                                    probability_volatility_16
                                ),
                                "negative_affect": temporal_record[
                                    "negative_affect"
                                ],
                                "stress_proxy": temporal_record["stress_proxy"],
                            }
                        csv_rows.append({
                            "frame_index": frame_index,
                            "timestamp_seconds": f"{timestamp:.6f}",
                            "track_id": face["track_id"],
                            "predicted_emotion": emotion,
                            "confidence": f"{confidence:.8f}",
                            "model_inference_time_ms": f"{model_time_per_face_ms:.6f}",
                            "model_batch_inference_time_ms": f"{model_batch_time_ms:.6f}",
                            "region_attention_mode": region_log["region_attention_mode"],
                            "region_attention_status": region_log["region_attention_status"],
                            "eye_weight": (
                                f"{region_log['eye_weight']:.8f}"
                                if region_log["eye_weight"] is not None else ""
                            ),
                            "eyebrow_weight": (
                                f"{region_log['eyebrow_weight']:.8f}"
                                if region_log["eyebrow_weight"] is not None else ""
                            ),
                            "mouth_weight": (
                                f"{region_log['mouth_weight']:.8f}"
                                if region_log["mouth_weight"] is not None else ""
                            ),
                            **{
                                f"probability_{class_name}": f"{probability:.8f}"
                                for class_name, probability in face_record[
                                    "probabilities"
                                ].items()
                            },
                            "x1": face["bbox"][0],
                            "y1": face["bbox"][1],
                            "x2": face["bbox"][2],
                            "y2": face["bbox"][3],
                        })
                        x1, y1, x2, y2 = face["bbox"]
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), (40, 220, 40), 3)
                        region_text = ""
                        if show_region_weights:
                            if region_weights is None:
                                region_text = " | ARWM off"
                            else:
                                region_text = (
                                    f" | E:{region_weights[face_number][0]:.2f}"
                                    f" B:{region_weights[face_number][1]:.2f}"
                                    f" M:{region_weights[face_number][2]:.2f}"
                                )
                        cv2.putText(
                            annotated,
                            f"ID {face['track_id']} {emotion} {confidence:.2f}{region_text}",
                            (x1, max(24, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (40, 220, 40),
                            2,
                        )
                    if frame_timing is not None:
                        frame_timing["prediction_postprocessing_time_ms"] = (
                            time.perf_counter() - postprocessing_start
                        ) * 1000.0
                elif faces and face_mode == "selected":
                    status = "selected_face_not_found"

                frame_record = {
                    "frame_index": frame_index,
                    "timestamp_seconds": timestamp,
                    "detection_status": status,
                    "detections": [
                        {
                            "track_id": int(face["track_id"]),
                            "bbox_xyxy": [int(value) for value in face["bbox"]],
                            "detector_confidence": float(
                                face["detection_confidence"]
                            ),
                        }
                        for face in faces
                    ],
                    "faces": predictions,
                }

            records.append(frame_record)
            if output_video is not None:
                elapsed_seconds = frame_index / effective_fps
                dashboard_frame = draw_video_dashboard(
                    annotated,
                    dashboard_snapshot,
                    elapsed_seconds=elapsed_seconds,
                    fps=effective_fps,
                )
                if frame_timing is not None:
                    frame_timing["annotation_dashboard_time_ms"] = (
                        time.perf_counter() - annotation_start
                    ) * 1000.0
                    video_write_start = time.perf_counter()
                output_video.write(dashboard_frame)
                if frame_timing is not None:
                    frame_timing["video_write_time_ms"] = (
                        time.perf_counter() - video_write_start
                    ) * 1000.0
            if write_diagnostics and (
                frame_index % sample_every == 0
                or (not sample_saved and frame_record["detection_status"] == "detected")
            ):
                sample_path = os.path.join(
                    sample_dir, f"frame_{frame_index:06d}.jpg"
                )
                sample_write_start = (
                    time.perf_counter()
                    if timing_collector is not None else None
                )
                if not cv2.imwrite(sample_path, annotated):
                    raise IOError(f"Could not save annotated sample: {sample_path}")
                if frame_timing is not None:
                    frame_timing["sample_write_time_ms"] = (
                        time.perf_counter() - sample_write_start
                    ) * 1000.0
                sample_saved = True
            if frame_timing is not None:
                frame_timing["frame_total_time_ms"] = (
                    time.perf_counter() - frame_start
                ) * 1000.0
                timing_collector["per_frame"].append(frame_timing)
            frame_index += 1
    finally:
        cap.release()
        if output_video is not None:
            output_video.release()

    track_summaries = {}
    all_labels = []
    all_confidences = []
    all_probability_changes = []
    all_region_weights = []
    for track_id, labels in sorted(labels_by_track.items()):
        all_labels.extend(labels)
        all_confidences.extend(confidences_by_track[track_id])
        counts = Counter(labels)
        transitions = sum(
            previous != current
            for previous, current in zip(labels, labels[1:])
        )
        prediction_count = len(labels)
        transition_rate = transitions / max(1, prediction_count - 1)
        probability_changes = [
            0.5 * np.abs(np.asarray(current) - np.asarray(previous)).sum()
            for previous, current in zip(
                probabilities_by_track[track_id],
                probabilities_by_track[track_id][1:],
            )
        ]
        all_probability_changes.extend(probability_changes)
        all_region_weights.extend(region_weights_by_track[track_id])
        track_summaries[str(track_id)] = {
            "prediction_count": prediction_count,
            "emotion_counts": dict(counts),
            "dominant_emotion": counts.most_common(1)[0][0],
            "dominant_emotion_fraction": counts.most_common(1)[0][1]
            / prediction_count,
            "mean_confidence": float(np.mean(confidences_by_track[track_id])),
            "label_transition_count": transitions,
            "label_transition_rate": float(transition_rate),
            "probability_volatility_mean_half_l1": (
                float(np.mean(probability_changes)) if probability_changes else 0.0
            ),
            "arwm_mean_weights_eye_eyebrow_mouth": (
                np.mean(region_weights_by_track[track_id], axis=0).astype(float).tolist()
                if region_weights_by_track[track_id] else None
            ),
            "nearly_constant_label_at_90_percent": (
                counts.most_common(1)[0][1] / prediction_count >= 0.9
            ),
            "high_label_fluctuation_at_50_percent": transition_rate >= 0.5,
        }

    result = {
        "video": str(video_path),
        "input": {
            "width": width,
            "height": height,
            "fps": fps,
            "source_frame_count": total_frames,
            "duration_seconds": total_frames / effective_fps,
        },
        "inference": {
            "face_mode": face_mode,
            "region_attention_mode": model.spatial_encoder.region_attention_mode,
            "selected_face_index": face_index if face_mode == "selected" else None,
            "frame_stride": frame_stride,
            "detection_max_side": detection_max_side,
            "image_size": image_size,
            "tensor_layout": "NCHW",
            "tensor_shape_per_face": [3, image_size, image_size],
            "preprocessing": (
                "BGR face crop -> grayscale -> replicated RGB -> validation "
                "resize -> ImageNet normalization -> CHW tensor"
            ),
            "normalization": {
                "mean": IMAGENET_MEAN,
                "std": IMAGENET_STD,
            },
            "processed_frames": processed_frames,
            "detected_frames": detected_frames,
            "no_face_frames": no_face_frames,
            "skipped_frames": total_frames - processed_frames,
            "temporal_analysis_enabled": True,
            "effective_fps_used_for_timestamps": effective_fps,
        },
        "prediction_diagnostics": {
            "predicted_face_count": len(all_labels),
            "emotion_counts": dict(Counter(all_labels)),
            "mean_confidence": (
                float(np.mean(all_confidences)) if all_confidences else None
            ),
            "tracks_nearly_constant_at_90_percent": sum(
                summary["nearly_constant_label_at_90_percent"]
                for summary in track_summaries.values()
            ),
            "tracks_high_fluctuation_at_50_percent": sum(
                summary["high_label_fluctuation_at_50_percent"]
                for summary in track_summaries.values()
            ),
            "mean_probability_volatility_half_l1": (
                float(np.mean(all_probability_changes))
                if all_probability_changes else None
            ),
            "mean_arwm_weights_eye_eyebrow_mouth": (
                np.mean(all_region_weights, axis=0).astype(float).tolist()
                if all_region_weights else None
            ),
            "tracks": track_summaries,
            "interpretation": (
                "These statistics describe model outputs only; without video "
                "ground-truth labels they are not accuracy measurements."
            ),
        },
        "frames": records,
    }
    result["configuration"] = {
        "path": str(Path(config_path).resolve()) if config_path else None,
        "image_size": image_size,
        "emotion_class_order": EMOTION_CLASSES,
        "stress_alpha": alpha,
        "stress_beta": beta,
    }
    if checkpoint_report is not None:
        result["checkpoint"] = checkpoint_report

    plot_names = []
    analysis = None
    primary_track = None
    if result["prediction_diagnostics"]["predicted_face_count"]:
        analysis, primary_track = reconstruct_temporal_analysis(
            result,
            alpha=alpha,
            beta=beta,
        )
        plot_names = write_temporal_plots(
            primary_track,
            analysis,
            output_dir,
            include_temporal_plot=True,
        )
        primary_track_data = analysis["tracks"][
            analysis["primary_plot_track_id"]
        ]
        stress = primary_track["stress"]
        result["stress_trajectory"] = {
            "description": (
                "Deterministic proxy derived from negative-affect probabilities "
                "and their temporal change; not a learned or validated stress score."
            ),
            "primary_plot_track_id": analysis["primary_plot_track_id"],
            "timestamps_seconds": primary_track["timestamps"].astype(float).tolist(),
            "stress_proxy": stress["raw_trajectory"].astype(float).tolist(),
            "smoothed_stress_proxy": stress["smoothed"].astype(float).tolist(),
            "negative_affect": stress["r"].astype(float).tolist(),
            "negative_affect_volatility": stress["v"].astype(float).tolist(),
        }
        result["calculated_stress_proxy"] = {
            "description": (
                "Deterministic proxy from emotion probabilities; not a validated "
                "or learned stress prediction."
            ),
            "primary_plot_track_id": analysis["primary_plot_track_id"],
            "tracks": {
                track_id: {
                    "mean": summary["stress_proxy_mean"],
                    "peak": summary["stress_proxy_peak"],
                }
                for track_id, summary in analysis["tracks"].items()
            },
        }
        result["temporal_analysis"] = analysis
    else:
        no_prediction_reason = (
            "No face detections produced emotion probabilities in this video."
        )
        plot_names = write_empty_temporal_plots(
            output_dir,
            no_prediction_reason,
        )
        result["visual_diagnostics"] = {
            "status": "unavailable",
            "reason": no_prediction_reason,
            "plots_contain_data": False,
        }
        result["temporal_analysis"] = {
            "status": "unavailable",
            "reason": no_prediction_reason,
            "learned_temporal_model_used": False,
        }
        result["calculated_stress_proxy"] = {
            "status": "unavailable",
            "reason": no_prediction_reason,
            "learned_stress_prediction": False,
        }

    result["temporal_model"] = {
        "learned_temporal_model_used": False,
        "reason": (
            "This inference path uses per-frame spatial predictions and computes "
            "time-based signals from their probabilities. The available training "
            "loop optimizes forward_frame, so this output does not treat saved "
            "temporal-head parameters as trained stress predictions."
        ),
    }
    result["artifacts"] = {
        "annotated_video": str((output_path / "annotated_video.mp4").resolve()),
        "emotion_timeline": str((output_path / "emotion_timeline.png").resolve()),
        "stress_trajectory": str((output_path / "stress_trajectory.png").resolve()),
        "temporal_analysis": str((output_path / "temporal_analysis.png").resolve()),
        "results_json": str((output_path / "results.json").resolve()),
        "prediction_count": result["prediction_diagnostics"]["predicted_face_count"],
        "stress_sample_count": len(
            result.get("stress_trajectory", {}).get("timestamps_seconds", [])
        ),
    }
    if temporal_analysis:
        result["temporal_analysis_requested"] = True

    # results.json and the five dashboard files are mandatory even when
    # ancillary diagnostic logging is disabled.
    results_json_path = output_path / "results.json"
    try:
        with results_json_path.open("w", encoding="utf-8") as result_file:
            json.dump(result, result_file, indent=2, allow_nan=False)
    except Exception as error:
        raise RuntimeError(
            f"Failed to generate required artifact results.json: {error}"
        ) from error

    if write_diagnostics:
        with open(os.path.join(output_dir, "inference_log.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, allow_nan=False)
        with open(
            os.path.join(output_dir, "prediction_probabilities.csv"),
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            fieldnames = [
                "frame_index", "timestamp_seconds", "track_id",
                "predicted_emotion", "confidence", "model_inference_time_ms",
                "model_batch_inference_time_ms", "region_attention_mode",
                "region_attention_status", "eye_weight", "eyebrow_weight",
                "mouth_weight",
                *[f"probability_{name}" for name in EMOTION_CLASSES],
                "x1", "y1", "x2", "y2",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        with open(
            os.path.join(output_dir, "diagnostic_report.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump({
                "video": str(video_path),
                "face_detection": {
                    "source_dimensions": [width, height],
                    "detection_max_side": detection_max_side,
                    "processed_frames": processed_frames,
                    "detected_frames": detected_frames,
                    "no_face_frames": no_face_frames,
                    "frame_stride": frame_stride,
                },
                "preprocessing": result["inference"]["preprocessing"],
                "checkpoint": "See checkpoint_load.json for model-load diagnostics.",
                "prediction_diagnostics": result["prediction_diagnostics"],
                "artifacts": [
                    "inference_log.json",
                    "prediction_probabilities.csv",
                    "annotated_samples/",
                    "annotated_video.mp4",
                    *plot_names,
                    "temporal_analysis.json" if plot_names else None,
                ],
                "temporal_analysis": result.get("temporal_analysis", {}).get(
                    "method"
                ),
                "classification_accuracy": (
                    "Not measured: this video has no supplied ground-truth labels."
                ),
                "domain_note": (
                    "If preprocessing and checkpoint loading are correct, FER2013 "
                    "to movie-footage domain shift may still limit classification."
                ),
            }, f, indent=2, allow_nan=False)
        if "temporal_analysis" in result and analysis is not None:
            with open(
                os.path.join(output_dir, "temporal_analysis.json"),
                "w",
                encoding="utf-8",
            ) as temporal_file:
                json.dump(
                    result["temporal_analysis"],
                    temporal_file,
                    indent=2,
                    allow_nan=False,
                )

    print(
        f"[InferVideo] Processed {processed_frames}/{frame_index} frames; "
        f"faces predicted={result['prediction_diagnostics']['predicted_face_count']}; "
        f"no-face frames={no_face_frames}"
    )
    for track_id, summary in track_summaries.items():
        print(
            f"[InferVideo] Track {track_id}: dominant={summary['dominant_emotion']}, "
            f"confidence={summary['mean_confidence']:.3f}, "
            f"label transitions={summary['label_transition_count']}"
        )
    required_artifacts = validate_required_video_artifacts(
        output_path,
        width,
        height,
    )
    print("[InferVideo] Required artifacts:")
    for filename in (
        "annotated_video.mp4",
        "emotion_timeline.png",
        "stress_trajectory.png",
        "temporal_analysis.png",
        "results.json",
    ):
        print(f"  {output_path / filename}")
    return required_artifacts


@torch.no_grad()
def _infer_video_legacy(
    model: torch.nn.Module,
    video_path: str,
    device: torch.device,
    image_size: int,
    sequence_length: int = 16,
    alpha: float = 0.7,
    beta: float  = 0.3,
    output_dir: str = "results/video_output",
    annotate_video: bool = True,
    temporal_analysis: bool = False,
) -> dict:
    """
    Runs the full Swin-FANE pipeline on a video file.

    Args:
        model:           Trained Swin-FANE framework.
        video_path:      Path to input video.
        device:          Computation device.
        image_size:      Configured model input resolution.
        sequence_length: Temporal window size T.
        alpha, beta:     Stress formulation weights.
        output_dir:      Output directory for results.
        annotate_video:  Whether to produce an annotated output video.

    Returns:
        Dict with per-frame emotions and stress trajectory.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.eval()

    # ---- Video info ----
    info = get_video_info(video_path)
    print(f"[InferVideo] Processing: {Path(video_path).name}")
    print(f"             Duration: {info['duration_sec']:.1f}s | "
          f"FPS: {info['fps']:.1f} | "
          f"Frames: {info['total_frames']}")

    # ---- Load frames ----
    print("[InferVideo] Loading and preprocessing frames...")
    dataset = VideoFrameDataset(
        video_path=video_path,
        image_size=image_size,
        sequence_length=sequence_length,
        use_mtcnn=True,
        skip_no_face=False,
        include_partial_window=temporal_analysis,
    )

    if len(dataset) == 0:
        print("[InferVideo] No valid sequences found. Exiting.")
        return {}

    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=False)

    # ---- Run inference ----
    print("[InferVideo] Running emotion recognition + stress estimation...")
    all_frame_probs  = []   # (T_total, K)
    all_frame_labels = []   # (T_total,)
    all_timestamps = []
    all_stress_seqs  = []   # stress score per sequence

    for batch in loader:
        x_seq = batch['image'].to(device)   # (B, T, C, H, W)
        if x_seq.dim() == 4:
            x_seq = x_seq.unsqueeze(1)

        output = model.forward_sequence(x_seq)

        B, T = output['frame_probs'].shape[:2]
        for b in range(B):
            valid_length = int(batch['valid_length'][b].item())
            probs_b  = output['frame_probs'][b, :valid_length].cpu().numpy()
            labels_b = output['frame_labels'][b, :valid_length].cpu().numpy()
            all_frame_probs.append(probs_b)
            all_frame_labels.append(labels_b)
            all_timestamps.extend(
                batch['timestamps'][b, :valid_length].cpu().numpy().tolist()
            )

        stress_b = output['stress_score'].cpu().numpy()
        all_stress_seqs.extend(stress_b[:, 0].tolist())

    # Flatten frame arrays
    all_frame_probs  = np.concatenate(all_frame_probs,  axis=0)  # (N_frames, K)
    all_frame_labels = np.concatenate(all_frame_labels, axis=0)  # (N_frames,)

    # ---- Compute full stress trajectory ----
    stress_info = compute_stress_trajectory(
        probs_sequence=all_frame_probs,
        alpha=alpha,
        beta=beta,
        smooth_window=7,
    )

    print(f"[InferVideo] Summary:")
    print(f"  Frames processed: {len(all_frame_labels)}")
    print(f"  Mean stress index: {stress_info['mean_stress']:.4f}")
    print(f"  Peak stress:       {stress_info['peak_stress']:.4f}")

    # Most frequent emotion
    mode_emotion = EMOTION_CLASSES[int(np.bincount(all_frame_labels).argmax())]
    print(f"  Dominant emotion:  {mode_emotion}")

    temporal_records = []
    transition_events = []
    persistence_runs = []
    if temporal_analysis:
        tracker = TemporalSignalTracker(alpha=alpha, beta=beta)
        temporal_records = [
            tracker.update(probabilities, timestamp)
            for probabilities, timestamp in zip(all_frame_probs, all_timestamps)
        ]
        transition_events, persistence_runs = summarize_temporal_records(
            temporal_records
        )
        timestamps = np.asarray(all_timestamps, dtype=np.float64)
        probability_volatility = np.asarray(
            [record['probability_volatility'] for record in temporal_records],
            dtype=np.float64,
        )
        stress_proxy = np.asarray(
            [record['stress_proxy'] for record in temporal_records],
            dtype=np.float64,
        )

    # ---- Save annotated video ----
    if annotate_video:
        _write_annotated_video(
            video_path=video_path,
            frame_labels=all_frame_labels,
            frame_probs=all_frame_probs,
            stress_traj=(stress_proxy if temporal_analysis else stress_info['smoothed']),
            output_path=os.path.join(output_dir, "annotated_video.mp4"),
            temporal_records=temporal_records if temporal_analysis else None,
        )

    # ---- Plot stress trajectory ----
    _plot_stress_trajectory(
        stress_info=stress_info,
        save_path=os.path.join(output_dir, "stress_trajectory.png"),
        timestamps=timestamps if temporal_analysis else None,
    )

    # ---- Plot emotion timeline heatmap ----
    _plot_emotion_timeline(
        probs=all_frame_probs,
        save_path=os.path.join(output_dir, "emotion_timeline.png"),
        timestamps=timestamps if temporal_analysis else None,
    )

    if temporal_analysis:
        _plot_temporal_analysis(
            probs=all_frame_probs,
            timestamps=timestamps,
            volatility=probability_volatility,
            stress_proxy=stress_proxy,
            transition_events=transition_events,
            persistence_runs=persistence_runs,
            save_path=os.path.join(output_dir, "temporal_analysis.png"),
        )

    # ---- Save JSON results ----
    results = {
        'video':             str(video_path),
        'total_frames':      int(len(all_frame_labels)),
        'mean_stress':       float(stress_info['mean_stress']),
        'peak_stress':       float(stress_info['peak_stress']),
        'dominant_emotion':  mode_emotion,
        'stress_trajectory': stress_info['smoothed'].tolist(),
        'negative_affect':   stress_info['r'].tolist(),
        'volatility':        stress_info['v'].tolist(),
        'frame_emotions':    [EMOTION_CLASSES[int(l)] for l in all_frame_labels],
    }
    if temporal_analysis:
        results['temporal_analysis'] = {
            'method': {
                'probability_volatility': '0.5 * L1(P_t - P_(t-1)); first frame is 0',
                'negative_affect_volatility': 'unchanged existing abs(r_t - r_(t-1))',
                'stress_proxy': 'unchanged existing alpha * r_t + beta * negative_affect_volatility',
                'learned_stress_prediction': False,
            },
            'time_steps': temporal_records,
            'transition_events': transition_events,
            'persistence_runs': persistence_runs,
        }

    with open(os.path.join(output_dir, "results.json"), 'w') as f:
        json.dump(results, f, indent=2)

    print(f"[InferVideo] Results saved to: {output_dir}")
    return results


def _write_annotated_video(
    video_path: str,
    frame_labels: np.ndarray,
    frame_probs: np.ndarray,
    stress_traj: np.ndarray,
    output_path: str,
    temporal_records: list = None,
):
    """Writes an annotated video with emotion labels and stress bar."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame_idx >= len(frame_labels):
            break

        emotion_name = EMOTION_CLASSES[int(frame_labels[frame_idx])]
        colour       = EMOTION_COLOURS_BGR.get(emotion_name, (200, 200, 200))
        stress_val   = float(stress_traj[min(frame_idx, len(stress_traj) - 1)])
        confidence   = float(frame_probs[frame_idx].max())

        # Emotion label box
        cv2.rectangle(frame, (10, 10), (280, 70), colour, -1)
        cv2.putText(frame, emotion_name.upper(), (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

        # Confidence score
        cv2.putText(frame, f"Conf: {confidence:.2f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 1)

        if temporal_records is not None:
            record = temporal_records[frame_idx]
            lines = [
                f"Transition: {record['last_transition']}",
                f"Persistence: {record['persistence_seconds']:.1f}s",
                f"Volatility: {record['probability_volatility']:.3f}",
                f"Stress proxy: {record['stress_proxy']:.3f}",
            ]
            for line_idx, text in enumerate(lines):
                cv2.putText(
                    frame, text, (10, 118 + line_idx * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (245, 245, 245), 1,
                )

        # Stress bar (bottom of frame)
        bar_w = int(w * stress_val)
        stress_colour = (0, int(255 * (1 - stress_val)), int(255 * stress_val))
        cv2.rectangle(frame, (0, h - 20), (bar_w, h), stress_colour, -1)
        cv2.putText(frame, f"Stress: {stress_val:.3f}", (10, h - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"[InferVideo] Annotated video saved: {output_path}")


def _plot_stress_trajectory(
    stress_info: dict, save_path: str, timestamps: np.ndarray = None
):
    """Plots the temporal stress trajectory with annotations."""
    T     = len(stress_info['smoothed'])
    time  = timestamps if timestamps is not None else np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)

    # Plot 1: Stress trajectory
    ax = axes[0]
    ax.fill_between(time, stress_info['smoothed'], alpha=0.3, color='#e74c3c')
    ax.plot(time, stress_info['smoothed'], color='#c0392b', linewidth=2)
    ax.axhline(y=stress_info['mean_stress'], color='gray', linestyle='--',
               label=f"Mean = {stress_info['mean_stress']:.3f}")
    ax.set_ylabel('Stress Score S(t)', fontsize=10)
    ax.set_title('Temporal Stress Trajectory', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Plot 2: Negative affect
    ax2 = axes[1]
    ax2.fill_between(time, stress_info['r'], alpha=0.3, color='#3498db')
    ax2.plot(time, stress_info['r'], color='#2980b9', linewidth=1.5)
    ax2.set_ylabel('Negative Affect r(t)', fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)

    # Plot 3: Volatility
    ax3 = axes[2]
    ax3.fill_between(time, stress_info['v'], alpha=0.3, color='#e67e22')
    ax3.plot(time, stress_info['v'], color='#d35400', linewidth=1.5)
    ax3.set_ylabel('Volatility v(t)', fontsize=10)
    ax3.set_xlabel('Time (seconds)' if timestamps is not None else 'Frame Index', fontsize=10)
    ax3.set_ylim(0, 1)
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_emotion_timeline(
    probs: np.ndarray, save_path: str, timestamps: np.ndarray = None
):
    """Plots per-frame emotion probability heatmap."""
    fig, ax = plt.subplots(figsize=(14, 4))
    if timestamps is None:
        im = ax.imshow(probs.T, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=1, origin='lower')
        ax.set_xlabel('Frame Index', fontsize=10)
    else:
        extent = [float(timestamps[0]), float(timestamps[-1]), -0.5, len(EMOTION_CLASSES) - 0.5]
        im = ax.imshow(probs.T, aspect='auto', cmap='YlOrRd',
                       vmin=0, vmax=1, origin='lower', extent=extent)
        ax.set_xlabel('Time (seconds)', fontsize=10)

    ax.set_yticks(range(len(EMOTION_CLASSES)))
    ax.set_yticklabels([c.capitalize() for c in EMOTION_CLASSES])
    ax.set_title('Per-Frame Emotion Probability Timeline', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Probability')

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def _plot_temporal_analysis(
    probs: np.ndarray,
    timestamps: np.ndarray,
    volatility: np.ndarray,
    stress_proxy: np.ndarray,
    transition_events: list,
    persistence_runs: list,
    save_path: str,
):
    """Plot probabilities, temporal signals, transitions, and persistence."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)

    ax = axes[0]
    image = ax.imshow(
        probs.T,
        aspect='auto',
        cmap='YlOrRd',
        vmin=0,
        vmax=1,
        origin='lower',
        extent=[float(timestamps[0]), float(timestamps[-1]), -0.5, len(EMOTION_CLASSES) - 0.5],
    )
    ax.set_yticks(range(len(EMOTION_CLASSES)))
    ax.set_yticklabels([name.capitalize() for name in EMOTION_CLASSES])
    ax.set_ylabel('Emotion probability')
    ax.set_title('Emotion Probabilities Over Video Time')
    fig.colorbar(image, ax=ax, label='Probability')

    ax = axes[1]
    ax.plot(timestamps, volatility, color='#2878a0', label='Full-probability volatility')
    ax.plot(timestamps, stress_proxy, color='#c44e52', label='Deterministic stress proxy')
    for event in transition_events:
        event_time = event['timestamp_seconds']
        ax.axvline(event_time, color='#555555', alpha=0.28, linewidth=0.8)
        ax.annotate(
            event['transition'],
            (event_time, 0.98),
            xycoords=('data', 'axes fraction'),
            xytext=(3, -3),
            textcoords='offset points',
            rotation=90,
            va='top',
            fontsize=7,
        )
    ax.set_ylabel('Signal value')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylim(0, 1)
    ax.set_title('Volatility, Stress Proxy, and Transition Events')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.25)

    ax = axes[2]
    for run in persistence_runs:
        color = EMOTION_COLOURS_BGR.get(run['emotion'], (100, 100, 100))[::-1]
        ax.barh(
            run['emotion'].capitalize(),
            run['duration_seconds'],
            left=run['start_seconds'],
            height=0.65,
            color=tuple(channel / 255.0 for channel in color),
        )
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Dominant emotion')
    ax.set_title('Emotion Persistence Runs')
    ax.grid(True, axis='x', alpha=0.25)

    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Infer Swin-FANE on video')
    parser.add_argument('--video',       type=str, required=True)
    parser.add_argument('--checkpoint',  type=str, required=True)
    parser.add_argument('--config',      type=str, required=True)
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results/<video-stem>)',
    )
    parser.add_argument('--no-annotate', action='store_true',
                        help='Legacy option; incompatible with required annotated output')
    parser.add_argument('--arwm', action='store_true',
                        help='Legacy alias for --arwm-mode adaptive')
    parser.add_argument(
        '--arwm-mode', choices=('off', 'fixed', 'adaptive', 'context_aware'), default=None,
        help='Expected trained model mode; must match the checkpoint',
    )
    parser.add_argument(
        '--face-mode',
        choices=('largest', 'selected', 'all'),
        default='largest',
        help='Predict the largest face, an area-ranked face, or every face',
    )
    parser.add_argument(
        '--face-index', type=int, default=0,
        help='Zero-based area rank used with --face-mode selected',
    )
    parser.add_argument(
        '--frame-stride', type=int, default=3,
        help='Run detection and inference every N source frames',
    )
    parser.add_argument(
        '--detection-max-side', type=int, default=1280,
        help='Maximum detector input dimension; boxes are mapped back to source size',
    )
    parser.add_argument(
        '--sample-every', type=int, default=30,
        help='Save an annotated sample every N source frames',
    )
    parser.add_argument(
        '--show-region-weights', action='store_true',
        help='Overlay actual region weights (or explicit ARWM off status)',
    )
    parser.add_argument(
        '--no-log', action='store_true',
        help='Skip ancillary log/CSV/sample files (required results.json is still written)',
    )
    parser.add_argument(
        '--temporal-analysis',
        action='store_true',
        help='Legacy opt-in; required temporal plots and analysis are now always included',
    )
    args = parser.parse_args()
    if args.no_annotate:
        parser.error(
            "--no-annotate cannot be used because annotated_video.mp4 is required"
        )
    for name, path in (
        ("video", args.video),
        ("checkpoint", args.checkpoint),
        ("configuration", args.config),
    ):
        if not Path(path).is_file():
            raise FileNotFoundError(f"{name.capitalize()} file not found: {path}")
    output_path = Path(args.output) if args.output else (
        Path("results") / Path(args.video).stem
    )
    if output_path.exists() and (
        not output_path.is_dir() or any(output_path.iterdir())
    ):
        raise FileExistsError(
            f"Output directory already contains files: {output_path}. "
            "Choose a fresh output directory to preserve prior results."
        )

    config = load_config(args.config)
    if args.arwm_mode:
        set_region_attention_mode(config, args.arwm_mode)
    elif args.arwm:
        set_region_attention_mode(config, 'adaptive')
    region_mode = get_region_attention_mode(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # The strict checkpoint load below restores every parameter, so avoid an
    # unnecessary ImageNet download while constructing the model.
    config.setdefault('model', {}).setdefault('swin', {})['pretrained'] = False
    model = build_framework(config).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=True
    )
    state_dict = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint must contain a model state dictionary: {args.checkpoint}"
        )
    validate_checkpoint_region_mode(checkpoint, region_mode, args.checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    model.eval()
    if model.training:
        raise RuntimeError("Model did not enter evaluation mode")
    checkpoint_report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get('epoch'),
        "checkpoint_metrics": checkpoint.get('metrics'),
        "loaded_state_entries": len(state_dict),
        "model_state_entries": len(model.state_dict()),
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "strict_load": True,
        "region_attention_mode": region_mode,
        "evaluation_mode": not model.training,
        "model_class": type(model).__name__,
        "swin_variant": config['model']['swin']['variant'],
        "arwm_enabled": (
            model.spatial_encoder.region_attention.enabled
            if model.spatial_encoder.region_attention is not None else False
        ),
        "arwm_fusion_weight": (
            model.spatial_encoder.region_attention.fusion_weight
            if model.spatial_encoder.region_attention is not None else None
        ),
        "configured_image_size": config['dataset']['image_size'],
        "emotion_class_order": EMOTION_CLASSES,
        "device": str(device),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    if not args.no_log:
        with open(
            output_path / "checkpoint_load.json",
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(checkpoint_report, f, indent=2, default=str)
    print(
        f"[InferVideo] Strict checkpoint load: {len(state_dict)} entries; "
        f"missing={incompatible.missing_keys}; "
        f"unexpected={incompatible.unexpected_keys}; "
        f"eval={not model.training}; device={device}"
    )

    infer_video(
        model=model,
        video_path=args.video,
        device=device,
        image_size=config['dataset']['image_size'],
        output_dir=str(output_path),
        face_mode=args.face_mode,
        face_index=args.face_index,
        frame_stride=args.frame_stride,
        detection_max_side=args.detection_max_side,
        sample_every=args.sample_every,
        annotate_video=True,
        write_diagnostics=not args.no_log,
        show_region_weights=args.show_region_weights,
        temporal_analysis=args.temporal_analysis,
        alpha=float(config.get('stress', {}).get('alpha', 0.7)),
        beta=float(config.get('stress', {}).get('beta', 0.3)),
        config_path=args.config,
        checkpoint_report=checkpoint_report,
    )
