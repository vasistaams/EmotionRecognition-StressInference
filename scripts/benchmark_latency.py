"""Reproducible model-only and end-to-end video inference benchmark."""

import argparse
import csv
import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import subprocess
import tempfile
import time
import urllib.request
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import cv2
import numpy as np
import torch

from data import video_dataset
from data.augmentations import get_val_transforms
from data.fer2013_dataset import FER2013FolderDataset
from models.full_framework import build_framework
from scripts.infer_video import infer_video
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config
from utils.face_inference import preprocess_face
from utils.stress_formulation import EMOTION_CLASSES
from utils.temporal_analysis import TemporalSignalTracker


DEFAULT_CHECKPOINT = (
    REPOSITORY_ROOT
    / "checkpoints"
    / "arwm_comparison_2026"
    / "context_aware"
    / "best_model.pth"
)
DEFAULT_CONFIG = (
    REPOSITORY_ROOT
    / "results"
    / "arwm_comparison_2026"
    / "context_aware"
    / "resolved_config.yaml"
)
DEFAULT_VIDEO = REPOSITORY_ROOT / "data" / "videos" / "testvideo1.mp4"
DEFAULT_OUTPUT_ROOT = (
    REPOSITORY_ROOT / "results" / "inference_benchmark" / "context_aware"
)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def resolve_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPOSITORY_ROOT / resolved
    return resolved.resolve()


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / timestamp
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def summarize(values: list) -> dict:
    measurements = np.asarray(values, dtype=np.float64)
    if measurements.size == 0:
        return {
            "mean": None,
            "median": None,
            "p95": None,
            "minimum": None,
            "maximum": None,
            "standard_deviation": None,
        }
    return {
        "mean": float(np.mean(measurements)),
        "median": float(np.median(measurements)),
        "p95": float(np.percentile(measurements, 95)),
        "minimum": float(np.min(measurements)),
        "maximum": float(np.max(measurements)),
        "standard_deviation": float(np.std(measurements, ddof=0)),
    }


def load_model(
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
) -> tuple:
    config = load_config(str(config_path))
    region_mode = get_region_attention_mode(config)
    if config["dataset"]["num_classes"] != len(EMOTION_CLASSES):
        raise ValueError(
            "Configuration class count does not match the model labels"
        )
    if config["dataset"].get("class_names") != EMOTION_CLASSES:
        raise ValueError(
            "Configuration class order does not match "
            f"{EMOTION_CLASSES!r}"
        )

    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    model = build_framework(config).to(device)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    if not isinstance(checkpoint, dict) or not isinstance(
        checkpoint.get("model"), dict
    ):
        raise ValueError(f"Checkpoint has no model state dictionary: {checkpoint_path}")
    checkpoint_mode = validate_checkpoint_region_mode(
        checkpoint,
        region_mode,
        str(checkpoint_path),
    )
    incompatible = model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    if model.training:
        raise RuntimeError("Model did not enter evaluation mode")
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError("Strict checkpoint loading reported incompatible keys")

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_region_attention_mode": checkpoint_mode,
        "config_region_attention_mode": region_mode,
        "strict_load": True,
        "missing_keys": incompatible.missing_keys,
        "unexpected_keys": incompatible.unexpected_keys,
        "evaluation_mode": not model.training,
        "model_class": type(model).__name__,
        "model_state_entries": len(model.state_dict()),
        "loaded_state_entries": len(checkpoint["model"]),
        "total_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "spatial_encoder_parameter_count": sum(
            parameter.numel()
            for parameter in model.spatial_encoder.parameters()
        ),
        "region_attention_mode": model.spatial_encoder.region_attention_mode,
    }
    del checkpoint
    return model, config, report


PAPER_VARIANTS = {
    "off": {
        "checkpoint": REPOSITORY_ROOT / "checkpoints" / "arwm_variants" / "off_20" / "best_model.pth",
        "config": REPOSITORY_ROOT / "configs" / "arwm" / "off_20.yaml",
    },
    "fixed": {
        "checkpoint": REPOSITORY_ROOT / "checkpoints" / "arwm_variants" / "fixed_20" / "best_model.pth",
        "config": REPOSITORY_ROOT / "configs" / "arwm" / "fixed_20.yaml",
    },
    "adaptive": {
        "checkpoint": REPOSITORY_ROOT / "checkpoints" / "best_model.pth",
        "config": REPOSITORY_ROOT / "configs" / "arwm" / "adaptive.yaml",
    },
}


def run_model_latency_batches(
    model: torch.nn.Module,
    device: torch.device,
    face_tensor: torch.Tensor,
    warmup_iterations: int,
    measured_iterations: int,
    batch_sizes: tuple = (1, 8),
) -> dict:
    results = {}
    model.eval()
    with torch.inference_mode():
        for batch_size in batch_sizes:
            if face_tensor.shape[0] < batch_size:
                repeats = (batch_size + face_tensor.shape[0] - 1) // face_tensor.shape[0]
                batch = face_tensor.repeat(repeats, 1, 1, 1)[:batch_size]
            else:
                batch = face_tensor[:batch_size]
            batch = batch.to(device)
            for _ in range(warmup_iterations):
                output = model.forward_frame(batch)
            synchronize(device)
            if output["probs"].shape != (batch_size, len(EMOTION_CLASSES)):
                raise RuntimeError(
                    f"Expected {(batch_size, len(EMOTION_CLASSES))} probabilities, "
                    f"got {tuple(output['probs'].shape)}"
                )

            timings_ms = []
            for _ in range(measured_iterations):
                synchronize(device)
                start = time.perf_counter()
                model.forward_frame(batch)
                synchronize(device)
                timings_ms.append((time.perf_counter() - start) * 1000.0)

            results[str(batch_size)] = {
                "batch_size": batch_size,
                "warmup_iterations": warmup_iterations,
                "timed_iterations": measured_iterations,
                "batch_latency_ms": summarize(timings_ms),
                "mean_ms_per_image": float(np.mean(timings_ms) / batch_size),
                "throughput_images_per_second": float(
                    1000.0 * batch_size / np.mean(timings_ms)
                ),
                "individual_batch_times_ms": timings_ms,
            }
    return results


def load_latency_inputs(config: dict, batch_size: int = 8) -> tuple:
    image_size = int(config["dataset"]["image_size"])
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    dataset = FER2013FolderDataset(
        root_dir=str(dataset_root),
        split="test",
        image_size=image_size,
        transform=get_val_transforms(image_size),
    )
    if len(dataset) < batch_size:
        raise RuntimeError(
            f"Need at least {batch_size} test images; found {len(dataset)}"
        )
    items = [dataset[index] for index in range(batch_size)]
    return torch.stack([item["image"] for item in items]), [
        item["path"] for item in items
    ]


FACE_DETECTOR_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_detector/"
    "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
)


def create_media_pipe_detector():
    import mediapipe as mp

    cache_dir = Path(tempfile.gettempdir()) / "mediapipe_models"
    model_path = cache_dir / "blaze_face_short_range.tflite"
    if not model_path.is_file():
        cache_dir.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(FACE_DETECTOR_MODEL_URL, timeout=30) as response:
            model_path.write_bytes(response.read())
    if model_path.stat().st_size == 0:
        raise RuntimeError(f"MediaPipe detector model is empty: {model_path}")

    options = mp.tasks.vision.FaceDetectorOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        min_detection_confidence=0.5,
    )
    detector = mp.tasks.vision.FaceDetector.create_from_options(options)
    model_hash = hashlib.sha256(model_path.read_bytes()).hexdigest()
    return mp, detector, model_hash


def media_pipe_bbox(
    frame_bgr: np.ndarray,
    detector,
    mediapipe_module,
    timestamp_ms: int,
) -> tuple:
    frame_height, frame_width = frame_bgr.shape[:2]
    rgb_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = mediapipe_module.Image(
        image_format=mediapipe_module.ImageFormat.SRGB,
        data=rgb_frame,
    )
    detection_result = detector.detect_for_video(image, timestamp_ms)
    detections = detection_result.detections or []
    candidates = []
    for detection in detections:
        relative = detection.bounding_box
        x1 = max(0, relative.origin_x)
        y1 = max(0, relative.origin_y)
        x2 = min(frame_width, relative.origin_x + relative.width)
        y2 = min(frame_height, relative.origin_y + relative.height)
        if x2 > x1 and y2 > y1:
            candidates.append(((x1, y1, x2, y2), (x2 - x1) * (y2 - y1)))
    return max(candidates, key=lambda candidate: candidate[1])[0] if candidates else None


def run_mediapipe_video_benchmark(
    model: torch.nn.Module,
    image_size: int,
    video_path: Path,
    device: torch.device,
    frame_stride: int,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Cannot open benchmark video: {video_path}")
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    reported_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if not np.isfinite(source_fps) or source_fps <= 0 or reported_frames <= 0:
        capture.release()
        raise ValueError(
            f"Invalid video metadata: fps={source_fps}, frames={reported_frames}"
        )

    transform = get_val_transforms(image_size)
    mp, detector, model_hash = create_media_pipe_detector()
    tracker = TemporalSignalTracker()
    stage_times = {
        "decode": [],
        "detect": [],
        "preprocess": [],
        "forward": [],
        "temporal": [],
    }
    source_frame_count = 0
    sampled_frame_count = 0
    face_count = 0
    model.eval()
    start_total = time.perf_counter()
    try:
        with torch.inference_mode():
            while True:
                stage_start = time.perf_counter()
                ok, frame_bgr = capture.read()
                if not ok:
                    break
                stage_times["decode"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                frame_index = source_frame_count
                source_frame_count += 1
                if frame_index % frame_stride:
                    continue
                sampled_frame_count += 1

                stage_start = time.perf_counter()
                bbox = media_pipe_bbox(
                    frame_bgr,
                    detector,
                    mp,
                    int(round(frame_index * 1000.0 / source_fps)),
                )
                stage_times["detect"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )
                if bbox is None:
                    continue
                face_count += 1

                stage_start = time.perf_counter()
                face_tensor = preprocess_face(
                    frame_bgr, bbox, transform, image_size=image_size
                ).unsqueeze(0).to(device)
                stage_times["preprocess"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                synchronize(device)
                stage_start = time.perf_counter()
                output = model.forward_frame(face_tensor)
                synchronize(device)
                stage_times["forward"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )

                stage_start = time.perf_counter()
                probabilities = (
                    output["probs"][0].detach().float().cpu().numpy()
                )
                tracker.update(
                    probabilities,
                    timestamp_seconds=frame_index / source_fps,
                )
                stage_times["temporal"].append(
                    (time.perf_counter() - stage_start) * 1000.0
                )
    finally:
        capture.release()
        detector.close()
    synchronize(device)
    elapsed_seconds = time.perf_counter() - start_total
    if source_frame_count != reported_frames:
        raise RuntimeError(
            f"Decoded {source_frame_count} frames but container metadata reports "
            f"{reported_frames}"
        )

    return {
        "detector": "MediaPipe Tasks FaceDetector, VIDEO mode",
        "detector_model": "BlazeFace short-range float16",
        "detector_model_asset_sha256": model_hash,
        "minimum_detection_confidence": 0.5,
        "face_selection": "largest valid detected bounding box",
        "frame_stride": frame_stride,
        "source_fps": source_fps,
        "source_frame_count": source_frame_count,
        "sampled_frame_count": sampled_frame_count,
        "frames_with_face": face_count,
        "elapsed_seconds": elapsed_seconds,
        "source_frames_per_second": source_frame_count / elapsed_seconds,
        "sampled_frames_per_second": sampled_frame_count / elapsed_seconds,
        "stage_mean_ms": {
            name: (float(np.mean(values)) if values else None)
            for name, values in stage_times.items()
        },
        "stage_sample_counts": {
            name: len(values) for name, values in stage_times.items()
        },
        "timing_scope": (
            "OpenCV decode of every source frame; detection and, when a face is "
            "found, crop/resize, model forward and temporal update on sampled "
            "frames. No display, video writing, or host-to-device transfer is "
            "excluded from end-to-end elapsed time."
        ),
    }


def collect_benchmark_environment(device: torch.device) -> dict:
    nvidia_smi = shutil.which("nvidia-smi")
    if device.type == "cuda" and nvidia_smi is None:
        raise FileNotFoundError("nvidia-smi is required to record GPU driver details")
    gpu_query = None
    if nvidia_smi:
        result = subprocess.run(
            [
                nvidia_smi,
                "--query-gpu=name,driver_version,pstate,power.limit",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        gpu_query = result.stdout.strip()
    power_scheme = None
    powercfg = shutil.which("powercfg")
    if powercfg:
        result = subprocess.run(
            [powercfg, "/GETACTIVESCHEME"],
            capture_output=True,
            text=True,
            check=True,
        )
        power_scheme = result.stdout.strip()

    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "pytorch_cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else None
        ),
        "gpu_total_memory_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda" else None
        ),
        "nvidia_smi_gpu_query": gpu_query,
        "windows_active_power_scheme": power_scheme,
        "inference_amp": False,
        "inference_dtype": "float32",
        "cuda_synchronized_around_timed_model_forwards": device.type == "cuda",
        "display_enabled": False,
    }


def load_validation_face(config: dict):
    image_size = int(config["dataset"]["image_size"])
    transform = get_val_transforms(image_size)
    dataset_root = Path(config["dataset"]["root"])
    if not dataset_root.is_absolute():
        dataset_root = REPOSITORY_ROOT / dataset_root
    dataset = FER2013FolderDataset(
        root_dir=str(dataset_root),
        split="test",
        image_size=image_size,
        transform=transform,
    )
    if not dataset.samples:
        raise RuntimeError(f"No FER2013 test images found in {dataset_root}")

    image_path = Path(dataset.samples[0]["path"])
    frame_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if frame_bgr is None or frame_bgr.size == 0:
        raise IOError(f"Could not read benchmark face image: {image_path}")
    height, width = frame_bgr.shape[:2]
    bbox = (0, 0, width, height)
    return image_path, frame_bgr, bbox, transform, image_size


def run_latency_benchmark(
    model: torch.nn.Module,
    device: torch.device,
    image_path: Path,
    frame_bgr: np.ndarray,
    bbox: tuple,
    transform,
    image_size: int,
    warmup_iterations: int,
    measured_iterations: int,
    output_dir: Path,
    write_measurements_csv: bool = True,
) -> dict:
    preprocessing_times = []
    model_input = None
    for _ in range(measured_iterations):
        start = time.perf_counter()
        face_tensor = preprocess_face(
            frame_bgr,
            bbox,
            transform,
            image_size=image_size,
        )
        preprocessing_times.append((time.perf_counter() - start) * 1000.0)
        model_input = face_tensor.unsqueeze(0)

    if model_input is None or model_input.shape != (1, 3, image_size, image_size):
        raise RuntimeError("Benchmark input is not exactly one 3-channel face")

    synchronize(device)
    transfer_start = time.perf_counter()
    model_input = model_input.to(device)
    synchronize(device)
    input_transfer_ms = (time.perf_counter() - transfer_start) * 1000.0

    with torch.inference_mode():
        for _ in range(warmup_iterations):
            output = model.forward_frame(model_input)
        synchronize(device)
        if output["probs"].shape != (1, len(EMOTION_CLASSES)):
            raise RuntimeError(
                f"Expected batch-size-1 class probabilities, got "
                f"{tuple(output['probs'].shape)}"
            )

        measured_times = []
        for _ in range(measured_iterations):
            synchronize(device)
            start = time.perf_counter()
            output = model.forward_frame(model_input)
            synchronize(device)
            measured_times.append((time.perf_counter() - start) * 1000.0)

    latency_stats = summarize(measured_times)
    preprocessing_stats = summarize(preprocessing_times)
    latency_csv = None
    if write_measurements_csv:
        latency_csv = output_dir / "latency_results.csv"
        with latency_csv.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(
                csv_file,
                fieldnames=(
                    "iteration",
                    "batch_size",
                    "model_forward_latency_ms",
                    "preprocessing_latency_ms",
                ),
            )
            writer.writeheader()
            writer.writerows(
                {
                    "iteration": index,
                    "batch_size": 1,
                    "model_forward_latency_ms": f"{measured_times[index]:.8f}",
                    "preprocessing_latency_ms": f"{preprocessing_times[index]:.8f}",
                }
                for index in range(measured_iterations)
            )

    results = {
        "benchmark": "batch_size_1_model_forward",
        "checkpoint_image": str(image_path),
        "batch_size": 1,
        "input_resolution": [image_size, image_size],
        "input_tensor_shape": list(model_input.shape),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda" else platform.processor() or "CPU"
        ),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "mixed_precision_enabled": False,
        "inference_dtype": str(model_input.dtype).replace("torch.", ""),
        "warmup_iterations": warmup_iterations,
        "measured_iterations": measured_iterations,
        "model_evaluation_mode": not model.training,
        "model_parameter_count": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "cuda_synchronized_before_and_after_each_timed_forward": (
            device.type == "cuda"
        ),
        "preprocessing": {
            "method": (
                "Existing preprocess_face pipeline: BGR crop, grayscale conversion, "
                "RGB replication, validation resize/normalization, CHW tensor"
            ),
            "iterations": len(preprocessing_times),
            "statistics_ms": preprocessing_stats,
            "device_input_transfer_ms": input_transfer_ms,
        },
        "model_forward_latency_ms": latency_stats,
        "model_throughput_images_per_second": (
            1000.0 / latency_stats["mean"] if latency_stats["mean"] else None
        ),
        "timing_scope": (
            "Only model.forward_frame on a preprocessed tensor of shape "
            "(1, 3, H, W); model and detector loading, face detection, "
            "preprocessing, and device transfer are excluded."
        ),
        "latency_csv": str(latency_csv),
    }
    if not write_measurements_csv:
        results["individual_measurements_ms"] = measured_times
    return results


def mean_if_available(values: list):
    return float(np.mean(values)) if values else None


def run_video_benchmark(
    model: torch.nn.Module,
    config: dict,
    checkpoint_report: dict,
    checkpoint_path: Path,
    config_path: Path,
    video_path: Path,
    device: torch.device,
    output_dir: Path,
) -> dict:
    if not video_dataset.MTCNN_AVAILABLE:
        raise ImportError("facenet-pytorch is required for video face detection")
    if not video_path.is_file():
        raise FileNotFoundError(f"Benchmark video not found: {video_path}")

    runs = []
    per_frame_rows = []
    for frame_stride in (1, 3):
        run_name = "every_frame" if frame_stride == 1 else "default_stride_3"
        inference_output = output_dir / f"video_{run_name}"
        timing_collector = {}
        if device.type == "cuda":
            synchronize(device)
        start = time.perf_counter()
        result = infer_video(
            model=model,
            video_path=str(video_path),
            device=device,
            image_size=int(config["dataset"]["image_size"]),
            output_dir=str(inference_output),
            face_mode="largest",
            face_index=0,
            frame_stride=frame_stride,
            detection_max_side=1280,
            sample_every=30,
            annotate_video=True,
            write_diagnostics=True,
            show_region_weights=False,
            temporal_analysis=False,
            alpha=float(config.get("stress", {}).get("alpha", 0.7)),
            beta=float(config.get("stress", {}).get("beta", 0.3)),
            config_path=str(config_path),
            checkpoint_report=checkpoint_report,
            timing_collector=timing_collector,
        )
        synchronize(device)
        elapsed_seconds = time.perf_counter() - start

        input_info = result["input"]
        inference_info = result["inference"]
        frames = timing_collector["per_frame"]
        face_detection_times = [
            frame["face_detection_time_ms"]
            for frame in frames
            if frame["face_detection_time_ms"] is not None
        ]
        preprocessing_times = [
            frame["preprocessing_time_ms"]
            for frame in frames
            if frame["preprocessing_time_ms"] is not None
        ]
        model_forward_times = [
            frame["model_forward_time_ms"]
            for frame in frames
            if frame["model_forward_time_ms"] is not None
        ]
        detected_face_count = sum(
            len(frame.get("detections", []))
            for frame in result["frames"]
            if frame["detection_status"] != "skipped_by_stride"
        )
        selected_face_count = int(
            result["prediction_diagnostics"]["predicted_face_count"]
        )
        source_frame_count = int(input_info["source_frame_count"])
        processed_frame_count = int(inference_info["processed_frames"])
        skipped_frame_count = int(inference_info["skipped_frames"])
        if len(frames) != source_frame_count:
            raise RuntimeError(
                f"Timing records contain {len(frames)} frames but source reports "
                f"{source_frame_count}"
            )

        run_metrics = {
            "configuration": run_name,
            "frame_stride": frame_stride,
            "is_existing_default_stride": frame_stride == 3,
            "face_mode": "largest",
            "video": str(video_path),
            "source_video_fps": float(input_info["fps"]),
            "source_frame_count": source_frame_count,
            "source_video_duration_seconds": float(
                input_info["duration_seconds"]
            ),
            "frames_processed": processed_frame_count,
            "frames_skipped": skipped_frame_count,
            "faces_detected_total": detected_face_count,
            "selected_faces_predicted": selected_face_count,
            "total_elapsed_processing_seconds": elapsed_seconds,
            "mean_face_detection_time_ms": mean_if_available(
                face_detection_times
            ),
            "mean_preprocessing_time_ms_per_selected_face": (
                mean_if_available(preprocessing_times)
            ),
            "mean_model_forward_time_ms_per_selected_face": (
                mean_if_available(model_forward_times)
            ),
            "source_frame_throughput_frames_per_second": (
                source_frame_count / elapsed_seconds if elapsed_seconds else None
            ),
            "analyzed_frame_throughput_frames_per_second": (
                processed_frame_count / elapsed_seconds
                if elapsed_seconds else None
            ),
            "cuda_synchronized_for_gpu_timing": timing_collector[
                "cuda_synchronized"
            ],
            "output_directory": str(inference_output),
            "end_to_end_scope": (
                "Elapsed around the complete existing infer_video call after model "
                "loading. Includes detector/transform setup, source-frame decoding, "
                "face detection, crop preprocessing, model inference, probability "
                "and temporal/stress processing, annotation/dashboard generation, "
                "annotated-video and diagnostic writing, plot generation, and "
                "artifact validation."
            ),
        }
        runs.append(run_metrics)
        for frame in frames:
            per_frame_rows.append({
                "configuration": run_name,
                "frame_stride": frame_stride,
                **frame,
            })

    timing_csv = output_dir / "video_benchmark_results.csv"
    timing_fields = (
        "configuration",
        "frame_stride",
        "frame_index",
        "processed",
        "detected_face_count",
        "selected_face_count",
        "decode_time_ms",
        "face_detection_time_ms",
        "preprocessing_time_ms",
        "model_forward_time_ms",
        "prediction_postprocessing_time_ms",
        "annotation_dashboard_time_ms",
        "video_write_time_ms",
        "sample_write_time_ms",
        "frame_total_time_ms",
    )
    with timing_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=timing_fields)
        writer.writeheader()
        for row in per_frame_rows:
            writer.writerow({
                field: (
                    f"{row[field]:.8f}"
                    if isinstance(row.get(field), float)
                    else row.get(field)
                )
                for field in timing_fields
            })

    return {
        "benchmark": "full_existing_video_inference_pipeline",
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "device": str(device),
        "video": str(video_path),
        "existing_infer_video_default_frame_stride": 3,
        "runs": runs,
        "per_frame_timing_csv": str(timing_csv),
        "per_frame_timing_count": len(per_frame_rows),
        "stage_timing_note": (
            "Face detection is averaged per analyzed source frame. Preprocessing "
            "and forward timing are averaged per selected face; face_mode=largest "
            "ensures one selected face at most per analyzed frame. Per-frame stage "
            "timing instrumentation is opt-in and does not change predictions."
        ),
    }


def format_value(value, digits: int = 3) -> str:
    if value is None:
        return "Not measured"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def write_summary(
    output_path: Path,
    latency: dict,
    video: dict,
    checkpoint_report: dict,
) -> None:
    stride_runs = {run["frame_stride"]: run for run in video["runs"]}
    latency_stats = latency["model_forward_latency_ms"]
    preprocessing_stats = latency["preprocessing"]["statistics_ms"]
    rows = [
        ("Batch size", latency["batch_size"], "images"),
        (
            "Input resolution",
            f"{latency['input_resolution'][0]} x {latency['input_resolution'][1]}",
            "pixels",
        ),
        ("Mean inference latency", latency_stats["mean"], "ms"),
        ("Median inference latency", latency_stats["median"], "ms"),
        ("P95 inference latency", latency_stats["p95"], "ms"),
        ("Minimum inference latency", latency_stats["minimum"], "ms"),
        ("Maximum inference latency", latency_stats["maximum"], "ms"),
        (
            "Standard deviation",
            latency_stats["standard_deviation"],
            "ms",
        ),
        (
            "Model throughput",
            latency["model_throughput_images_per_second"],
            "images/sec",
        ),
    ]
    for stride in (1, 3):
        run = stride_runs.get(stride)
        label = "every-frame" if stride == 1 else "default stride=3"
        rows.extend([
            (
                f"End-to-end video FPS ({label}; source frames)",
                (
                    run["source_frame_throughput_frames_per_second"]
                    if run else None
                ),
                "frames/sec",
            ),
            (
                f"Analyzed-frame FPS ({label})",
                (
                    run["analyzed_frame_throughput_frames_per_second"]
                    if run else None
                ),
                "frames/sec",
            ),
            (
                f"Average face detection time ({label})",
                run["mean_face_detection_time_ms"] if run else None,
                "ms",
            ),
            (
                f"Average preprocessing time per selected face ({label})",
                (
                    run["mean_preprocessing_time_ms_per_selected_face"]
                    if run else None
                ),
                "ms",
            ),
            (
                f"Average model-forward time per selected face ({label})",
                (
                    run["mean_model_forward_time_ms_per_selected_face"]
                    if run else None
                ),
                "ms",
            ),
        ])
    rows.append(
        (
            "Preprocessing time (FER2013 test face; mean)",
            preprocessing_stats["mean"],
            "ms",
        )
    )
    rows.append(
        (
            "GPU device",
            latency["device_name"],
            "device",
        )
    )

    run_one = stride_runs.get(1)
    run_three = stride_runs.get(3)
    lines = [
        "# Context-Aware ARWM Inference Benchmark",
        "",
        f"- Run directory: `{output_path.parent}`",
        f"- Checkpoint: `{checkpoint_report['checkpoint']}`",
        f"- Checkpoint epoch: `{checkpoint_report['checkpoint_epoch']}`",
        "- Configuration: resolved Context-Aware ARWM YAML",
        f"- Model mode: `{checkpoint_report['region_attention_mode']}`",
        f"- Strict checkpoint load: `{checkpoint_report['strict_load']}`",
        f"- Model evaluation mode: `{checkpoint_report['evaluation_mode']}`",
        "",
        "## Model-only latency and video pipeline results",
        "",
        "| Metric | Measured value | Unit |",
        "|---|---:|---|",
    ]
    lines.extend(
        f"| {metric} | {format_value(value)} | {unit} |"
        for metric, value, unit in rows
    )
    lines.extend([
        "",
        "## Video configuration and counts",
        "",
        "| Configuration | Source FPS | Source frames | Duration (s) | "
        "Processed | Skipped | Faces detected | Elapsed (s) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in (run_one, run_three):
        if run is None:
            continue
        lines.append(
            "| {configuration} | {fps} | {source} | {duration} | {processed} | "
            "{skipped} | {faces} | {elapsed} |".format(
                configuration=run["configuration"],
                fps=format_value(run["source_video_fps"]),
                source=run["source_frame_count"],
                duration=format_value(
                    run["source_video_duration_seconds"]
                ),
                processed=run["frames_processed"],
                skipped=run["frames_skipped"],
                faces=run["faces_detected_total"],
                elapsed=format_value(
                    run["total_elapsed_processing_seconds"]
                ),
            )
        )
    lines.extend([
        "",
        "## Measurement protocol",
        "",
        f"- Latency was measured at batch size 1 with "
        f"{latency['warmup_iterations']} warm-up and "
        f"{latency['measured_iterations']} measured forward passes.",
        "- Model-only timings use `time.perf_counter()` and synchronize CUDA "
        "immediately before and after each timed forward pass when applicable.",
        "- The forward pass is full precision; automatic mixed precision is "
        "disabled to match the existing `infer_video.py` inference path. The "
        "training configuration's AMP setting is not an inference setting.",
        "- Preprocessing was timed separately using the repository's existing "
        "`preprocess_face` and validation transform on a fixed FER2013 test face.",
        "- The video elapsed time surrounds the complete existing inference call "
        "after model loading. It includes decoding, face detection, preprocessing, "
        "inference, probability/temporal/stress processing, annotated video and "
        "diagnostic output writing, plot generation, and artifact validation.",
        "- Source-frame FPS is source frame count divided by full elapsed time. "
        "Analyzed-frame FPS is processed frame count divided by the same elapsed "
        "time; these differ when frame stride is greater than 1.",
        "- Per-stage timing is opt-in and recorded in "
        "`video_benchmark_results.csv`. Model loading is excluded from all "
        "reported inference timings.",
        "",
        "## Outputs",
        "",
        "- `latency_results.json` and `latency_results.csv`",
        "- `video_benchmark_results.json` and `video_benchmark_results.csv`",
        "- `benchmark_environment.json`",
        "- Each video run also writes the complete inference artifacts under its "
        "own `video_every_frame` or `video_default_stride_3` directory.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def benchmark_variant(
    variant: str,
    checkpoint_path: Path,
    config_path: Path,
    video_path: Path,
    output_dir: Path,
    device: torch.device,
    warmup_iterations: int,
    measured_iterations: int,
) -> dict:
    model, config, checkpoint_report = load_model(
        checkpoint_path, config_path, device
    )
    image_size = int(config["dataset"]["image_size"])
    face_tensor, image_paths = load_latency_inputs(config)
    model_latency = run_model_latency_batches(
        model=model,
        device=device,
        face_tensor=face_tensor,
        warmup_iterations=warmup_iterations,
        measured_iterations=measured_iterations,
    )
    video_results = {
        str(stride): run_mediapipe_video_benchmark(
            model=model,
            image_size=image_size,
            video_path=video_path,
            device=device,
            frame_stride=stride,
        )
        for stride in (1, 3)
    }
    result = {
        "variant": variant,
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "checkpoint_report": checkpoint_report,
        "model_latency": model_latency,
        "video": video_results,
        "inference_amp": False,
        "inference_dtype": "float32",
        "training_mixed_precision_configured": bool(
            config.get("training", {}).get("mixed_precision", False)
        ),
        "benchmark_input": {
            "image_paths": image_paths,
            "resolution": [image_size, image_size],
            "batch8_input": (
                "Eight distinct preprocessed official-test-folder images; "
                "host-to-device transfer is outside the timed model forward."
            ),
        },
    }
    del model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def write_comparison_summary(
    output_path: Path,
    results: dict,
    environment: dict,
) -> None:
    lines = [
        "# ARWM latency and video benchmark",
        "",
        f"- GPU: {environment['gpu_name']}",
        f"- Driver: {environment['nvidia_smi_gpu_query']}",
        f"- PyTorch/CUDA: {environment['pytorch_version']} / "
        f"{environment['pytorch_cuda_runtime']}",
        f"- Power state / scheme: {environment['nvidia_smi_gpu_query']} / "
        f"{environment['windows_active_power_scheme']}",
        "- Inference AMP/fp16: disabled; FP32",
        "- Model-only: `forward_frame`, batch sizes 1 and 8, "
        "50 warm-up and 500 timed forwards, synchronized around each timed "
        "CUDA forward; loading, preprocessing, and host-to-device transfer excluded.",
        "",
        "## Model-only latency",
        "",
        "| Variant | Batch 1 mean / median / p95 (ms) | Batch 8 mean / median / p95 (ms) | Batch 8 mean (ms/image) |",
        "|---|---:|---:|---:|",
    ]
    for variant, result in results.items():
        b1 = result["model_latency"]["1"]["batch_latency_ms"]
        b8 = result["model_latency"]["8"]["batch_latency_ms"]
        lines.append(
            f"| {variant} | {b1['mean']:.3f} / {b1['median']:.3f} / "
            f"{b1['p95']:.3f} | {b8['mean']:.3f} / {b8['median']:.3f} / "
            f"{b8['p95']:.3f} | {result['model_latency']['8']['mean_ms_per_image']:.3f} |"
        )
    lines.extend([
        "",
        "## Video pipeline",
        "",
        "| Variant | Sampling | Source FPS | Sampled FPS | Decode ms | Detect ms | Preprocess ms | Forward ms | Temporal ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for variant, result in results.items():
        for stride, video in result["video"].items():
            stage = video["stage_mean_ms"]
            lines.append(
                f"| {variant} | every {stride} frame(s) | "
                f"{video['source_frames_per_second']:.3f} | "
                f"{video['sampled_frames_per_second']:.3f} | "
                f"{stage['decode'] if stage['decode'] is not None else float('nan'):.3f} | "
                f"{stage['detect'] if stage['detect'] is not None else float('nan'):.3f} | "
                f"{stage['preprocess'] if stage['preprocess'] is not None else float('nan'):.3f} | "
                f"{stage['forward'] if stage['forward'] is not None else float('nan'):.3f} | "
                f"{stage['temporal'] if stage['temporal'] is not None else float('nan'):.3f} |"
            )
    lines.extend([
        "",
        "The video path decodes all source frames. Detection, preprocessing, "
        "forward, and temporal update run only on sampled frames with a detected "
        "largest face. The benchmark has no display or video writer.",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants",
        nargs="+",
        choices=tuple(PAPER_VARIANTS),
        default=list(PAPER_VARIANTS),
    )
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument(
        "--output-root",
        default=str(
            REPOSITORY_ROOT
            / "results"
            / "inference_benchmark"
            / "arwm_variants"
        ),
    )
    parser.add_argument("--warmup-iterations", type=int, default=50)
    parser.add_argument("--measured-iterations", type=int, default=500)
    args = parser.parse_args()
    if args.warmup_iterations < 1 or args.measured_iterations < 1:
        parser.error("warm-up and measured iteration counts must be positive")

    video_path = resolve_path(args.video)
    if not video_path.is_file():
        raise FileNotFoundError(f"Benchmark video not found: {video_path}")
    for variant in args.variants:
        for label in ("checkpoint", "config"):
            path = PAPER_VARIANTS[variant][label]
            if not path.is_file():
                raise FileNotFoundError(
                    f"{variant} {label} not found: {path}"
                )

    try:
        import mediapipe
    except ImportError as error:
        raise ImportError(
            "MediaPipe is required for the requested video benchmark. "
            "Install the dependency from requirements.txt."
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = create_run_directory(resolve_path(args.output_root))
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise IOError(f"Cannot open benchmark video: {video_path}")
    video_metadata = {
        "fps": float(video_capture.get(cv2.CAP_PROP_FPS)),
        "frame_count": int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    video_capture.release()
    video_metadata["duration_seconds"] = (
        video_metadata["frame_count"] / video_metadata["fps"]
        if video_metadata["fps"] else None
    )
    environment = collect_benchmark_environment(device)
    environment.update({
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "opencv_version": cv2.__version__,
        "numpy_version": np.__version__,
        "mediapipe_version": mediapipe.__version__,
        "device": str(device),
        "video": str(video_path),
        "video_metadata": video_metadata,
        "warmup_iterations_per_batch": args.warmup_iterations,
        "measured_iterations_per_batch": args.measured_iterations,
        "batch_sizes": [1, 8],
    })
    results = {}
    for variant in args.variants:
        settings = PAPER_VARIANTS[variant]
        result = benchmark_variant(
            variant=variant,
            checkpoint_path=settings["checkpoint"],
            config_path=settings["config"],
            video_path=video_path,
            output_dir=output_dir,
            device=device,
            warmup_iterations=args.warmup_iterations,
            measured_iterations=args.measured_iterations,
        )
        results[variant] = result
        write_json(output_dir / f"{variant}_results.json", result)
        print(
            f"[Benchmark] {variant}: "
            f"batch-1 mean="
            f"{result['model_latency']['1']['batch_latency_ms']['mean']:.3f} ms; "
            f"batch-8 mean/image="
            f"{result['model_latency']['8']['mean_ms_per_image']:.3f} ms"
        )

    write_json(output_dir / "benchmark_environment.json", environment)
    write_json(output_dir / "benchmark_results.json", results)
    write_comparison_summary(
        output_dir / "benchmark_summary.md",
        results,
        environment,
    )
    print(f"[Benchmark] Results written to {output_dir}")


if __name__ == "__main__":
    main()
