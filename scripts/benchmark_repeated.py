"""Run three independent latency and interleaved full-video benchmark repetitions."""

import argparse
import csv
import gc
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch

from scripts.benchmark_latency import (
    DEFAULT_CHECKPOINT,
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_VIDEO,
    load_model,
    load_validation_face,
    resolve_path,
    run_latency_benchmark,
    synchronize,
    write_json,
)
from scripts.infer_video import infer_video


LATENCY_REPETITIONS = 3
LATENCY_WARMUP_ITERATIONS = 30
LATENCY_MEASURED_ITERATIONS = 200
VIDEO_RUN_ORDER = (
    (1, 1),
    (3, 1),
    (3, 2),
    (1, 2),
    (1, 3),
    (3, 3),
)
VIDEO_CSV_FIELDS = (
    "repetition",
    "frame_stride",
    "run_order",
    "status",
    "source_frame_count",
    "processed_frame_count",
    "skipped_frame_count",
    "total_elapsed_seconds",
    "source_frame_fps",
    "analyzed_frame_fps",
    "faces_detected",
    "selected_faces_predicted",
    "mean_face_detection_ms",
    "mean_preprocessing_ms_per_selected_face",
    "mean_model_forward_ms_per_selected_face",
    "mean_video_write_ms_per_source_frame",
    "total_video_write_seconds",
    "gpu_device",
    "output_directory",
    "gpu_before_json",
    "gpu_after_json",
    "error",
)
FRAME_CSV_FIELDS = (
    "repetition",
    "frame_stride",
    "run_order",
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


def gpu_snapshot(stage: str, run_id: str = None) -> dict:
    snapshot = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "stage": stage,
        "run_id": run_id,
        "nvidia_smi_available": False,
        "gpu_query": None,
        "compute_processes": None,
        "throttle_status": None,
        "error": None,
    }
    executable = shutil.which("nvidia-smi")
    if executable is None:
        snapshot["error"] = "nvidia-smi not found on PATH"
        return snapshot

    snapshot["nvidia_smi_available"] = True
    gpu_query = (
        "--query-gpu=index,name,temperature.gpu,utilization.gpu,memory.used,"
        "memory.total,power.draw,clocks.sm"
    )
    try:
        result = subprocess.run(
            [
                executable,
                gpu_query,
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        snapshot["gpu_query"] = result.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        snapshot["error"] = f"GPU telemetry query failed: {error}"

    try:
        processes = subprocess.run(
            [
                executable,
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        snapshot["compute_processes"] = processes.stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        snapshot["compute_processes"] = f"unavailable: {error}"

    try:
        performance = subprocess.run(
            [executable, "-q", "-d", "PERFORMANCE"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        snapshot["throttle_status"] = [
            line.strip()
            for line in performance.stdout.splitlines()
            if "clocks event reasons" in line.lower()
            or "active" in line.lower()
            or "thermal" in line.lower()
        ]
    except (OSError, subprocess.SubprocessError) as error:
        snapshot["throttle_status"] = f"unavailable: {error}"
    return snapshot


def checkpoint_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "modified_time_ns": stat.st_mtime_ns,
    }


def create_repetition_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    candidate = output_root / f"repeated_{timestamp}"
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"repeated_{timestamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def directory_manifest(path: Path) -> dict:
    if not path.exists():
        return {"exists": False, "files": {}}
    files = {}
    for item in path.rglob("*"):
        if item.is_file():
            stat = item.stat()
            files[str(item.relative_to(path))] = (
                stat.st_size,
                stat.st_mtime_ns,
            )
    return {"exists": True, "files": files}


def append_csv(path: Path, fields: tuple, row: dict) -> None:
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def sample_stats(values: list) -> dict:
    finite_values = [float(value) for value in values if value is not None]
    if not finite_values:
        return {"mean": None, "standard_deviation": None, "n": 0}
    return {
        "mean": float(np.mean(finite_values)),
        "standard_deviation": (
            float(np.std(finite_values, ddof=1))
            if len(finite_values) > 1 else None
        ),
        "n": len(finite_values),
    }


def summarize_gpu_telemetry(snapshots: list, benchmark_pid: int) -> dict:
    temperatures = []
    utilizations = []
    memory_used = []
    status_counts = {}
    compute_processes = set()
    for snapshot in snapshots:
        raw_gpu = snapshot.get("gpu_query")
        if raw_gpu:
            fields = [
                value.strip()
                for value in raw_gpu.splitlines()[0].split(",")
            ]
            if len(fields) >= 6:
                if fields[2].isdigit():
                    temperatures.append(int(fields[2]))
                if fields[3].isdigit():
                    utilizations.append(int(fields[3]))
                if fields[4].isdigit():
                    memory_used.append(int(fields[4]))
        raw_processes = snapshot.get("compute_processes")
        if raw_processes:
            for line in raw_processes.splitlines():
                fields = [value.strip() for value in line.split(",", 2)]
                if fields and fields[0].isdigit():
                    compute_processes.add(int(fields[0]))
        for line in snapshot.get("throttle_status", []):
            if line.strip() == "Clocks Event Reasons Counters":
                break
            if ":" not in line:
                continue
            key, value = (part.strip() for part in line.split(":", 1))
            if value in {"Active", "Not Active"}:
                status_counts[f"{key}: {value}"] = (
                    status_counts.get(f"{key}: {value}", 0) + 1
                )

    observations = []
    software_thermal_count = status_counts.get("SW Thermal Slowdown: Active", 0)
    hardware_thermal_count = status_counts.get("HW Thermal Slowdown: Active", 0)
    if software_thermal_count:
        observations.append(
            "nvidia-smi reported software thermal slowdown active in "
            f"{software_thermal_count}/{len(snapshots)} telemetry snapshots; "
            f"hardware thermal slowdown was active in "
            f"{hardware_thermal_count}/{len(snapshots)} snapshots. Runs were "
            "retained without exclusions."
        )
    external_processes = sorted(
        process_id
        for process_id in compute_processes
        if process_id != benchmark_pid
    )
    if external_processes:
        observations.append(
            "GPU compute process IDs other than the benchmark process were "
            f"observed: {external_processes}."
        )
    else:
        observations.append(
            "No GPU compute process other than the benchmark process was "
            "reported in the captured compute-process snapshots."
        )

    def range_summary(values: list) -> dict:
        if not values:
            return {"minimum": None, "maximum": None, "mean": None}
        return {
            "minimum": min(values),
            "maximum": max(values),
            "mean": float(np.mean(values)),
        }

    return {
        "snapshot_count": len(snapshots),
        "gpu_temperature_c": range_summary(temperatures),
        "gpu_utilization_percent": range_summary(utilizations),
        "gpu_memory_used_mib": range_summary(memory_used),
        "throttle_status_snapshot_counts": status_counts,
        "compute_process_ids": sorted(compute_processes),
        "benchmark_process_id": benchmark_pid,
        "observations": observations,
    }


def frame_stage_means(frames: list) -> dict:
    detection = [
        row["face_detection_time_ms"]
        for row in frames
        if row["face_detection_time_ms"] is not None
    ]
    preprocessing = [
        row["preprocessing_time_ms"]
        for row in frames
        if row["preprocessing_time_ms"] is not None
    ]
    model_forward = [
        row["model_forward_time_ms"]
        for row in frames
        if row["model_forward_time_ms"] is not None
    ]
    video_writes = [
        row["video_write_time_ms"]
        for row in frames
        if row["video_write_time_ms"] is not None
    ]
    return {
        "mean_face_detection_ms": (
            float(np.mean(detection)) if detection else None
        ),
        "mean_preprocessing_ms_per_selected_face": (
            float(np.mean(preprocessing)) if preprocessing else None
        ),
        "mean_model_forward_ms_per_selected_face": (
            float(np.mean(model_forward)) if model_forward else None
        ),
        "mean_video_write_ms_per_source_frame": (
            float(np.mean(video_writes)) if video_writes else None
        ),
        "total_video_write_seconds": (
            float(sum(video_writes) / 1000.0) if video_writes else None
        ),
    }


def run_latency_repetition(
    repetition: int,
    checkpoint_path: Path,
    config_path: Path,
    device: torch.device,
    image: tuple,
    output_dir: Path,
) -> tuple:
    run_id = f"latency_repetition_{repetition}"
    before = gpu_snapshot("before_latency_repetition", run_id)
    model, _, checkpoint_report = load_model(
        checkpoint_path,
        config_path,
        device,
    )
    image_path, frame_bgr, bbox, transform, image_size = image
    result = run_latency_benchmark(
        model=model,
        device=device,
        image_path=image_path,
        frame_bgr=frame_bgr,
        bbox=bbox,
        transform=transform,
        image_size=image_size,
        warmup_iterations=LATENCY_WARMUP_ITERATIONS,
        measured_iterations=LATENCY_MEASURED_ITERATIONS,
        output_dir=output_dir,
        write_measurements_csv=False,
    )
    measurements = result.pop("individual_measurements_ms")
    if len(measurements) != LATENCY_MEASURED_ITERATIONS:
        raise RuntimeError(
            f"Expected {LATENCY_MEASURED_ITERATIONS} measured iterations, "
            f"found {len(measurements)}"
        )
    stats = result["model_forward_latency_ms"]
    stats["standard_deviation"] = float(
        np.std(np.asarray(measurements, dtype=np.float64), ddof=1)
    )
    result.update({
        "repetition": repetition,
        "warmup_iterations": LATENCY_WARMUP_ITERATIONS,
        "measured_iterations": LATENCY_MEASURED_ITERATIONS,
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "checkpoint_region_attention_mode": (
            checkpoint_report["checkpoint_region_attention_mode"]
        ),
        "config_region_attention_mode": (
            checkpoint_report["config_region_attention_mode"]
        ),
        "strict_load": checkpoint_report["strict_load"],
        "missing_keys": checkpoint_report["missing_keys"],
        "unexpected_keys": checkpoint_report["unexpected_keys"],
        "model_evaluation_mode": not model.training,
        "gpu_before": before,
        "gpu_after": gpu_snapshot("after_latency_repetition", run_id),
        "measurements": measurements,
    })
    result.pop("latency_csv", None)
    try:
        return result, measurements
    finally:
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()


def run_video_repetition(
    repetition: int,
    frame_stride: int,
    run_order: int,
    checkpoint_path: Path,
    config_path: Path,
    video_path: Path,
    device: torch.device,
    output_dir: Path,
    videos_csv: Path,
    frame_csv: Path,
) -> dict:
    run_id = f"stride_{frame_stride}_repetition_{repetition}"
    inference_dir = output_dir / f"video_stride_{frame_stride}_repetition_{repetition}"
    record = {
        "repetition": repetition,
        "frame_stride": frame_stride,
        "run_order": run_order,
        "status": "failed",
        "output_directory": str(inference_dir),
        "error": None,
    }
    before = gpu_snapshot("before_video_repetition", run_id)
    record["gpu_before_json"] = before
    model = None
    try:
        model, config, checkpoint_report = load_model(
            checkpoint_path,
            config_path,
            device,
        )
        collector = {}
        synchronize(device)
        start = time.perf_counter()
        result = infer_video(
            model=model,
            video_path=str(video_path),
            device=device,
            image_size=int(config["dataset"]["image_size"]),
            output_dir=str(inference_dir),
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
            timing_collector=collector,
        )
        synchronize(device)
        elapsed = time.perf_counter() - start
        frames = collector["per_frame"]
        input_info = result["input"]
        inference_info = result["inference"]
        source_count = int(input_info["source_frame_count"])
        processed_count = int(inference_info["processed_frames"])
        skipped_count = int(inference_info["skipped_frames"])
        if len(frames) != source_count:
            raise RuntimeError(
                f"Expected {source_count} frame timing records, got {len(frames)}"
            )
        expected_processed = len(range(0, source_count, frame_stride))
        if processed_count != expected_processed:
            raise RuntimeError(
                f"Stride {frame_stride} processed {processed_count} frames; "
                f"expected {expected_processed}"
            )
        if processed_count + skipped_count != source_count:
            raise RuntimeError("Processed and skipped frame counts do not add up")
        for frame in frames:
            append_csv(
                frame_csv,
                FRAME_CSV_FIELDS,
                {
                    "repetition": repetition,
                    "frame_stride": frame_stride,
                    "run_order": run_order,
                    **frame,
                },
            )
        stage_metrics = frame_stage_means(frames)
        record.update({
            "status": "completed",
            "source_frame_count": source_count,
            "source_video_fps": float(input_info["fps"]),
            "source_video_duration_seconds": float(
                input_info["duration_seconds"]
            ),
            "processed_frame_count": processed_count,
            "skipped_frame_count": skipped_count,
            "total_elapsed_seconds": elapsed,
            "source_frame_fps": source_count / elapsed,
            "analyzed_frame_fps": processed_count / elapsed,
            "faces_detected": sum(
                len(frame.get("detections", []))
                for frame in result["frames"]
                if frame["detection_status"] != "skipped_by_stride"
            ),
            "selected_faces_predicted": int(
                result["prediction_diagnostics"]["predicted_face_count"]
            ),
            "gpu_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else platform.processor() or "CPU"
            ),
            "cuda_synchronized": device.type == "cuda",
            "strict_checkpoint_load": checkpoint_report["strict_load"],
            "region_attention_mode": checkpoint_report[
                "region_attention_mode"
            ],
            "evaluation_mode": not model.training,
            "stage_metrics": stage_metrics,
            "timing_collector": (
                "Optional instrumentation in infer_video; output_video.write "
                "duration is measured per source frame."
            ),
        })
    except Exception as error:
        record["error"] = f"{type(error).__name__}: {error}"
        record["traceback"] = traceback.format_exc()
    finally:
        record["gpu_after_json"] = gpu_snapshot(
            "after_video_repetition",
            run_id,
        )
        if model is not None:
            del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        append_csv(
            videos_csv,
            VIDEO_CSV_FIELDS,
            {
                field: (
                    json.dumps(record.get(field), allow_nan=False)
                    if field in {"gpu_before_json", "gpu_after_json"}
                    else record.get(field)
                )
                for field in VIDEO_CSV_FIELDS
            },
        )
    return record


def repetition_aggregate(rows: list, metric: str) -> dict:
    values = [
        float(row[metric])
        for row in rows
        if row.get("status", "completed") == "completed"
        and row.get(metric) is not None
    ]
    return sample_stats(values)


def video_aggregates(video_runs: list) -> dict:
    aggregates = {}
    metric_names = (
        "source_frame_fps",
        "analyzed_frame_fps",
        "total_elapsed_seconds",
        "faces_detected",
    )
    for stride in (1, 3):
        runs = [
            run for run in video_runs
            if run["frame_stride"] == stride
            and run["status"] == "completed"
        ]
        aggregates[f"stride_{stride}"] = {
            "completed_repetitions": len(runs),
            "failed_repetitions": sum(
                run["frame_stride"] == stride and run["status"] != "completed"
                for run in video_runs
            ),
            "metrics": {
                name: repetition_aggregate(runs, name)
                for name in metric_names
            },
            "stage_metrics": {
                stage: sample_stats([
                    run.get("stage_metrics", {}).get(stage)
                    for run in runs
                ])
                for stage in (
                    "mean_face_detection_ms",
                    "mean_preprocessing_ms_per_selected_face",
                    "mean_model_forward_ms_per_selected_face",
                    "mean_video_write_ms_per_source_frame",
                    "total_video_write_seconds",
                )
            },
        }
    return aggregates


def create_latency_csv(output_path: Path, repetitions: list) -> None:
    fields = (
        "repetition",
        "iteration",
        "batch_size",
        "latency_ms",
        "device",
        "precision",
        "input_shape",
    )
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)
        writer.writeheader()
        for result in repetitions:
            for iteration, latency in enumerate(result["measurements"], start=1):
                writer.writerow({
                    "repetition": result["repetition"],
                    "iteration": iteration,
                    "batch_size": 1,
                    "latency_ms": f"{latency:.8f}",
                    "device": result["device"],
                    "precision": result["inference_dtype"],
                    "input_shape": json.dumps(result["input_tensor_shape"]),
                })


def write_summary_markdown(
    path: Path,
    latency_aggregate: dict,
    video_aggregate: dict,
    latency_runs: list,
    video_runs: list,
    environment: dict,
) -> None:
    latency = latency_aggregate
    stride_one = video_aggregate["stride_1"]
    stride_three = video_aggregate["stride_3"]
    metrics = [
        ("Batch-size-1 inference latency", latency["mean_of_repetition_means"], "ms"),
        ("Median inference latency", latency["mean_of_repetition_medians"], "ms"),
        ("P95 inference latency", latency["mean_of_repetition_p95"], "ms"),
        ("Model throughput", latency["mean_throughput_images_per_second"], "images/sec"),
        ("End-to-end FPS (stride 1)", stride_one["metrics"]["source_frame_fps"]["mean"], "frames/sec"),
        ("End-to-end FPS (stride 3)", stride_three["metrics"]["source_frame_fps"]["mean"], "source frames/sec"),
        ("Analyzed-frame FPS (stride 3)", stride_three["metrics"]["analyzed_frame_fps"]["mean"], "frames/sec"),
        ("Face detection time", stride_one["stage_metrics"]["mean_face_detection_ms"]["mean"], "ms"),
        ("Preprocessing time", stride_one["stage_metrics"]["mean_preprocessing_ms_per_selected_face"]["mean"], "ms"),
        ("Model-forward time", stride_one["stage_metrics"]["mean_model_forward_ms_per_selected_face"]["mean"], "ms"),
        ("Video writing time", stride_one["stage_metrics"]["mean_video_write_ms_per_source_frame"]["mean"], "ms/source frame"),
    ]

    def show(value) -> str:
        return "Not measured" if value is None else f"{value:.4f}"

    lines = [
        "# Repeated Context-Aware ARWM Inference Benchmark",
        "",
        f"- Output directory: `{path.parent}`",
        f"- Checkpoint: `{environment['checkpoint']}`",
        f"- Configuration: `{environment['configuration']}`",
        f"- Video: `{environment['video']}`",
        f"- Device: `{environment['device_name']}`",
        f"- PyTorch / CUDA: `{environment['pytorch_version']}` / `{environment['cuda_version']}`",
        "- Inference precision: FP32; mixed precision disabled.",
        "- All reported repetition-level standard deviations use sample standard deviation "
        "(ddof=1). Within-repetition latency SD uses sample standard deviation across "
        "that repetition's 200 measured iterations.",
        "- Frame-level timings are summarized within each video run; uncertainty is "
        "then calculated across the three complete video repetitions, not across frames.",
        "",
        "## Research-paper results",
        "",
        "| Metric | Mean | Standard deviation | Unit |",
        "|---|---:|---:|---|",
    ]
    for name, value, unit in metrics:
        if name == "Batch-size-1 inference latency":
            std = latency["standard_deviation_of_repetition_means"]
        elif name == "Median inference latency":
            std = latency["standard_deviation_of_repetition_medians"]
        elif name == "P95 inference latency":
            std = latency["standard_deviation_of_repetition_p95"]
        elif name == "Model throughput":
            std = latency["standard_deviation_of_repetition_throughputs"]
        else:
            metric_key = (
                "source_frame_fps" if name.startswith("End-to-end FPS")
                else "analyzed_frame_fps" if name.startswith("Analyzed-frame")
                else None
            )
            if metric_key:
                stride_group = stride_one if "stride 1" in name else stride_three
                std = stride_group["metrics"][metric_key]["standard_deviation"]
            else:
                stage_key = {
                    "Face detection time": "mean_face_detection_ms",
                    "Preprocessing time": (
                        "mean_preprocessing_ms_per_selected_face"
                    ),
                    "Model-forward time": (
                        "mean_model_forward_ms_per_selected_face"
                    ),
                    "Video writing time": (
                        "mean_video_write_ms_per_source_frame"
                    ),
                }[name]
                std = stride_one["stage_metrics"][stage_key][
                    "standard_deviation"
                ]
        lines.append(f"| {name} | {show(value)} | {show(std)} | {unit} |")

    lines.extend([
        "",
        "## Latency repetitions",
        "",
        "| Repetition | Mean (ms) | Median (ms) | P95 (ms) | Min (ms) | "
        "Max (ms) | SD (ms) | Throughput (images/sec) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for result in latency_runs:
        stats = result["model_forward_latency_ms"]
        lines.append(
            f"| {result['repetition']} | {show(stats['mean'])} | "
            f"{show(stats['median'])} | {show(stats['p95'])} | "
            f"{show(stats['minimum'])} | {show(stats['maximum'])} | "
            f"{show(stats['standard_deviation'])} | "
            f"{show(result['model_throughput_images_per_second'])} |"
        )
    lines.extend([
        "",
        "## Video repetitions",
        "",
        "| Order | Stride | Repetition | Processed/skipped | Elapsed (s) | "
        "Source FPS | Analyzed FPS | Detection (ms) | Preprocess (ms) | "
        "Forward (ms) | Write (ms/frame) |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for run in sorted(video_runs, key=lambda item: item["run_order"]):
        if run["status"] != "completed":
            lines.append(
                f"| {run['run_order']} | {run['frame_stride']} | "
                f"{run['repetition']} | Failed | Not measured | Not measured | "
                f"Not measured | Not measured | Not measured | Not measured | "
                f"Not measured |"
            )
            continue
        stage = run["stage_metrics"]
        lines.append(
            f"| {run['run_order']} | {run['frame_stride']} | "
            f"{run['repetition']} | {run['processed_frame_count']}/"
            f"{run['skipped_frame_count']} | "
            f"{show(run['total_elapsed_seconds'])} | "
            f"{show(run['source_frame_fps'])} | "
            f"{show(run['analyzed_frame_fps'])} | "
            f"{show(stage['mean_face_detection_ms'])} | "
            f"{show(stage['mean_preprocessing_ms_per_selected_face'])} | "
            f"{show(stage['mean_model_forward_ms_per_selected_face'])} | "
            f"{show(stage['mean_video_write_ms_per_source_frame'])} |"
        )
    lines.extend([
        "",
        "## Pooled latency statistics",
        "",
        f"- Pooled median across all 600 raw timings: "
        f"{show(latency['pooled_median_ms'])} ms.",
        f"- Pooled P95 across all 600 raw timings: "
        f"{show(latency['pooled_p95_ms'])} ms.",
        f"- Mean of repetition-level means: "
        f"{show(latency['mean_of_repetition_means'])} ms; sample SD across "
        f"repetition means: "
        f"{show(latency['standard_deviation_of_repetition_means'])} ms.",
        "- Video FPS is source-frame count / elapsed time; analyzed-frame FPS is "
        "processed-frame count / the same elapsed time. Stride-3 metrics are "
        "reported separately.",
        "- Complete video elapsed time includes decoding, face detection, face "
        "preprocessing, forward inference, prediction/temporal/stress processing, "
        "dashboard annotation, output video and diagnostics writing, plot "
        "generation, and artifact validation. Model loading occurs outside this "
        "elapsed interval.",
        "",
        "## Hardware telemetry and protocol",
        "",
        f"- GPU: `{environment['device_name']}`; VRAM: "
        f"{environment['gpu_total_memory_bytes']} bytes.",
        "- GPU temperature, utilization, memory usage, compute processes, and "
        "performance/throttle status were sampled before and after each run when "
        "`nvidia-smi` exposed those fields. See `benchmark_environment.json`.",
        *[
            f"- Telemetry observation: {observation}"
            for observation in environment.get(
                "gpu_telemetry_summary", {}
            ).get("observations", [])
        ],
        "- Video run order: " + ", ".join(
            f"stride {stride} repetition {rep}"
            for stride, rep in VIDEO_RUN_ORDER
        ) + ".",
        "- No measured run is silently omitted; failures, if any, are recorded "
        "with an error in the repetition CSV/JSON.",
        "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--video", default=str(DEFAULT_VIDEO))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    args = parser.parse_args()

    checkpoint_path = resolve_path(args.checkpoint)
    config_path = resolve_path(args.config)
    video_path = resolve_path(args.video)
    output_root = resolve_path(args.output_root)
    for label, path in (
        ("checkpoint", checkpoint_path),
        ("configuration", config_path),
        ("video", video_path),
    ):
        if not path.is_file():
            raise FileNotFoundError(f"{label.capitalize()} not found: {path}")

    previous_benchmark_dir = output_root / "20261105_205312"
    initial_checkpoint_signature = checkpoint_signature(checkpoint_path)
    initial_previous_manifest = directory_manifest(previous_benchmark_dir)
    initial_gpu = gpu_snapshot("benchmark_start")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError(
            "CUDA is not available; this repeated benchmark requires the "
            "previously detected CUDA device."
        )

    output_dir = create_repetition_directory(output_root)
    environment = {
        "timestamp_local": datetime.now().astimezone().isoformat(),
        "checkpoint": str(checkpoint_path),
        "checkpoint_initial_signature": initial_checkpoint_signature,
        "configuration": str(config_path),
        "video": str(video_path),
        "video_size_bytes": video_path.stat().st_size,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_bytes": int(
            torch.cuda.get_device_properties(device).total_memory
        ),
        "initial_gpu_telemetry": initial_gpu,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "logical_cpu_count": os.cpu_count(),
        "benchmark_process_id": os.getpid(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "numpy_version": np.__version__,
        "precision": "float32",
        "mixed_precision_enabled": False,
        "model_configuration": "resolved Context-Aware ARWM; input 160x160",
        "latency_warmup_iterations": LATENCY_WARMUP_ITERATIONS,
        "latency_measured_iterations_per_repetition": (
            LATENCY_MEASURED_ITERATIONS
        ),
        "latency_batch_size": 1,
        "latency_input_shape": [1, 3, 160, 160],
        "cuda_synchronize_before_and_after_each_timed_forward": True,
        "video_run_order": [
            {"order": index, "frame_stride": stride, "repetition": repetition}
            for index, (stride, repetition) in enumerate(VIDEO_RUN_ORDER, start=1)
        ],
        "standard_deviation_convention": (
            "Sample SD (ddof=1) across the three independent repetition-level "
            "results; within-repetition latency SD uses sample SD across its "
            "200 timings."
        ),
        "existing_benchmark_directory": str(previous_benchmark_dir),
        "existing_benchmark_manifest_before": initial_previous_manifest,
    }
    write_json(output_dir / "benchmark_environment.json", environment)
    print(f"[RepeatedBenchmark] Output directory: {output_dir}")
    print(f"[RepeatedBenchmark] Device: {environment['device_name']}")
    if initial_gpu.get("compute_processes"):
        print(
            "[RepeatedBenchmark] GPU compute processes at start: "
            f"{initial_gpu['compute_processes']}"
        )

    latency_runs = []
    all_measurements = []
    validation_image = None
    video_runs = []
    videos_csv = output_dir / "video_repetitions.csv"
    frame_csv = output_dir / "video_all_frame_timings.csv"
    try:
        for repetition in range(1, LATENCY_REPETITIONS + 1):
            print(
                f"[RepeatedBenchmark] Latency repetition "
                f"{repetition}/{LATENCY_REPETITIONS}"
            )
            if validation_image is None:
                model, config, _ = load_model(
                    checkpoint_path,
                    config_path,
                    device,
                )
                validation_image = load_validation_face(config)
                del model
                gc.collect()
                torch.cuda.empty_cache()
            result, measurements = run_latency_repetition(
                repetition=repetition,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                device=device,
                image=validation_image,
                output_dir=output_dir,
            )
            all_measurements.extend(measurements)
            latency_runs.append(result)
            saved_result = {
                key: value
                for key, value in result.items()
                if key != "measurements"
            }
            write_json(
                output_dir / f"latency_repetition_{repetition}.json",
                saved_result,
            )
            print(
                f"[RepeatedBenchmark] Repetition {repetition}: "
                f"mean={result['model_forward_latency_ms']['mean']:.4f} ms"
            )

        create_latency_csv(
            output_dir / "latency_all_measurements.csv",
            latency_runs,
        )

        for run_order, (stride, repetition) in enumerate(VIDEO_RUN_ORDER, start=1):
            print(
                f"[RepeatedBenchmark] Video run {run_order}/6: "
                f"stride={stride}, repetition={repetition}"
            )
            run = run_video_repetition(
                repetition=repetition,
                frame_stride=stride,
                run_order=run_order,
                checkpoint_path=checkpoint_path,
                config_path=config_path,
                video_path=video_path,
                device=device,
                output_dir=output_dir,
                videos_csv=videos_csv,
                frame_csv=frame_csv,
            )
            video_runs.append(run)
            if run["status"] == "completed":
                print(
                    f"[RepeatedBenchmark] Completed: "
                    f"{run['source_frame_fps']:.4f} source FPS, "
                    f"{run['analyzed_frame_fps']:.4f} analyzed FPS"
                )
            else:
                print(
                    f"[RepeatedBenchmark] FAILED: {run['error']}"
                )
    finally:
        environment["final_gpu_telemetry"] = gpu_snapshot("benchmark_end")
        environment["checkpoint_final_signature"] = checkpoint_signature(
            checkpoint_path
        )
        environment["previous_benchmark_manifest_after"] = directory_manifest(
            previous_benchmark_dir
        )
        environment["original_benchmark_unchanged"] = (
            environment["existing_benchmark_manifest_before"]
            == environment["previous_benchmark_manifest_after"]
        )
        environment["checkpoint_unchanged"] = (
            environment["checkpoint_initial_signature"]
            == environment["checkpoint_final_signature"]
        )
        write_json(output_dir / "benchmark_environment.json", environment)

    if len(all_measurements) == (
        LATENCY_REPETITIONS * LATENCY_MEASURED_ITERATIONS
    ):
        pooled = np.asarray(all_measurements, dtype=np.float64)
        latency_aggregate = {
            "repetition_count": LATENCY_REPETITIONS,
            "measured_iterations_per_repetition": LATENCY_MEASURED_ITERATIONS,
            "total_individual_measurements": int(pooled.size),
            "standard_deviation_convention": "sample (ddof=1)",
            "mean_of_repetition_means": float(np.mean([
                run["model_forward_latency_ms"]["mean"]
                for run in latency_runs
            ])),
            "standard_deviation_of_repetition_means": float(np.std(
                [run["model_forward_latency_ms"]["mean"] for run in latency_runs],
                ddof=1,
            )),
            "mean_of_repetition_medians": float(np.mean([
                run["model_forward_latency_ms"]["median"]
                for run in latency_runs
            ])),
            "standard_deviation_of_repetition_medians": float(np.std(
                [run["model_forward_latency_ms"]["median"] for run in latency_runs],
                ddof=1,
            )),
            "mean_of_repetition_p95": float(np.mean([
                run["model_forward_latency_ms"]["p95"]
                for run in latency_runs
            ])),
            "standard_deviation_of_repetition_p95": float(np.std(
                [run["model_forward_latency_ms"]["p95"] for run in latency_runs],
                ddof=1,
            )),
            "pooled_median_ms": float(np.median(pooled)),
            "pooled_p95_ms": float(np.percentile(pooled, 95)),
            "mean_throughput_images_per_second": float(np.mean([
                run["model_throughput_images_per_second"]
                for run in latency_runs
            ])),
            "standard_deviation_of_repetition_throughputs": float(np.std(
                [
                    run["model_throughput_images_per_second"]
                    for run in latency_runs
                ],
                ddof=1,
            )),
        }
    else:
        latency_aggregate = {
            "repetition_count": len(latency_runs),
            "total_individual_measurements": len(all_measurements),
            "expected_individual_measurements": (
                LATENCY_REPETITIONS * LATENCY_MEASURED_ITERATIONS
            ),
            "error": "Incomplete latency measurements; aggregates not fabricated.",
        }

    aggregate = {
        "latency": latency_aggregate,
        "video": video_aggregates(video_runs),
    }
    telemetry_snapshots = []
    for run in latency_runs:
        telemetry_snapshots.extend(
            [run["gpu_before"], run["gpu_after"]]
        )
    for run in video_runs:
        telemetry_snapshots.extend([
            run["gpu_before_json"],
            run["gpu_after_json"],
        ])
    environment["gpu_telemetry_summary"] = summarize_gpu_telemetry(
        telemetry_snapshots,
        os.getpid(),
    )
    write_json(output_dir / "benchmark_environment.json", environment)
    summary = {
        "benchmark": "repeated_context_aware_arwm_inference",
        "output_directory": str(output_dir),
        "checkpoint": str(checkpoint_path),
        "configuration": str(config_path),
        "video": str(video_path),
        "device": environment["device_name"],
        "latency_repetitions": [
            {
                key: value for key, value in run.items()
                if key != "measurements"
            }
            for run in latency_runs
        ],
        "video_repetitions": video_runs,
        "aggregates": aggregate,
        "environmental_observations": environment[
            "gpu_telemetry_summary"
        ]["observations"],
        "run_order": environment["video_run_order"],
        "notes": [
            "Latency batch size is exactly 1 and every repetition reloads "
            "the checkpoint.",
            "Video runs reload the checkpoint independently and write to "
            "separate output directories.",
            "Video timing includes the complete infer_video call and output "
            "writing; model loading is excluded from elapsed time.",
            "Sample standard deviation (ddof=1) is used across three "
            "independent repetitions.",
            "No run is discarded. Failed runs, if any, are recorded as failed.",
        ],
    }
    write_json(output_dir / "repeated_benchmark_summary.json", summary)
    write_summary_markdown(
        output_dir / "repeated_benchmark_summary.md",
        latency_aggregate,
        aggregate["video"],
        latency_runs,
        video_runs,
        environment,
    )
    print(json.dumps({
        "output_directory": str(output_dir),
        "latency_aggregate": latency_aggregate,
        "video_aggregate": aggregate["video"],
        "video_failed_runs": [
            {
                "stride": run["frame_stride"],
                "repetition": run["repetition"],
                "error": run["error"],
            }
            for run in video_runs
            if run["status"] != "completed"
        ],
        "checkpoint_unchanged": environment["checkpoint_unchanged"],
        "original_benchmark_unchanged": environment[
            "original_benchmark_unchanged"
        ],
    }, indent=2))


if __name__ == "__main__":
    main()
