"""Benchmark model-only and end-to-end video inference on one machine."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from data import video_dataset
from data.augmentations import get_val_transforms
from models.full_framework import build_framework
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config, set_region_attention_mode
from utils.face_inference import detect_faces, preprocess_face, select_faces
from utils.stress_formulation import EMOTION_CLASSES


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--arwm-mode", choices=("off", "fixed", "adaptive"))
    parser.add_argument("--face-mode", choices=("largest", "selected", "all"), default="all")
    parser.add_argument("--face-index", type=int, default=0)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--detection-max-side", type=int, default=1280)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-iterations", type=int, default=5)
    parser.add_argument("--timed-frames", type=int, default=120)
    parser.add_argument("--show-region-weights", action="store_true",
                        help="Measure overlay drawing overhead for actual weights")
    args = parser.parse_args()
    if args.frame_stride < 1 or args.detection_max_side < 1:
        raise ValueError("frame stride and detection max side must be positive")
    if args.batch_size < 1 or args.warmup_iterations < 0 or args.timed_frames < 1:
        raise ValueError("batch size/timed frames must be positive and warmup non-negative")
    if not video_dataset.MTCNN_AVAILABLE:
        raise ImportError("facenet-pytorch is required for video benchmarking")

    config = load_config(args.config)
    if args.arwm_mode:
        set_region_attention_mode(config, args.arwm_mode)
    mode = get_region_attention_mode(config)
    config.setdefault("model", {}).setdefault("swin", {})["pretrained"] = False
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_framework(config).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_checkpoint_region_mode(checkpoint, mode, args.checkpoint)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()
    detector = video_dataset.MTCNN(
        image_size=config["dataset"]["image_size"],
        keep_all=True,
        min_face_size=40,
        device="cpu",
    )
    transform = get_val_transforms(config["dataset"]["image_size"])
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_index = 0
    processed_index = 0
    model_times = []
    face_count = 0

    def next_processed_frame():
        nonlocal frame_index, processed_index
        while True:
            success, frame = cap.read()
            if not success:
                return None
            current_index = frame_index
            frame_index += 1
            if current_index % args.frame_stride == 0:
                processed_index += 1
                return current_index, frame

    def process_frame(index, frame, collect_timing):
        nonlocal face_count
        faces = detect_faces(frame, detector, max_side=args.detection_max_side)
        selected = select_faces(faces, args.face_mode, args.face_index)
        if not selected:
            return
        inputs = torch.stack([
            preprocess_face(
                frame, face["bbox"], transform, config["dataset"]["image_size"]
            )
            for face in selected
        ])
        for offset in range(0, len(inputs), args.batch_size):
            batch = inputs[offset:offset + args.batch_size].to(device)
            synchronize(device)
            start = time.perf_counter()
            with torch.inference_mode():
                output = model.forward_frame(batch)
            synchronize(device)
            duration = time.perf_counter() - start
            if collect_timing:
                model_times.append(duration)
                face_count += len(batch)
            probabilities = output["probs"].detach().cpu().numpy()
            if args.show_region_weights:
                region_weights = output.get("region_weights")
                if region_weights is not None:
                    region_weights = region_weights.detach().cpu().numpy()
                for local_index, prediction in enumerate(
                    probabilities.argmax(axis=-1)
                ):
                    face = selected[offset + local_index]
                    confidence = probabilities[local_index, prediction]
                    if region_weights is None:
                        text = (
                            f"{EMOTION_CLASSES[prediction]} {confidence:.2f} "
                            "ARWM off"
                        )
                    else:
                        eye, eyebrow, mouth = region_weights[local_index]
                        text = (
                            f"{EMOTION_CLASSES[prediction]} {confidence:.2f} "
                            f"E:{eye:.2f} B:{eyebrow:.2f} M:{mouth:.2f}"
                        )
                    x1, y1, x2, y2 = face["bbox"]
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame, text, (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2,
                    )

    try:
        for _ in range(args.warmup_iterations):
            item = next_processed_frame()
            if item is None:
                break
            process_frame(*item, collect_timing=False)
        warmup_completed = processed_index
        if args.warmup_iterations and not warmup_completed:
            raise RuntimeError("Video ended before any warm-up frame was processed")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_index = 0
        processed_index = 0
        model_times.clear()
        face_count = 0
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        end_to_end_start = time.perf_counter()
        timed_frame_count = 0
        while timed_frame_count < args.timed_frames:
            item = next_processed_frame()
            if item is None:
                break
            process_frame(*item, collect_timing=True)
            timed_frame_count += 1
        synchronize(device)
        end_to_end_seconds = time.perf_counter() - end_to_end_start
    finally:
        cap.release()

    model_seconds = float(sum(model_times))
    report = {
        "video": args.video,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "region_attention_mode": mode,
        "device": str(device),
        "hardware": {
            "processor": platform.processor(),
            "platform": platform.platform(),
            "cpu_threads": torch.get_num_threads(),
            "cuda_device": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda" else None
            ),
        },
        "video_resolution": [width, height],
        "image_resolution": config["dataset"]["image_size"],
        "frame_stride": args.frame_stride,
        "batch_size": args.batch_size,
        "warmup_iterations_requested": args.warmup_iterations,
        "warmup_frames_completed": warmup_completed,
        "timed_processed_frames": timed_frame_count,
        "predicted_face_count": face_count,
        "show_region_weights_overlay": args.show_region_weights,
        "timing_method": (
            "perf_counter with CUDA synchronization where applicable. Model-only "
            "latency covers forward_frame calls only. End-to-end includes video "
            "decode, detection, crop preprocessing, forward passes and optional "
            "OpenCV overlay drawing; excludes display and encoded video writing."
        ),
        "model_only": {
            "forward_calls": len(model_times),
            "mean_batch_latency_ms": (
                float(np.mean(model_times) * 1000) if model_times else None
            ),
            "p50_batch_latency_ms": (
                float(np.percentile(model_times, 50) * 1000) if model_times else None
            ),
            "p95_batch_latency_ms": (
                float(np.percentile(model_times, 95) * 1000) if model_times else None
            ),
            "amortized_ms_per_face": (
                model_seconds * 1000 / face_count if face_count else None
            ),
            "faces_per_second": (
                face_count / model_seconds if model_seconds else None
            ),
        },
        "end_to_end": {
            "elapsed_seconds": end_to_end_seconds,
            "processed_source_frames_per_second": (
                timed_frame_count / end_to_end_seconds if end_to_end_seconds else None
            ),
            "predicted_faces_per_second": (
                face_count / end_to_end_seconds if end_to_end_seconds else None
            ),
        },
        "peak_gpu_memory_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda" else None
        ),
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved benchmark report: {output_path}")


if __name__ == "__main__":
    main()
