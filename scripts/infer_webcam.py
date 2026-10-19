"""
Live webcam inference for the Swin-FANE framework.

Usage:
    python scripts/infer_webcam.py \\
        --checkpoint checkpoints/fer2013_weighted_freeze3_lr1e4/best_model.pth \\
        --config configs/config.yaml

Press q in the video window to stop inference.
"""

import argparse
import csv
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
import torch

from data import video_dataset
from data.augmentations import get_val_transforms
from models.full_framework import build_framework
from utils.stress_formulation import EMOTION_CLASSES
from utils.temporal_analysis import TemporalSignalTracker
from utils.face_inference import (
    FaceTracker,
    detect_faces,
    preprocess_face,
    region_weight_log_record,
    select_faces,
)
from utils.checkpointing import validate_checkpoint_region_mode
from utils.configuration import get_region_attention_mode, load_config, set_region_attention_mode


@torch.inference_mode()
def infer_webcam(
    model: torch.nn.Module,
    device: torch.device,
    image_size: int,
    sequence_length: int,
    camera_index: int = 0,
    frame_skip: int = 3,
    camera_width: int = 640,
    camera_height: int = 480,
    temporal_analysis: bool = False,
    face_mode: str = "largest",
    face_index: int = 0,
    detection_max_side: int = 1280,
    show_region_weights: bool = False,
    overlay: bool = True,
    log_csv: str = None,
) -> None:
    """Run configured Swin-FANE inference on frames from a webcam."""
    if frame_skip < 1:
        raise ValueError("frame_skip must be at least 1")

    if not video_dataset.MTCNN_AVAILABLE:
        raise ImportError(
            "facenet-pytorch is required for webcam face detection. "
            "Install it with: pip install facenet-pytorch"
        )

    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise IOError(f"Cannot open webcam at index {camera_index}")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, camera_width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_height)

    face_detector = video_dataset.MTCNN(
        image_size=image_size,
        keep_all=True,
        min_face_size=40,
        device='cpu',
    )
    transform = get_val_transforms(image_size)
    face_tracker = FaceTracker(max_gap=max(3, frame_skip * 3))
    frame_buffer = deque(maxlen=sequence_length)
    last_emotion = "No face"
    last_stress = 0.0
    frame_number = 0
    processed_frame_number = 0
    session_start = time.monotonic()
    temporal_tracker = TemporalSignalTracker() if temporal_analysis else None
    last_confidence = 0.0
    last_transition = "No transition yet"
    last_persistence = 0.0
    last_volatility = 0.0
    last_timestamp = 0.0
    last_probability_lines = ["", ""]
    last_faces = []
    log_rows = []

    model.eval()
    try:
        while True:
            ret, frame_bgr = camera.read()
            if not ret:
                print("[InferWebcam] Could not read a webcam frame.")
                break

            if overlay:
                cv2.putText(
                    frame_bgr,
                    f"Emotion: {last_emotion}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    frame_bgr,
                    f"{'Stress proxy' if temporal_analysis else 'Stress'}: {last_stress:.3f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 200, 255),
                    2,
                )
            if temporal_analysis and overlay:
                overlay_lines = [
                    f"Time: {last_timestamp:.2f}s",
                    last_probability_lines[0],
                    last_probability_lines[1],
                    f"Confidence: {last_confidence:.2f}",
                    f"Transition: {last_transition}",
                    f"Persistence: {last_persistence:.1f}s",
                    f"Volatility: {last_volatility:.3f}",
                ]
                for line_index, text in enumerate(overlay_lines):
                    cv2.putText(
                        frame_bgr, text, (20, 112 + line_index * 24),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (245, 245, 245), 1,
                    )
            if overlay:
                for face in last_faces:
                    x1, y1, x2, y2 = face["bbox"]
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    region_line = ""
                    if show_region_weights:
                        if face["region_weights"] is None:
                            region_line = " ARWM off"
                        else:
                            weights = face["region_weights"]
                            region_line = (
                                f" E:{weights[0]:.2f} B:{weights[1]:.2f}"
                                f" M:{weights[2]:.2f}"
                            )
                    cv2.putText(
                        frame_bgr,
                        f"ID {face['track_id']} {face['emotion']} "
                        f"{face['confidence']:.2f}{region_line}",
                        (x1, max(20, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.55,
                        (0, 255, 0),
                        2,
                    )
            cv2.imshow("Swin-FANE Webcam Inference", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_number += 1
            if frame_number % frame_skip != 0:
                continue

            processed_frame_number += 1
            faces = detect_faces(
                frame_bgr, face_detector, max_side=detection_max_side
            )
            face_tracker.update(faces, processed_frame_number)
            selected = select_faces(faces, face_mode, face_index)
            if not selected:
                last_faces = []
                last_emotion = "No face"
                last_stress = 0.0
                last_confidence = 0.0
                last_probability_lines = ["", ""]
                frame_buffer.clear()
                continue

            face_tensors = torch.stack([
                preprocess_face(frame_bgr, face["bbox"], transform, image_size)
                for face in selected
            ]).to(device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_start = time.perf_counter()
            frame_output = model.forward_frame(face_tensors)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_batch_time_ms = (time.perf_counter() - model_start) * 1000.0
            frame_probabilities = frame_output["probs"].cpu().numpy()
            region_weights = frame_output.get("region_weights")
            if region_weights is not None:
                region_weights = region_weights.detach().cpu().numpy()
            last_faces = []
            timestamp = time.monotonic() - session_start
            for face_number, (face, probabilities) in enumerate(
                zip(selected, frame_probabilities)
            ):
                predicted_index = int(np.argmax(probabilities))
                weights = (
                    region_weights[face_number].astype(float).tolist()
                    if region_weights is not None else None
                )
                face_result = {
                    "bbox": face["bbox"],
                    "track_id": face["track_id"],
                    "emotion": EMOTION_CLASSES[predicted_index],
                    "confidence": float(probabilities[predicted_index]),
                    "region_weights": weights,
                }
                last_faces.append(face_result)
                if log_csv:
                    region_log = region_weight_log_record(
                        model.spatial_encoder.region_attention_mode,
                        (
                            region_weights[face_number]
                            if region_weights is not None else None
                        ),
                    )
                    log_rows.append({
                        "frame_index": frame_number,
                        "timestamp_seconds": f"{timestamp:.6f}",
                        "track_id": face["track_id"],
                        "predicted_emotion": face_result["emotion"],
                        "confidence": f"{face_result['confidence']:.8f}",
                        "model_inference_time_ms": f"{model_batch_time_ms / len(selected):.6f}",
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
                            f"probability_{name}": f"{float(probabilities[i]):.8f}"
                            for i, name in enumerate(EMOTION_CLASSES)
                        },
                        "x1": face["bbox"][0],
                        "y1": face["bbox"][1],
                        "x2": face["bbox"][2],
                        "y2": face["bbox"][3],
                    })

            last_probs = frame_probabilities[0]
            last_emotion = EMOTION_CLASSES[int(np.argmax(last_probs))]
            frame_buffer.append(face_tensors[0].cpu())
            sequence = list(frame_buffer)
            while len(sequence) < sequence_length:
                sequence.insert(0, sequence[0])
            x_seq = torch.stack(sequence).unsqueeze(0).to(device)
            output = model.forward_sequence(x_seq)
            if temporal_analysis:
                record = temporal_tracker.update(
                    last_probs, timestamp
                )
                last_timestamp = record['timestamp_seconds']
                probability_labels = ["A", "D", "F", "H", "Sa", "Su", "N"]
                compact_probs = [
                    f"{label}:{probability:.2f}"
                    for label, probability in zip(probability_labels, last_probs)
                ]
                last_probability_lines = [
                    "P " + " ".join(compact_probs[:4]),
                    "P " + " ".join(compact_probs[4:]),
                ]
                last_stress = record['stress_proxy']
                last_confidence = float(last_probs.max())
                last_transition = record['last_transition']
                last_persistence = record['persistence_seconds']
                last_volatility = record['probability_volatility']
            else:
                last_stress = float(output['stress_score'][0, 0].item())
    finally:
        camera.release()
        cv2.destroyAllWindows()
        if log_csv:
            log_path = Path(log_csv)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            fieldnames = [
                "frame_index", "timestamp_seconds", "track_id",
                "predicted_emotion", "confidence", "model_inference_time_ms",
                "model_batch_inference_time_ms", "region_attention_mode",
                "region_attention_status", "eye_weight", "eyebrow_weight",
                "mouth_weight",
                *[f"probability_{name}" for name in EMOTION_CLASSES],
                "x1", "y1", "x2", "y2",
            ]
            with log_path.open("w", newline="", encoding="utf-8") as csv_file:
                writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(log_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description='Infer Swin-FANE from a webcam')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--frame-skip', type=int, default=3)
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--arwm', action='store_true',
                        help='Legacy alias for --arwm-mode adaptive')
    parser.add_argument('--arwm-mode', choices=('off', 'fixed', 'adaptive'))
    parser.add_argument('--temporal-analysis', action='store_true',
                        help='Display temporal signals and deterministic stress proxy')
    parser.add_argument(
        '--face-mode', choices=('largest', 'selected', 'all'), default='largest'
    )
    parser.add_argument('--face-index', type=int, default=0)
    parser.add_argument('--detection-max-side', type=int, default=1280)
    parser.add_argument('--show-region-weights', action='store_true')
    parser.add_argument('--no-overlay', action='store_true',
                        help='Disable all display annotations')
    parser.add_argument('--log-csv', type=str, default=None,
                        help='Optional per-face CSV diagnostics path')
    parser.add_argument('--no-log', action='store_true',
                        help='Disable per-face diagnostic logging')
    args = parser.parse_args()

    config = load_config(args.config)
    if args.arwm_mode:
        set_region_attention_mode(config, args.arwm_mode)
    elif args.arwm:
        set_region_attention_mode(config, 'adaptive')
    region_mode = get_region_attention_mode(config)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config.setdefault('model', {}).setdefault('swin', {})['pretrained'] = False
    model = build_framework(config).to(device)
    checkpoint = torch.load(
        args.checkpoint, map_location=device, weights_only=False
    )
    state_dict = checkpoint.get('model') if isinstance(checkpoint, dict) else None
    if not isinstance(state_dict, dict):
        raise ValueError(
            f"Checkpoint must contain a model state dictionary: {args.checkpoint}"
        )
    validate_checkpoint_region_mode(checkpoint, region_mode, args.checkpoint)
    incompatible = model.load_state_dict(state_dict, strict=True)
    model.eval()
    print(
        f"[InferWebcam] Strict checkpoint load: {len(state_dict)} entries; "
        f"missing={incompatible.missing_keys}; "
        f"unexpected={incompatible.unexpected_keys}"
    )

    infer_webcam(
        model=model,
        device=device,
        image_size=config['dataset']['image_size'],
        sequence_length=config['dataset']['sequence_length'],
        camera_index=args.camera,
        frame_skip=args.frame_skip,
        camera_width=args.camera_width,
        camera_height=args.camera_height,
        temporal_analysis=args.temporal_analysis,
        face_mode=args.face_mode,
        face_index=args.face_index,
        detection_max_side=args.detection_max_side,
        show_region_weights=args.show_region_weights,
        overlay=not args.no_overlay,
        log_csv=None if args.no_log else args.log_csv,
    )


if __name__ == '__main__':
    main()
