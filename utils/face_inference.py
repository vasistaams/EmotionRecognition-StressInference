"""Shared face detection and FER2013-compatible preprocessing for inference."""

from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
from PIL import Image


def detect_faces(
    frame_bgr: np.ndarray,
    detector,
    max_side: int = 1280,
    min_probability: float = 0.5,
) -> List[Dict]:
    """Detect faces on a downscaled copy and return boxes in source coordinates."""
    if max_side < 1:
        raise ValueError("max_side must be positive")

    height, width = frame_bgr.shape[:2]
    scale = min(1.0, max_side / max(height, width))
    if scale < 1.0:
        detection_frame = cv2.resize(
            frame_bgr,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_frame = frame_bgr

    detection_rgb = cv2.cvtColor(detection_frame, cv2.COLOR_BGR2RGB)
    boxes, probabilities = detector.detect(Image.fromarray(detection_rgb))
    if boxes is None:
        return []

    detection_height, detection_width = detection_frame.shape[:2]
    scale_x = width / detection_width
    scale_y = height / detection_height
    faces = []
    for index, box in enumerate(boxes):
        probability = (
            float(probabilities[index])
            if probabilities is not None and index < len(probabilities)
            else 0.0
        )
        if not np.isfinite(probability) or probability < min_probability:
            continue
        x1, y1, x2, y2 = (
            int(round(float(box[0]) * scale_x)),
            int(round(float(box[1]) * scale_y)),
            int(round(float(box[2]) * scale_x)),
            int(round(float(box[3]) * scale_y)),
        )
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        if x2 > x1 and y2 > y1:
            faces.append({
                "bbox": (x1, y1, x2, y2),
                "detection_confidence": probability,
            })
    return faces


def select_faces(
    faces: Sequence[Dict],
    mode: str,
    selected_index: int = 0,
) -> List[Dict]:
    """Select all detections, the largest, or an area-ranked detection."""
    if mode not in {"largest", "selected", "all"}:
        raise ValueError(f"Unknown face mode: {mode}")
    if selected_index < 0:
        raise ValueError("selected_index must be non-negative")
    ranked = sorted(
        faces,
        key=lambda face: (
            (face["bbox"][2] - face["bbox"][0])
            * (face["bbox"][3] - face["bbox"][1])
        ),
        reverse=True,
    )
    if mode == "all":
        return ranked
    if mode == "largest":
        return ranked[:1]
    return ranked[selected_index:selected_index + 1]


def preprocess_face(
    frame_bgr: np.ndarray,
    bbox: Tuple[int, int, int, int],
    transform,
    image_size: Optional[int] = None,
) -> torch.Tensor:
    """Crop a face, replicate grayscale into RGB, then apply validation transforms."""
    x1, y1, x2, y2 = bbox
    face_bgr = frame_bgr[y1:y2, x1:x2]
    if face_bgr.size == 0:
        raise ValueError(f"Empty face crop for bounding box {bbox}")
    face_gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    face_rgb = cv2.cvtColor(face_gray, cv2.COLOR_GRAY2RGB)
    tensor = transform(image=face_rgb)["image"]
    if not isinstance(tensor, torch.Tensor) or tensor.ndim != 3:
        raise TypeError("Validation transform must return a CHW torch.Tensor")
    if tensor.shape[0] != 3:
        raise ValueError(f"Expected a 3-channel tensor, got {tuple(tensor.shape)}")
    if image_size is not None and tuple(tensor.shape[-2:]) != (
        image_size,
        image_size,
    ):
        raise ValueError(
            f"Expected a {image_size}x{image_size} tensor, got "
            f"{tuple(tensor.shape[-2:])}"
        )
    return tensor


def region_weight_log_record(mode: str, weights=None) -> Dict:
    """Build mode-aware region-weight fields without inventing off-mode values."""
    if mode not in {"off", "fixed", "adaptive"}:
        raise ValueError(f"Unknown region-attention mode: {mode}")
    if mode == "off":
        if weights is not None:
            raise ValueError("ARWM-off output must not include region weights")
        values = None
    else:
        if weights is None:
            raise ValueError(f"{mode} mode requires actual region weights")
        values_array = np.asarray(weights, dtype=float)
        if values_array.shape != (3,) or not np.isfinite(values_array).all():
            raise ValueError(
                f"Expected three finite region weights, got {values_array.shape}"
            )
        values = {
            name: float(values_array[index])
            for index, name in enumerate(("eye", "eyebrow", "mouth"))
        }
    return {
        "region_attention_mode": mode,
        "region_attention_status": "disabled" if values is None else "active",
        "arwm_region_weights": values,
        "eye_weight": None if values is None else values["eye"],
        "eyebrow_weight": None if values is None else values["eyebrow"],
        "mouth_weight": None if values is None else values["mouth"],
    }


def _box_iou(
    first: Tuple[int, int, int, int],
    second: Tuple[int, int, int, int],
) -> float:
    x1 = max(first[0], second[0])
    y1 = max(first[1], second[1])
    x2 = min(first[2], second[2])
    y2 = min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


class FaceTracker:
    """Associate nearby-frame boxes with stable IDs using greedy IoU matching."""

    def __init__(self, max_gap: int = 3, min_iou: float = 0.2):
        self.max_gap = max_gap
        self.min_iou = min_iou
        self.next_id = 1
        self.tracks: Dict[int, Tuple[Tuple[int, int, int, int], int]] = {}

    def update(self, faces: List[Dict], frame_index: int) -> None:
        active = {
            track_id: box
            for track_id, (box, last_frame) in self.tracks.items()
            if frame_index - last_frame <= self.max_gap
        }
        pairs = sorted(
            (
                (_box_iou(face["bbox"], box), face_index, track_id)
                for face_index, face in enumerate(faces)
                for track_id, box in active.items()
            ),
            reverse=True,
        )
        used_faces = set()
        used_tracks = set()
        for overlap, face_index, track_id in pairs:
            if (
                overlap < self.min_iou
                or face_index in used_faces
                or track_id in used_tracks
            ):
                continue
            faces[face_index]["track_id"] = track_id
            used_faces.add(face_index)
            used_tracks.add(track_id)

        for face_index, face in enumerate(faces):
            if face_index not in used_faces:
                face["track_id"] = self.next_id
                self.next_id += 1
            self.tracks[face["track_id"]] = (face["bbox"], frame_index)

        self.tracks = {
            track_id: state
            for track_id, state in self.tracks.items()
            if frame_index - state[1] <= self.max_gap
        }
