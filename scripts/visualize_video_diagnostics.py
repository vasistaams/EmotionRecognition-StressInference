"""Create face-crop contact sheets and ARWM token-mask overlays."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.swin_fane import AdaptiveRegionWeightingModule


REGIONS = (
    ("eye", np.array([1.0, 0.78, 0.0], dtype=np.float32)),
    ("eyebrow", np.array([0.0, 0.78, 1.0], dtype=np.float32)),
    ("mouth", np.array([1.0, 0.12, 0.18], dtype=np.float32)),
)


def select_examples(records, limit: int = 16):
    """Choose low-confidence, emotion-covering and multi-face examples."""
    candidates = []
    by_emotion = {}
    multi_face = []
    for record in records:
        faces = record.get("faces", [])
        for face in faces:
            item = {
                "frame_index": record["frame_index"],
                "timestamp_seconds": record["timestamp_seconds"],
                "face": face,
                "detection_count": len(record.get("detections", [])),
            }
            candidates.append(item)
            emotion = face["predicted_emotion"]
            if emotion not in by_emotion or face["confidence"] > by_emotion[emotion]["face"]["confidence"]:
                by_emotion[emotion] = item
            if len(record.get("detections", [])) > 1:
                multi_face.append(item)

    selected = []
    seen = set()

    def add(item, reason):
        key = (item["frame_index"], item["face"]["track_id"])
        if key in seen or len(selected) >= limit:
            return
        selected.append({**item, "reason": reason})
        seen.add(key)

    for item in sorted(candidates, key=lambda value: value["face"]["confidence"])[:4]:
        add(item, "low confidence")
    for emotion, item in sorted(by_emotion.items()):
        add(item, f"emotion: {emotion}")
    multi_frames = {}
    for item in multi_face:
        frame_index = item["frame_index"]
        if frame_index not in multi_frames or item["face"]["confidence"] > multi_frames[frame_index]["face"]["confidence"]:
            multi_frames[frame_index] = item
    for item in sorted(
        multi_frames.values(),
        key=lambda value: (-value["detection_count"], value["frame_index"]),
    ):
        add(item, f"{item['detection_count']} faces in frame")
    return selected


def read_crop(cap, item):
    frame_index = int(item["frame_index"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    if not ok:
        raise IOError(f"Could not read frame {frame_index}")
    x1, y1, x2, y2 = item["face"]["bbox_xyxy"]
    height, width = frame.shape[:2]
    x1, x2 = max(0, min(x1, width)), max(0, min(x2, width))
    y1, y2 = max(0, min(y1, height)), max(0, min(y2, height))
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError(f"Empty crop at frame {frame_index}: {(x1, y1, x2, y2)}")
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (160, 160), interpolation=cv2.INTER_AREA)


def make_contact_sheet(cap, examples, output_path):
    columns = 4
    rows = math.ceil(len(examples) / columns)
    fig, axes = plt.subplots(rows, columns, figsize=(16, rows * 4.3), squeeze=False)
    for axis, item in zip(axes.flat, examples):
        gray = read_crop(cap, item)
        face = item["face"]
        axis.imshow(gray, cmap="gray", vmin=0, vmax=255)
        axis.set_title(
            f"{item['reason']}\nframe {item['frame_index']} | "
            f"ID {face['track_id']} | pred={face['predicted_emotion']} "
            f"| p={face['confidence']:.2f}",
            fontsize=9,
        )
        axis.axis("off")
    for axis in axes.flat[len(examples):]:
        axis.axis("off")
    fig.suptitle(
        "testvideo2 face crops — low p is model confidence, not classification accuracy",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def make_mask_visualization(cap, examples, output_path):
    module = AdaptiveRegionWeightingModule(
        feature_dim=1024,
        hidden_dim=128,
        enabled=True,
    )
    masks = {
        name: module._make_region_mask(5, name).reshape(5, 5).numpy()
        for name, _ in REGIONS
    }
    fig, axes = plt.subplots(len(examples), 4, figsize=(14, 3.6 * len(examples)), squeeze=False)
    for row, item in enumerate(examples):
        gray = read_crop(cap, item)
        base = np.repeat((gray.astype(np.float32) / 255.0)[..., None], 3, axis=2)
        axes[row, 0].imshow(gray, cmap="gray", vmin=0, vmax=255)
        face = item["face"]
        weights = face.get("arwm_region_weights") or {}
        axes[row, 0].set_title(
            f"frame {item['frame_index']} | ID {face['track_id']} | "
            f"pred={face['predicted_emotion']} (p={face['confidence']:.2f})"
        )
        for column, (name, color) in enumerate(REGIONS, start=1):
            token_mask = masks[name]
            expanded = cv2.resize(
                token_mask,
                (160, 160),
                interpolation=cv2.INTER_NEAREST,
            )
            alpha = (0.65 * expanded)[..., None]
            overlay = base * (1.0 - alpha) + color[None, None, :] * alpha
            axes[row, column].imshow(np.clip(overlay, 0, 1))
            for coordinate in range(32, 160, 32):
                axes[row, column].axvline(coordinate - 0.5, color="white", linewidth=0.6)
                axes[row, column].axhline(coordinate - 0.5, color="white", linewidth=0.6)
            axes[row, column].set_title(
                f"{name}: mask prior | weight={weights.get(name, float('nan')):.3f}"
            )
        for axis in axes[row]:
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle(
        "ARWM 5x5 geometric priors (overlay opacity follows each mask value)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--log", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=16)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("--limit must be positive")

    with open(args.log, encoding="utf-8") as file:
        log = json.load(file)
    examples = select_examples(log["frames"], args.limit)
    if not examples:
        raise RuntimeError("No predicted faces found in inference log")
    os.makedirs(args.output_dir, exist_ok=True)
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {args.video}")
    try:
        make_contact_sheet(
            cap, examples, os.path.join(args.output_dir, "face_contact_sheet.jpg")
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
            if all((entry["frame_index"], entry["face"]["track_id"]) != key for entry in mask_examples):
                mask_examples.append(item)
        make_mask_visualization(
            cap,
            mask_examples,
            os.path.join(args.output_dir, "arwm_token_masks.jpg"),
        )
    finally:
        cap.release()
    print(f"Saved {len(examples)} crop examples to {args.output_dir}")
    print(f"Saved: {os.path.join(args.output_dir, 'face_contact_sheet.jpg')}")
    print(f"Saved: {os.path.join(args.output_dir, 'arwm_token_masks.jpg')}")


if __name__ == "__main__":
    main()
