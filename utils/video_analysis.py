"""Reconstruct time-based emotion and stress-proxy diagnostics from video logs."""

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from utils.stress_formulation import EMOTION_CLASSES, compute_stress_trajectory
from utils.temporal_analysis import TemporalSignalTracker, summarize_temporal_records


EMOTION_COLOURS = {
    "anger": "#c0392b",
    "disgust": "#8e44ad",
    "fear": "#2980b9",
    "happiness": "#27ae60",
    "sadness": "#d35400",
    "surprise": "#16a085",
    "neutral": "#7f8c8d",
}


def _save_figure(figure, path: Path, artifact_name: str, dpi: int = 150) -> None:
    try:
        figure.savefig(path, dpi=dpi, bbox_inches="tight")
    except Exception as error:
        raise RuntimeError(
            f"Failed to generate required artifact {artifact_name}: {error}"
        ) from error
    finally:
        plt.close(figure)
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Required artifact {artifact_name} was not created or is empty: {path}"
        )


def _track_predictions(results: dict) -> Dict[str, List[dict]]:
    tracks = defaultdict(list)
    for frame in results.get("frames", []):
        for face in frame.get("faces", []):
            probabilities = face.get("probabilities")
            if not isinstance(probabilities, dict):
                raise ValueError(
                    "Every predicted face must include its class probabilities"
                )
            if set(probabilities) != set(EMOTION_CLASSES):
                raise ValueError(
                    "Face probability keys do not match the configured model "
                    f"classes: expected {EMOTION_CLASSES}"
                )
            vector = np.asarray(
                [probabilities[name] for name in EMOTION_CLASSES],
                dtype=np.float64,
            )
            if vector.shape != (len(EMOTION_CLASSES),) or not np.isfinite(vector).all():
                raise ValueError("Face probabilities must be seven finite values")
            timestamp = float(face["timestamp_seconds"])
            if not np.isfinite(timestamp):
                raise ValueError("Prediction timestamps must be finite")
            tracks[str(face["track_id"])].append({
                "frame_index": int(face["frame_index"]),
                "timestamp_seconds": timestamp,
                "probabilities": vector,
            })

    for predictions in tracks.values():
        predictions.sort(
            key=lambda item: (item["timestamp_seconds"], item["frame_index"])
        )
    return dict(tracks)


def reconstruct_temporal_analysis(
    results: dict,
    alpha: float = 0.7,
    beta: float = 0.3,
) -> Tuple[dict, dict]:
    """Compute deterministic temporal signals independently for each face track."""
    tracks = _track_predictions(results)
    if not tracks:
        raise ValueError("Inference results contain no face predictions")

    track_data = {}
    arrays_by_track = {}
    for track_id, predictions in tracks.items():
        tracker = TemporalSignalTracker(alpha=alpha, beta=beta)
        records = [
            tracker.update(item["probabilities"], item["timestamp_seconds"])
            for item in predictions
        ]
        probabilities = np.stack([item["probabilities"] for item in predictions])
        timestamps = np.asarray(
            [item["timestamp_seconds"] for item in predictions],
            dtype=np.float64,
        )
        stress = compute_stress_trajectory(
            probabilities,
            alpha=alpha,
            beta=beta,
            smooth_window=min(7, len(predictions)),
        )
        transitions, persistence = summarize_temporal_records(records)
        track_data[track_id] = {
            "prediction_count": len(predictions),
            "start_seconds": float(timestamps[0]),
            "end_seconds": float(timestamps[-1]),
            "emotion_transitions": transitions,
            "transition_count": len(transitions),
            "persistence_runs": persistence,
            "probability_volatility_mean_half_l1": float(np.mean([
                record["probability_volatility"] for record in records
            ])),
            "negative_affect_volatility_mean": float(np.mean([
                record["negative_affect_volatility"] for record in records
            ])),
            "stress_proxy_mean": float(np.mean(stress["raw_trajectory"])),
            "stress_proxy_peak": float(np.max(stress["raw_trajectory"])),
            "time_steps": records,
        }
        arrays_by_track[track_id] = {
            "timestamps": timestamps,
            "probabilities": probabilities,
            "records": records,
            "stress": stress,
        }

    primary_track_id = min(
        tracks,
        key=lambda track_id: (-len(tracks[track_id]), int(track_id)),
    )
    analysis = {
        "method": {
            "scope": (
                "Signals are calculated independently per face track from saved "
                "predictions; plots show the track with the most predictions."
            ),
            "probability_volatility": "0.5 * L1(P_t - P_(t-1)); first prediction is 0",
            "negative_affect_indices": [0, 1, 2, 4],
            "negative_affect_volatility": "abs(r_t - r_(t-1)); first prediction is 0",
            "stress_proxy": (
                "alpha * negative_affect + beta * negative_affect_volatility"
            ),
            "alpha": float(alpha),
            "beta": float(beta),
            "learned_stress_prediction": False,
            "classification_accuracy": (
                "Not measured: no video ground-truth labels were supplied."
            ),
        },
        "primary_plot_track_id": primary_track_id,
        "tracks": track_data,
    }
    return analysis, arrays_by_track[primary_track_id]


def write_temporal_plots(
    primary_track: dict,
    temporal_analysis: dict,
    output_dir: str,
    include_temporal_plot: bool = True,
    dpi: int = 150,
) -> List[str]:
    """Write probability and stress plots, optionally adding full temporal signals."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    timestamps = primary_track["timestamps"]
    probabilities = primary_track["probabilities"]
    records = primary_track["records"]
    stress = primary_track["stress"]
    track_id = temporal_analysis["primary_plot_track_id"]
    generated = []

    figure, axis = plt.subplots(figsize=(14, 5))
    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    axis.step(
        timestamps,
        predicted,
        where="mid",
        color="#34495e",
        linewidth=1.2,
        alpha=0.8,
    )
    for class_index, emotion in enumerate(EMOTION_CLASSES):
        indices = np.flatnonzero(predicted == class_index)
        if indices.size:
            axis.scatter(
                timestamps[indices],
                predicted[indices],
                color=EMOTION_COLOURS[emotion],
                s=24,
                label=emotion.capitalize(),
                zorder=3,
            )
    axis.set_yticks(range(len(EMOTION_CLASSES)))
    axis.set_yticklabels([name.capitalize() for name in EMOTION_CLASSES])
    axis.set_ylim(-0.5, len(EMOTION_CLASSES) - 0.5)
    axis.set_xlabel("Time (seconds)")
    axis.set_ylabel("Predicted emotion")
    axis.set_title(f"Emotion timeline and confidence — face track {track_id}")
    confidence_axis = axis.twinx()
    confidence_axis.plot(
        timestamps,
        confidence,
        color="#2c3e50",
        alpha=0.55,
        linewidth=1.0,
        label="Confidence",
    )
    confidence_axis.set_ylim(0, 1)
    confidence_axis.set_ylabel("Confidence")
    handles, labels = axis.get_legend_handles_labels()
    confidence_handles, confidence_labels = (
        confidence_axis.get_legend_handles_labels()
    )
    axis.legend(
        handles + confidence_handles,
        labels + confidence_labels,
        loc="upper left",
        ncol=min(4, len(handles) + len(confidence_handles)),
        fontsize=8,
    )
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    _save_figure(
        figure,
        output_path / "emotion_timeline.png",
        "emotion_timeline.png",
        dpi=dpi,
    )
    generated.append("emotion_timeline.png")

    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    axes[0].fill_between(timestamps, stress["smoothed"], alpha=0.3, color="#e74c3c")
    axes[0].plot(timestamps, stress["smoothed"], color="#c0392b", linewidth=1.8)
    axes[0].axhline(
        y=stress["mean_stress"],
        color="gray",
        linestyle="--",
        label=f"Mean proxy = {stress['mean_stress']:.3f}",
    )
    axes[0].set_ylabel("Calculated stress proxy")
    axes[0].set_title(
        "Calculated stress proxy (not a validated stress measurement)"
    )
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[1].plot(timestamps, stress["r"], color="#2980b9")
    axes[1].set_ylabel("Negative affect")
    axes[1].set_ylim(0, 1)
    axes[2].plot(timestamps, stress["v"], color="#d35400")
    axes[2].set_ylabel("Negative-affect volatility")
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.grid(True, alpha=0.25)
    figure.tight_layout()
    _save_figure(
        figure,
        output_path / "stress_trajectory.png",
        "stress_trajectory.png",
        dpi=dpi,
    )
    generated.append("stress_trajectory.png")

    if not include_temporal_plot:
        return generated

    figure, axes = plt.subplots(3, 1, figsize=(14, 11), constrained_layout=True)
    image = axes[0].imshow(
        probabilities.T,
        aspect="auto",
        interpolation="nearest",
        cmap="YlOrRd",
        vmin=0,
        vmax=1,
        origin="lower",
        extent=[
            float(timestamps[0]),
            float(timestamps[-1]) if len(timestamps) > 1 else float(timestamps[0] + 1),
            -0.5,
            len(EMOTION_CLASSES) - 0.5,
        ],
    )
    axes[0].set_yticks(range(len(EMOTION_CLASSES)))
    axes[0].set_yticklabels([name.capitalize() for name in EMOTION_CLASSES])
    axes[0].set_ylabel("Emotion probability")
    axes[0].set_title(f"Emotion probabilities — face track {track_id}")
    figure.colorbar(image, ax=axes[0], label="Probability")

    volatility = np.asarray(
        [record["probability_volatility"] for record in records],
        dtype=np.float64,
    )
    stress_proxy = np.asarray(
        [record["stress_proxy"] for record in records],
        dtype=np.float64,
    )
    axes[1].plot(timestamps, volatility, color="#2878a0", label="Full-probability volatility")
    axes[1].plot(timestamps, stress_proxy, color="#c44e52", label="Calculated stress proxy")
    for event in temporal_analysis["tracks"][track_id]["emotion_transitions"]:
        event_time = event["timestamp_seconds"]
        axes[1].axvline(event_time, color="#555555", alpha=0.25, linewidth=0.7)
    axes[1].set_ylabel("Signal value")
    axes[1].set_ylim(0, 1)
    axes[1].set_title("Volatility, calculated stress proxy, and transitions")
    axes[1].legend(loc="upper right")
    axes[1].grid(True, alpha=0.25)

    for run in temporal_analysis["tracks"][track_id]["persistence_runs"]:
        axes[2].barh(
            run["emotion"].capitalize(),
            run["duration_seconds"],
            left=run["start_seconds"],
            height=0.65,
            color=EMOTION_COLOURS.get(run["emotion"], "#777777"),
        )
    axes[2].set_xlabel("Time (seconds)")
    axes[2].set_ylabel("Dominant emotion")
    axes[2].set_title("Emotion persistence")
    axes[2].grid(True, axis="x", alpha=0.25)
    _save_figure(
        figure,
        output_path / "temporal_analysis.png",
        "temporal_analysis.png",
        dpi=dpi,
    )
    generated.append("temporal_analysis.png")
    return generated


def write_empty_temporal_plots(output_dir: str, reason: str) -> List[str]:
    """Create required, explicitly data-empty plots when no face was predicted."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    generated = []
    titles = (
        ("emotion_timeline.png", "Emotion timeline"),
        ("stress_trajectory.png", "Calculated stress-proxy trajectory"),
        ("temporal_analysis.png", "Temporal analysis"),
    )
    for filename, title in titles:
        figure, axis = plt.subplots(figsize=(12, 4))
        axis.set_title(title)
        axis.text(
            0.5,
            0.5,
            f"No prediction data available\n{reason}",
            horizontalalignment="center",
            verticalalignment="center",
            transform=axis.transAxes,
        )
        axis.set_axis_off()
        _save_figure(figure, output_path / filename, filename)
        generated.append(filename)
    return generated
