"""Online temporal signals derived from seven-class emotion probabilities."""

from typing import Dict, List, Optional, Tuple

import numpy as np

from utils.stress_formulation import EMOTION_CLASSES, NEGATIVE_EMOTION_INDICES


class TemporalSignalTracker:
    """Track emotion transitions, persistence, volatility, and proxy stress."""

    def __init__(self, alpha: float = 0.7, beta: float = 0.3):
        self.alpha = alpha
        self.beta = beta
        self.previous_probs: Optional[np.ndarray] = None
        self.previous_negative_affect: Optional[float] = None
        self.previous_emotion: Optional[str] = None
        self.run_start_seconds: Optional[float] = None
        self.last_timestamp: Optional[float] = None
        self.last_transition = "No transition yet"

    def update(self, probabilities: np.ndarray, timestamp_seconds: float) -> Dict:
        """Add one frame prediction and return its temporal measurements."""
        probs = np.asarray(probabilities, dtype=np.float64)
        if probs.shape != (len(EMOTION_CLASSES),):
            raise ValueError(
                f"Expected {len(EMOTION_CLASSES)} class probabilities; got {probs.shape}"
            )
        if not np.isfinite(probs).all():
            raise ValueError("Probabilities must contain only finite values")

        timestamp = float(timestamp_seconds)
        if not np.isfinite(timestamp):
            raise ValueError("Timestamp must be finite")
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            raise ValueError("Timestamps must be non-decreasing")

        emotion = EMOTION_CLASSES[int(np.argmax(probs))]
        transition = None
        if self.previous_emotion is None:
            self.run_start_seconds = timestamp
        elif emotion != self.previous_emotion:
            transition = f"{self.previous_emotion.upper()} -> {emotion.upper()}"
            self.last_transition = transition
            self.run_start_seconds = timestamp

        negative_affect = float(probs[NEGATIVE_EMOTION_INDICES].sum())
        if self.previous_probs is None:
            volatility = 0.0
            negative_volatility = 0.0
        else:
            volatility = float(0.5 * np.abs(probs - self.previous_probs).sum())
            negative_volatility = abs(
                negative_affect - self.previous_negative_affect
            )

        stress_proxy = (
            self.alpha * negative_affect
            + self.beta * negative_volatility
        )
        persistence = max(0.0, timestamp - self.run_start_seconds)

        record = {
            "timestamp_seconds": timestamp,
            "dominant_emotion": emotion,
            "emotion_probabilities": {
                name: float(probs[index])
                for index, name in enumerate(EMOTION_CLASSES)
            },
            "transition": transition,
            "last_transition": self.last_transition,
            "persistence_seconds": float(persistence),
            "probability_volatility": volatility,
            "negative_affect": negative_affect,
            "negative_affect_volatility": float(negative_volatility),
            "stress_proxy": float(stress_proxy),
        }

        self.previous_probs = probs.copy()
        self.previous_negative_affect = negative_affect
        self.previous_emotion = emotion
        self.last_timestamp = timestamp
        return record


def summarize_temporal_records(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Return transition events and per-run persistence durations."""
    transition_events = [
        {
            "timestamp_seconds": record["timestamp_seconds"],
            "transition": record["transition"],
        }
        for record in records
        if record["transition"] is not None
    ]

    if not records:
        return transition_events, []

    timestamps = np.asarray(
        [record["timestamp_seconds"] for record in records], dtype=np.float64
    )
    intervals = np.diff(timestamps)
    positive_intervals = intervals[intervals > 0]
    final_interval = (
        float(np.median(positive_intervals)) if positive_intervals.size else 0.0
    )

    runs = []
    start = 0
    for index in range(1, len(records) + 1):
        at_end = index == len(records)
        changed = (
            not at_end
            and records[index]["dominant_emotion"]
            != records[start]["dominant_emotion"]
        )
        if not at_end and not changed:
            continue

        end_seconds = (
            timestamps[index]
            if changed
            else timestamps[index - 1] + final_interval
        )
        runs.append({
            "emotion": records[start]["dominant_emotion"],
            "start_seconds": float(timestamps[start]),
            "end_seconds": float(end_seconds),
            "duration_seconds": float(max(0.0, end_seconds - timestamps[start])),
            "frame_count": index - start,
        })
        start = index

    return transition_events, runs