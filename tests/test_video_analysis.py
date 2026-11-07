import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import cv2

from utils.stress_formulation import EMOTION_CLASSES
from utils.video_analysis import (
    reconstruct_temporal_analysis,
    write_empty_temporal_plots,
    write_temporal_plots,
)
from scripts.infer_video import draw_video_dashboard


def prediction(frame_index, timestamp, track_id, emotion_index):
    probabilities = np.zeros(len(EMOTION_CLASSES), dtype=float)
    probabilities[emotion_index] = 1.0
    return {
        "frame_index": frame_index,
        "timestamp_seconds": timestamp,
        "faces": [{
            "track_id": track_id,
            "frame_index": frame_index,
            "timestamp_seconds": timestamp,
            "probabilities": dict(zip(EMOTION_CLASSES, probabilities.tolist())),
        }],
    }


class VideoAnalysisTests(unittest.TestCase):
    def test_reconstructs_temporal_signals_per_track_from_timestamps(self):
        logged_results = {
            "frames": [
                prediction(0, 0.0, 1, 0),
                prediction(3, 0.5, 1, 3),
                prediction(6, 1.5, 1, 3),
                prediction(9, 2.0, 1, 6),
                prediction(0, 0.0, 2, 4),
            ]
        }
        analysis, primary = reconstruct_temporal_analysis(
            logged_results, alpha=0.7, beta=0.3
        )

        self.assertEqual(analysis["primary_plot_track_id"], "1")
        self.assertEqual(analysis["tracks"]["1"]["prediction_count"], 4)
        self.assertEqual(analysis["tracks"]["1"]["transition_count"], 2)
        self.assertEqual(analysis["tracks"]["2"]["transition_count"], 0)
        self.assertAlmostEqual(
            analysis["tracks"]["1"]["persistence_runs"][0]["duration_seconds"],
            0.5,
        )
        self.assertTrue(
            np.allclose(primary["timestamps"], [0.0, 0.5, 1.5, 2.0])
        )
        self.assertFalse(analysis["method"]["learned_stress_prediction"])
        json.dumps(analysis, allow_nan=False)

    def test_plot_generation_writes_expected_artifacts(self):
        logged_results = {
            "frames": [
                prediction(0, 0.0, 1, 0),
                prediction(3, 0.5, 1, 3),
                prediction(6, 1.0, 1, 4),
            ]
        }
        analysis, primary = reconstruct_temporal_analysis(logged_results)
        with tempfile.TemporaryDirectory() as temporary_dir:
            files = write_temporal_plots(primary, analysis, temporary_dir)
            self.assertEqual(
                files,
                [
                    "emotion_timeline.png",
                    "stress_trajectory.png",
                    "temporal_analysis.png",
                ],
            )
            for filename in files:
                path = Path(temporary_dir) / filename
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertIsNotNone(cv2.imread(str(path)))

    def test_no_prediction_plots_are_explicitly_empty_and_readable(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            files = write_empty_temporal_plots(
                temporary_dir,
                "No face detections.",
            )
            self.assertEqual(
                files,
                [
                    "emotion_timeline.png",
                    "stress_trajectory.png",
                    "temporal_analysis.png",
                ],
            )
            for filename in files:
                path = Path(temporary_dir) / filename
                self.assertTrue(path.is_file())
                self.assertGreater(path.stat().st_size, 0)
                self.assertIsNotNone(cv2.imread(str(path)))

    def test_dashboard_preserves_source_frame_and_adds_panel_and_footer(self):
        frame = np.full((80, 120, 3), (10, 20, 30), dtype=np.uint8)
        dashboard = draw_video_dashboard(
            frame,
            {
                "track_id": 4,
                "emotion": "anger",
                "confidence": 0.8,
                "last_transition": "No transition yet",
                "transition_count": 0,
                "persistence_seconds": 1.25,
                "probability_volatility_16": 0.15,
                "negative_affect": 0.7,
                "stress_proxy": 0.68,
            },
            elapsed_seconds=2.0,
            fps=30.0,
        )
        self.assertEqual(dashboard.shape, (80 + 150 + 42, 120, 3))
        self.assertTrue(
            np.array_equal(dashboard[150:230], frame)
        )
        self.assertTrue(np.any(dashboard[:150]))
        self.assertTrue(np.any(dashboard[230:]))


if __name__ == "__main__":
    unittest.main()
