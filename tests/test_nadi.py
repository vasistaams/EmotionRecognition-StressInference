import unittest

import numpy as np

from utils.stress_formulation import EMOTION_CLASSES, NEGATIVE_EMOTION_INDICES
from utils.temporal_analysis import TemporalSignalTracker, compute_nadi


class NadiTests(unittest.TestCase):
    def test_class_order_and_negative_indices_match_fer2013(self):
        self.assertEqual(
            EMOTION_CLASSES,
            [
                "anger",
                "disgust",
                "fear",
                "happiness",
                "sadness",
                "surprise",
                "neutral",
            ],
        )
        self.assertEqual(NEGATIVE_EMOTION_INDICES, [0, 1, 2, 4])

    def test_hand_computed_nadi_and_tracker_record(self):
        previous = np.array([0.1, 0.1, 0.0, 0.4, 0.0, 0.2, 0.2])
        current = np.array([0.2, 0.1, 0.1, 0.3, 0.1, 0.1, 0.1])
        tracker = TemporalSignalTracker()
        first = tracker.update(previous, 0.0)
        second = tracker.update(current, 1.0)

        self.assertAlmostEqual(first["nadi"], 0.14)
        self.assertAlmostEqual(second["negative_affect"], 0.5)
        self.assertAlmostEqual(second["negative_affect_volatility"], 0.3)
        self.assertAlmostEqual(second["nadi"], 0.44)
        self.assertEqual(second["stress_proxy"], second["nadi"])
        self.assertAlmostEqual(compute_nadi(0.5, 0.2), 0.44)

    def test_custom_weights_and_nonfinite_inputs(self):
        self.assertAlmostEqual(compute_nadi(0.5, 0.2, alpha=0.6, beta=0.4), 0.42)
        with self.assertRaises(ValueError):
            compute_nadi(float("nan"), 0.2)


if __name__ == "__main__":
    unittest.main()
