import unittest

import numpy as np

from mania_difficulty.models.tabular import SUMMARY_FEATURE_NAMES, summarize_sequence


class TabularFeatureTests(unittest.TestCase):
    def test_summarize_sequence_returns_stable_numeric_vector(self):
        features = np.asarray(
            [
                [0.00, 0.00, 0.00, 0.0, 0.0, 0.50],
                [0.10, 0.10, 0.33, 1.0, 0.4, 0.25],
                [0.20, 0.10, 0.66, 0.0, 0.0, 0.75],
                [1.00, 0.80, 1.00, 0.0, 0.0, 1.00],
            ],
            dtype=np.float32,
        )

        summary = summarize_sequence(features)

        self.assertEqual(summary.shape, (len(SUMMARY_FEATURE_NAMES),))
        self.assertFalse(np.isnan(summary).any())
        self.assertGreater(summary[SUMMARY_FEATURE_NAMES.index("note_count_log")], 0)
        self.assertGreater(summary[SUMMARY_FEATURE_NAMES.index("ln_ratio")], 0)
        self.assertGreater(summary[SUMMARY_FEATURE_NAMES.index("chord_4_ratio")], 0)

    def test_summarize_sequence_handles_empty_input(self):
        summary = summarize_sequence(np.empty((0, 6), dtype=np.float32))

        self.assertEqual(summary.shape, (len(SUMMARY_FEATURE_NAMES),))
        self.assertTrue(np.all(summary == 0))


if __name__ == "__main__":
    unittest.main()
