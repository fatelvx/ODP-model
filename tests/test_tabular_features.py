import unittest

import numpy as np

from mania_difficulty.models.tabular import (
    METADATA_FEATURE_NAMES,
    SUMMARY_FEATURE_NAMES,
    feature_names_for_set,
    metadata_features_from_row,
    summarize_sequence,
)


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

    def test_summarize_sequence_detects_density_and_jack_patterns(self):
        rows = []
        time_sec = 0.0
        previous_time = 0.0
        for index in range(20):
            if index < 10:
                time_sec += 0.5
                column = index % 4
                chord = 1
            else:
                time_sec += 0.05
                column = 0
                chord = 2
            rows.append(
                [
                    time_sec / 6.0,
                    time_sec - previous_time,
                    column / 3.0,
                    0.0,
                    0.0,
                    chord / 4.0,
                ]
            )
            previous_time = time_sec
        features = np.asarray(rows, dtype=np.float32)

        names = feature_names_for_set("burst")
        summary = summarize_sequence(features, feature_set="burst")

        self.assertGreater(summary[names.index("peak_notes_per_sec_1s")], 10)
        self.assertGreater(summary[names.index("jack_ratio_100ms")], 0.4)
        self.assertGreater(summary[names.index("burst_chord_2_ratio")], 0.3)

    def test_core_metadata_feature_set_appends_map_settings(self):
        features = np.asarray(
            [
                [0.00, 0.00, 0.00, 0.0, 0.0, 0.25],
                [1.00, 0.50, 1.00, 0.0, 0.0, 0.50],
            ],
            dtype=np.float32,
        )

        names = feature_names_for_set("core_metadata")
        summary = summarize_sequence(
            features,
            feature_set="core_metadata",
            metadata={
                "keys": "4",
                "hp_drain_rate": "6.5",
                "overall_difficulty": "8",
                "approach_rate": "5",
                "difficulty_rating": "6.2",
                "length_ms": "120000",
                "bpm": "180",
            },
        )

        self.assertEqual(summary.shape, (len(SUMMARY_FEATURE_NAMES) + len(METADATA_FEATURE_NAMES),))
        self.assertEqual(summary[names.index("metadata_keys")], 4.0)
        self.assertEqual(summary[names.index("metadata_hp_drain_rate")], 6.5)
        self.assertEqual(summary[names.index("metadata_overall_difficulty")], 8.0)
        self.assertEqual(summary[names.index("metadata_length_sec")], 120.0)

    def test_metadata_features_from_row_defaults_missing_values_to_zero(self):
        values = metadata_features_from_row({})

        self.assertEqual(values.shape, (len(METADATA_FEATURE_NAMES),))
        self.assertTrue(np.all(values == 0.0))


if __name__ == "__main__":
    unittest.main()
