import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mania_difficulty.tools.audit_dataset import audit_dataset, write_html_report


class AuditDatasetTests(unittest.TestCase):
    def test_audit_dataset_counts_missing_sequences_and_target_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "labels.csv"
            sequences_dir = tmp_path / "sequences"
            sequences_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "beatmapset_id": 10,
                        "mean_acc": 0.90,
                        "acc_std": 0.02,
                        "skill_gap": 0.05,
                        "score_count": 100,
                    },
                    {
                        "beatmap_id": 2,
                        "beatmapset_id": 10,
                        "mean_acc": 0.80,
                        "acc_std": 0.03,
                        "skill_gap": 0.07,
                        "score_count": 70,
                    },
                    {
                        "beatmap_id": 3,
                        "beatmapset_id": 11,
                        "mean_acc": 0.70,
                        "acc_std": 0.04,
                        "skill_gap": 0.09,
                        "score_count": 40,
                    },
                ]
            ).to_csv(labels_csv, index=False)
            np.save(sequences_dir / "1.npy", np.zeros((5, 6), dtype=np.float32))
            np.save(sequences_dir / "2.npy", np.zeros((9, 6), dtype=np.float32))

            summary, missing = audit_dataset(labels_csv, sequences_dir, max_notes=6)

        self.assertEqual(summary["label_rows"], 3)
        self.assertEqual(summary["usable_rows"], 2)
        self.assertEqual(summary["missing_sequence_count"], 1)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["sequence_length"]["max"], 9)
        self.assertEqual(summary["sequence_truncation"]["max_notes"], 6)
        self.assertEqual(summary["sequence_truncation"]["truncated_rows"], 1)
        self.assertAlmostEqual(summary["sequence_truncation"]["truncated_rate"], 0.5)
        self.assertAlmostEqual(summary["targets"]["mean_acc"]["mean"], 0.85)
        self.assertEqual(summary["label_reliability"]["full_top100_rows"], 1)
        self.assertAlmostEqual(summary["label_reliability"]["full_top100_rate"], 0.5)
        self.assertEqual(summary["label_reliability"]["low_score_count_threshold"], 80)
        self.assertEqual(summary["label_reliability"]["low_score_count_rows"], 1)
        self.assertAlmostEqual(summary["label_reliability"]["low_score_count_rate"], 0.5)
        warning_codes = {warning["code"] for warning in summary["quality_warnings"]}
        self.assertIn("missing_sequences", warning_codes)
        self.assertIn("small_usable_dataset", warning_codes)
        self.assertIn("low_full_top100_rate", warning_codes)
        self.assertIn("high_low_score_count_rate", warning_codes)
        self.assertIn("sequence_truncation", warning_codes)
        self.assertEqual(missing[0]["beatmap_id"], 3)

    def test_audit_html_report_includes_label_reliability(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp)
            summary = {
                "target_missing": [],
                "label_rows": 2,
                "usable_rows": 2,
                "missing_sequence_count": 0,
                "extra_sequence_count": 0,
                "duplicate_beatmap_id_count": 0,
                "group_count": 1,
                "targets": {},
                "sequence_length": {"count": 2, "min": 5, "median": 7, "mean": 7, "max": 9, "std": 2},
                "sequence_truncation": {
                    "max_notes": 6,
                    "truncated_rows": 1,
                    "truncated_rate": 0.5,
                    "max_notes_over_limit": 3,
                },
                "score_count": {"count": 2, "min": 70, "median": 85, "mean": 85, "max": 100, "std": 15},
                "quality_warnings": [
                    {
                        "code": "low_full_top100_rate",
                        "severity": "warning",
                        "message": "Only 50.00% of usable maps have full top100 score coverage.",
                    }
                ],
                "label_reliability": {
                    "score_count_available": True,
                    "low_score_count_threshold": 80,
                    "low_score_count_rows": 1,
                    "low_score_count_rate": 0.5,
                    "full_top100_rows": 1,
                    "full_top100_rate": 0.5,
                },
            }

            write_html_report(summary, out_dir)
            html = (out_dir / "dataset_audit.html").read_text(encoding="utf-8")

        self.assertIn("Label Reliability", html)
        self.assertIn("Full Top100 Rate", html)
        self.assertIn("Low Score Count Rate", html)
        self.assertIn("Dataset Quality Warnings", html)
        self.assertIn("low_full_top100_rate", html)
        self.assertIn("Only 50.00%", html)
        self.assertIn("Sequence Truncation", html)
        self.assertIn("Truncated Rate", html)


if __name__ == "__main__":
    unittest.main()
