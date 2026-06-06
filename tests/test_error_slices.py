import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.error_analysis import write_error_slices


class ErrorSliceTests(unittest.TestCase):
    def test_write_error_slices_outputs_overall_and_metadata_bins(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "labels.csv"
            predictions_csv = tmp_path / "predictions.csv"
            out_csv = tmp_path / "error_slices.csv"

            pd.DataFrame(
                [
                    {"beatmap_id": 1, "score_count": 30, "num_notes": 100},
                    {"beatmap_id": 2, "score_count": 60, "num_notes": 200},
                    {"beatmap_id": 3, "score_count": 90, "num_notes": 300},
                    {"beatmap_id": 4, "score_count": 120, "num_notes": 400},
                ]
            ).to_csv(labels_csv, index=False)
            pd.DataFrame(
                [
                    {"beatmap_id": 1, "actual_mean_acc": 0.90, "pred_mean_acc": 0.85},
                    {"beatmap_id": 2, "actual_mean_acc": 0.80, "pred_mean_acc": 0.90},
                    {"beatmap_id": 3, "actual_mean_acc": 0.70, "pred_mean_acc": 0.60},
                    {"beatmap_id": 4, "actual_mean_acc": 0.60, "pred_mean_acc": 0.65},
                ]
            ).to_csv(predictions_csv, index=False)

            write_error_slices(out_csv, labels_csv, predictions_csv, target_column="mean_acc")
            rows = pd.read_csv(out_csv)

        overall = rows[(rows["slice_column"] == "overall") & (rows["slice_value"] == "all")].iloc[0]
        self.assertEqual(int(overall["count"]), 4)
        self.assertAlmostEqual(overall["mae"], 0.075)
        self.assertIn("score_count", set(rows["slice_column"]))
        self.assertIn("num_notes", set(rows["slice_column"]))
        self.assertIn("low", set(rows["slice_value"]))
        self.assertIn("high", set(rows["slice_value"]))


if __name__ == "__main__":
    unittest.main()
