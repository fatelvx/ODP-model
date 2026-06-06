import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.train import write_prediction_rankings


class PredictionRankingsTests(unittest.TestCase):
    def test_write_prediction_rankings_exports_hardest_easiest_and_error_sections(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_csv = root / "labels.csv"
            predictions_csv = root / "predictions.csv"
            out_csv = root / "prediction_rankings.csv"
            pd.DataFrame(
                [
                    {"beatmap_id": 1, "title": "easy", "artist": "a", "mapper": "m", "version": "v"},
                    {"beatmap_id": 2, "title": "hard", "artist": "a", "mapper": "m", "version": "v"},
                    {"beatmap_id": 3, "title": "wrong", "artist": "a", "mapper": "m", "version": "v"},
                ]
            ).to_csv(labels_csv, index=False)
            pd.DataFrame(
                [
                    {"beatmap_id": 1, "actual_mean_acc": 0.95, "pred_mean_acc": 0.90, "error_mean_acc": -0.05},
                    {"beatmap_id": 2, "actual_mean_acc": 0.70, "pred_mean_acc": 0.60, "error_mean_acc": -0.10},
                    {"beatmap_id": 3, "actual_mean_acc": 0.88, "pred_mean_acc": 0.80, "error_mean_acc": -0.50},
                ]
            ).to_csv(predictions_csv, index=False)

            write_prediction_rankings(out_csv, labels_csv, predictions_csv, target_column="mean_acc", top_n=1)

            rankings = pd.read_csv(out_csv)

        self.assertEqual(rankings["ranking_section"].tolist(), [
            "predicted_hardest",
            "predicted_easiest",
            "largest_abs_error",
        ])
        self.assertEqual(rankings["beatmap_id"].tolist(), [2, 1, 3])
        self.assertEqual(rankings["rank"].tolist(), [1, 1, 1])
        self.assertIn("pred_mean_acc", rankings.columns)
        self.assertIn("abs_error_mean_acc", rankings.columns)


if __name__ == "__main__":
    unittest.main()
