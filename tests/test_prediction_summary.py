import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.train import write_prediction_summary


class PredictionSummaryTests(unittest.TestCase):
    def test_write_prediction_summary_exports_target_bias_and_error_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions_csv = root / "predictions.csv"
            out_csv = root / "prediction_summary.csv"
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "actual_mean_acc": 0.8,
                        "pred_mean_acc": 0.7,
                        "actual_acc_std": 0.1,
                        "pred_acc_std": 0.2,
                    },
                    {
                        "beatmap_id": 2,
                        "actual_mean_acc": 0.6,
                        "pred_mean_acc": 0.7,
                        "actual_acc_std": 0.3,
                        "pred_acc_std": 0.1,
                    },
                ]
            ).to_csv(predictions_csv, index=False)

            write_prediction_summary(out_csv, predictions_csv, ["mean_acc", "acc_std"])

            summary = pd.read_csv(out_csv)

        self.assertEqual(summary["target"].tolist(), ["mean_acc", "acc_std"])
        self.assertEqual(summary["count"].tolist(), [2, 2])
        self.assertAlmostEqual(summary.loc[0, "actual_mean"], 0.7)
        self.assertAlmostEqual(summary.loc[0, "pred_mean"], 0.7)
        self.assertAlmostEqual(summary.loc[0, "bias"], 0.0)
        self.assertAlmostEqual(summary.loc[0, "mae"], 0.1)
        self.assertAlmostEqual(summary.loc[1, "bias"], -0.05)
        self.assertAlmostEqual(summary.loc[1, "mae"], 0.15)
        self.assertAlmostEqual(summary.loc[1, "max_abs_error"], 0.2)


if __name__ == "__main__":
    unittest.main()
