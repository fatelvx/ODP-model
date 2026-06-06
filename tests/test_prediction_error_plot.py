import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.visualize import plot_prediction_errors


class PredictionErrorPlotTests(unittest.TestCase):
    def test_plot_prediction_errors_writes_residual_png(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            predictions_csv = root / "predictions.csv"
            out_png = root / "prediction_errors.png"
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "actual_mean_acc": 0.80,
                        "pred_mean_acc": 0.75,
                        "actual_acc_std": 0.10,
                        "pred_acc_std": 0.12,
                    },
                    {
                        "beatmap_id": 2,
                        "actual_mean_acc": 0.60,
                        "pred_mean_acc": 0.70,
                        "actual_acc_std": 0.30,
                        "pred_acc_std": 0.20,
                    },
                    {
                        "beatmap_id": 3,
                        "actual_mean_acc": 0.70,
                        "pred_mean_acc": 0.69,
                        "actual_acc_std": 0.20,
                        "pred_acc_std": 0.24,
                    },
                ]
            ).to_csv(predictions_csv, index=False)

            plot_prediction_errors(predictions_csv, ["mean_acc", "acc_std"], out_png)

            self.assertTrue(out_png.exists())
            self.assertGreater(out_png.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
