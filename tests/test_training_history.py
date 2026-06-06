import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.train import append_history, write_history_header


class TrainingHistoryTests(unittest.TestCase):
    def test_history_writer_records_training_performance_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"

            write_history_header(path)
            append_history(
                path,
                1,
                0.5,
                0.4,
                val_mean_mae=0.12,
                val_mean_pairwise_order_accuracy=0.75,
                epoch_seconds=1.25,
                lr=0.001,
                cuda_max_memory_mb=123.5,
            )

            history = pd.read_csv(path)

        self.assertIn("val_mean_mae", history.columns)
        self.assertIn("val_mean_pairwise_order_accuracy", history.columns)
        self.assertIn("epoch_seconds", history.columns)
        self.assertIn("lr", history.columns)
        self.assertIn("cuda_max_memory_mb", history.columns)
        self.assertEqual(float(history.loc[0, "val_mean_mae"]), 0.12)
        self.assertEqual(float(history.loc[0, "val_mean_pairwise_order_accuracy"]), 0.75)
        self.assertEqual(float(history.loc[0, "epoch_seconds"]), 1.25)
        self.assertEqual(float(history.loc[0, "lr"]), 0.001)
        self.assertEqual(float(history.loc[0, "cuda_max_memory_mb"]), 123.5)

    def test_history_writer_migrates_older_three_column_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "history.csv"
            path.write_text(
                "epoch,train_loss,val_loss\n1,0.5,0.4\n",
                encoding="utf-8",
            )

            append_history(path, 2, 0.3, 0.35, epoch_seconds=2.0, lr=0.0005)

            history = pd.read_csv(path)

        self.assertEqual(list(history.columns), [
            "epoch",
            "train_loss",
            "val_loss",
            "val_mean_mae",
            "val_mean_pairwise_order_accuracy",
            "epoch_seconds",
            "lr",
            "cuda_max_memory_mb",
        ])
        self.assertEqual(len(history), 2)
        self.assertTrue(pd.isna(history.loc[0, "val_mean_mae"]))
        self.assertTrue(pd.isna(history.loc[0, "val_mean_pairwise_order_accuracy"]))
        self.assertTrue(pd.isna(history.loc[0, "epoch_seconds"]))
        self.assertEqual(float(history.loc[1, "epoch_seconds"]), 2.0)


if __name__ == "__main__":
    unittest.main()
