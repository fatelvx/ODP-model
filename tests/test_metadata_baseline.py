import unittest

import numpy as np
import pandas as pd

from mania_difficulty.train import metadata_linear_baseline_predictions


class MetadataBaselineTests(unittest.TestCase):
    def test_metadata_linear_baseline_predictions_fit_multi_target_line(self):
        labels = pd.DataFrame(
            [
                {"difficulty_rating": 1.0, "mean_acc": 0.95, "acc_std": 0.02},
                {"difficulty_rating": 2.0, "mean_acc": 0.90, "acc_std": 0.04},
                {"difficulty_rating": 3.0, "mean_acc": 0.85, "acc_std": 0.06},
                {"difficulty_rating": 4.0, "mean_acc": 0.80, "acc_std": 0.08},
            ]
        )

        baseline = metadata_linear_baseline_predictions(
            labels,
            train_indices=[0, 1, 2],
            eval_indices=[3],
            target_columns=["mean_acc", "acc_std"],
            metadata_column="difficulty_rating",
        )

        self.assertIsNotNone(baseline)
        np.testing.assert_allclose(baseline, np.asarray([[0.80, 0.08]], dtype=np.float32), atol=1e-6)

    def test_metadata_linear_baseline_predictions_skip_missing_or_constant_metadata(self):
        labels = pd.DataFrame(
            [
                {"difficulty_rating": 1.0, "mean_acc": 0.95},
                {"difficulty_rating": 1.0, "mean_acc": 0.90},
                {"difficulty_rating": 1.0, "mean_acc": 0.85},
            ]
        )

        baseline = metadata_linear_baseline_predictions(
            labels,
            train_indices=[0, 1],
            eval_indices=[2],
            target_columns=["mean_acc"],
            metadata_column="difficulty_rating",
        )

        self.assertIsNone(baseline)


if __name__ == "__main__":
    unittest.main()
