import unittest

import numpy as np

from mania_difficulty.metrics import regression_report


class MetricsTests(unittest.TestCase):
    def test_regression_report_includes_baseline_improvement(self):
        actual = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        pred = np.asarray([[1.0], [2.0], [2.0]], dtype=np.float32)
        baseline = np.asarray([[1.0], [1.0], [1.0]], dtype=np.float32)

        report = regression_report(actual, pred, ["target"], baseline_pred=baseline)

        self.assertAlmostEqual(report["target"]["mae"], 1 / 3)
        self.assertAlmostEqual(report["target"]["baseline_mae"], 1.0)
        self.assertAlmostEqual(report["target"]["mae_improvement_vs_baseline"], 2 / 3)
        self.assertGreater(report["target"]["mae_improvement_pct"], 0)

    def test_regression_report_includes_named_baselines(self):
        actual = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        pred = np.asarray([[1.0], [2.0], [2.0]], dtype=np.float32)
        difficulty_baseline = np.asarray([[1.0], [2.0], [1.0]], dtype=np.float32)

        report = regression_report(
            actual,
            pred,
            ["target"],
            named_baselines={"difficulty_rating": difficulty_baseline},
        )

        self.assertAlmostEqual(report["target"]["difficulty_rating_baseline_mae"], 2 / 3)
        self.assertAlmostEqual(
            report["target"]["mae_improvement_vs_difficulty_rating_baseline"],
            1 / 3,
        )
        self.assertAlmostEqual(
            report["target"]["mae_improvement_pct_vs_difficulty_rating_baseline"],
            0.5,
        )

    def test_regression_report_includes_rank_and_pairwise_order_metrics(self):
        actual = np.asarray([[0.9], [0.8], [0.7], [0.6]], dtype=np.float32)
        good_pred = np.asarray([[0.91], [0.79], [0.71], [0.59]], dtype=np.float32)
        bad_pred = good_pred[::-1]

        good_report = regression_report(actual, good_pred, ["mean_acc"])
        bad_report = regression_report(actual, bad_pred, ["mean_acc"])

        self.assertAlmostEqual(good_report["mean_acc"]["spearman"], 1.0)
        self.assertAlmostEqual(good_report["mean_acc"]["pairwise_order_accuracy"], 1.0)
        self.assertAlmostEqual(bad_report["mean_acc"]["spearman"], -1.0)
        self.assertAlmostEqual(bad_report["mean_acc"]["pairwise_order_accuracy"], 0.0)


if __name__ == "__main__":
    unittest.main()
