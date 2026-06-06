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


if __name__ == "__main__":
    unittest.main()
