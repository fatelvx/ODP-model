import argparse
import unittest

from mania_difficulty.tools.sweep_selection import (
    parse_selection_metric,
    parse_summary_selection_metric,
    selection_sort_ascending,
    selection_sort_value,
)


class SweepSelectionTests(unittest.TestCase):
    def test_selection_sort_value_treats_higher_rank_metrics_as_better(self):
        row = {"mean_pairwise_order_accuracy": 0.9, "mean_mae": 0.1}

        self.assertLess(selection_sort_value(row, "mean_pairwise_order_accuracy"), 0)
        self.assertEqual(selection_sort_value(row, "mean_mae"), 0.1)
        self.assertFalse(selection_sort_ascending("mean_pairwise_order_accuracy"))
        self.assertTrue(selection_sort_ascending("mean_mae"))

    def test_parse_selection_metric_rejects_unknown_metric(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_selection_metric("whatever")

    def test_parse_summary_selection_metric_rejects_neural_only_loss_metric(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_summary_selection_metric("best_val_loss")


if __name__ == "__main__":
    unittest.main()
