import unittest

from mania_difficulty.labels import compute_accuracy_labels


class AccuracyLabelTests(unittest.TestCase):
    def test_compute_accuracy_labels_returns_distribution_descriptors(self):
        labels = compute_accuracy_labels([0.90, 0.95, 0.80, 1.00, 0.85])

        self.assertEqual(round(labels.mean_acc, 4), 0.9000)
        self.assertGreater(labels.acc_std, 0)
        self.assertEqual(round(labels.skill_gap, 4), round(1.00 - ((0.85 + 0.80) / 2), 4))
        self.assertEqual(labels.score_count, 5)


if __name__ == "__main__":
    unittest.main()
