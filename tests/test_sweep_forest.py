import unittest
from types import SimpleNamespace

import numpy as np

from mania_difficulty.tools import sweep_forest
from mania_difficulty.tools.sweep_forest import choose_best_candidate, evaluate_candidate, forest_grid


class SweepForestTests(unittest.TestCase):
    def test_forest_grid_expands_all_parameter_combinations(self):
        candidates = forest_grid(
            trees=[50, 100],
            min_samples_leaf=[1, 3],
            max_features=["sqrt", 0.75],
        )

        self.assertEqual(len(candidates), 8)
        self.assertEqual(candidates[0]["forest_trees"], 50)
        self.assertEqual(candidates[0]["forest_min_samples_leaf"], 1)
        self.assertEqual(candidates[0]["forest_max_features"], "sqrt")

    def test_choose_best_candidate_prefers_lowest_mean_mae_then_simpler_model(self):
        rows = [
            {"candidate_id": "big", "mean_mae": 0.1, "forest_trees": 300, "forest_min_samples_leaf": 1},
            {"candidate_id": "small", "mean_mae": 0.1, "forest_trees": 100, "forest_min_samples_leaf": 3},
            {"candidate_id": "bad", "mean_mae": 0.2, "forest_trees": 50, "forest_min_samples_leaf": 1},
        ]

        best = choose_best_candidate(rows)

        self.assertEqual(best["candidate_id"], "small")

    def test_choose_best_candidate_can_optimize_pairwise_order_accuracy(self):
        rows = [
            {
                "candidate_id": "low_mae",
                "mean_mae": 0.05,
                "mean_pairwise_order_accuracy": 0.6,
                "forest_trees": 50,
                "forest_min_samples_leaf": 2,
            },
            {
                "candidate_id": "better_order",
                "mean_mae": 0.08,
                "mean_pairwise_order_accuracy": 0.9,
                "forest_trees": 100,
                "forest_min_samples_leaf": 2,
            },
        ]

        best = choose_best_candidate(rows, selection_metric="mean_pairwise_order_accuracy")

        self.assertEqual(best["candidate_id"], "better_order")

    def test_evaluate_candidate_passes_fold_sample_weights_when_provided(self):
        class Recorder:
            def __init__(self):
                self.fit_kwargs = None
                self.y_mean = None

            def fit(self, x, y, **kwargs):
                self.fit_kwargs = kwargs
                self.y_mean = np.mean(y, axis=0)

            def predict(self, x):
                return np.tile(self.y_mean, (len(x), 1))

        recorders = []

        def fake_model(args, *, seed):
            model = Recorder()
            recorders.append(model)
            return model

        original_model_factory = sweep_forest.create_tabular_forest_model
        sweep_forest.create_tabular_forest_model = fake_model
        try:
            evaluate_candidate(
                SimpleNamespace(workers=0, seed=42),
                {
                    "candidate_id": "weighted",
                    "feature_set": "core",
                    "forest_trees": 50,
                    "forest_min_samples_leaf": 2,
                    "forest_max_features": "sqrt",
                },
                np.asarray([[0.9], [0.8], [0.7], [0.6]], dtype="float32"),
                ["mean_acc"],
                [([0, 1], [2, 3]), ([2, 3], [0, 1])],
                {"core": np.ones((4, 2), dtype="float32")},
                sample_weights=np.asarray([1.0, 0.5, 0.25, 0.75], dtype="float32"),
            )
        finally:
            sweep_forest.create_tabular_forest_model = original_model_factory

        self.assertEqual(recorders[0].fit_kwargs["sample_weight"].tolist(), [1.0, 0.5])
        self.assertEqual(recorders[1].fit_kwargs["sample_weight"].tolist(), [0.25, 0.75])


if __name__ == "__main__":
    unittest.main()
