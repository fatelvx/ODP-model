import unittest

from mania_difficulty.tools.sweep_forest import choose_best_candidate, forest_grid


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


if __name__ == "__main__":
    unittest.main()
