import unittest

from mania_difficulty.tools.sweep_neural import choose_best_candidate, neural_grid


class SweepNeuralTests(unittest.TestCase):
    def test_neural_grid_expands_model_specific_parameter_combinations(self):
        candidates = neural_grid(
            models=["summary", "lstm"],
            lrs=[0.001],
            weight_decays=[0.0001],
            batch_sizes=[16],
            summary_hidden_dims=[64, 96],
            summary_dropouts=[0.1],
            lstm_embed_dims=[16],
            lstm_hidden_dims=[32, 64],
            lstm_layers=[1],
            lstm_dropouts=[0.0],
            lstm_head_dropouts=[0.2],
        )

        self.assertEqual(len(candidates), 4)
        self.assertEqual(candidates[0]["model"], "summary")
        self.assertEqual(candidates[0]["summary_hidden_dim"], 64)
        self.assertEqual(candidates[-1]["model"], "lstm")
        self.assertEqual(candidates[-1]["lstm_hidden_dim"], 64)

    def test_choose_best_candidate_prefers_lowest_mean_mae_then_smaller_model(self):
        rows = [
            {"candidate_id": "large", "mean_mae": 0.1, "model_size_score": 500},
            {"candidate_id": "small", "mean_mae": 0.1, "model_size_score": 100},
            {"candidate_id": "bad", "mean_mae": 0.2, "model_size_score": 10},
        ]

        best = choose_best_candidate(rows)

        self.assertEqual(best["candidate_id"], "small")

    def test_choose_best_candidate_can_optimize_pairwise_order_accuracy(self):
        rows = [
            {
                "candidate_id": "low_mae",
                "mean_mae": 0.05,
                "mean_pairwise_order_accuracy": 0.55,
                "model_size_score": 50,
            },
            {
                "candidate_id": "better_order",
                "mean_mae": 0.08,
                "mean_pairwise_order_accuracy": 0.9,
                "model_size_score": 80,
            },
        ]

        best = choose_best_candidate(rows, selection_metric="mean_pairwise_order_accuracy")

        self.assertEqual(best["candidate_id"], "better_order")


if __name__ == "__main__":
    unittest.main()
