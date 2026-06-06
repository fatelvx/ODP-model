import unittest
from pathlib import Path
from types import SimpleNamespace

from mania_difficulty.tools.sweep_neural import (
    candidate_train_args,
    choose_best_candidate,
    neural_grid,
    summarize_run,
)


class SweepNeuralTests(unittest.TestCase):
    def test_neural_grid_expands_model_specific_parameter_combinations(self):
        candidates = neural_grid(
            models=["summary", "lstm", "transformer"],
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
            transformer_embed_dims=[32],
            transformer_heads=[4],
            transformer_layers=[1],
            transformer_ff_dims=[64],
            transformer_dropouts=[0.1],
            transformer_head_dropouts=[0.2],
        )

        self.assertEqual(len(candidates), 5)
        self.assertEqual(candidates[0]["model"], "summary")
        self.assertEqual(candidates[0]["summary_hidden_dim"], 64)
        self.assertEqual(candidates[-1]["model"], "transformer")
        self.assertEqual(candidates[-1]["transformer_embed_dim"], 32)

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

    def test_summarize_run_includes_gradient_clip_norm(self):
        summary, _ = summarize_run(
            {"candidate_id": "lstm", "model": "lstm"},
            {
                "mean_acc": {
                    "mae": 0.1,
                    "r2": 0.2,
                    "spearman": 0.3,
                    "pairwise_order_accuracy": 0.75,
                    "baseline_mae": 0.2,
                    "mae_improvement_pct": 0.5,
                },
                "_run": {
                    "grad_accum_steps": 2,
                    "grad_clip_norm": 0.75,
                    "effective_batch_size": 32,
                },
            },
            Path("outputs/runs/lstm"),
        )

        self.assertEqual(summary["grad_clip_norm"], 0.75)

    def test_candidate_train_args_passes_loader_options(self):
        base_args = SimpleNamespace(
            labels=Path("labels.csv"),
            sequences=Path("sequences"),
            run_prefix="sweep",
            targets="mean_acc,acc_std,skill_gap",
            epochs=2,
            patience=1,
            max_notes=800,
            group_column="beatmapset_id",
            seed=42,
            device="cuda",
            loss_weights=[1.0, 0.5, 0.5],
            sample_weight_column="score_count",
            sample_weight_min=0.25,
            sample_weight_max_value=100.0,
            workers=0,
            loader_workers=2,
            pin_memory="auto",
            loader_prefetch_factor=3,
            amp="auto",
            grad_accum_steps=2,
            grad_clip_norm=0.75,
            checkpoint_metric="val_mean_mae",
            checkpoint_backup_dir=Path("drive/checkpoints"),
            lstm_embed_dims=[32],
            lstm_hidden_dims=[64],
            lstm_layers=[1],
            lstm_dropouts=[0.1],
            lstm_head_dropouts=[0.2],
            summary_hidden_dims=[96],
            summary_dropouts=[0.1],
            transformer_embed_dims=[32],
            transformer_heads=[4],
            transformer_layers=[1],
            transformer_ff_dims=[64],
            transformer_dropouts=[0.1],
            transformer_head_dropouts=[0.2],
        )
        candidate = {
            "candidate_id": "summary",
            "model": "summary",
            "summary_hidden_dim": 96,
            "summary_dropout": 0.1,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 16,
        }

        train_args = candidate_train_args(base_args, candidate)

        self.assertEqual(train_args.loader_workers, 2)
        self.assertEqual(train_args.pin_memory, "auto")
        self.assertEqual(train_args.loader_prefetch_factor, 3)
        self.assertEqual(train_args.amp, "auto")
        self.assertEqual(train_args.grad_accum_steps, 2)
        self.assertEqual(train_args.grad_clip_norm, 0.75)
        self.assertEqual(train_args.checkpoint_metric, "val_mean_mae")
        self.assertEqual(train_args.checkpoint_backup_dir, Path("drive/checkpoints"))
        self.assertEqual(train_args.sample_weight_column, "score_count")
        self.assertEqual(train_args.sample_weight_min, 0.25)
        self.assertEqual(train_args.sample_weight_max_value, 100.0)

    def test_candidate_train_args_passes_transformer_options(self):
        base_args = SimpleNamespace(
            labels=Path("labels.csv"),
            sequences=Path("sequences"),
            run_prefix="sweep",
            targets="mean_acc,acc_std,skill_gap",
            epochs=2,
            patience=1,
            max_notes=800,
            group_column="beatmapset_id",
            seed=42,
            device="cuda",
            loss_weights=[1.0, 0.5, 0.5],
            workers=0,
            loader_workers=2,
            pin_memory="auto",
            loader_prefetch_factor=3,
            amp="auto",
            grad_accum_steps=2,
            checkpoint_metric="val_mean_mae",
            checkpoint_backup_dir=Path("drive/checkpoints"),
            lstm_embed_dims=[32],
            lstm_hidden_dims=[64],
            lstm_layers=[1],
            lstm_dropouts=[0.1],
            lstm_head_dropouts=[0.2],
            summary_hidden_dims=[96],
            summary_dropouts=[0.1],
            transformer_embed_dims=[32],
            transformer_heads=[4],
            transformer_layers=[1],
            transformer_ff_dims=[64],
            transformer_dropouts=[0.1],
            transformer_head_dropouts=[0.2],
        )
        candidate = {
            "candidate_id": "transformer",
            "model": "transformer",
            "transformer_embed_dim": 48,
            "transformer_heads": 6,
            "transformer_layers": 2,
            "transformer_ff_dim": 192,
            "transformer_dropout": 0.15,
            "transformer_head_dropout": 0.25,
            "lr": 0.001,
            "weight_decay": 0.0001,
            "batch_size": 16,
        }

        train_args = candidate_train_args(base_args, candidate)

        self.assertEqual(train_args.model, "transformer")
        self.assertEqual(train_args.transformer_embed_dim, 48)
        self.assertEqual(train_args.transformer_heads, 6)
        self.assertEqual(train_args.transformer_layers, 2)
        self.assertEqual(train_args.transformer_ff_dim, 192)
        self.assertEqual(train_args.transformer_dropout, 0.15)
        self.assertEqual(train_args.transformer_head_dropout, 0.25)


if __name__ == "__main__":
    unittest.main()
