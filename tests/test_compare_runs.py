import json
import tempfile
import unittest
from pathlib import Path

from mania_difficulty.tools.compare_runs import run_metrics_rows


class CompareRunsTests(unittest.TestCase):
    def test_run_metrics_rows_includes_holdout_and_cv_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run_a"
            run_dir.mkdir()
            metrics = {
                "mean_acc": {
                    "mae": 0.2,
                    "r2": 0.1,
                    "spearman": 0.8,
                    "pairwise_order_accuracy": 0.75,
                    "baseline_mae": 0.3,
                    "mae_improvement_pct": 0.333,
                    "difficulty_rating_baseline_mae": 0.25,
                    "mae_improvement_pct_vs_difficulty_rating_baseline": 0.2,
                },
                "_run": {
                    "model_name": "summary",
                    "evaluation": "holdout",
                    "seed": 42,
                    "device": "cuda",
                    "requested_device": "auto",
                    "amp_enabled": True,
                    "effective_batch_size": 64,
                    "grad_accum_steps": 2,
                    "grad_clip_norm": 0.75,
                    "loss": "huber",
                    "huber_delta": 0.5,
                    "sample_weight_column": "score_count",
                    "sample_weight_train_mean": 0.72,
                    "sample_weight_train_downweighted_rate": 0.4,
                    "checkpoint_metric": "val_mean_mae",
                    "best_checkpoint_score": 0.12,
                    "epochs_completed": 3,
                    "stop_reason": "early_stopping",
                    "git_commit": "abc1234",
                    "git_dirty": False,
                },
            }
            cv_metrics = {
                "mean_acc": {
                    "mae": 0.25,
                    "r2": 0.05,
                    "baseline_mae": 0.31,
                    "mae_improvement_pct": 0.194,
                },
                "_run": {
                    "model_name": "tabular_forest",
                    "evaluation": "cv_oof",
                    "seed": 42,
                    "cv_folds": 3,
                },
            }
            (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
            (run_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics), encoding="utf-8")

            rows = run_metrics_rows(run_dir)

        self.assertEqual([row["evaluation"] for row in rows], ["holdout", "cv_oof"])
        self.assertEqual(rows[1]["cv_folds"], 3)
        self.assertEqual(rows[0]["baseline_mae"], 0.3)
        self.assertEqual(rows[0]["difficulty_rating_baseline_mae"], 0.25)
        self.assertEqual(rows[0]["mae_improvement_pct_vs_difficulty_rating_baseline"], 0.2)
        self.assertEqual(rows[0]["spearman"], 0.8)
        self.assertEqual(rows[0]["pairwise_order_accuracy"], 0.75)
        self.assertEqual(rows[0]["device"], "cuda")
        self.assertTrue(rows[0]["amp_enabled"])
        self.assertEqual(rows[0]["effective_batch_size"], 64)
        self.assertEqual(rows[0]["grad_accum_steps"], 2)
        self.assertEqual(rows[0]["grad_clip_norm"], 0.75)
        self.assertEqual(rows[0]["loss"], "huber")
        self.assertEqual(rows[0]["huber_delta"], 0.5)
        self.assertEqual(rows[0]["sample_weight_column"], "score_count")
        self.assertEqual(rows[0]["sample_weight_train_mean"], 0.72)
        self.assertEqual(rows[0]["sample_weight_train_downweighted_rate"], 0.4)
        self.assertEqual(rows[0]["checkpoint_metric"], "val_mean_mae")
        self.assertEqual(rows[0]["best_checkpoint_score"], 0.12)
        self.assertEqual(rows[0]["epochs_completed"], 3)
        self.assertEqual(rows[0]["stop_reason"], "early_stopping")
        self.assertEqual(rows[0]["git_commit"], "abc1234")
        self.assertFalse(rows[0]["git_dirty"])


if __name__ == "__main__":
    unittest.main()
