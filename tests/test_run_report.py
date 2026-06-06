import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.visualize import write_run_report


class RunReportTests(unittest.TestCase):
    def test_run_report_embeds_pairwise_human_review_preview(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            pd.DataFrame(
                [
                    {
                        "review_reason": "pairwise_rank_disagreement",
                        "model_harder_beatmap_id": 1,
                        "observed_harder_beatmap_id": 2,
                        "disagreement_strength": 0.25,
                    }
                ]
            ).to_csv(run_dir / "human_pair_review.csv", index=False)
            (run_dir / "embedding_projection.png").write_bytes(b"png")
            (run_dir / "embedding_report.html").write_text("<html>embeddings</html>", encoding="utf-8")
            (run_dir / "attention_map.png").write_bytes(b"png")
            (run_dir / "attention_report.html").write_text("<html>attention</html>", encoding="utf-8")
            pd.DataFrame(
                [
                    {
                        "ranking_section": "predicted_hardest",
                        "rank": 1,
                        "beatmap_id": 1,
                        "title": "hard map",
                        "pred_mean_acc": 0.75,
                    }
                ]
            ).to_csv(run_dir / "prediction_rankings.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "slice_column": "overall",
                        "slice_value": "all",
                        "count": 4,
                        "actual_mean": 0.8,
                        "pred_mean": 0.78,
                        "mae": 0.08,
                        "bias": -0.02,
                        "max_abs_error": 0.15,
                    },
                    {
                        "slice_column": "score_count",
                        "slice_value": "high",
                        "count": 2,
                        "actual_mean": 0.7,
                        "pred_mean": 0.9,
                        "mae": 0.25,
                        "bias": 0.2,
                        "max_abs_error": 0.3,
                    },
                    {
                        "slice_column": "num_notes",
                        "slice_value": "low",
                        "count": 2,
                        "actual_mean": 0.9,
                        "pred_mean": 0.85,
                        "mae": 0.05,
                        "bias": -0.05,
                        "max_abs_error": 0.06,
                    },
                ]
            ).to_csv(run_dir / "error_slices.csv", index=False)
            pd.DataFrame(
                [
                    {
                        "epoch": 1,
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "val_mean_mae": 0.18,
                        "val_mean_pairwise_order_accuracy": 0.55,
                        "epoch_seconds": 1.0,
                        "lr": 0.001,
                        "cuda_max_memory_mb": 100.0,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.3,
                        "val_loss": 0.35,
                        "val_mean_mae": 0.12,
                        "val_mean_pairwise_order_accuracy": 0.72,
                        "epoch_seconds": 2.0,
                        "lr": 0.0005,
                        "cuda_max_memory_mb": 150.0,
                    },
                    {
                        "epoch": 3,
                        "train_loss": 0.2,
                        "val_loss": 0.5,
                        "val_mean_mae": 0.14,
                        "val_mean_pairwise_order_accuracy": 0.68,
                        "epoch_seconds": 3.0,
                        "lr": 0.0001,
                        "cuda_max_memory_mb": 125.0,
                    },
                ]
            ).to_csv(run_dir / "history.csv", index=False)
            metrics_path = run_dir / "metrics.json"
            metrics_path.write_text(
                json.dumps(
                    {
                        "mean_acc": {
                            "mae": 0.1,
                            "r2": 0.2,
                            "spearman": 0.3,
                            "pairwise_order_accuracy": 0.4,
                            "baseline_mae": 0.2,
                            "mae_improvement_vs_baseline": 0.1,
                            "mae_improvement_pct": 0.5,
                            "difficulty_rating_baseline_mae": 0.15,
                            "mae_improvement_vs_difficulty_rating_baseline": 0.05,
                            "mae_improvement_pct_vs_difficulty_rating_baseline": 1 / 3,
                        },
                        "_run": {
                            "model_name": "summary",
                            "device": "cuda",
                            "requested_device": "auto",
                            "torch_version": "2.8.0+cu128",
                            "cuda_available": True,
                            "cuda_device_name": "NVIDIA T4",
                            "cuda_device_count": 1,
                            "amp": "auto",
                            "amp_enabled": False,
                            "batch_size": 8,
                            "grad_accum_steps": 2,
                            "effective_batch_size": 16,
                            "sample_weight_column": "score_count",
                            "sample_weight_min": 0.25,
                            "sample_weight_max_value": 100.0,
                            "checkpoint_metric": "val_mean_pairwise_order_accuracy",
                            "best_checkpoint_score": 0.72,
                            "best_epoch": 2,
                            "epochs_requested": 5,
                            "epochs_completed": 3,
                            "stop_reason": "early_stopping",
                            "early_stopped": True,
                            "patience_left": 0,
                            "resume": True,
                            "resumed_from_epoch": 2,
                            "checkpoint_backup_dir": "drive/checkpoints/run_a",
                            "restored_from_backup": True,
                        },
                    }
                ),
                encoding="utf-8",
            )

            write_run_report(run_dir, target_columns=["mean_acc"], metrics_path=metrics_path)

            report = (run_dir / "run_report.html").read_text(encoding="utf-8")

        self.assertIn("Pairwise Human Review", report)
        self.assertIn("pairwise_rank_disagreement", report)
        self.assertIn("Embedding Projection", report)
        self.assertIn("embedding_report.html", report)
        self.assertIn("Transformer Attention Map", report)
        self.assertIn("attention_report.html", report)
        self.assertIn("Prediction Rankings", report)
        self.assertIn("predicted_hardest", report)
        self.assertIn("Worst Error Slices", report)
        self.assertIn("score_count: high", report)
        self.assertIn("0.250000", report)
        self.assertIn("Device", report)
        self.assertIn("cuda", report)
        self.assertIn("Requested Device", report)
        self.assertIn("Torch Version", report)
        self.assertIn("2.8.0+cu128", report)
        self.assertIn("CUDA Available", report)
        self.assertIn("CUDA Device", report)
        self.assertIn("NVIDIA T4", report)
        self.assertIn("CUDA Device Count", report)
        self.assertIn("AMP Enabled", report)
        self.assertIn("False", report)
        self.assertIn("Grad Accum Steps", report)
        self.assertIn("Effective Batch Size", report)
        self.assertIn("16", report)
        self.assertIn("Sample Weight Column", report)
        self.assertIn("score_count", report)
        self.assertIn("Sample Weight Min", report)
        self.assertIn("Sample Weight Max Value", report)
        self.assertIn("Checkpoint Metric", report)
        self.assertIn("val_mean_pairwise_order_accuracy", report)
        self.assertIn("Best Checkpoint Score", report)
        self.assertIn("<tr><th>Best Checkpoint Score</th><td>72.00%</td></tr>", report)
        self.assertIn("Checkpoint Best Epoch", report)
        self.assertIn("Epochs Requested", report)
        self.assertIn("Epochs Completed", report)
        self.assertIn("Stop Reason", report)
        self.assertIn("early_stopping", report)
        self.assertIn("Early Stopped", report)
        self.assertIn("Patience Left", report)
        self.assertIn("Training Health", report)
        self.assertIn("Best Epoch", report)
        self.assertIn("Best Val MAE", report)
        self.assertIn("0.120000", report)
        self.assertIn("Best Val Pairwise Order", report)
        self.assertIn("72.00%", report)
        self.assertIn("Generalization Gap", report)
        self.assertIn("Overfit Signal", report)
        self.assertIn("Possible", report)
        self.assertIn("Resume", report)
        self.assertIn("Resumed From Epoch", report)
        self.assertIn("Checkpoint Backup Dir", report)
        self.assertIn("Restored From Backup", report)
        self.assertIn("Training Performance", report)
        self.assertIn("Average Epoch Seconds", report)
        self.assertIn("Peak CUDA Memory MB", report)
        self.assertIn("Model Verdict", report)
        self.assertIn("Targets Beating Baseline", report)
        self.assertIn("Targets Beating Difficulty Rating", report)
        self.assertIn("Mean Difficulty Rating Improvement", report)
        self.assertIn("Next Action", report)
        self.assertIn("ranking signal is weak", report)
        self.assertIn("Difficulty Rating Baseline MAE", report)
        self.assertIn("Difficulty Rating Improvement", report)
        self.assertIn("33.33%", report)
        self.assertIn("Mean Pairwise Order", report)
        self.assertIn("Weakest Target", report)


if __name__ == "__main__":
    unittest.main()
