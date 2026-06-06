import json
import tempfile
import unittest
from pathlib import Path

from mania_difficulty.tools.build_dashboard import write_dashboard


class BuildDashboardTests(unittest.TestCase):
    def test_write_dashboard_links_runs_reports_and_audit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run_dir = root / "outputs" / "runs" / "run_a"
            audit_dir = root / "outputs" / "dataset_audit"
            sweep_dir = root / "outputs" / "forest_sweep"
            run_dir.mkdir(parents=True)
            audit_dir.mkdir(parents=True)
            sweep_dir.mkdir(parents=True)
            (run_dir / "run_report.html").write_text("<html>run</html>", encoding="utf-8")
            (run_dir / "last_checkpoint.pt").write_bytes(b"checkpoint")
            (run_dir / "learning_curve.png").write_bytes(b"png")
            (run_dir / "prediction_scatter.png").write_bytes(b"png")
            (run_dir / "prediction_rankings.csv").write_text(
                "ranking_section,rank,beatmap_id\npredicted_hardest,1,1\n",
                encoding="utf-8",
            )
            (run_dir / "error_slices.csv").write_text(
                "\n".join(
                    [
                        "slice_column,slice_value,count,actual_mean,pred_mean,mae,bias,max_abs_error",
                        "overall,all,4,0.8,0.78,0.08,-0.02,0.15",
                        "score_count,high,2,0.7,0.9,0.25,0.2,0.3",
                        "num_notes,low,2,0.9,0.85,0.05,-0.05,0.06",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "embedding_projection.png").write_bytes(b"png")
            (run_dir / "embedding_report.html").write_text("<html>embeddings</html>", encoding="utf-8")
            (run_dir / "embedding_projection.csv").write_text(
                "beatmap_id,projection_x,projection_y\n1,0.0,1.0\n",
                encoding="utf-8",
            )
            (run_dir / "attention_map.png").write_bytes(b"png")
            (run_dir / "attention_report.html").write_text("<html>attention</html>", encoding="utf-8")
            (run_dir / "attention_map.csv").write_text(
                "note_index,attention\n0,1.0\n",
                encoding="utf-8",
            )
            (run_dir / "human_pair_judgment_template.csv").write_text(
                "\n".join(
                    [
                        "model_harder_beatmap_id,observed_harder_beatmap_id,human_harder_beatmap_id",
                        "11,22,11",
                        "33,44,44",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "history.csv").write_text(
                "\n".join(
                    [
                        (
                            "epoch,train_loss,val_loss,val_mean_mae,"
                            "val_mean_pairwise_order_accuracy,epoch_seconds,lr,cuda_max_memory_mb"
                        ),
                        "1,0.5,0.4,0.18,0.55,1.0,0.001,100.0",
                        "2,0.3,0.35,0.12,0.72,2.0,0.0005,150.0",
                        "3,0.2,0.5,0.14,0.68,3.0,0.0001,125.0",
                    ]
                ),
                encoding="utf-8",
            )
            (run_dir / "metrics.json").write_text(
                json.dumps(
                    {
                        "mean_acc": {
                            "mae": 0.1,
                            "spearman": 0.5,
                            "pairwise_order_accuracy": 0.75,
                            "difficulty_rating_baseline_mae": 0.15,
                            "mae_improvement_vs_difficulty_rating_baseline": 0.05,
                            "mae_improvement_pct_vs_difficulty_rating_baseline": 1 / 3,
                        },
                        "acc_std": {
                            "mae": 0.2,
                            "spearman": 0.4,
                            "pairwise_order_accuracy": 0.6,
                            "baseline_mae": 0.1,
                            "mae_improvement_vs_baseline": -0.1,
                            "mae_improvement_pct": -1.0,
                        },
                        "_run": {
                            "model_name": "summary",
                            "evaluation": "holdout",
                            "seed": 42,
                            "checkpoint_metric": "val_mean_pairwise_order_accuracy",
                            "best_checkpoint_score": 0.72,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (audit_dir / "dataset_audit.html").write_text("<html>audit</html>", encoding="utf-8")
            (audit_dir / "dataset_audit.json").write_text(
                json.dumps(
                    {
                        "usable_rows": 10,
                        "missing_sequence_count": 1,
                        "group_count": 3,
                        "label_reliability": {
                            "score_count_available": True,
                            "usable_score_count_rows": 10,
                            "low_score_count_threshold": 80,
                            "low_score_count_rows": 2,
                            "low_score_count_rate": 0.2,
                            "full_top100_rows": 6,
                            "full_top100_rate": 0.6,
                            "min_score_count": 50,
                            "median_score_count": 95,
                        },
                    }
                ),
                encoding="utf-8",
            )
            (sweep_dir / "sweep_report.html").write_text("<html>sweep</html>", encoding="utf-8")
            (sweep_dir / "best_params.json").write_text(
                json.dumps({"candidate_id": "best_forest", "selection_metric": "mean_mae"}),
                encoding="utf-8",
            )

            out_html = root / "outputs" / "dashboard.html"
            write_dashboard(
                out_html,
                run_dirs=[run_dir],
                audit_dir=audit_dir,
                forest_sweep_dir=sweep_dir,
            )

            html = out_html.read_text(encoding="utf-8")

        self.assertIn("run_a", html)
        self.assertIn("run_report.html", html)
        self.assertIn("last_checkpoint.pt", html)
        self.assertIn("prediction rankings", html)
        self.assertIn("prediction_rankings.csv", html)
        self.assertIn("learning_curve.png", html)
        self.assertIn("embedding_report.html", html)
        self.assertIn("embedding_projection.png", html)
        self.assertIn("attention_report.html", html)
        self.assertIn("attention_map.png", html)
        self.assertIn("mean_acc", html)
        self.assertIn("Dataset Audit", html)
        self.assertIn("Label Reliability", html)
        self.assertIn("Full Top100 Rate", html)
        self.assertIn("60.00%", html)
        self.assertIn("Low Score Count Rate", html)
        self.assertIn("20.00%", html)
        self.assertIn("best_forest", html)
        self.assertIn("Human Judgment Scores", html)
        self.assertIn("model_agreement_rate", html)
        self.assertIn("0.500000", html)
        self.assertIn("Training Health", html)
        self.assertIn("Best Epoch", html)
        self.assertIn("Best Val MAE", html)
        self.assertIn("Best Val Pairwise Order", html)
        self.assertIn("Overfit Signal", html)
        self.assertIn("Training Performance", html)
        self.assertIn("Peak CUDA Memory MB", html)
        self.assertIn("Model Verdict", html)
        self.assertIn("Targets Beating Baseline", html)
        self.assertIn("Targets Beating Difficulty Rating", html)
        self.assertIn("Mean Difficulty Rating Improvement", html)
        self.assertIn("Next Action", html)
        self.assertIn("Weakest Target", html)
        self.assertIn("Worst Error Slices", html)
        self.assertIn("score_count: high", html)
        self.assertIn("0.250000", html)
        self.assertIn("Checkpoint Selection", html)
        self.assertIn("Checkpoint Metric", html)
        self.assertIn("val_mean_pairwise_order_accuracy", html)
        self.assertIn("72.00%", html)


if __name__ == "__main__":
    unittest.main()
