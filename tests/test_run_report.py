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
                        "epoch": 1,
                        "train_loss": 0.5,
                        "val_loss": 0.4,
                        "epoch_seconds": 1.0,
                        "lr": 0.001,
                        "cuda_max_memory_mb": 100.0,
                    },
                    {
                        "epoch": 2,
                        "train_loss": 0.3,
                        "val_loss": 0.35,
                        "epoch_seconds": 2.0,
                        "lr": 0.0005,
                        "cuda_max_memory_mb": 150.0,
                    },
                    {
                        "epoch": 3,
                        "train_loss": 0.2,
                        "val_loss": 0.5,
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
                        "mean_acc": {"mae": 0.1, "r2": 0.2, "spearman": 0.3, "pairwise_order_accuracy": 0.4},
                        "_run": {
                            "model_name": "summary",
                            "amp": "auto",
                            "amp_enabled": False,
                            "batch_size": 8,
                            "grad_accum_steps": 2,
                            "effective_batch_size": 16,
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
        self.assertIn("AMP Enabled", report)
        self.assertIn("False", report)
        self.assertIn("Grad Accum Steps", report)
        self.assertIn("Effective Batch Size", report)
        self.assertIn("16", report)
        self.assertIn("Training Health", report)
        self.assertIn("Best Epoch", report)
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


if __name__ == "__main__":
    unittest.main()
