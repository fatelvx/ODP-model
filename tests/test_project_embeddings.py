import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mania_difficulty.models.lstm import LSTMDifficultyModel
from mania_difficulty.models.summary import SummaryDifficultyModel
from mania_difficulty.tools.project_embeddings import write_embedding_projection


class ProjectEmbeddingsTests(unittest.TestCase):
    def test_models_expose_pre_head_embeddings(self):
        features = torch.rand(2, 5, 6)
        lengths = torch.tensor([5, 3], dtype=torch.long)

        summary = SummaryDifficultyModel(hidden_dim=8, output_dim=3)
        lstm = LSTMDifficultyModel(embed_dim=4, hidden_dim=6, num_layers=1, output_dim=3)

        self.assertEqual(tuple(summary.encode(features, lengths).shape), (2, 4))
        self.assertEqual(tuple(lstm.encode(features, lengths).shape), (2, 12))

    def test_write_embedding_projection_exports_csv_png_and_html(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels_csv = root / "labels.csv"
            sequences_dir = root / "sequences"
            checkpoint_path = root / "best_model.pt"
            sequences_dir.mkdir()

            labels = pd.DataFrame(
                [
                    {"beatmap_id": 1, "mean_acc": 0.92, "acc_std": 0.02, "skill_gap": 0.12},
                    {"beatmap_id": 2, "mean_acc": 0.86, "acc_std": 0.04, "skill_gap": 0.18},
                    {"beatmap_id": 3, "mean_acc": 0.78, "acc_std": 0.06, "skill_gap": 0.24},
                ]
            )
            labels.to_csv(labels_csv, index=False)
            for beatmap_id in labels["beatmap_id"]:
                np.save(sequences_dir / f"{beatmap_id}.npy", np.random.rand(8, 6).astype("float32"))

            model = SummaryDifficultyModel(hidden_dim=8, output_dim=3)
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "summary",
                    "model_config": model.config,
                    "target_columns": ["mean_acc", "acc_std", "skill_gap"],
                    "target_mean": [0.0, 0.0, 0.0],
                    "target_std": [1.0, 1.0, 1.0],
                    "max_notes": 3000,
                },
                checkpoint_path,
            )

            out_csv = root / "embedding_projection.csv"
            out_png = root / "embedding_projection.png"
            out_html = root / "embedding_report.html"
            write_embedding_projection(
                checkpoint_path=checkpoint_path,
                labels_csv=labels_csv,
                sequences_dir=sequences_dir,
                out_csv=out_csv,
                out_png=out_png,
                out_html=out_html,
                method="pca",
                device_name="cpu",
            )

            projection = pd.read_csv(out_csv)
            report = out_html.read_text(encoding="utf-8")
            png_exists = out_png.exists()

        self.assertIn("projection_x", projection.columns)
        self.assertIn("projection_y", projection.columns)
        self.assertIn("embedding_norm", projection.columns)
        self.assertEqual(len(projection), 3)
        self.assertTrue(png_exists)
        self.assertIn("Embedding Projection", report)
        self.assertIn("embedding_projection.csv", report)


if __name__ == "__main__":
    unittest.main()
