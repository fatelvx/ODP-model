import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from mania_difficulty.models.transformer import TransformerDifficultyModel
from mania_difficulty.tools.attention_map import write_attention_map


class AttentionMapTests(unittest.TestCase):
    def test_write_attention_map_exports_note_csv_png_html_and_refreshes_run_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequences_dir = root / "sequences"
            checkpoint_path = root / "best_model.pt"
            sequences_dir.mkdir()

            beatmap_id = 123
            sequence = np.asarray(
                [
                    [0.0, 0.0, 0.0, 0.0, 0.00, 0.25],
                    [0.2, 0.5, 0.3333, 0.0, 0.00, 0.50],
                    [0.4, 0.5, 0.6667, 1.0, 0.25, 0.25],
                    [0.7, 0.8, 1.0, 0.0, 0.00, 0.75],
                ],
                dtype="float32",
            )
            np.save(sequences_dir / f"{beatmap_id}.npy", sequence)

            model = TransformerDifficultyModel(
                embed_dim=16,
                num_heads=4,
                num_layers=1,
                ff_dim=32,
                output_dim=3,
                max_positions=16,
                dropout=0.0,
                head_dropout=0.0,
            )
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": "transformer",
                    "model_config": model.config,
                    "target_columns": ["mean_acc", "acc_std", "skill_gap"],
                    "target_mean": [0.0, 0.0, 0.0],
                    "target_std": [1.0, 1.0, 1.0],
                    "max_notes": 16,
                },
                checkpoint_path,
            )

            out_csv = root / "attention_map.csv"
            out_png = root / "attention_map.png"
            out_html = root / "attention_report.html"
            write_attention_map(
                checkpoint_path=checkpoint_path,
                beatmap_id=beatmap_id,
                sequences_dir=sequences_dir,
                out_csv=out_csv,
                out_png=out_png,
                out_html=out_html,
                device_name="cpu",
            )

            frame = pd.read_csv(out_csv)
            report = out_html.read_text(encoding="utf-8")
            run_report = (root / "run_report.html").read_text(encoding="utf-8")
            png_exists = out_png.exists()

        self.assertEqual(len(frame), 4)
        self.assertIn("attention", frame.columns)
        self.assertIn("column", frame.columns)
        self.assertIn("time_fraction", frame.columns)
        self.assertAlmostEqual(float(frame["attention"].sum()), 1.0, places=5)
        self.assertTrue(png_exists)
        self.assertIn("Transformer Attention Map", report)
        self.assertIn("attention_map.csv", report)
        self.assertIn("Transformer Attention Map", run_report)


if __name__ == "__main__":
    unittest.main()
