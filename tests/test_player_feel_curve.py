import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mania_difficulty.tools.player_feel_curve import (
    compute_player_feel_curve,
    summarize_player_feel_curve,
    write_player_feel_curves,
)


class PlayerFeelCurveTests(unittest.TestCase):
    def test_compute_player_feel_curve_detects_jack_chord_and_ln_pressure(self):
        sequence = np.asarray(
            [
                [0.00, 0.00, 0.0, 0.0, 0.0, 0.25],
                [0.05, 0.20, 0.0, 0.0, 0.0, 0.25],
                [0.10, 0.20, 0.0, 0.0, 0.0, 0.25],
                [0.25, 0.60, 1.0, 0.0, 0.0, 0.50],
                [0.25, 0.00, 0.7, 0.0, 0.0, 0.50],
                [0.50, 1.00, 0.3, 1.0, 1.2, 0.25],
            ],
            dtype=np.float32,
        )
        metadata = {
            "beatmap_id": 123,
            "title": "Pressure",
            "length_ms": 4000,
            "keys": 4,
            "overall_difficulty": 8,
        }

        curve = compute_player_feel_curve(sequence, metadata, window_sec=2.0, step_sec=1.0)
        summary = summarize_player_feel_curve(curve, metadata)

        self.assertIn("feel_strain", curve.columns)
        self.assertGreater(curve["jack_density"].max(), 0.0)
        self.assertGreater(curve["chord_load"].max(), 0.0)
        self.assertGreater(curve["ln_load"].max(), 0.0)
        self.assertGreater(summary["peak_strain"], summary["mean_strain"])
        self.assertIn(summary["dominant_skill"], {"rice", "jack", "chord", "ln", "stamina", "accuracy"})

    def test_write_player_feel_curves_outputs_curve_and_summary_csvs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.csv"
            sequences = root / "sequences"
            out_dir = root / "out"
            sequences.mkdir()
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "title": "A",
                        "artist": "x",
                        "version": "4K",
                        "length_ms": 2000,
                        "keys": 4,
                        "overall_difficulty": 7,
                    }
                ]
            ).to_csv(labels, index=False)
            np.save(
                sequences / "1.npy",
                np.asarray(
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.25],
                        [0.5, 1.0, 0.3, 1.0, 0.5, 0.25],
                    ],
                    dtype=np.float32,
                ),
            )

            write_player_feel_curves(labels, sequences, out_dir, window_sec=1.0, step_sec=0.5)

            curve = pd.read_csv(out_dir / "player_feel_curve.csv")
            summary = pd.read_csv(out_dir / "player_feel_summary.csv")

        self.assertEqual(curve["beatmap_id"].unique().tolist(), [1])
        self.assertEqual(summary.loc[0, "beatmap_id"], 1)
        self.assertGreater(summary.loc[0, "peak_strain"], 0.0)


if __name__ == "__main__":
    unittest.main()
