import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mania_difficulty.player_stages import write_default_player_stages
from mania_difficulty.tools.generate_player_feel_pairs import generate_player_feel_pairs


class PlayerFeelPairGenerationTests(unittest.TestCase):
    def test_generate_player_feel_pairs_outputs_map_and_segment_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            labels = root / "labels.csv"
            sequences = root / "sequences"
            stages = root / "stages.csv"
            out = root / "pairs.csv"
            sequences.mkdir()
            write_default_player_stages(stages)
            rows = []
            for index in range(4):
                beatmap_id = 100 + index
                rows.append(
                    {
                        "beatmap_id": beatmap_id,
                        "title": f"M{index}",
                        "artist": "a",
                        "version": "4K",
                        "length_ms": 4000,
                        "keys": 4,
                        "overall_difficulty": 6 + index,
                    }
                )
                sequence = np.asarray(
                    [
                        [0.00, 0.00, 0.0, 0.0, 0.0, 0.25],
                        [0.10, 0.10, 0.0, 0.0, 0.0, 0.25],
                        [0.20, 0.10, 1.0, 0.0, 0.0, 0.50 + index * 0.1],
                        [0.55, 0.80, 0.5, 1.0, 0.3 + index * 0.1, 0.25],
                    ],
                    dtype=np.float32,
                )
                np.save(sequences / f"{beatmap_id}.npy", sequence)
            pd.DataFrame(rows).to_csv(labels, index=False)

            generate_player_feel_pairs(
                labels,
                sequences,
                stages,
                out,
                max_pairs=8,
                stage_ids=["beginner", "dan_ready"],
                window_sec=1.0,
                step_sec=0.5,
            )
            pairs = pd.read_csv(out)

        self.assertGreaterEqual(len(pairs), 2)
        self.assertIn("map", set(pairs["scope"]))
        self.assertIn("segment", set(pairs["scope"]))
        self.assertIn("harder_choice", pairs.columns)
        self.assertTrue(set(pairs["player_stage"]).issubset({"beginner", "dan_ready"}))


if __name__ == "__main__":
    unittest.main()
