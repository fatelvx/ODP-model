import tempfile
import unittest
from pathlib import Path

import numpy as np

from mania_difficulty.data.dataset import ManiaDifficultyDataset, collate_batch


class SampleWeightTests(unittest.TestCase):
    def test_dataset_reads_score_count_sample_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequences = root / "sequences"
            sequences.mkdir()
            np.save(sequences / "1.npy", np.ones((2, 6), dtype=np.float32))
            np.save(sequences / "2.npy", np.ones((2, 6), dtype=np.float32))
            labels = root / "labels.csv"
            labels.write_text(
                "\n".join(
                    [
                        "beatmap_id,mean_acc,acc_std,skill_gap,score_count",
                        "1,0.9,0.01,0.03,50",
                        "2,0.8,0.02,0.04,10",
                    ]
                ),
                encoding="utf-8",
            )

            dataset = ManiaDifficultyDataset(
                labels,
                sequences,
                sample_weight_column="score_count",
                sample_weight_min=0.25,
                sample_weight_max_value=100,
            )

            first = dataset[0]
            second = dataset[1]

        self.assertAlmostEqual(first["sample_weight"], 0.5)
        self.assertAlmostEqual(second["sample_weight"], 0.25)

    def test_collate_batch_preserves_sample_weights(self):
        batch = collate_batch(
            [
                {
                    "beatmap_id": 1,
                    "features": np.ones((2, 6), dtype=np.float32),
                    "targets": np.asarray([0.9, 0.01, 0.03], dtype=np.float32),
                    "sample_weight": 0.5,
                },
                {
                    "beatmap_id": 2,
                    "features": np.ones((3, 6), dtype=np.float32),
                    "targets": np.asarray([0.8, 0.02, 0.04], dtype=np.float32),
                    "sample_weight": 1.0,
                },
            ]
        )

        self.assertEqual(batch.sample_weights.tolist(), [0.5, 1.0])


if __name__ == "__main__":
    unittest.main()
