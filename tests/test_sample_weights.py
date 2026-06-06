import tempfile
import unittest
from pathlib import Path

import numpy as np

from mania_difficulty.data.dataset import ManiaDifficultyDataset, collate_batch
from mania_difficulty.train import fit_tabular_model, sample_weight_summary, tabular_sample_weights


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

    def test_sample_weight_summary_counts_downweighted_training_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequences = root / "sequences"
            sequences.mkdir()
            for beatmap_id in [1, 2, 3]:
                np.save(sequences / f"{beatmap_id}.npy", np.ones((2, 6), dtype=np.float32))
            labels = root / "labels.csv"
            labels.write_text(
                "\n".join(
                    [
                        "beatmap_id,mean_acc,acc_std,skill_gap,score_count",
                        "1,0.9,0.01,0.03,100",
                        "2,0.8,0.02,0.04,50",
                        "3,0.7,0.03,0.05,10",
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

            summary = sample_weight_summary(dataset, [0, 1, 2], prefix="train")

        self.assertEqual(summary["sample_weight_train_count"], 3)
        self.assertEqual(summary["sample_weight_train_downweighted_count"], 2)
        self.assertAlmostEqual(summary["sample_weight_train_mean"], (1.0 + 0.5 + 0.25) / 3)
        self.assertAlmostEqual(summary["sample_weight_train_min"], 0.25)
        self.assertAlmostEqual(summary["sample_weight_train_downweighted_rate"], 2 / 3)

    def test_tabular_sample_weights_match_dataset_reliability_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sequences = root / "sequences"
            sequences.mkdir()
            for beatmap_id in [1, 2, 3]:
                np.save(sequences / f"{beatmap_id}.npy", np.ones((2, 6), dtype=np.float32))
            labels = root / "labels.csv"
            labels.write_text(
                "\n".join(
                    [
                        "beatmap_id,mean_acc,acc_std,skill_gap,score_count",
                        "1,0.9,0.01,0.03,100",
                        "2,0.8,0.02,0.04,50",
                        "3,0.7,0.03,0.05,10",
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

            weights = tabular_sample_weights(dataset, [0, 1, 2])

        self.assertEqual(weights.tolist(), [1.0, 0.5, 0.25])

    def test_fit_tabular_model_passes_sample_weight_to_regressor(self):
        class Recorder:
            def __init__(self):
                self.fit_kwargs = None

            def fit(self, x, y, **kwargs):
                self.fit_kwargs = kwargs

        model = Recorder()
        weights = np.asarray([1.0, 0.5], dtype="float32")

        fit_tabular_model(
            model,
            np.ones((2, 2), dtype="float32"),
            np.ones((2, 1), dtype="float32"),
            sample_weights=weights,
        )

        self.assertIs(model.fit_kwargs["sample_weight"], weights)


if __name__ == "__main__":
    unittest.main()
