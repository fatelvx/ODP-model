import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from mania_difficulty.tools.audit_dataset import audit_dataset


class AuditDatasetTests(unittest.TestCase):
    def test_audit_dataset_counts_missing_sequences_and_target_stats(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "labels.csv"
            sequences_dir = tmp_path / "sequences"
            sequences_dir.mkdir()
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "beatmapset_id": 10,
                        "mean_acc": 0.90,
                        "acc_std": 0.02,
                        "skill_gap": 0.05,
                        "score_count": 100,
                    },
                    {
                        "beatmap_id": 2,
                        "beatmapset_id": 10,
                        "mean_acc": 0.80,
                        "acc_std": 0.03,
                        "skill_gap": 0.07,
                        "score_count": 70,
                    },
                    {
                        "beatmap_id": 3,
                        "beatmapset_id": 11,
                        "mean_acc": 0.70,
                        "acc_std": 0.04,
                        "skill_gap": 0.09,
                        "score_count": 40,
                    },
                ]
            ).to_csv(labels_csv, index=False)
            np.save(sequences_dir / "1.npy", np.zeros((5, 6), dtype=np.float32))
            np.save(sequences_dir / "2.npy", np.zeros((9, 6), dtype=np.float32))

            summary, missing = audit_dataset(labels_csv, sequences_dir)

        self.assertEqual(summary["label_rows"], 3)
        self.assertEqual(summary["usable_rows"], 2)
        self.assertEqual(summary["missing_sequence_count"], 1)
        self.assertEqual(summary["group_count"], 2)
        self.assertEqual(summary["sequence_length"]["max"], 9)
        self.assertAlmostEqual(summary["targets"]["mean_acc"]["mean"], 0.85)
        self.assertEqual(missing[0]["beatmap_id"], 3)


if __name__ == "__main__":
    unittest.main()
