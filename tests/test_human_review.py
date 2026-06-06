import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.train import write_human_review, write_pairwise_review


class HumanReviewTests(unittest.TestCase):
    def test_human_review_includes_largest_error_for_each_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "labels.csv"
            predictions_csv = tmp_path / "predictions.csv"
            out_csv = tmp_path / "human_review.csv"

            pd.DataFrame(
                [
                    {"beatmap_id": 1, "title": "Alpha", "artist": "a", "mapper": "m", "version": "v"},
                    {"beatmap_id": 2, "title": "Beta", "artist": "b", "mapper": "m", "version": "v"},
                    {"beatmap_id": 3, "title": "Gamma", "artist": "c", "mapper": "m", "version": "v"},
                ]
            ).to_csv(labels_csv, index=False)
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "actual_mean_acc": 0.90,
                        "pred_mean_acc": 0.89,
                        "error_mean_acc": -0.01,
                        "actual_acc_std": 0.05,
                        "pred_acc_std": 0.06,
                        "error_acc_std": 0.01,
                    },
                    {
                        "beatmap_id": 2,
                        "actual_mean_acc": 0.80,
                        "pred_mean_acc": 0.79,
                        "error_mean_acc": -0.01,
                        "actual_acc_std": 0.30,
                        "pred_acc_std": 0.05,
                        "error_acc_std": -0.25,
                    },
                    {
                        "beatmap_id": 3,
                        "actual_mean_acc": 0.70,
                        "pred_mean_acc": 0.65,
                        "error_mean_acc": -0.05,
                        "actual_acc_std": 0.20,
                        "pred_acc_std": 0.19,
                        "error_acc_std": -0.01,
                    },
                ]
            ).to_csv(predictions_csv, index=False)

            write_human_review(
                out_csv,
                labels_csv,
                predictions_csv,
                ["mean_acc", "acc_std"],
                top_n=1,
            )

            rows = pd.read_csv(out_csv)

        acc_std_rows = rows[rows["review_reason"] == "largest_acc_std_disagreement"]
        self.assertEqual(acc_std_rows["beatmap_id"].tolist(), [2])
        self.assertAlmostEqual(float(acc_std_rows.iloc[0]["error_acc_std"]), -0.25)

    def test_pairwise_review_writes_rank_disagreement_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            labels_csv = tmp_path / "labels.csv"
            predictions_csv = tmp_path / "predictions.csv"
            out_csv = tmp_path / "human_pair_review.csv"

            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "title": "Alpha",
                        "artist": "Artist A",
                        "mapper": "Mapper A",
                        "version": "Hard",
                    },
                    {
                        "beatmap_id": 2,
                        "title": "Beta",
                        "artist": "Artist B",
                        "mapper": "Mapper B",
                        "version": "Insane",
                    },
                    {
                        "beatmap_id": 3,
                        "title": "Gamma",
                        "artist": "Artist C",
                        "mapper": "Mapper C",
                        "version": "Easy",
                    },
                ]
            ).to_csv(labels_csv, index=False)
            pd.DataFrame(
                [
                    {"beatmap_id": 1, "actual_mean_acc": 0.90, "pred_mean_acc": 0.70},
                    {"beatmap_id": 2, "actual_mean_acc": 0.80, "pred_mean_acc": 0.92},
                    {"beatmap_id": 3, "actual_mean_acc": 0.95, "pred_mean_acc": 0.93},
                ]
            ).to_csv(predictions_csv, index=False)

            write_pairwise_review(
                out_csv,
                labels_csv,
                predictions_csv,
                target_column="mean_acc",
                top_n=5,
            )

            rows = pd.read_csv(out_csv)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows.loc[0, "model_harder_beatmap_id"], 1)
        self.assertEqual(rows.loc[0, "observed_harder_beatmap_id"], 2)
        self.assertEqual(rows.loc[0, "model_harder_title"], "Alpha")
        self.assertGreater(rows.loc[0, "disagreement_strength"], 0)


if __name__ == "__main__":
    unittest.main()
