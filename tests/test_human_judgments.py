import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.human_judgments import score_pair_judgments, write_pair_judgment_template


class HumanJudgmentTests(unittest.TestCase):
    def test_write_pair_judgment_template_adds_fillable_columns(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pair_review_csv = tmp_path / "human_pair_review.csv"
            template_csv = tmp_path / "human_pair_judgment_template.csv"
            pd.DataFrame(
                [
                    {
                        "review_reason": "pairwise_rank_disagreement",
                        "model_harder_beatmap_id": 1,
                        "observed_harder_beatmap_id": 2,
                    }
                ]
            ).to_csv(pair_review_csv, index=False)

            write_pair_judgment_template(template_csv, pair_review_csv)
            rows = pd.read_csv(template_csv)

        self.assertIn("human_harder_beatmap_id", rows.columns)
        self.assertIn("human_confidence", rows.columns)
        self.assertIn("human_notes", rows.columns)
        self.assertEqual(rows.loc[0, "model_harder_beatmap_id"], 1)

    def test_score_pair_judgments_counts_model_and_proxy_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            judgments_csv = Path(tmp) / "judgments.csv"
            pd.DataFrame(
                [
                    {
                        "model_harder_beatmap_id": 1,
                        "observed_harder_beatmap_id": 2,
                        "human_harder_beatmap_id": 1,
                    },
                    {
                        "model_harder_beatmap_id": 3,
                        "observed_harder_beatmap_id": 4,
                        "human_harder_beatmap_id": 4,
                    },
                    {
                        "model_harder_beatmap_id": 5,
                        "observed_harder_beatmap_id": 6,
                        "human_harder_beatmap_id": "",
                    },
                ]
            ).to_csv(judgments_csv, index=False)

            score = score_pair_judgments(judgments_csv)

        self.assertEqual(score["judged_count"], 2)
        self.assertEqual(score["model_agree_count"], 1)
        self.assertEqual(score["proxy_agree_count"], 1)
        self.assertAlmostEqual(score["model_agreement_rate"], 0.5)
        self.assertEqual(score["unjudged_count"], 1)


if __name__ == "__main__":
    unittest.main()
