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

            write_run_report(run_dir, target_columns=["mean_acc"])

            report = (run_dir / "run_report.html").read_text(encoding="utf-8")

        self.assertIn("Pairwise Human Review", report)
        self.assertIn("pairwise_rank_disagreement", report)


if __name__ == "__main__":
    unittest.main()
