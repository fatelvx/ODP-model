import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.player_stages import write_default_player_stages
from mania_difficulty.tools.train_feel_ranker import train_feel_ranker


class TrainFeelRankerTests(unittest.TestCase):
    def test_train_feel_ranker_uses_confidence_and_skips_uncertain_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            stages = root / "stages.csv"
            summary = root / "summary.csv"
            curves = root / "curves.csv"
            judgments = root / "judgments.csv"
            out_dir = root / "out"
            write_default_player_stages(stages)
            pd.DataFrame(
                [
                    {
                        "beatmap_id": 1,
                        "peak_strain": 4.0,
                        "mean_strain": 3.0,
                        "p90_strain": 3.8,
                        "fatigue_area": 20.0,
                        "rice_score": 4.0,
                        "jack_score": 0.2,
                        "chord_score": 0.2,
                        "ln_score": 0.1,
                        "stamina_score": 3.0,
                        "accuracy_score": 4.1,
                    },
                    {
                        "beatmap_id": 2,
                        "peak_strain": 1.0,
                        "mean_strain": 0.8,
                        "p90_strain": 0.9,
                        "fatigue_area": 4.0,
                        "rice_score": 1.0,
                        "jack_score": 0.1,
                        "chord_score": 0.1,
                        "ln_score": 0.0,
                        "stamina_score": 0.8,
                        "accuracy_score": 1.0,
                    },
                    {
                        "beatmap_id": 3,
                        "peak_strain": 0.7,
                        "mean_strain": 0.5,
                        "p90_strain": 0.6,
                        "fatigue_area": 3.0,
                        "rice_score": 0.6,
                        "jack_score": 0.0,
                        "chord_score": 0.1,
                        "ln_score": 0.0,
                        "stamina_score": 0.5,
                        "accuracy_score": 0.7,
                    },
                ]
            ).to_csv(summary, index=False)
            pd.DataFrame(
                [
                    {"beatmap_id": 1, "start_sec": 0.0, "end_sec": 2.0, "feel_strain": 4.0},
                    {"beatmap_id": 2, "start_sec": 0.0, "end_sec": 2.0, "feel_strain": 1.0},
                    {"beatmap_id": 3, "start_sec": 0.0, "end_sec": 2.0, "feel_strain": 0.7},
                ]
            ).to_csv(curves, index=False)
            pd.DataFrame(
                [
                    {
                        "pair_id": "p1",
                        "scope": "map",
                        "player_stage": "beginner",
                        "a_beatmap_id": 1,
                        "b_beatmap_id": 2,
                        "harder_choice": "A",
                        "confidence": 3,
                    },
                    {
                        "pair_id": "p2",
                        "scope": "map",
                        "player_stage": "beginner",
                        "a_beatmap_id": 3,
                        "b_beatmap_id": 1,
                        "harder_choice": "B",
                        "confidence": 2,
                    },
                    {
                        "pair_id": "p3",
                        "scope": "map",
                        "player_stage": "beginner",
                        "a_beatmap_id": 1,
                        "b_beatmap_id": 3,
                        "harder_choice": "uncertain",
                        "confidence": 9,
                    },
                ]
            ).to_csv(judgments, index=False)

            metrics = train_feel_ranker(
                judgments,
                summary,
                curves,
                stages,
                out_dir,
                test_size=0.0,
                seed=7,
            )

            self.assertEqual(metrics["usable_judgments"], 2)
            self.assertEqual(metrics["ignored_judgments"], 1)
            self.assertGreaterEqual(metrics["train_agreement"], 0.5)
            self.assertTrue((out_dir / "feel_ranker.joblib").exists())
            self.assertTrue((out_dir / "feel_ranker_report.html").exists())


if __name__ == "__main__":
    unittest.main()
