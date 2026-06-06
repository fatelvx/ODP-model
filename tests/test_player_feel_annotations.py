import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.player_feel_annotations import (
    ANNOTATION_COLUMNS,
    normalize_harder_choice,
    read_player_feel_pairs,
    usable_judgment_rows,
    write_empty_player_feel_template,
)


class PlayerFeelAnnotationTests(unittest.TestCase):
    def test_template_and_reader_keep_uncertain_out_of_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pairs.csv"
            write_empty_player_feel_template(path)
            pd.DataFrame(
                [
                    {
                        "pair_id": "p1",
                        "scope": "map",
                        "player_stage": "beginner",
                        "a_beatmap_id": 1,
                        "b_beatmap_id": 2,
                        "harder_choice": "A",
                        "confidence": 2,
                    },
                    {
                        "pair_id": "p2",
                        "scope": "segment",
                        "player_stage": "dan_ready",
                        "a_beatmap_id": 3,
                        "b_beatmap_id": 4,
                        "harder_choice": "out_of_range",
                        "confidence": 5,
                    },
                ],
                columns=ANNOTATION_COLUMNS,
            ).to_csv(path, index=False)

            pairs = read_player_feel_pairs(path)
            usable = usable_judgment_rows(pairs)

        self.assertEqual(list(pairs.columns), ANNOTATION_COLUMNS)
        self.assertEqual(normalize_harder_choice("left"), "a")
        self.assertIsNone(normalize_harder_choice("uncertain"))
        self.assertEqual(len(usable), 1)
        self.assertEqual(usable.loc[0, "normalized_harder_choice"], "a")
        self.assertEqual(usable.loc[0, "sample_weight"], 2.0)


if __name__ == "__main__":
    unittest.main()
