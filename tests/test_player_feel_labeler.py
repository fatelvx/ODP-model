import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.player_feel_annotations import ANNOTATION_COLUMNS
from mania_difficulty.tools.serve_player_feel_labeler import (
    LabelerStore,
    label_status,
)


class PlayerFeelLabelerTests(unittest.TestCase):
    def test_label_status_tracks_judged_and_out_of_range_rows(self):
        self.assertEqual(label_status({"harder_choice": "A"}), "judged")
        self.assertEqual(label_status({"harder_choice": "out_of_range"}), "out_of_range")
        self.assertEqual(label_status({"harder_choice": "uncertain"}), "uncertain")
        self.assertEqual(label_status({"harder_choice": ""}), "open")

    def test_store_reads_filters_and_saves_judgment(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_csv = Path(tmp) / "pairs.csv"
            pd.DataFrame(
                [
                    {
                        "pair_id": "p1",
                        "scope": "map",
                        "player_stage": "beginner",
                        "a_beatmap_id": 1,
                        "a_title": "A",
                        "a_peak_strain": 2.0,
                        "b_beatmap_id": 2,
                        "b_title": "B",
                        "b_peak_strain": 1.0,
                        "harder_choice": "",
                    },
                    {
                        "pair_id": "p2",
                        "scope": "segment",
                        "player_stage": "dan_ready",
                        "a_beatmap_id": 3,
                        "a_title": "C",
                        "b_beatmap_id": 4,
                        "b_title": "D",
                        "harder_choice": "out_of_range",
                    },
                ],
                columns=ANNOTATION_COLUMNS,
            ).to_csv(pairs_csv, index=False)
            store = LabelerStore(pairs_csv)

            state = store.state(stage="beginner", scope="", status="open", index=0)
            self.assertEqual(state["total_count"], 2)
            self.assertEqual(state["filtered_count"], 1)
            self.assertEqual(state["current"]["pair_id"], "p1")

            saved = store.save(
                {
                    "pair_id": "p1",
                    "harder_choice": "A",
                    "confidence": 3,
                    "reason_tags": "jack,reading",
                    "notes": "left feels harder",
                }
            )

            self.assertEqual(saved["status"], "ok")
            rows = pd.read_csv(pairs_csv).fillna("")
            self.assertEqual(rows.loc[0, "harder_choice"], "A")
            self.assertEqual(rows.loc[0, "reason_tags"], "jack,reading")
            self.assertIn("judged_count", store.state())

    def test_store_handles_json_post_body(self):
        with tempfile.TemporaryDirectory() as tmp:
            pairs_csv = Path(tmp) / "pairs.csv"
            pd.DataFrame(
                [{"pair_id": "p1", "scope": "map", "player_stage": "beginner"}],
                columns=ANNOTATION_COLUMNS,
            ).to_csv(pairs_csv, index=False)
            store = LabelerStore(pairs_csv)
            payload = json.dumps({"pair_id": "p1", "harder_choice": "uncertain"}).encode("utf-8")

            result = store.save_json(payload)

            self.assertEqual(result["status"], "ok")
            self.assertEqual(result["pair"]["harder_choice"], "uncertain")


if __name__ == "__main__":
    unittest.main()
