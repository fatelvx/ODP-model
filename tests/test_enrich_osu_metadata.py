import tempfile
import unittest
from pathlib import Path

from mania_difficulty.tools.enrich_osu_metadata import enrich_rows


class EnrichOsuMetadataTests(unittest.TestCase):
    def test_enrich_rows_adds_mode_keys_hp_od_and_ar(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            osu_dir = Path(temp_dir)
            (osu_dir / "123.osu").write_text(
                """
osu file format v14

[General]
Mode: 3

[Difficulty]
HPDrainRate:7
CircleSize:4
OverallDifficulty:8.5
ApproachRate:5
""",
                encoding="utf-8",
            )

            rows = enrich_rows([{"beatmap_id": "123", "title": "x"}], osu_dir)

        self.assertEqual(rows[0]["mode"], 3.0)
        self.assertEqual(rows[0]["keys"], 4.0)
        self.assertEqual(rows[0]["hp_drain_rate"], 7.0)
        self.assertEqual(rows[0]["overall_difficulty"], 8.5)
        self.assertEqual(rows[0]["approach_rate"], 5.0)


if __name__ == "__main__":
    unittest.main()
