import tempfile
import unittest
from pathlib import Path

import pandas as pd

from mania_difficulty.player_stages import (
    STAGE_COLUMNS,
    load_player_stages,
    stage_vector,
    validate_player_stages,
    write_default_player_stages,
)


class PlayerStageTests(unittest.TestCase):
    def test_default_player_stages_validate_and_include_growth_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stages.csv"

            write_default_player_stages(path)
            stages = load_player_stages(path)

        self.assertEqual(list(stages.columns), STAGE_COLUMNS)
        self.assertIn("readless_beginner", set(stages["stage_id"]))
        self.assertIn("dan_ready", set(stages["stage_id"]))
        beginner = stage_vector(stages, "beginner")
        dan_ready = stage_vector(stages, "dan_ready")
        self.assertLess(beginner["reading"], dan_ready["reading"])
        self.assertLess(beginner["stamina"], dan_ready["stamina"])

    def test_validate_player_stages_rejects_missing_or_out_of_range_values(self):
        stages = pd.DataFrame(
            [
                {
                    "stage_id": "bad",
                    "reading": 1.2,
                    "speed": 0.5,
                    "stamina": 0.5,
                    "jack": 0.5,
                    "chord": 0.5,
                    "ln": 0.5,
                    "accuracy": 0.5,
                    "pattern_memory": 0.5,
                }
            ]
        )

        with self.assertRaises(ValueError):
            validate_player_stages(stages)


if __name__ == "__main__":
    unittest.main()
