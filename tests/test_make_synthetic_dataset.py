import unittest

import numpy as np

from mania_difficulty.tools.make_synthetic_dataset import make_map


class MakeSyntheticDatasetTests(unittest.TestCase):
    def test_make_map_includes_difficulty_rating_metadata(self):
        _, label = make_map(np.random.default_rng(42), 900000)

        self.assertIn("difficulty_rating", label)
        self.assertGreater(label["difficulty_rating"], 0)


if __name__ == "__main__":
    unittest.main()
