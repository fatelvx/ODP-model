import unittest

from mania_difficulty.data.parse_notes import events_to_features, parse_osu_text


SAMPLE_OSU = """
osu file format v14

[HitObjects]
64,192,1000,1,0,0:0:0:0:
192,192,1000,1,0,0:0:0:0:
320,192,1500,128,0,2100:0:0:0:0:
448,192,1900,1,0,0:0:0:0:
"""


class ParseNotesTests(unittest.TestCase):
    def test_parse_osu_text_extracts_columns_ln_and_chords(self):
        events = parse_osu_text(SAMPLE_OSU, keys=4)

        self.assertEqual([event.column for event in events], [0, 1, 2, 3])
        self.assertEqual([event.chord_size for event in events], [2, 2, 1, 1])
        self.assertEqual(events[2].is_ln, 1)
        self.assertEqual(events[2].ln_length_ms, 600)

    def test_events_to_features_normalizes_expected_columns(self):
        features = events_to_features(parse_osu_text(SAMPLE_OSU, keys=4), keys=4)

        self.assertEqual(len(features), 4)
        self.assertEqual(features[0][1], 0.0)
        self.assertEqual(features[1][1], 0.0)
        self.assertEqual(features[2][1], 0.5)
        self.assertEqual(features[2][4], 0.6)
        self.assertEqual(features[0][5], 0.5)


if __name__ == "__main__":
    unittest.main()
