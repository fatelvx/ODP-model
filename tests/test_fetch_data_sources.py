import unittest

from mania_difficulty.data.fetch_maps import fetch_ranked_mania_maps, flatten_search_page
from mania_difficulty.data.fetch_scores import fetch_score_labels


class RecorderClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, path, params=None):
        self.calls.append((path, dict(params or {})))
        return self.responses.pop(0)


class FetchDataSourcesTests(unittest.TestCase):
    def test_flatten_search_page_keeps_only_ranked_mania_target_keys(self):
        page = {
            "beatmapsets": [
                {
                    "id": 10,
                    "title": "T",
                    "artist": "A",
                    "creator": "M",
                    "beatmaps": [
                        {
                            "id": 1,
                            "mode": "mania",
                            "status": "ranked",
                            "cs": 4,
                            "version": "4K",
                            "total_length": 120,
                        },
                        {
                            "id": 2,
                            "mode": "mania",
                            "status": "ranked",
                            "cs": 7,
                            "version": "7K",
                        },
                        {"id": 3, "mode": "osu", "status": "ranked", "cs": 4},
                    ],
                }
            ]
        }

        rows = flatten_search_page(page, keys=4)

        self.assertEqual([row["beatmap_id"] for row in rows], [1])
        self.assertEqual(rows[0]["mode"], 3)
        self.assertEqual(rows[0]["keys"], 4)

    def test_fetch_ranked_mania_maps_queries_requested_key_mode(self):
        client = RecorderClient(
            [
                {
                    "beatmapsets": [
                        {
                            "id": 10,
                            "title": "T",
                            "artist": "A",
                            "creator": "M",
                            "beatmaps": [
                                {
                                    "id": 1,
                                    "mode": "mania",
                                    "status": "ranked",
                                    "cs": 7,
                                    "version": "7K",
                                    "total_length": 120,
                                }
                            ],
                        }
                    ],
                    "cursor_string": None,
                }
            ]
        )

        rows = fetch_ranked_mania_maps(client, target=1, keys=7)

        self.assertEqual(rows[0]["keys"], 7)
        self.assertEqual(client.calls[0][0], "beatmapsets/search")
        self.assertEqual(client.calls[0][1]["m"], 3)
        self.assertEqual(client.calls[0][1]["q"], "keys=7")

    def test_fetch_score_labels_requests_mania_top100_and_records_source_mode(self):
        client = RecorderClient(
            [
                {
                    "scores": [
                        {"accuracy": 0.99},
                        {"accuracy": 0.95},
                        {"accuracy": 0.90},
                    ]
                }
            ]
        )

        rows = fetch_score_labels(
            client,
            [{"beatmap_id": "1", "title": "x"}],
            min_scores=3,
        )

        self.assertEqual(client.calls[0][0], "beatmaps/1/scores")
        self.assertEqual(client.calls[0][1]["mode"], "mania")
        self.assertEqual(client.calls[0][1]["limit"], 100)
        self.assertEqual(rows[0]["score_mode"], "mania")
        self.assertEqual(rows[0]["score_limit"], 100)


if __name__ == "__main__":
    unittest.main()
