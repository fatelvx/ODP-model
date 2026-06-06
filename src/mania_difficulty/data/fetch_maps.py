from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mania_difficulty.data.osu_api import OsuApiClient


def _numeric_equal(value: Any, expected: int) -> bool:
    if value is None:
        return False
    try:
        return int(float(value)) == expected
    except (TypeError, ValueError):
        return False


def is_ranked_mania_keymode(beatmap: dict[str, Any], *, keys: int = 4) -> bool:
    mode = beatmap.get("mode") or beatmap.get("mode_int")
    status = beatmap.get("status") or beatmap.get("ranked")
    circle_size = beatmap.get("cs")

    mode_ok = mode == "mania" or mode == 3
    status_ok = status == "ranked" or status == 1
    keys_ok = _numeric_equal(circle_size, keys)
    return mode_ok and status_ok and keys_ok


def is_ranked_4k_mania(beatmap: dict[str, Any]) -> bool:
    return is_ranked_mania_keymode(beatmap, keys=4)


def flatten_search_page(page: dict[str, Any], *, keys: int = 4) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beatmapset in page.get("beatmapsets", []):
        for beatmap in beatmapset.get("beatmaps", []):
            if not is_ranked_mania_keymode(beatmap, keys=keys):
                continue
            rows.append(
                {
                    "beatmap_id": beatmap.get("id"),
                    "beatmapset_id": beatmapset.get("id"),
                    "title": beatmapset.get("title", ""),
                    "artist": beatmapset.get("artist", ""),
                    "mapper": (beatmapset.get("creator") or ""),
                    "version": beatmap.get("version", ""),
                    "length_ms": int(float(beatmap.get("total_length", 0))) * 1000,
                    "bpm": beatmap.get("bpm") or beatmapset.get("bpm") or "",
                    "difficulty_rating": beatmap.get("difficulty_rating", ""),
                    "mode": 3,
                    "circle_size": beatmap.get("cs", ""),
                    "keys": int(float(beatmap.get("cs", keys))),
                    "hp_drain_rate": beatmap.get("drain", ""),
                    "overall_difficulty": beatmap.get("accuracy", ""),
                    "approach_rate": beatmap.get("ar", ""),
                }
            )
    return rows


def fetch_ranked_mania_maps(
    client: OsuApiClient,
    target: int,
    *,
    keys: int = 4,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    cursor_string: str | None = None

    with tqdm(total=target, desc="maps") as progress:
        while len(rows) < target:
            params = {
                "m": 3,
                "s": "ranked",
                "q": f"keys={keys}",
            }
            if cursor_string:
                params["cursor_string"] = cursor_string

            page = client.get("beatmapsets/search", params=params)
            page_rows = flatten_search_page(page, keys=keys)
            for row in page_rows:
                beatmap_id = int(row["beatmap_id"])
                if beatmap_id in seen:
                    continue
                seen.add(beatmap_id)
                rows.append(row)
                progress.update(1)
                if len(rows) >= target:
                    break

            next_cursor = page.get("cursor_string")
            if not next_cursor or next_cursor == cursor_string:
                break
            cursor_string = next_cursor

    return rows


def fetch_ranked_4k_maps(client: OsuApiClient, target: int) -> list[dict[str, Any]]:
    return fetch_ranked_mania_maps(client, target, keys=4)


def write_csv(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "beatmap_id",
        "beatmapset_id",
        "title",
        "artist",
        "mapper",
        "version",
        "length_ms",
        "bpm",
        "difficulty_rating",
        "mode",
        "circle_size",
        "keys",
        "hp_drain_rate",
        "overall_difficulty",
        "approach_rate",
    ]
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ranked mania map metadata for one key mode.")
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--keys", type=int, default=4, help="Mania key mode to fetch, e.g. 4 or 7.")
    parser.add_argument("--out", type=Path, default=Path("data/raw/beatmaps.csv"))
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    client = OsuApiClient.from_client_credentials(sleep_seconds=args.sleep)
    rows = fetch_ranked_mania_maps(client, args.target, keys=args.keys)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} maps to {args.out}")


if __name__ == "__main__":
    main()
