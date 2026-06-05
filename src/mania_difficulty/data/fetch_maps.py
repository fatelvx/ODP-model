from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mania_difficulty.data.osu_api import OsuApiClient


def is_ranked_4k_mania(beatmap: dict[str, Any]) -> bool:
    mode = beatmap.get("mode") or beatmap.get("mode_int")
    status = beatmap.get("status") or beatmap.get("ranked")
    circle_size = beatmap.get("cs")

    mode_ok = mode == "mania" or mode == 3
    status_ok = status == "ranked" or status == 1
    keys_ok = circle_size is not None and int(float(circle_size)) == 4
    return mode_ok and status_ok and keys_ok


def flatten_search_page(page: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beatmapset in page.get("beatmapsets", []):
        for beatmap in beatmapset.get("beatmaps", []):
            if not is_ranked_4k_mania(beatmap):
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
                }
            )
    return rows


def fetch_ranked_4k_maps(client: OsuApiClient, target: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[int] = set()
    cursor_string: str | None = None

    with tqdm(total=target, desc="maps") as progress:
        while len(rows) < target:
            params = {
                "m": 3,
                "s": "ranked",
                "q": "keys=4",
            }
            if cursor_string:
                params["cursor_string"] = cursor_string

            page = client.get("beatmapsets/search", params=params)
            page_rows = flatten_search_page(page)
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
    ]
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch ranked 4K mania map metadata.")
    parser.add_argument("--target", type=int, default=2000)
    parser.add_argument("--out", type=Path, default=Path("data/raw/beatmaps.csv"))
    parser.add_argument("--sleep", type=float, default=0.25)
    args = parser.parse_args()

    client = OsuApiClient.from_client_credentials(sleep_seconds=args.sleep)
    rows = fetch_ranked_4k_maps(client, args.target)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} maps to {args.out}")


if __name__ == "__main__":
    main()
