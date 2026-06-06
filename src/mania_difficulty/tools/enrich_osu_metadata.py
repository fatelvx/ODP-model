from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mania_difficulty.data.parse_notes import read_osu_metadata


METADATA_COLUMNS = [
    "mode",
    "circle_size",
    "keys",
    "hp_drain_rate",
    "overall_difficulty",
    "approach_rate",
]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def enrich_rows(rows: list[dict[str, Any]], osu_dir: Path) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for row in tqdm(rows, desc="metadata"):
        next_row = dict(row)
        osu_path = osu_dir / f"{row['beatmap_id']}.osu"
        metadata = read_osu_metadata(osu_path) if osu_path.exists() else {}
        for column in METADATA_COLUMNS:
            value = metadata.get(column)
            next_row[column] = "" if value is None else value
        enriched.append(next_row)
    return enriched


def write_rows(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No rows to write.")
    fieldnames = list(rows[0].keys())
    for column in METADATA_COLUMNS:
        if column not in fieldnames:
            fieldnames.append(column)
    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich labels/maps CSV with .osu mode/key/HP/OD/AR metadata.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--osu-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    rows = enrich_rows(read_rows(args.labels), args.osu_dir)
    write_rows(rows, args.out)
    print(f"Wrote {len(rows)} enriched rows to {args.out}")


if __name__ == "__main__":
    main()
