from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_MIRRORS = [
    "https://osu.ppy.sh/osu/{beatmap_id}",
    "https://api.chimu.moe/v1/osu/{beatmap_id}",
]


def read_map_ids(path: Path) -> list[str]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return [row["beatmap_id"] for row in csv.DictReader(file)]


def download_osu_file(beatmap_id: str, out_path: Path, mirrors: list[str]) -> bool:
    for mirror in mirrors:
        url = mirror.format(beatmap_id=beatmap_id)
        try:
            response = requests.get(url, timeout=45)
            if response.status_code != 200 or "[HitObjects]" not in response.text:
                continue
            out_path.write_text(response.text, encoding="utf-8")
            return True
        except requests.RequestException:
            continue
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Download .osu files from mirrors.")
    parser.add_argument("--maps", type=Path, default=Path("data/raw/beatmaps.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/raw/osu"))
    parser.add_argument("--sleep", type=float, default=1.0)
    parser.add_argument(
        "--mirror",
        action="append",
        default=None,
        help="URL template containing {beatmap_id}. Can be passed multiple times.",
    )
    args = parser.parse_args()

    mirrors = args.mirror or DEFAULT_MIRRORS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    successes = 0
    failures: list[str] = []
    for beatmap_id in tqdm(read_map_ids(args.maps), desc=".osu"):
        out_path = args.out_dir / f"{beatmap_id}.osu"
        if out_path.exists():
            successes += 1
            continue
        if download_osu_file(beatmap_id, out_path, mirrors):
            successes += 1
        else:
            failures.append(beatmap_id)
        time.sleep(args.sleep)

    if failures:
        failure_path = args.out_dir / "download_failures.txt"
        failure_path.write_text("\n".join(failures), encoding="utf-8")
        print(f"{len(failures)} downloads failed. See {failure_path}")
    print(f"Downloaded or reused {successes} .osu files in {args.out_dir}")


if __name__ == "__main__":
    main()
