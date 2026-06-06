from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from tqdm import tqdm

from mania_difficulty.data.osu_api import OsuApiClient
from mania_difficulty.labels import compute_accuracy_labels


def read_map_ids(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def extract_accuracies(payload: dict[str, Any]) -> list[float]:
    accuracies: list[float] = []
    for score in payload.get("scores", []):
        accuracy = score.get("accuracy")
        if accuracy is None:
            continue
        accuracies.append(float(accuracy))
    return accuracies


def fetch_score_labels(
    client: OsuApiClient,
    maps: list[dict[str, str]],
    *,
    min_scores: int,
    mods: str | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for beatmap in tqdm(maps, desc="scores"):
        beatmap_id = beatmap["beatmap_id"]
        score_mode = "mania"
        score_limit = 100
        params: dict[str, Any] = {"mode": score_mode, "limit": score_limit}
        if mods:
            params["mods"] = mods

        payload = client.get(f"beatmaps/{beatmap_id}/scores", params=params)
        accuracies = extract_accuracies(payload)
        if len(accuracies) < min_scores:
            continue

        labels = compute_accuracy_labels(accuracies)
        rows.append(
            {
                **beatmap,
                "mean_acc": labels.mean_acc,
                "acc_std": labels.acc_std,
                "skill_gap": labels.skill_gap,
                "median_acc": labels.median_acc,
                "p10_acc": labels.p10_acc,
                "p90_acc": labels.p90_acc,
                "score_count": labels.score_count,
                "score_mode": score_mode,
                "score_limit": score_limit,
            }
        )
    return rows


def write_labels(rows: list[dict[str, Any]], out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No labels to write. Lower --min-scores or check API access.")

    with out.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch top score labels for beatmaps.")
    parser.add_argument("--maps", type=Path, default=Path("data/raw/beatmaps.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/processed/labels.csv"))
    parser.add_argument("--min-scores", type=int, default=30)
    parser.add_argument("--mods", type=str, default="")
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    client = OsuApiClient.from_client_credentials(sleep_seconds=args.sleep)
    rows = fetch_score_labels(
        client,
        read_map_ids(args.maps),
        min_scores=args.min_scores,
        mods=args.mods or None,
    )
    write_labels(rows, args.out)
    print(f"Wrote {len(rows)} labeled maps to {args.out}")


if __name__ == "__main__":
    main()
