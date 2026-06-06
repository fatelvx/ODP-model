from __future__ import annotations

import argparse
from itertools import cycle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from mania_difficulty.player_feel_annotations import ANNOTATION_COLUMNS
from mania_difficulty.player_stages import load_player_stages
from mania_difficulty.tools.player_feel_curve import (
    compute_player_feel_curve,
    numeric_value,
    summarize_player_feel_curve,
)


PRESSURE_SKILLS = {
    "reading": "reading_pressure",
    "speed": "speed_pressure",
    "stamina": "stamina_pressure",
    "jack": "jack_pressure",
    "chord": "chord_pressure",
    "ln": "ln_pressure",
    "accuracy": "accuracy_pressure",
}
DOMINANT_PRESSURE_SKILLS = {
    key: value for key, value in PRESSURE_SKILLS.items() if key not in {"accuracy", "stamina"}
}


def dominant_skill_from_row(row: pd.Series) -> str:
    scores = {
        skill: float(row.get(column, 0.0) or 0.0)
        for skill, column in DOMINANT_PRESSURE_SKILLS.items()
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def build_feel_frames(
    labels_csv: Path,
    sequences_dir: Path,
    *,
    window_sec: float,
    step_sec: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    labels = pd.read_csv(labels_csv)
    curves: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for row in tqdm(labels.to_dict("records"), desc="feel-pairs"):
        beatmap_id = int(numeric_value(row, "beatmap_id", default=0.0))
        sequence_path = sequences_dir / f"{beatmap_id}.npy"
        if not sequence_path.exists():
            continue
        sequence = np.load(sequence_path).astype("float32")
        curve = compute_player_feel_curve(sequence, row, window_sec=window_sec, step_sec=step_sec)
        if curve.empty:
            continue
        curves.append(curve)
        summaries.append(summarize_player_feel_curve(curve, row))
    curve_frame = pd.concat(curves, ignore_index=True) if curves else pd.DataFrame()
    return labels, curve_frame, pd.DataFrame(summaries)


def _identity(row: pd.Series, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_beatmap_id": int(row["beatmap_id"]),
        f"{prefix}_title": row.get("title", ""),
        f"{prefix}_artist": row.get("artist", ""),
        f"{prefix}_version": row.get("version", ""),
    }


def _map_side(row: pd.Series, prefix: str) -> dict[str, Any]:
    return {
        **_identity(row, prefix),
        f"{prefix}_start_sec": "",
        f"{prefix}_end_sec": "",
        f"{prefix}_dominant_skill": row.get("dominant_skill", ""),
        f"{prefix}_peak_strain": float(row.get("peak_strain", 0.0) or 0.0),
    }


def _segment_side(row: pd.Series, label_row: pd.Series, prefix: str) -> dict[str, Any]:
    return {
        **_identity(label_row, prefix),
        f"{prefix}_start_sec": float(row.get("start_sec", 0.0) or 0.0),
        f"{prefix}_end_sec": float(row.get("end_sec", 0.0) or 0.0),
        f"{prefix}_dominant_skill": dominant_skill_from_row(row),
        f"{prefix}_peak_strain": float(row.get("feel_strain", 0.0) or 0.0),
    }


def map_pair_candidates(summaries: pd.DataFrame) -> list[tuple[pd.Series, pd.Series]]:
    if len(summaries) < 2:
        return []
    ranked = summaries.sort_values(["peak_strain", "beatmap_id"]).reset_index(drop=True)
    pairs: list[tuple[pd.Series, pd.Series]] = []
    for index in range(len(ranked) - 1):
        left = ranked.iloc[index]
        right = ranked.iloc[index + 1]
        pairs.append((left, right))
    return sorted(
        pairs,
        key=lambda pair: (
            abs(float(pair[0]["peak_strain"]) - float(pair[1]["peak_strain"])),
            pair[0].get("dominant_skill", "") == pair[1].get("dominant_skill", ""),
        ),
    )


def segment_pair_candidates(
    curves: pd.DataFrame,
    labels: pd.DataFrame,
) -> list[tuple[pd.Series, pd.Series, pd.Series, pd.Series]]:
    if curves.empty:
        return []
    labels_by_id = labels.set_index("beatmap_id", drop=False)
    top_segments = (
        curves.sort_values(["beatmap_id", "feel_strain"], ascending=[True, False])
        .groupby("beatmap_id", as_index=False)
        .head(2)
        .sort_values(["feel_strain", "beatmap_id"])
        .reset_index(drop=True)
    )
    pairs: list[tuple[pd.Series, pd.Series, pd.Series, pd.Series]] = []
    for index in range(len(top_segments) - 1):
        left = top_segments.iloc[index]
        right = top_segments.iloc[index + 1]
        left_id = int(left["beatmap_id"])
        right_id = int(right["beatmap_id"])
        if left_id not in labels_by_id.index or right_id not in labels_by_id.index:
            continue
        pairs.append((left, labels_by_id.loc[left_id], right, labels_by_id.loc[right_id]))
    return sorted(
        pairs,
        key=lambda pair: abs(float(pair[0]["feel_strain"]) - float(pair[2]["feel_strain"])),
    )


def _blank_judgment_fields() -> dict[str, str]:
    return {"harder_choice": "", "confidence": "", "reason_tags": "", "notes": ""}


def generate_player_feel_pairs(
    labels_csv: Path,
    sequences_dir: Path,
    stages_csv: Path,
    out_csv: Path,
    *,
    max_pairs: int = 80,
    stage_ids: list[str] | None = None,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
) -> pd.DataFrame:
    stages = load_player_stages(stages_csv)
    selected_stages = stage_ids or stages["stage_id"].tolist()
    labels, curves, summaries = build_feel_frames(
        labels_csv,
        sequences_dir,
        window_sec=window_sec,
        step_sec=step_sec,
    )

    rows: list[dict[str, Any]] = []
    stage_cycle = cycle(selected_stages)
    for index, (left, right) in enumerate(map_pair_candidates(summaries)):
        if len(rows) >= max_pairs:
            break
        if index % 2:
            left, right = right, left
        stage_id = next(stage_cycle)
        rows.append(
            {
                "pair_id": f"map_{stage_id}_{index:04d}",
                "scope": "map",
                "player_stage": stage_id,
                **_map_side(left, "a"),
                **_map_side(right, "b"),
                **_blank_judgment_fields(),
            }
        )

    for index, (left, left_label, right, right_label) in enumerate(segment_pair_candidates(curves, labels)):
        if len(rows) >= max_pairs:
            break
        if index % 2:
            left, left_label, right, right_label = right, right_label, left, left_label
        stage_id = next(stage_cycle)
        rows.append(
            {
                "pair_id": f"segment_{stage_id}_{index:04d}",
                "scope": "segment",
                "player_stage": stage_id,
                **_segment_side(left, left_label, "a"),
                **_segment_side(right, right_label, "b"),
                **_blank_judgment_fields(),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    for column in ANNOTATION_COLUMNS:
        if column not in frame.columns:
            frame[column] = ""
    frame = frame[ANNOTATION_COLUMNS]
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    return frame


def parse_stage_ids(value: str) -> list[str] | None:
    ids = [item.strip() for item in value.split(",") if item.strip()]
    return ids or None


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 4K player-feel pairwise annotation candidates.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--player-stages", type=Path, default=Path("data/player_stages_4k.csv"))
    parser.add_argument("--out", type=Path, default=Path("data/annotations/player_feel_pairs.csv"))
    parser.add_argument("--max-pairs", type=int, default=80)
    parser.add_argument("--stage-ids", type=parse_stage_ids, default=None)
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    args = parser.parse_args()

    frame = generate_player_feel_pairs(
        args.labels,
        args.sequences,
        args.player_stages,
        args.out,
        max_pairs=args.max_pairs,
        stage_ids=args.stage_ids,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
    )
    print(f"Wrote {len(frame)} player-feel pairs to {args.out}")


if __name__ == "__main__":
    main()
