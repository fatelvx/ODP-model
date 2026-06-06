from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd


ANNOTATION_COLUMNS = [
    "pair_id",
    "scope",
    "player_stage",
    "a_beatmap_id",
    "a_start_sec",
    "a_end_sec",
    "a_title",
    "a_artist",
    "a_version",
    "a_dominant_skill",
    "a_peak_strain",
    "b_beatmap_id",
    "b_start_sec",
    "b_end_sec",
    "b_title",
    "b_artist",
    "b_version",
    "b_dominant_skill",
    "b_peak_strain",
    "harder_choice",
    "confidence",
    "reason_tags",
    "notes",
]

CHOICE_A = {"a", "left", "first", "1"}
CHOICE_B = {"b", "right", "second", "2"}
UNCERTAIN_CHOICES = {"", "tie", "same", "equal", "skip", "unknown", "uncertain", "out_of_range"}


def normalize_harder_choice(value: object) -> str | None:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    if text in CHOICE_A:
        return "a"
    if text in CHOICE_B:
        return "b"
    if text in UNCERTAIN_CHOICES:
        return None
    return None


def normalize_confidence(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return 1.0
    try:
        confidence = float(str(value).strip())
    except ValueError:
        return 1.0
    return max(0.0, min(confidence, 5.0))


def ensure_annotation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in ANNOTATION_COLUMNS:
        if column not in output.columns:
            output[column] = ""
    return output[ANNOTATION_COLUMNS]


def read_player_feel_pairs(path: Path) -> pd.DataFrame:
    return ensure_annotation_columns(pd.read_csv(path))


def usable_judgment_rows(frame: pd.DataFrame) -> pd.DataFrame:
    output = ensure_annotation_columns(frame)
    output["normalized_harder_choice"] = output["harder_choice"].apply(normalize_harder_choice)
    output["sample_weight"] = output["confidence"].apply(normalize_confidence)
    return output[
        output["normalized_harder_choice"].isin(["a", "b"]) & (output["sample_weight"] > 0.0)
    ].reset_index(drop=True)


def write_empty_player_feel_template(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=ANNOTATION_COLUMNS)
        writer.writeheader()
