from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


STAGE_COLUMNS = [
    "stage_id",
    "reading",
    "speed",
    "stamina",
    "jack",
    "chord",
    "ln",
    "accuracy",
    "pattern_memory",
]

ABILITY_COLUMNS = STAGE_COLUMNS[1:]

DEFAULT_STAGE_ROWS = [
    {
        "stage_id": "readless_beginner",
        "reading": 0.05,
        "speed": 0.08,
        "stamina": 0.05,
        "jack": 0.03,
        "chord": 0.04,
        "ln": 0.02,
        "accuracy": 0.05,
        "pattern_memory": 0.00,
    },
    {
        "stage_id": "beginner",
        "reading": 0.18,
        "speed": 0.20,
        "stamina": 0.16,
        "jack": 0.12,
        "chord": 0.13,
        "ln": 0.10,
        "accuracy": 0.16,
        "pattern_memory": 0.08,
    },
    {
        "stage_id": "novice",
        "reading": 0.35,
        "speed": 0.38,
        "stamina": 0.32,
        "jack": 0.28,
        "chord": 0.30,
        "ln": 0.24,
        "accuracy": 0.34,
        "pattern_memory": 0.22,
    },
    {
        "stage_id": "intermediate",
        "reading": 0.55,
        "speed": 0.58,
        "stamina": 0.53,
        "jack": 0.48,
        "chord": 0.50,
        "ln": 0.45,
        "accuracy": 0.55,
        "pattern_memory": 0.44,
    },
    {
        "stage_id": "advanced",
        "reading": 0.74,
        "speed": 0.76,
        "stamina": 0.72,
        "jack": 0.70,
        "chord": 0.70,
        "ln": 0.66,
        "accuracy": 0.74,
        "pattern_memory": 0.66,
    },
    {
        "stage_id": "dan_ready",
        "reading": 0.88,
        "speed": 0.90,
        "stamina": 0.88,
        "jack": 0.86,
        "chord": 0.85,
        "ln": 0.82,
        "accuracy": 0.88,
        "pattern_memory": 0.82,
    },
]


def validate_player_stages(stages: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in STAGE_COLUMNS if column not in stages.columns]
    if missing:
        raise ValueError(f"Missing player stage columns: {', '.join(missing)}")
    output = stages[STAGE_COLUMNS].copy()
    if output["stage_id"].isna().any() or (output["stage_id"].astype(str).str.strip() == "").any():
        raise ValueError("stage_id must be non-empty.")
    if output["stage_id"].duplicated().any():
        raise ValueError("stage_id values must be unique.")
    for column in ABILITY_COLUMNS:
        values = pd.to_numeric(output[column], errors="coerce")
        if values.isna().any():
            raise ValueError(f"{column} must be numeric.")
        if ((values < 0.0) | (values > 1.0)).any():
            raise ValueError(f"{column} must be between 0 and 1.")
        output[column] = values.astype(float)
    output["stage_id"] = output["stage_id"].astype(str)
    return output


def load_player_stages(path: Path) -> pd.DataFrame:
    return validate_player_stages(pd.read_csv(path))


def write_default_player_stages(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=STAGE_COLUMNS)
        writer.writeheader()
        writer.writerows(DEFAULT_STAGE_ROWS)


def stage_vector(stages: pd.DataFrame, stage_id: str) -> dict[str, float]:
    validated = validate_player_stages(stages)
    matches = validated[validated["stage_id"] == stage_id]
    if matches.empty:
        raise KeyError(f"Unknown player stage: {stage_id}")
    row = matches.iloc[0]
    return {column: float(row[column]) for column in ABILITY_COLUMNS}


def stage_feature_array(stage: dict[str, Any]) -> np.ndarray:
    return np.asarray([float(stage[column]) for column in ABILITY_COLUMNS], dtype=np.float32)
