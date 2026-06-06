from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_SLICE_COLUMNS = ("score_count", "num_notes", "length_ms", "bpm")


def error_slice_fields() -> list[str]:
    return [
        "slice_column",
        "slice_value",
        "count",
        "actual_mean",
        "pred_mean",
        "mae",
        "bias",
        "max_abs_error",
    ]


def slice_metrics(frame: pd.DataFrame, target_column: str) -> dict[str, object]:
    actual_column = f"actual_{target_column}"
    pred_column = f"pred_{target_column}"
    error = frame[pred_column] - frame[actual_column]
    abs_error = error.abs()
    return {
        "count": int(len(frame)),
        "actual_mean": float(frame[actual_column].mean()),
        "pred_mean": float(frame[pred_column].mean()),
        "mae": float(abs_error.mean()),
        "bias": float(error.mean()),
        "max_abs_error": float(abs_error.max()),
    }


def quantile_labels(size: int) -> list[str]:
    if size <= 1:
        return ["all"]
    if size == 2:
        return ["low", "high"]
    return ["low", "mid", "high"]


def add_quantile_bins(frame: pd.DataFrame, column: str) -> pd.Series:
    numeric = pd.to_numeric(frame[column], errors="coerce")
    valid = numeric.dropna()
    if len(valid) <= 1:
        return pd.Series(["all" for _ in range(len(frame))], index=frame.index, dtype=object)
    bin_count = min(3, len(valid))
    labels = quantile_labels(bin_count)
    ranked = numeric.rank(method="first")
    binned = pd.qcut(ranked, q=bin_count, labels=labels, duplicates="drop")
    return binned.astype(object).where(numeric.notna(), "missing")


def write_error_slices(
    path: Path,
    labels_csv: Path,
    predictions_csv: Path,
    *,
    target_column: str = "mean_acc",
    slice_columns: tuple[str, ...] = DEFAULT_SLICE_COLUMNS,
) -> None:
    predictions = pd.read_csv(predictions_csv)
    labels = pd.read_csv(labels_csv)
    actual_column = f"actual_{target_column}"
    pred_column = f"pred_{target_column}"
    fieldnames = error_slice_fields()
    if actual_column not in predictions.columns or pred_column not in predictions.columns:
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
        return

    frame = predictions.merge(labels, on="beatmap_id", how="left", suffixes=("", "_label"))
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna(subset=[actual_column, pred_column])
    rows: list[dict[str, object]] = []
    if not frame.empty:
        rows.append({"slice_column": "overall", "slice_value": "all", **slice_metrics(frame, target_column)})

    for column in slice_columns:
        if column not in frame.columns or frame.empty:
            continue
        working = frame.copy()
        working["_slice_value"] = add_quantile_bins(working, column)
        for slice_value, group in working.groupby("_slice_value", sort=False):
            if group.empty:
                continue
            rows.append(
                {
                    "slice_column": column,
                    "slice_value": str(slice_value),
                    **slice_metrics(group, target_column),
                }
            )

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
