from __future__ import annotations

import argparse
import math
from typing import Any


LOWER_IS_BETTER = {"mean_mae", "best_val_loss", "test_loss"}
HIGHER_IS_BETTER = {
    "mean_r2",
    "mean_spearman",
    "mean_pairwise_order_accuracy",
    "mean_improvement_pct",
}
SELECTION_METRICS = tuple(sorted(LOWER_IS_BETTER | HIGHER_IS_BETTER))
SUMMARY_SELECTION_METRICS = tuple(
    metric for metric in SELECTION_METRICS if metric not in {"best_val_loss", "test_loss"}
)


def parse_selection_metric(value: str) -> str:
    if value not in SELECTION_METRICS:
        allowed = ", ".join(SELECTION_METRICS)
        raise argparse.ArgumentTypeError(f"Unknown selection metric: {value}. Use one of: {allowed}")
    return value


def parse_summary_selection_metric(value: str) -> str:
    if value not in SUMMARY_SELECTION_METRICS:
        allowed = ", ".join(SUMMARY_SELECTION_METRICS)
        raise argparse.ArgumentTypeError(f"Unknown selection metric: {value}. Use one of: {allowed}")
    return value


def selection_sort_value(row: dict[str, Any], selection_metric: str) -> float:
    try:
        metric_value = float(row[selection_metric])
    except (KeyError, TypeError, ValueError):
        return math.inf
    if not math.isfinite(metric_value):
        return math.inf
    if selection_metric in HIGHER_IS_BETTER:
        return -metric_value
    return metric_value


def selection_sort_ascending(selection_metric: str) -> bool:
    return selection_metric in LOWER_IS_BETTER
