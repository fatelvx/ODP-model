from __future__ import annotations

import numpy as np


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    residual = float(np.sum((y_true - y_pred) ** 2))
    total = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if total == 0:
        return 0.0
    return 1.0 - residual / total


def regression_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_columns: list[str],
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, column in enumerate(target_columns):
        report[column] = {
            "mae": mean_absolute_error(y_true[:, index], y_pred[:, index]),
            "r2": r2_score(y_true[:, index], y_pred[:, index]),
        }
    return report
