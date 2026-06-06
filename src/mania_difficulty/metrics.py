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
    *,
    baseline_pred: np.ndarray | None = None,
) -> dict[str, dict[str, float]]:
    report: dict[str, dict[str, float]] = {}
    for index, column in enumerate(target_columns):
        mae = mean_absolute_error(y_true[:, index], y_pred[:, index])
        report[column] = {
            "mae": mae,
            "r2": r2_score(y_true[:, index], y_pred[:, index]),
        }
        if baseline_pred is not None:
            baseline_mae = mean_absolute_error(y_true[:, index], baseline_pred[:, index])
            report[column]["baseline_mae"] = baseline_mae
            report[column]["mae_improvement_vs_baseline"] = baseline_mae - mae
            report[column]["mae_improvement_pct"] = (
                ((baseline_mae - mae) / baseline_mae) if baseline_mae else 0.0
            )
    return report
