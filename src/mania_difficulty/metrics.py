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


def average_ranks(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def spearman_rank_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    true_ranks = average_ranks(y_true)
    pred_ranks = average_ranks(y_pred)
    true_std = float(np.std(true_ranks))
    pred_std = float(np.std(pred_ranks))
    if true_std == 0.0 or pred_std == 0.0:
        return 0.0
    correlation = float(np.corrcoef(true_ranks, pred_ranks)[0, 1])
    return correlation if np.isfinite(correlation) else 0.0


def pairwise_order_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    comparable = 0
    correct = 0
    for index in range(len(y_true) - 1):
        true_gap = y_true[index] - y_true[index + 1 :]
        pred_gap = y_pred[index] - y_pred[index + 1 :]
        mask = (true_gap != 0) & (pred_gap != 0)
        if not np.any(mask):
            continue
        comparable += int(np.sum(mask))
        correct += int(np.sum(np.sign(true_gap[mask]) == np.sign(pred_gap[mask])))
    if comparable == 0:
        return 0.0
    return correct / comparable


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
            "spearman": spearman_rank_correlation(y_true[:, index], y_pred[:, index]),
            "pairwise_order_accuracy": pairwise_order_accuracy(y_true[:, index], y_pred[:, index]),
        }
        if baseline_pred is not None:
            baseline_mae = mean_absolute_error(y_true[:, index], baseline_pred[:, index])
            report[column]["baseline_mae"] = baseline_mae
            report[column]["mae_improvement_vs_baseline"] = baseline_mae - mae
            report[column]["mae_improvement_pct"] = (
                ((baseline_mae - mae) / baseline_mae) if baseline_mae else 0.0
            )
    return report
