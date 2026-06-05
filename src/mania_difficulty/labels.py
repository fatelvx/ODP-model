from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, pstdev


@dataclass(frozen=True)
class AccuracyLabels:
    mean_acc: float
    acc_std: float
    skill_gap: float
    median_acc: float
    p10_acc: float
    p90_acc: float
    score_count: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        raise ValueError("percentile() requires at least one value")
    if q < 0 or q > 1:
        raise ValueError("q must be in [0, 1]")

    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = q * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def compute_accuracy_labels(accuracies: list[float]) -> AccuracyLabels:
    if not accuracies:
        raise ValueError("compute_accuracy_labels() requires at least one score")

    values = sorted((float(acc) for acc in accuracies), reverse=True)
    top_n = max(1, round(len(values) * 0.10))
    bottom_n = max(1, round(len(values) * 0.50))
    top_acc = mean(values[:top_n])
    bottom_acc = mean(values[-bottom_n:])

    return AccuracyLabels(
        mean_acc=mean(values),
        acc_std=pstdev(values) if len(values) > 1 else 0.0,
        skill_gap=top_acc - bottom_acc,
        median_acc=percentile(values, 0.50),
        p10_acc=percentile(values, 0.10),
        p90_acc=percentile(values, 0.90),
        score_count=len(values),
    )
