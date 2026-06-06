from __future__ import annotations

import numpy as np


SUMMARY_FEATURE_NAMES = [
    "note_count_log",
    "duration_sec_log",
    "notes_per_sec",
    "delta_mean",
    "delta_std",
    "delta_p10",
    "delta_p50",
    "delta_p90",
    "short_gap_50ms_ratio",
    "short_gap_100ms_ratio",
    "short_gap_200ms_ratio",
    "column_mean",
    "column_std",
    "column_change_mean",
    "same_column_ratio",
    "jump_column_ratio",
    "column_entropy",
    "column_imbalance",
    "ln_ratio",
    "ln_length_mean",
    "ln_length_std",
    "ln_length_max",
    "ln_total_per_sec",
    "chord_mean",
    "chord_std",
    "chord_max",
    "chord_2_ratio",
    "chord_3_ratio",
    "chord_4_ratio",
]


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def summarize_sequence(features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 6 or len(array) == 0:
        return np.zeros(len(SUMMARY_FEATURE_NAMES), dtype=np.float32)

    note_count = float(len(array))
    deltas = np.clip(array[:, 1], 0.0, None)
    columns = np.clip(array[:, 2], 0.0, 1.0)
    is_ln = np.clip(array[:, 3], 0.0, 1.0)
    ln_lengths = np.clip(array[:, 4], 0.0, None)
    chord_fraction = np.clip(array[:, 5], 0.0, None)
    chord_sizes = chord_fraction * 4.0

    duration_sec = float(np.sum(deltas))
    if duration_sec <= 0:
        duration_sec = float(max(array[-1, 0], 1e-6))
    notes_per_sec = _safe_ratio(note_count, duration_sec)

    nonzero_deltas = deltas[deltas > 0]
    if len(nonzero_deltas) == 0:
        nonzero_deltas = np.array([0.0], dtype=np.float32)

    if len(columns) > 1:
        column_changes = np.abs(np.diff(columns))
        same_column_ratio = float(np.mean(column_changes < 1e-6))
        jump_column_ratio = float(np.mean(column_changes >= (2.0 / 3.0)))
        column_change_mean = float(np.mean(column_changes))
    else:
        same_column_ratio = 0.0
        jump_column_ratio = 0.0
        column_change_mean = 0.0

    bins = np.clip(np.rint(columns * 3).astype(int), 0, 3)
    counts = np.bincount(bins, minlength=4).astype(np.float32)
    proportions = counts / max(float(np.sum(counts)), 1.0)
    nonzero_proportions = proportions[proportions > 0]
    column_entropy = float(-np.sum(nonzero_proportions * np.log2(nonzero_proportions)) / 2.0)
    column_imbalance = float(np.max(proportions) - np.min(proportions))

    ln_mask = is_ln > 0.5
    ln_values = ln_lengths[ln_mask]
    if len(ln_values) == 0:
        ln_values = np.array([0.0], dtype=np.float32)

    values = [
        np.log1p(note_count),
        np.log1p(duration_sec),
        notes_per_sec,
        float(np.mean(nonzero_deltas)),
        float(np.std(nonzero_deltas)),
        float(np.percentile(nonzero_deltas, 10)),
        float(np.percentile(nonzero_deltas, 50)),
        float(np.percentile(nonzero_deltas, 90)),
        float(np.mean(nonzero_deltas <= 0.05)),
        float(np.mean(nonzero_deltas <= 0.10)),
        float(np.mean(nonzero_deltas <= 0.20)),
        float(np.mean(columns)),
        float(np.std(columns)),
        column_change_mean,
        same_column_ratio,
        jump_column_ratio,
        column_entropy,
        column_imbalance,
        float(np.mean(ln_mask)),
        float(np.mean(ln_values)),
        float(np.std(ln_values)),
        float(np.max(ln_values)),
        _safe_ratio(float(np.sum(ln_lengths)), duration_sec),
        float(np.mean(chord_sizes)),
        float(np.std(chord_sizes)),
        float(np.max(chord_sizes)),
        float(np.mean(chord_sizes >= 2.0)),
        float(np.mean(chord_sizes >= 3.0)),
        float(np.mean(chord_sizes >= 4.0)),
    ]
    return np.asarray(values, dtype=np.float32)


def summarize_sequences(sequences: list[np.ndarray]) -> np.ndarray:
    if not sequences:
        return np.empty((0, len(SUMMARY_FEATURE_NAMES)), dtype=np.float32)
    return np.stack([summarize_sequence(sequence) for sequence in sequences]).astype(np.float32)
