from __future__ import annotations

import numpy as np


CORE_FEATURE_NAMES = [
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

BURST_FEATURE_NAMES = [
    "peak_notes_per_sec_1s",
    "peak_notes_per_sec_2s",
    "mean_notes_per_sec_1s",
    "burst_10nps_ratio",
    "burst_15nps_ratio",
    "jack_ratio_100ms",
    "jack_ratio_150ms",
    "trill_like_ratio_120ms",
    "burst_chord_2_ratio",
    "burst_chord_3_ratio",
]

SUMMARY_FEATURE_NAMES = CORE_FEATURE_NAMES
EXTENDED_FEATURE_NAMES = CORE_FEATURE_NAMES + BURST_FEATURE_NAMES


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def feature_names_for_set(feature_set: str) -> list[str]:
    if feature_set == "core":
        return list(CORE_FEATURE_NAMES)
    if feature_set == "burst":
        return list(EXTENDED_FEATURE_NAMES)
    raise ValueError(f"Unknown feature set: {feature_set}")


def _window_density(times: np.ndarray, window_sec: float) -> tuple[float, float, np.ndarray]:
    if len(times) == 0:
        empty = np.zeros(0, dtype=np.float32)
        return 0.0, 0.0, empty

    counts = np.zeros(len(times), dtype=np.float32)
    right = 0
    for left, start in enumerate(times):
        while right < len(times) and times[right] - start <= window_sec:
            right += 1
        counts[left] = right - left
    densities = counts / max(window_sec, 1e-6)
    return float(np.max(densities)), float(np.mean(densities)), densities


def summarize_sequence(features: np.ndarray, *, feature_set: str = "core") -> np.ndarray:
    feature_names = feature_names_for_set(feature_set)
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 6 or len(array) == 0:
        return np.zeros(len(feature_names), dtype=np.float32)

    note_count = float(len(array))
    deltas = np.clip(array[:, 1], 0.0, None)
    columns = np.clip(array[:, 2], 0.0, 1.0)
    is_ln = np.clip(array[:, 3], 0.0, 1.0)
    ln_lengths = np.clip(array[:, 4], 0.0, None)
    chord_fraction = np.clip(array[:, 5], 0.0, None)
    chord_sizes = chord_fraction * 4.0
    times = np.cumsum(deltas)

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
        adjacent_same = column_changes < 1e-6
        adjacent_short = deltas[1:] <= 0.10
        adjacent_medium = deltas[1:] <= 0.15
        jack_ratio_100ms = float(np.mean(adjacent_same & adjacent_short))
        jack_ratio_150ms = float(np.mean(adjacent_same & adjacent_medium))

        if len(columns) > 2:
            alternating_back = (np.abs(columns[2:] - columns[:-2]) < 1e-6) & (
                np.abs(columns[1:-1] - columns[:-2]) > 1e-6
            )
            fast_pair = (deltas[1:-1] <= 0.12) & (deltas[2:] <= 0.12)
            trill_like_ratio_120ms = float(np.mean(alternating_back & fast_pair))
        else:
            trill_like_ratio_120ms = 0.0
    else:
        same_column_ratio = 0.0
        jump_column_ratio = 0.0
        column_change_mean = 0.0
        jack_ratio_100ms = 0.0
        jack_ratio_150ms = 0.0
        trill_like_ratio_120ms = 0.0

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

    peak_nps_1s, mean_nps_1s, density_1s = _window_density(times, 1.0)
    peak_nps_2s, _, _ = _window_density(times, 2.0)
    burst_mask = density_1s >= 10.0
    burst_10nps_ratio = float(np.mean(density_1s >= 10.0)) if len(density_1s) else 0.0
    burst_15nps_ratio = float(np.mean(density_1s >= 15.0)) if len(density_1s) else 0.0
    burst_chord_2_ratio = float(np.mean(chord_sizes[burst_mask] >= 2.0)) if np.any(burst_mask) else 0.0
    burst_chord_3_ratio = float(np.mean(chord_sizes[burst_mask] >= 3.0)) if np.any(burst_mask) else 0.0

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
        peak_nps_1s,
        peak_nps_2s,
        mean_nps_1s,
        burst_10nps_ratio,
        burst_15nps_ratio,
        jack_ratio_100ms,
        jack_ratio_150ms,
        trill_like_ratio_120ms,
        burst_chord_2_ratio,
        burst_chord_3_ratio,
    ]
    values_array = np.asarray(values, dtype=np.float32)
    if feature_set == "core":
        return values_array[: len(CORE_FEATURE_NAMES)]
    return values_array


def summarize_sequences(sequences: list[np.ndarray], *, feature_set: str = "core") -> np.ndarray:
    if not sequences:
        return np.empty((0, len(feature_names_for_set(feature_set))), dtype=np.float32)
    return np.stack(
        [summarize_sequence(sequence, feature_set=feature_set) for sequence in sequences]
    ).astype(np.float32)
