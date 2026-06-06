from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


def numeric_value(row: Mapping[str, Any], *columns: str, default: float = 0.0) -> float:
    for column in columns:
        value = row.get(column)
        if value in (None, ""):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return default


def event_times_sec(sequence: np.ndarray, row: Mapping[str, Any]) -> np.ndarray:
    length_sec = numeric_value(row, "length_sec", default=0.0)
    if length_sec <= 0:
        length_sec = numeric_value(row, "length_ms", default=0.0) / 1000.0
    if length_sec <= 0:
        length_sec = 1.0
    return np.clip(sequence[:, 0].astype("float64"), 0.0, None) * length_sec


def sequence_columns(sequence: np.ndarray, keys: int) -> np.ndarray:
    return np.rint(sequence[:, 2].astype("float64") * max(1, keys - 1)).astype(int)


def jack_flags(times_sec: np.ndarray, columns: np.ndarray, *, threshold_sec: float = 0.22) -> np.ndarray:
    flags = np.zeros(len(times_sec), dtype=bool)
    previous_by_column: dict[int, float] = {}
    for index, (time_sec, column) in enumerate(zip(times_sec, columns, strict=True)):
        previous_time = previous_by_column.get(int(column))
        if previous_time is not None and 0.0 < time_sec - previous_time <= threshold_sec:
            flags[index] = True
        previous_by_column[int(column)] = float(time_sec)
    return flags


def window_starts(duration_sec: float, *, window_sec: float, step_sec: float) -> np.ndarray:
    if duration_sec <= window_sec:
        return np.asarray([0.0], dtype="float64")
    return np.arange(0.0, duration_sec - window_sec + step_sec, step_sec, dtype="float64")


def compute_player_feel_curve(
    sequence: np.ndarray,
    row: Mapping[str, Any],
    *,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
) -> pd.DataFrame:
    if len(sequence) == 0:
        return pd.DataFrame()

    keys = max(1, int(numeric_value(row, "keys", "circle_size", "cs", default=4.0)))
    od = numeric_value(row, "overall_difficulty", "od", "accuracy", default=5.0)
    beatmap_id = int(numeric_value(row, "beatmap_id", default=0.0))
    times_sec = event_times_sec(sequence, row)
    duration_sec = max(float(times_sec.max(initial=0.0)), numeric_value(row, "length_ms", default=0.0) / 1000.0)
    duration_sec = max(duration_sec, window_sec)

    columns = sequence_columns(sequence, keys)
    chords = np.maximum(1.0, np.rint(sequence[:, 5].astype("float64") * keys))
    is_ln = sequence[:, 3].astype("float64") > 0.5
    ln_lengths = np.clip(sequence[:, 4].astype("float64"), 0.0, None)
    deltas = np.clip(sequence[:, 1].astype("float64"), 0.0, None)
    jacks = jack_flags(times_sec, columns)
    bursts = (deltas > 0.0) & (deltas <= 0.12)
    non_ln = ~is_ln
    od_multiplier = 1.0 + max(0.0, od - 5.0) / 20.0

    rows: list[dict[str, Any]] = []
    for start_sec in window_starts(duration_sec, window_sec=window_sec, step_sec=step_sec):
        end_sec = start_sec + window_sec
        mask = (times_sec >= start_sec) & (times_sec < end_sec)
        note_count = int(mask.sum())
        note_density = note_count / window_sec
        rice_density = float(non_ln[mask].sum()) / window_sec
        jack_density = float(jacks[mask].sum()) / window_sec
        chord_load = float(np.maximum(chords[mask] - 1.0, 0.0).sum()) / window_sec
        ln_density = float(is_ln[mask].sum()) / window_sec
        ln_load = float(ln_lengths[mask].sum()) / window_sec
        burst_density = float(bursts[mask].sum()) / window_sec
        reading_pressure = max(0.0, note_density - 1.0) * 0.35 + chord_load * 0.25
        speed_pressure = rice_density * 0.35 + burst_density * 0.90
        stamina_pressure = note_density * 0.25 + ln_load * 0.15
        jack_pressure = jack_density * 1.30
        chord_pressure = chord_load * 1.10
        ln_pressure = ln_density * 0.55 + ln_load * 0.45
        accuracy_pressure = (
            speed_pressure * 0.35
            + chord_pressure * 0.25
            + ln_pressure * 0.20
            + max(0.0, od - 5.0) * 0.10
        )

        feel_strain = (
            reading_pressure
            + speed_pressure
            + stamina_pressure
            + jack_pressure
            + chord_pressure
            + ln_pressure
            + accuracy_pressure
        ) * od_multiplier
        rows.append(
            {
                "beatmap_id": beatmap_id,
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "note_count": note_count,
                "note_density": note_density,
                "rice_density": rice_density,
                "burst_density": burst_density,
                "jack_density": jack_density,
                "chord_load": chord_load,
                "ln_density": ln_density,
                "ln_load": ln_load,
                "od_multiplier": od_multiplier,
                "reading_pressure": reading_pressure,
                "speed_pressure": speed_pressure,
                "stamina_pressure": stamina_pressure,
                "jack_pressure": jack_pressure,
                "chord_pressure": chord_pressure,
                "ln_pressure": ln_pressure,
                "accuracy_pressure": accuracy_pressure,
                "feel_strain": feel_strain,
            }
        )
    return pd.DataFrame(rows)


def _safe_quantile(values: pd.Series, quantile: float) -> float:
    if values.empty:
        return 0.0
    return float(values.quantile(quantile))


def summarize_player_feel_curve(curve: pd.DataFrame, row: Mapping[str, Any]) -> dict[str, Any]:
    if curve.empty:
        return {
            "beatmap_id": int(numeric_value(row, "beatmap_id", default=0.0)),
            "peak_strain": 0.0,
            "mean_strain": 0.0,
            "p90_strain": 0.0,
            "fatigue_area": 0.0,
            "peak_time_sec": 0.0,
            "dominant_skill": "",
        }
    peak_index = curve["feel_strain"].idxmax()
    skill_scores = {
        "reading": _safe_quantile(curve["reading_pressure"], 0.90),
        "speed": _safe_quantile(curve["speed_pressure"], 0.90),
        "rice": _safe_quantile(curve["rice_density"], 0.90),
        "jack": float(curve["jack_density"].max()),
        "chord": _safe_quantile(curve["chord_load"], 0.90),
        "ln": _safe_quantile(curve["ln_load"] + curve["ln_density"], 0.90),
        "stamina": float(curve["feel_strain"].mean()),
        "accuracy": _safe_quantile(curve["feel_strain"] * curve["od_multiplier"], 0.90),
    }
    dominant_candidates = {
        key: skill_scores[key]
        for key in ("reading", "speed", "jack", "chord", "ln")
    }
    dominant_skill = max(dominant_candidates.items(), key=lambda item: item[1])[0]
    return {
        "beatmap_id": int(numeric_value(row, "beatmap_id", default=0.0)),
        "title": row.get("title", ""),
        "artist": row.get("artist", ""),
        "version": row.get("version", ""),
        "keys": int(numeric_value(row, "keys", "circle_size", "cs", default=4.0)),
        "peak_strain": float(curve.loc[peak_index, "feel_strain"]),
        "mean_strain": float(curve["feel_strain"].mean()),
        "p90_strain": _safe_quantile(curve["feel_strain"], 0.90),
        "fatigue_area": float(curve["feel_strain"].sum()),
        "peak_time_sec": float(curve.loc[peak_index, "start_sec"]),
        "dominant_skill": dominant_skill,
        **{f"{skill}_score": float(score) for skill, score in skill_scores.items()},
        "reading_pressure": _safe_quantile(curve["reading_pressure"], 0.90),
        "speed_pressure": _safe_quantile(curve["speed_pressure"], 0.90),
        "stamina_pressure": _safe_quantile(curve["stamina_pressure"], 0.90),
        "jack_pressure": _safe_quantile(curve["jack_pressure"], 0.90),
        "chord_pressure": _safe_quantile(curve["chord_pressure"], 0.90),
        "ln_pressure": _safe_quantile(curve["ln_pressure"], 0.90),
        "accuracy_pressure": _safe_quantile(curve["accuracy_pressure"], 0.90),
    }


def plot_top_curves(curves: pd.DataFrame, summaries: pd.DataFrame, out_path: Path, *, top_n: int = 12) -> None:
    if curves.empty or summaries.empty:
        return
    top_ids = summaries.sort_values("peak_strain", ascending=False).head(top_n)["beatmap_id"].tolist()
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for beatmap_id in top_ids:
        curve = curves[curves["beatmap_id"] == beatmap_id]
        if curve.empty:
            continue
        label = str(beatmap_id)
        title_rows = summaries[summaries["beatmap_id"] == beatmap_id]
        if not title_rows.empty:
            label = f"{beatmap_id} {title_rows.iloc[0].get('dominant_skill', '')}"
        ax.plot(curve["start_sec"], curve["feel_strain"], linewidth=1.4, alpha=0.8, label=label)
    ax.set_title("Top Player-Feel Difficulty Curves")
    ax.set_xlabel("Time (sec)")
    ax.set_ylabel("Feel strain")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_player_feel_curves(
    labels_csv: Path,
    sequences_dir: Path,
    out_dir: Path,
    *,
    window_sec: float = 2.0,
    step_sec: float = 0.5,
    top_n: int = 12,
) -> None:
    labels = pd.read_csv(labels_csv)
    curve_frames: list[pd.DataFrame] = []
    summaries: list[dict[str, Any]] = []
    for row in tqdm(labels.to_dict("records"), desc="feel"):
        beatmap_id = int(numeric_value(row, "beatmap_id", default=0.0))
        sequence_path = sequences_dir / f"{beatmap_id}.npy"
        if not sequence_path.exists():
            continue
        sequence = np.load(sequence_path).astype("float32")
        curve = compute_player_feel_curve(sequence, row, window_sec=window_sec, step_sec=step_sec)
        if curve.empty:
            continue
        curve_frames.append(curve)
        summaries.append(summarize_player_feel_curve(curve, row))

    out_dir.mkdir(parents=True, exist_ok=True)
    curves = pd.concat(curve_frames, ignore_index=True) if curve_frames else pd.DataFrame()
    summary = pd.DataFrame(summaries)
    curves.to_csv(out_dir / "player_feel_curve.csv", index=False, encoding="utf-8")
    summary.to_csv(out_dir / "player_feel_summary.csv", index=False, encoding="utf-8")
    plot_top_curves(curves, summary, out_dir / "player_feel_curves.png", top_n=top_n)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract local osu!mania player-feel difficulty curves.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/player_feel_curves"))
    parser.add_argument("--window-sec", type=float, default=2.0)
    parser.add_argument("--step-sec", type=float, default=0.5)
    parser.add_argument("--top-n", type=int, default=12)
    args = parser.parse_args()

    write_player_feel_curves(
        args.labels,
        args.sequences,
        args.out_dir,
        window_sec=args.window_sec,
        step_sec=args.step_sec,
        top_n=args.top_n,
    )
    print(f"Wrote player-feel curves to {args.out_dir}")


if __name__ == "__main__":
    main()
