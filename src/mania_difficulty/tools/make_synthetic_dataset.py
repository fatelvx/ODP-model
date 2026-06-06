from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def make_map(rng: np.random.Generator, beatmap_id: int) -> tuple[np.ndarray, dict[str, float]]:
    note_count = int(rng.integers(180, 900))
    density = float(rng.uniform(0.6, 2.4))
    ln_rate = float(rng.uniform(0.0, 0.45))
    chord_rate = float(rng.uniform(0.05, 0.55))
    column_bias = float(rng.uniform(-0.35, 0.35))

    deltas = rng.exponential(1.0 / density, size=note_count).clip(0.035, 1.8)
    times = np.cumsum(deltas)
    time_norm = times / max(times[-1], 1e-6)
    columns = rng.choice(4, size=note_count, p=_column_probs(column_bias)) / 3.0
    is_ln = rng.random(note_count) < ln_rate
    ln_length = np.where(is_ln, rng.uniform(0.08, 0.9, size=note_count), 0.0)
    chord_size = np.where(rng.random(note_count) < chord_rate, rng.integers(2, 5, size=note_count), 1) / 4.0

    features = np.stack(
        [time_norm, deltas, columns, is_ln.astype(float), ln_length, chord_size],
        axis=1,
    ).astype("float32")

    strain = (
        0.22 * math.log1p(note_count / 120)
        + 0.20 * density
        + 0.55 * ln_rate
        + 0.38 * chord_rate
        + 0.18 * abs(column_bias)
    )
    noise = float(rng.normal(0, 0.012))
    mean_acc = float(np.clip(1.01 - 0.105 * strain + noise, 0.65, 0.995))
    acc_std = float(np.clip(0.012 + 0.035 * strain + rng.normal(0, 0.004), 0.003, 0.14))
    skill_gap = float(np.clip(0.02 + 0.065 * strain + rng.normal(0, 0.006), 0.005, 0.22))
    difficulty_rating = float(np.clip(0.5 + 2.8 * strain + rng.normal(0, 0.08), 0.5, 10.0))

    label = {
        "beatmap_id": beatmap_id,
        "difficulty_rating": difficulty_rating,
        "mean_acc": mean_acc,
        "acc_std": acc_std,
        "skill_gap": skill_gap,
        "median_acc": mean_acc + 0.003,
        "p10_acc": mean_acc - acc_std,
        "p90_acc": mean_acc + acc_std,
        "score_count": 100,
        "num_notes": note_count,
        "length_ms": int(times[-1] * 1000),
        "bpm": 120 + density * 40,
    }
    return features, label


def _column_probs(bias: float) -> np.ndarray:
    base = np.array([0.25 - bias / 4, 0.25 + bias / 4, 0.25 + bias / 4, 0.25 - bias / 4])
    base = np.clip(base, 0.05, 0.7)
    return base / base.sum()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a synthetic dataset for smoke tests.")
    parser.add_argument("--maps", type=int, default=96)
    parser.add_argument("--out", type=Path, default=Path("data/processed/synthetic"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    seq_dir = args.out / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)
    labels = []
    for index in range(args.maps):
        beatmap_id = 900000 + index
        features, label = make_map(rng, beatmap_id)
        np.save(seq_dir / f"{beatmap_id}.npy", features)
        labels.append(label)

    labels_path = args.out / "labels.csv"
    with labels_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(labels[0].keys()))
        writer.writeheader()
        writer.writerows(labels)

    print(f"Wrote {len(labels)} synthetic maps to {args.out}")


if __name__ == "__main__":
    main()
