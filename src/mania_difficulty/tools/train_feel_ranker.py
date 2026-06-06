from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mania_difficulty.player_feel_annotations import read_player_feel_pairs, usable_judgment_rows
from mania_difficulty.player_stages import ABILITY_COLUMNS, load_player_stages, stage_vector


SUMMARY_FEATURES = [
    "peak_strain",
    "mean_strain",
    "p90_strain",
    "fatigue_area",
    "reading_score",
    "speed_score",
    "rice_score",
    "jack_score",
    "chord_score",
    "ln_score",
    "stamina_score",
    "accuracy_score",
    "reading_pressure",
    "speed_pressure",
    "stamina_pressure",
    "jack_pressure",
    "chord_pressure",
    "ln_pressure",
    "accuracy_pressure",
]

PRESSURE_TO_ABILITY = {
    "reading_pressure": "reading",
    "speed_pressure": "speed",
    "stamina_pressure": "stamina",
    "jack_pressure": "jack",
    "chord_pressure": "chord",
    "ln_pressure": "ln",
    "accuracy_pressure": "accuracy",
}


def _float(value: object, default: float = 0.0) -> float:
    if pd.isna(value) or value == "":
        return default
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _summary_row(summaries: pd.DataFrame, beatmap_id: int) -> pd.Series:
    matches = summaries[summaries["beatmap_id"].astype(int) == int(beatmap_id)]
    if matches.empty:
        raise KeyError(f"No player-feel summary for beatmap_id {beatmap_id}")
    return matches.iloc[0]


def segment_summary(
    curves: pd.DataFrame,
    summaries: pd.DataFrame,
    beatmap_id: int,
    start_sec: float | None,
    end_sec: float | None,
) -> dict[str, float]:
    base = {column: _float(_summary_row(summaries, beatmap_id).get(column, 0.0)) for column in SUMMARY_FEATURES}
    if start_sec is None or end_sec is None or curves.empty:
        return base
    rows = curves[curves["beatmap_id"].astype(int) == int(beatmap_id)]
    rows = rows[(pd.to_numeric(rows["start_sec"], errors="coerce") >= start_sec) & (pd.to_numeric(rows["end_sec"], errors="coerce") <= end_sec)]
    if rows.empty:
        return base
    base.update(
        {
            "peak_strain": float(pd.to_numeric(rows["feel_strain"], errors="coerce").max()),
            "mean_strain": float(pd.to_numeric(rows["feel_strain"], errors="coerce").mean()),
            "p90_strain": float(pd.to_numeric(rows["feel_strain"], errors="coerce").quantile(0.90)),
            "fatigue_area": float(pd.to_numeric(rows["feel_strain"], errors="coerce").sum()),
        }
    )
    for pressure in PRESSURE_TO_ABILITY:
        if pressure in rows.columns:
            base[pressure] = float(pd.to_numeric(rows[pressure], errors="coerce").quantile(0.90))
    return base


def item_features(
    row: pd.Series,
    side: str,
    summaries: pd.DataFrame,
    curves: pd.DataFrame,
    stages: pd.DataFrame,
) -> dict[str, float]:
    beatmap_id = int(_float(row[f"{side}_beatmap_id"]))
    start_raw = row.get(f"{side}_start_sec", "")
    end_raw = row.get(f"{side}_end_sec", "")
    start_sec = None if pd.isna(start_raw) or str(start_raw).strip() == "" else _float(start_raw)
    end_sec = None if pd.isna(end_raw) or str(end_raw).strip() == "" else _float(end_raw)
    stage = stage_vector(stages, str(row["player_stage"]))
    base = segment_summary(curves, summaries, beatmap_id, start_sec, end_sec)

    features = {f"raw_{column}": float(base.get(column, 0.0)) for column in SUMMARY_FEATURES}
    for ability in ABILITY_COLUMNS:
        features[f"ability_{ability}"] = float(stage[ability])
    for pressure, ability in PRESSURE_TO_ABILITY.items():
        features[f"stage_{pressure}"] = float(base.get(pressure, 0.0)) * (1.0 - float(stage[ability]))
    features["stage_pattern_memory_pressure"] = float(base.get("peak_strain", 0.0)) * (
        1.0 - float(stage["pattern_memory"])
    )
    features["stage_total_pressure"] = sum(
        value for key, value in features.items() if key.startswith("stage_") and key.endswith("_pressure")
    )
    return features


def build_pairwise_training_frame(
    judgments: pd.DataFrame,
    summaries: pd.DataFrame,
    curves: pd.DataFrame,
    stages: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    usable = usable_judgment_rows(judgments)
    feature_rows: list[dict[str, float]] = []
    targets: list[int] = []
    weights: list[float] = []
    kept_rows: list[dict[str, Any]] = []
    for _, row in usable.iterrows():
        try:
            a_features = item_features(row, "a", summaries, curves, stages)
            b_features = item_features(row, "b", summaries, curves, stages)
        except (KeyError, ValueError):
            continue
        feature_names = sorted(set(a_features) | set(b_features))
        feature_rows.append(
            {name: float(a_features.get(name, 0.0)) - float(b_features.get(name, 0.0)) for name in feature_names}
        )
        targets.append(1 if row["normalized_harder_choice"] == "a" else 0)
        weights.append(float(row["sample_weight"]))
        kept_rows.append(row.to_dict())
    if not feature_rows:
        return pd.DataFrame(), np.asarray([], dtype=int), np.asarray([], dtype=float), pd.DataFrame()
    frame = pd.DataFrame(feature_rows).fillna(0.0)
    return frame, np.asarray(targets, dtype=int), np.asarray(weights, dtype=float), pd.DataFrame(kept_rows)


def agreement(model: Pipeline, features: pd.DataFrame, targets: np.ndarray) -> float:
    if len(features) == 0:
        return 0.0
    predictions = model.predict(features)
    return float(np.mean(predictions == targets))


def write_report(out_html: Path, metrics: dict[str, Any], predictions: pd.DataFrame) -> None:
    rows = "".join(
        f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>"
        for key, value in metrics.items()
    )
    preview = predictions.head(30).to_html(index=False, float_format=lambda value: f"{value:.4f}")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>player-feel ranker</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Player-Feel Ranker</h1>
  <table><tbody>{rows}</tbody></table>
  <h2>Pair Predictions</h2>
  {preview}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def train_feel_ranker(
    judgments_csv: Path,
    summary_csv: Path,
    curves_csv: Path,
    stages_csv: Path,
    out_dir: Path,
    *,
    test_size: float = 0.25,
    seed: int = 42,
) -> dict[str, Any]:
    judgments = read_player_feel_pairs(judgments_csv)
    summaries = pd.read_csv(summary_csv)
    curves = pd.read_csv(curves_csv) if curves_csv.exists() else pd.DataFrame()
    stages = load_player_stages(stages_csv)
    features, targets, weights, kept = build_pairwise_training_frame(judgments, summaries, curves, stages)
    ignored = int(len(judgments) - len(kept))
    if len(features) < 2:
        raise RuntimeError("Need at least two usable player-feel pair judgments.")
    if len(set(targets.tolist())) < 2:
        raise RuntimeError("Need both A-harder and B-harder examples for logistic pairwise training.")

    indices = np.arange(len(features))
    if test_size > 0 and len(features) >= 4:
        train_idx, test_idx = train_test_split(
            indices,
            test_size=test_size,
            random_state=seed,
            stratify=targets if min(np.bincount(targets)) >= 2 else None,
        )
    else:
        train_idx = indices
        test_idx = np.asarray([], dtype=int)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("ranker", LogisticRegression(max_iter=1000, random_state=seed)),
        ]
    )
    model.fit(features.iloc[train_idx], targets[train_idx], ranker__sample_weight=weights[train_idx])

    score = model.predict_proba(features)[:, 1]
    predictions = kept.copy()
    predictions["target_a_harder"] = targets
    predictions["pred_a_harder_probability"] = score
    predictions["predicted_choice"] = np.where(score >= 0.5, "a", "b")
    predictions["correct"] = predictions["predicted_choice"] == predictions["normalized_harder_choice"]

    metrics = {
        "usable_judgments": int(len(features)),
        "ignored_judgments": ignored,
        "feature_count": int(features.shape[1]),
        "train_rows": int(len(train_idx)),
        "test_rows": int(len(test_idx)),
        "train_agreement": agreement(model, features.iloc[train_idx], targets[train_idx]),
        "holdout_agreement": agreement(model, features.iloc[test_idx], targets[test_idx]) if len(test_idx) else 0.0,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_columns": features.columns.tolist(),
            "metrics": metrics,
        },
        out_dir / "feel_ranker.joblib",
    )
    predictions.to_csv(out_dir / "pair_predictions.csv", index=False, encoding="utf-8")
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_report(out_dir / "feel_ranker_report.html", metrics, predictions)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a 4K player-feel pairwise ranker.")
    parser.add_argument("--judgments", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--curves", type=Path, required=True)
    parser.add_argument("--player-stages", type=Path, default=Path("data/player_stages_4k.csv"))
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/player_feel_ranker"))
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    metrics = train_feel_ranker(
        args.judgments,
        args.summary,
        args.curves,
        args.player_stages,
        args.out_dir,
        test_size=args.test_size,
        seed=args.seed,
    )
    print(metrics)
    print(f"Wrote player-feel ranker to {args.out_dir}")


if __name__ == "__main__":
    main()
