from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from mania_difficulty.data.dataset import DEFAULT_TARGET_COLUMNS, ManiaDifficultyDataset
from mania_difficulty.metrics import regression_report
from mania_difficulty.train import (
    create_tabular_forest_model,
    cross_validation_splits,
    dataset_groups,
    parse_max_features,
    repeat_baseline,
    tabular_arrays,
)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    parsed = [int(item) for item in items]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def parse_max_features_list(value: str) -> list[str | float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one max_features value.")
    return [parse_max_features(item) for item in items]


def parse_feature_sets(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"core", "burst"}
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown feature sets: {invalid}")
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one feature set.")
    return items


def forest_grid(
    *,
    trees: list[int],
    min_samples_leaf: list[int],
    max_features: list[str | float],
    feature_sets: list[str] | None = None,
) -> list[dict[str, Any]]:
    feature_sets = feature_sets or ["core"]
    candidates = []
    for feature_set in feature_sets:
        for tree_count in trees:
            for leaf_count in min_samples_leaf:
                for feature_count in max_features:
                    candidates.append(
                        {
                            "candidate_id": (
                                f"{feature_set}_trees{tree_count}_leaf{leaf_count}_feat{feature_count}"
                            ),
                            "feature_set": feature_set,
                            "forest_trees": tree_count,
                            "forest_min_samples_leaf": leaf_count,
                            "forest_max_features": feature_count,
                        }
                    )
    return candidates


def choose_best_candidate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not summary_rows:
        raise ValueError("No candidate rows to choose from.")
    return sorted(
        summary_rows,
        key=lambda row: (
            float(row["mean_mae"]),
            int(row["forest_trees"]),
            -int(row["forest_min_samples_leaf"]),
            str(row.get("feature_set", "")),
            str(row.get("forest_max_features", "")),
        ),
    )[0]


def candidate_args(base_args: argparse.Namespace, candidate: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        forest_trees=int(candidate["forest_trees"]),
        forest_min_samples_leaf=int(candidate["forest_min_samples_leaf"]),
        forest_max_features=candidate["forest_max_features"],
        feature_set=candidate.get("feature_set", "core"),
        workers=base_args.workers,
    )


def evaluate_candidate(
    base_args: argparse.Namespace,
    candidate: dict[str, Any],
    y_all: np.ndarray,
    target_columns: list[str],
    splits: list[tuple[list[int], list[int]]],
    feature_cache: dict[str, np.ndarray],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    x_all = feature_cache[candidate.get("feature_set", "core")]
    oof_pred = np.zeros_like(y_all, dtype="float32")
    oof_baseline = np.zeros_like(y_all, dtype="float32")
    fold_rows = []
    model_args = candidate_args(base_args, candidate)

    for fold_index, (train_idx, val_idx) in enumerate(splits, start=1):
        train_idx_array = np.asarray(train_idx, dtype=int)
        val_idx_array = np.asarray(val_idx, dtype=int)
        model = create_tabular_forest_model(model_args, seed=base_args.seed + fold_index)
        model.fit(x_all[train_idx_array], y_all[train_idx_array])
        fold_pred = model.predict(x_all[val_idx_array]).astype("float32")
        fold_baseline = repeat_baseline(y_all[val_idx_array], y_all[train_idx_array].mean(axis=0))
        oof_pred[val_idx_array] = fold_pred
        oof_baseline[val_idx_array] = fold_baseline

        fold_report = regression_report(
            y_all[val_idx_array],
            fold_pred,
            target_columns,
            baseline_pred=fold_baseline,
        )
        for target in target_columns:
            fold_rows.append(
                {
                    **candidate,
                    "fold": fold_index,
                    "target": target,
                    "val_size": len(val_idx_array),
                    **fold_report[target],
                }
            )

    report = regression_report(y_all, oof_pred, target_columns, baseline_pred=oof_baseline)
    target_rows = [
        {
            **candidate,
            "target": target,
            **report[target],
        }
        for target in target_columns
    ]
    summary = {
        **candidate,
        "mean_mae": float(np.mean([row["mae"] for row in target_rows])),
        "mean_r2": float(np.mean([row["r2"] for row in target_rows])),
        "mean_baseline_mae": float(np.mean([row["baseline_mae"] for row in target_rows])),
        "mean_improvement_pct": float(np.mean([row["mae_improvement_pct"] for row in target_rows])),
    }
    return summary, target_rows + fold_rows


def write_html_report(summary_rows: list[dict[str, Any]], best: dict[str, Any], out_html: Path) -> None:
    frame = pd.DataFrame(summary_rows).sort_values(["mean_mae", "forest_trees"])
    table = frame.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>forest sweep report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Forest Parameter Sweep</h1>
  <p>Best candidate: <code>{html.escape(str(best["candidate_id"]))}</code></p>
  <p>Lower mean MAE is better. Positive mean improvement means the model beats the train-mean baseline.</p>
  {table}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep tabular_forest parameters with group-aware CV.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/forest_sweep"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--max-notes", type=int, default=3000)
    parser.add_argument("--group-column", default="beatmapset_id")
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trees", type=parse_int_list, default=[200, 400])
    parser.add_argument("--min-samples-leaf", type=parse_int_list, default=[1, 2, 4])
    parser.add_argument("--max-features", type=parse_max_features_list, default=["sqrt", 0.75])
    parser.add_argument("--feature-sets", type=parse_feature_sets, default=["core", "burst"])
    parser.add_argument("--workers", type=int, default=-1)
    args = parser.parse_args()

    target_columns = [column.strip() for column in args.targets.split(",") if column.strip()]
    dataset = ManiaDifficultyDataset(args.labels, args.sequences, target_columns=target_columns)
    groups = dataset_groups(dataset, args.group_column)
    splits = cross_validation_splits(len(dataset), groups=groups, folds=args.cv_folds, seed=args.seed)
    if not splits:
        raise RuntimeError("Need at least 2 CV folds for sweep.")

    _, y_all, _ = tabular_arrays(
        dataset,
        list(range(len(dataset))),
        max_notes=args.max_notes,
        feature_set="core",
    )
    feature_cache = {
        feature_set: tabular_arrays(
            dataset,
            list(range(len(dataset))),
            max_notes=args.max_notes,
            feature_set=feature_set,
        )[0]
        for feature_set in args.feature_sets
    }
    candidates = forest_grid(
        trees=args.trees,
        min_samples_leaf=args.min_samples_leaf,
        max_features=args.max_features,
        feature_sets=args.feature_sets,
    )

    summary_rows = []
    detail_rows = []
    for candidate in candidates:
        print(f"Evaluating {candidate['candidate_id']}")
        summary, details = evaluate_candidate(
            args,
            candidate,
            y_all,
            target_columns,
            splits,
            feature_cache,
        )
        summary_rows.append(summary)
        detail_rows.extend(details)

    best = choose_best_candidate(summary_rows)
    best_params = {
        **best,
        "cv_folds": args.cv_folds,
        "split_strategy": f"group:{args.group_column}" if groups else "random_map",
        "group_column": args.group_column if groups else "",
        "group_count": len(set(groups)) if groups else 0,
        "sample_size": len(dataset),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.out_dir / "sweep_summary.csv"
    detail_csv = args.out_dir / "sweep_details.csv"
    pd.DataFrame(summary_rows).sort_values(["mean_mae", "forest_trees"]).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding="utf-8")
    (args.out_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_html_report(summary_rows, best, args.out_dir / "sweep_report.html")
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_csv}, {detail_csv}, and {args.out_dir / 'sweep_report.html'}")


if __name__ == "__main__":
    main()
