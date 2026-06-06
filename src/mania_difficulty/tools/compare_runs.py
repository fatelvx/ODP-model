from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def metrics_rows(run_dir: Path, metrics_path: Path, evaluation: str) -> list[dict[str, object]]:
    if not metrics_path.exists():
        raise FileNotFoundError(f"No {metrics_path.name} found in {run_dir}")

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    run_info = metrics.get("_run", {})
    rows = []
    for target, values in metrics.items():
        if target.startswith("_") or not isinstance(values, dict):
            continue
        rows.append(
            {
                "run": run_dir.name,
                "model_name": run_info.get("model_name"),
                "seed": run_info.get("seed"),
                "evaluation": run_info.get("evaluation", evaluation),
                "feature_set": run_info.get("feature_set"),
                "split_strategy": run_info.get("split_strategy"),
                "group_column": run_info.get("group_column"),
                "group_count": run_info.get("group_count"),
                "target": target,
                "mae": values.get("mae"),
                "r2": values.get("r2"),
                "spearman": values.get("spearman"),
                "pairwise_order_accuracy": values.get("pairwise_order_accuracy"),
                "baseline_mae": values.get("baseline_mae"),
                "mae_improvement_vs_baseline": values.get("mae_improvement_vs_baseline"),
                "mae_improvement_pct": values.get("mae_improvement_pct"),
                "difficulty_rating_baseline_mae": values.get("difficulty_rating_baseline_mae"),
                "mae_improvement_vs_difficulty_rating_baseline": values.get(
                    "mae_improvement_vs_difficulty_rating_baseline"
                ),
                "mae_improvement_pct_vs_difficulty_rating_baseline": values.get(
                    "mae_improvement_pct_vs_difficulty_rating_baseline"
                ),
                "best_val_loss": run_info.get("best_val_loss"),
                "test_loss": run_info.get("test_loss"),
                "cv_folds": run_info.get("cv_folds"),
                "train_size": run_info.get("train_size"),
                "val_size": run_info.get("val_size"),
                "test_size": run_info.get("test_size"),
                "sample_size": run_info.get("sample_size"),
            }
        )
    return rows


def run_metrics_rows(run_dir: Path) -> list[dict[str, object]]:
    rows = metrics_rows(run_dir, run_dir / "metrics.json", "holdout")
    cv_metrics_path = run_dir / "cv_metrics.json"
    if cv_metrics_path.exists():
        rows.extend(metrics_rows(run_dir, cv_metrics_path, "cv_oof"))
    return rows


def write_comparison_report(rows: list[dict[str, object]], out_html: Path) -> None:
    frame = pd.DataFrame(rows).sort_values(["evaluation", "target", "mae", "run"])
    table = frame.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mania difficulty run comparison</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  </style>
</head>
<body>
  <h1>Run Comparison</h1>
  <p>Lower MAE is better. Positive improvement means the model beats the named baseline: train-mean for <code>mae_improvement_pct</code>, or difficulty-rating linear fit for difficulty-rating columns.</p>
  {table}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare saved training runs by metrics.json.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out-csv", type=Path, default=Path("outputs/run_comparison.csv"))
    parser.add_argument("--out-html", type=Path, default=Path("outputs/run_comparison.html"))
    args = parser.parse_args()

    rows = []
    for run_dir in args.runs:
        rows.extend(run_metrics_rows(run_dir))

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows).sort_values(["evaluation", "target", "mae", "run"])
    frame.to_csv(args.out_csv, index=False, encoding="utf-8")
    write_comparison_report(rows, args.out_html)
    print(frame.to_string(index=False))
    print(f"Wrote {args.out_csv} and {args.out_html}")


if __name__ == "__main__":
    main()
