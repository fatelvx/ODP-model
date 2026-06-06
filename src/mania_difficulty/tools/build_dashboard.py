from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd

from mania_difficulty.human_judgments import score_pair_judgments
from mania_difficulty.tools.compare_runs import run_metrics_rows
from mania_difficulty.visualize import (
    checkpoint_score_text,
    metric_float,
    model_verdict_html,
    model_verdict_summary,
    training_health_html,
    training_health_summary,
    training_performance_html,
    training_performance_summary,
    worst_error_slices_html,
)

DECISION_COLUMNS = [
    "run",
    "model_name",
    "evaluation",
    "device",
    "amp_enabled",
    "effective_batch_size",
    "grad_clip_norm",
    "sample_weight_column",
    "sample_weight_train_mean",
    "sample_weight_train_downweighted_rate",
    "human_judged_pairs",
    "human_judgment_coverage_rate",
    "human_model_agreement_rate",
    "human_proxy_agreement_rate",
    "human_model_vs_proxy_delta",
    "epochs_completed",
    "stop_reason",
    "early_stopped",
    "curve_best_epoch",
    "curve_best_val_loss",
    "curve_best_val_mean_mae",
    "curve_best_val_pairwise_order",
    "curve_final_val_mean_mae",
    "curve_final_val_pairwise_order",
    "curve_overfit_signal",
    "avg_epoch_seconds",
    "peak_cuda_memory_mb",
    "targets",
    "mean_mae",
    "mean_pairwise_order_accuracy",
    "targets_beating_baseline",
    "targets_beating_difficulty_rating",
    "weakest_target",
    "calibration_mean_abs_bias",
    "calibration_worst_bias_target",
    "calibration_worst_bias",
    "calibration_mean_pred_std_ratio",
    "calibration_warning",
    "training_adjustment",
    "next_action",
]

JUDGMENT_FILES = [
    ("holdout", "human_pair_judgment_template.csv"),
    ("cv_oof", "cv_human_pair_judgment_template.csv"),
    ("checkpoint_eval", "eval_human_pair_judgment_template.csv"),
]

PREDICTION_SUMMARY_FILES = {
    "holdout": "prediction_summary.csv",
    "cv_oof": "cv_prediction_summary.csv",
    "checkpoint_eval": "eval_prediction_summary.csv",
}


def href(path: Path, out_html: Path) -> str:
    return html.escape(os.path.relpath(path, start=out_html.parent).replace("\\", "/"))


def link(path: Path, label: str, out_html: Path) -> str:
    if not path.exists():
        return ""
    return f'<a href="{href(path, out_html)}">{html.escape(label)}</a>'


def image(path: Path, alt: str, out_html: Path) -> str:
    if not path.exists():
        return ""
    return f'<p><img src="{href(path, out_html)}" alt="{html.escape(alt)}"></p>'


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def format_percent(value: Any) -> str:
    try:
        return f"{float(value):.2%}"
    except (TypeError, ValueError):
        return str(value)


def prediction_calibration_summary(run_dir: Path, evaluation: str) -> dict[str, object]:
    filename = PREDICTION_SUMMARY_FILES.get(evaluation)
    if not filename:
        return {}
    path = run_dir / filename
    if not path.exists():
        return {}
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    if not {"target", "bias"}.issubset(frame.columns):
        return {}

    frame = frame.copy()
    frame["bias"] = pd.to_numeric(frame["bias"], errors="coerce")
    frame = frame.dropna(subset=["bias"])
    if frame.empty:
        return {}

    frame["_abs_bias"] = frame["bias"].abs()
    worst_bias_row = frame.loc[frame["_abs_bias"].idxmax()]
    summary: dict[str, object] = {
        "calibration_mean_abs_bias": float(frame["_abs_bias"].mean()),
        "calibration_worst_bias_target": str(worst_bias_row["target"]),
        "calibration_worst_bias": float(worst_bias_row["bias"]),
    }

    if "mae" in frame.columns:
        frame["mae"] = pd.to_numeric(frame["mae"], errors="coerce")
        worst_mae = metric_float(worst_bias_row.get("mae"))
        if worst_mae is not None and worst_mae > 1e-9:
            summary["calibration_worst_bias_to_mae"] = (
                abs(float(worst_bias_row["bias"])) / worst_mae
            )

    if {"actual_std", "pred_std"}.issubset(frame.columns):
        frame["actual_std"] = pd.to_numeric(frame["actual_std"], errors="coerce")
        frame["pred_std"] = pd.to_numeric(frame["pred_std"], errors="coerce")
        spread = frame.dropna(subset=["actual_std", "pred_std"])
        spread = spread[spread["actual_std"] > 1e-9].copy()
        if not spread.empty:
            spread["_pred_std_ratio"] = spread["pred_std"] / spread["actual_std"]
            summary["calibration_mean_pred_std_ratio"] = float(spread["_pred_std_ratio"].mean())
            low_spread_row = spread.loc[spread["_pred_std_ratio"].idxmin()]
            summary["calibration_low_spread_target"] = str(low_spread_row["target"])
            summary["calibration_low_spread_ratio"] = float(low_spread_row["_pred_std_ratio"])

    warning = calibration_warning_text(summary)
    if warning:
        summary["calibration_warning"] = warning
    return summary


def calibration_warning_text(summary: dict[str, object]) -> str:
    spread_ratio = metric_float(summary.get("calibration_mean_pred_std_ratio"))
    if spread_ratio is not None and spread_ratio < 0.35:
        target = str(
            summary.get("calibration_low_spread_target")
            or summary.get("calibration_worst_bias_target")
            or "the weakest target"
        )
        return (
            "Predictions are too compressed toward the mean; inspect "
            f"{target} features, train longer, or raise model capacity."
        )

    bias_ratio = metric_float(summary.get("calibration_worst_bias_to_mae"))
    if bias_ratio is not None and bias_ratio >= 0.75:
        target = str(summary.get("calibration_worst_bias_target") or "the weakest target")
        bias = metric_float(summary.get("calibration_worst_bias"))
        direction = "high" if bias is not None and bias > 0 else "low"
        return (
            f"{target} predictions are biased {direction}; check label quality "
            "and feature coverage before a longer run."
        )
    return ""


def audit_label_reliability_html(summary: dict[str, Any]) -> str:
    reliability = summary.get("label_reliability", {})
    if not isinstance(reliability, dict) or not reliability.get("score_count_available"):
        return ""

    fields = [
        ("Full Top100 Rows", "full_top100_rows", str),
        ("Full Top100 Rate", "full_top100_rate", format_percent),
        ("Low Score Count Rows", "low_score_count_rows", str),
        ("Low Score Count Rate", "low_score_count_rate", format_percent),
        ("Low Score Count Threshold", "low_score_count_threshold", str),
        ("Min Score Count", "min_score_count", str),
        ("Median Score Count", "median_score_count", str),
    ]
    rows = []
    for label, key, formatter in fields:
        if key not in reliability:
            continue
        rows.append(
            f"<tr><th>{html.escape(label)}</th>"
            f"<td>{html.escape(formatter(reliability.get(key)))}</td></tr>"
        )
    if not rows:
        return ""
    return f"<h3>Label Reliability</h3><table><tbody>{''.join(rows)}</tbody></table>"


def audit_section(audit_dir: Path | None, out_html: Path) -> str:
    if not audit_dir or not audit_dir.exists():
        return ""
    summary = load_json(audit_dir / "dataset_audit.json")
    rows = []
    for label, key in [
        ("Usable rows", "usable_rows"),
        ("Missing sequences", "missing_sequence_count"),
        ("Group count", "group_count"),
    ]:
        rows.append(f"<tr><th>{html.escape(label)}</th><td>{html.escape(str(summary.get(key, '')))}</td></tr>")
    links = " ".join(
        item
        for item in [
            link(audit_dir / "dataset_audit.html", "HTML report", out_html),
            link(audit_dir / "dataset_audit.json", "JSON", out_html),
            link(audit_dir / "missing_sequences.csv", "missing sequences", out_html),
        ]
        if item
    )
    return (
        "<section><h2>Dataset Audit</h2>"
        f"<table><tbody>{''.join(rows)}</tbody></table>"
        f"{audit_label_reliability_html(summary)}"
        f"<p>{links}</p>"
        f"{image(audit_dir / 'dataset_distributions.png', 'Dataset distributions', out_html)}"
        "</section>"
    )


def best_params_section(title: str, sweep_dir: Path | None, report_name: str, out_html: Path) -> str:
    if not sweep_dir or not sweep_dir.exists():
        return ""
    best = load_json(sweep_dir / "best_params.json")
    if not best:
        return ""
    rows = []
    for key in [
        "candidate_id",
        "selection_metric",
        "mean_mae",
        "mean_pairwise_order_accuracy",
        "mean_spearman",
        "mean_improvement_pct",
        "feature_set",
        "model",
    ]:
        if key in best:
            rows.append(f"<tr><th>{html.escape(key)}</th><td>{html.escape(str(best[key]))}</td></tr>")
    report_link = link(sweep_dir / report_name, "HTML report", out_html)
    summary_link = link(sweep_dir / "sweep_summary.csv", "summary CSV", out_html)
    if report_name.startswith("neural"):
        summary_link = link(sweep_dir / "neural_sweep_summary.csv", "summary CSV", out_html)
    return (
        f"<section><h2>{html.escape(title)}</h2>"
        f"<table><tbody>{''.join(rows)}</tbody></table>"
        f"<p>{' '.join(item for item in [report_link, summary_link] if item)}</p>"
        "</section>"
    )


def metrics_table(run_dirs: list[Path]) -> str:
    rows = []
    for run_dir in run_dirs:
        try:
            rows.extend(run_metrics_rows(run_dir))
        except FileNotFoundError:
            continue
    if not rows:
        return "<p>No run metrics found.</p>"
    frame = pd.DataFrame(rows)
    keep_columns = [
        "run",
        "model_name",
        "evaluation",
        "target",
        "mae",
        "spearman",
        "pairwise_order_accuracy",
        "mae_improvement_pct",
        "difficulty_rating_baseline_mae",
        "mae_improvement_pct_vs_difficulty_rating_baseline",
    ]
    keep_columns = [column for column in keep_columns if column in frame.columns]
    return frame[keep_columns].sort_values(["evaluation", "target", "mae", "run"]).to_html(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )


def run_decision_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    judgment_by_evaluation = human_judgment_summary_by_evaluation(run_dir)
    health = training_health_summary(run_dir / "history.csv")
    performance = training_performance_summary(run_dir / "history.csv")
    for metrics_name, default_evaluation in [
        ("metrics.json", "holdout"),
        ("cv_metrics.json", "cv_oof"),
    ]:
        metrics = load_json(run_dir / metrics_name)
        if not metrics:
            continue
        summary = model_verdict_summary(metrics)
        if not summary:
            continue
        run_info = metrics.get("_run", {})
        if not isinstance(run_info, dict):
            run_info = {}
        evaluation = run_info.get("evaluation", default_evaluation)
        judgment = judgment_by_evaluation.get(str(evaluation), {})
        calibration = prediction_calibration_summary(run_dir, str(evaluation))
        baseline_target_count = int(summary.get("baseline_target_count", 0))
        difficulty_target_count = int(summary.get("difficulty_baseline_target_count", 0))
        rows.append(
            {
                "run": run_dir.name,
                "model_name": run_info.get("model_name", ""),
                "evaluation": evaluation,
                "device": run_info.get("device", ""),
                "amp_enabled": run_info.get("amp_enabled", ""),
                "effective_batch_size": run_info.get("effective_batch_size", ""),
                "grad_clip_norm": run_info.get("grad_clip_norm", ""),
                "sample_weight_column": run_info.get("sample_weight_column", ""),
                "sample_weight_train_mean": run_info.get("sample_weight_train_mean", ""),
                "sample_weight_train_downweighted_rate": run_info.get(
                    "sample_weight_train_downweighted_rate",
                    "",
                ),
                "human_judged_pairs": judgment.get("human_judged_pairs", ""),
                "human_judgment_coverage_rate": judgment.get(
                    "human_judgment_coverage_rate",
                    "",
                ),
                "human_model_agreement_rate": judgment.get(
                    "human_model_agreement_rate",
                    "",
                ),
                "human_proxy_agreement_rate": judgment.get(
                    "human_proxy_agreement_rate",
                    "",
                ),
                "human_model_vs_proxy_delta": judgment.get(
                    "human_model_vs_proxy_delta",
                    "",
                ),
                "epochs_completed": run_info.get("epochs_completed", ""),
                "stop_reason": run_info.get("stop_reason", ""),
                "early_stopped": run_info.get("early_stopped", ""),
                "curve_best_epoch": health.get("best_epoch", ""),
                "curve_best_val_loss": health.get("best_val_loss", ""),
                "curve_best_val_mean_mae": health.get("best_val_mean_mae", ""),
                "curve_best_val_pairwise_order": health.get("best_val_pairwise_order", ""),
                "curve_final_val_mean_mae": health.get("final_val_mean_mae", ""),
                "curve_final_val_pairwise_order": health.get(
                    "final_val_pairwise_order",
                    "",
                ),
                "curve_overfit_signal": health.get("overfit_signal", ""),
                "avg_epoch_seconds": performance.get("average_epoch_seconds", ""),
                "peak_cuda_memory_mb": performance.get("peak_cuda_memory_mb", ""),
                "targets": summary.get("target_count", ""),
                "mean_mae": summary.get("mean_mae", ""),
                "mean_pairwise_order_accuracy": summary.get(
                    "mean_pairwise_order_accuracy",
                    "",
                ),
                "targets_beating_baseline": (
                    f"{summary.get('targets_beating_baseline', 0)} / {baseline_target_count}"
                    if baseline_target_count
                    else ""
                ),
                "targets_beating_difficulty_rating": (
                    f"{summary.get('targets_beating_difficulty_rating_baseline', 0)}"
                    f" / {difficulty_target_count}"
                    if difficulty_target_count
                    else ""
                ),
                "weakest_target": summary.get("weakest_target", ""),
                "calibration_mean_abs_bias": calibration.get("calibration_mean_abs_bias", ""),
                "calibration_worst_bias_target": calibration.get(
                    "calibration_worst_bias_target",
                    "",
                ),
                "calibration_worst_bias": calibration.get("calibration_worst_bias", ""),
                "calibration_mean_pred_std_ratio": calibration.get(
                    "calibration_mean_pred_std_ratio",
                    "",
                ),
                "calibration_warning": calibration.get("calibration_warning", ""),
                "training_adjustment": training_adjustment_text(
                    summary,
                    health,
                    performance,
                    run_info,
                    calibration,
                ),
                "next_action": summary.get("next_action", ""),
            }
        )
    return rows


def run_decision_table(run_dirs: list[Path]) -> str:
    frame = run_decision_frame(run_dirs)
    if frame.empty:
        return "<p>No model verdicts found.</p>"
    return frame.to_html(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )


def run_decision_frame(run_dirs: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_dir in run_dirs:
        rows.extend(run_decision_rows(run_dir))
    return pd.DataFrame(rows, columns=DECISION_COLUMNS).sort_values(["evaluation", "run"])


def write_run_decision_summary_csv(run_dirs: list[Path], out_csv: Path) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    run_decision_frame(run_dirs).to_csv(out_csv, index=False, encoding="utf-8")


def training_adjustment_text(
    verdict: dict[str, object],
    health: dict[str, object],
    performance: dict[str, object],
    run_info: dict[str, object],
    calibration: dict[str, object] | None = None,
) -> str:
    notes: list[str] = []
    if str(health.get("overfit_signal", "")).lower() == "possible":
        notes.append(
            "Increase regularization or reduce patience/epochs before a longer run."
        )

    calibration_warning = str((calibration or {}).get("calibration_warning", "")).strip()
    if calibration_warning:
        notes.append(calibration_warning)

    pairwise = metric_float(verdict.get("mean_pairwise_order_accuracy"))
    if pairwise is not None and pairwise < 0.55:
        notes.append("Improve labels/human judgments before spending more GPU time.")
    elif pairwise is not None and pairwise < 0.70:
        notes.append("Run a small LR/dropout/hidden-size sweep before final training.")

    model_name = str(run_info.get("model_name", "")).lower()
    device = str(run_info.get("device", "")).lower()
    if device == "cpu" and model_name not in {"tabular_forest", "forest"}:
        notes.append("Move neural training to CUDA/Colab for the next real run.")

    peak_memory = metric_float(performance.get("peak_cuda_memory_mb"))
    if peak_memory is not None and peak_memory >= 14000:
        notes.append("Lower batch size or raise grad_accum_steps to protect CUDA memory.")

    avg_epoch_seconds = metric_float(performance.get("average_epoch_seconds"))
    if avg_epoch_seconds is not None and avg_epoch_seconds >= 120:
        notes.append("Limit sweep candidates or lower max-notes if iteration is too slow.")

    downweighted_rate = metric_float(run_info.get("sample_weight_train_downweighted_rate"))
    if downweighted_rate is not None and downweighted_rate >= 0.50:
        notes.append("Review low score-count labels before trusting a long weighted run.")

    if notes:
        return " ".join(notes)
    return "Keep these settings as a baseline and compare the next run against it."


def human_judgment_summary_by_evaluation(run_dir: Path) -> dict[str, dict[str, object]]:
    rows: dict[str, dict[str, object]] = {}
    for evaluation, filename in JUDGMENT_FILES:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            score = score_pair_judgments(path)
        except (KeyError, ValueError, pd.errors.EmptyDataError):
            continue
        rows[evaluation] = {
            "human_judged_pairs": (
                f"{score['judged_count']} / {score['row_count']}"
                if score["row_count"]
                else "0 / 0"
            ),
            "human_judgment_coverage_rate": score["judgment_coverage_rate"],
            "human_model_agreement_rate": score["model_agreement_rate"],
            "human_proxy_agreement_rate": score["proxy_agreement_rate"],
            "human_model_vs_proxy_delta": score["model_vs_proxy_agreement_delta"],
        }
    return rows


def human_judgment_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation, filename in JUDGMENT_FILES:
        path = run_dir / filename
        if not path.exists():
            continue
        try:
            score = score_pair_judgments(path)
        except (KeyError, ValueError, pd.errors.EmptyDataError) as exc:
            rows.append({"evaluation": evaluation, "file": filename, "error": str(exc)})
            continue
        rows.append(
            {
                "evaluation": evaluation,
                "file": filename,
                "judged_count": score["judged_count"],
                "row_count": score["row_count"],
                "invalid_choice_count": score["invalid_choice_count"],
                "judgment_coverage_rate": score["judgment_coverage_rate"],
                "model_agreement_rate": score["model_agreement_rate"],
                "proxy_agreement_rate": score["proxy_agreement_rate"],
                "model_vs_proxy_agreement_delta": score["model_vs_proxy_agreement_delta"],
            }
        )
    return rows


def human_judgment_table(run_dir: Path) -> str:
    rows = human_judgment_rows(run_dir)
    if not rows:
        return ""
    return (
        "<h4>Human Judgment Scores</h4>"
        + pd.DataFrame(rows).to_html(index=False, float_format=lambda value: f"{value:.6f}")
    )


def run_verdict_html(run_dir: Path) -> str:
    metrics = load_json(run_dir / "metrics.json")
    if not metrics:
        return ""
    return model_verdict_html(metrics, heading_level=4)


def checkpoint_selection_html(run_dir: Path) -> str:
    metrics = load_json(run_dir / "metrics.json")
    run_info = metrics.get("_run", {})
    if not isinstance(run_info, dict):
        return ""
    checkpoint_metric = run_info.get("checkpoint_metric", "")
    if not checkpoint_metric:
        return ""
    rows = [
        ("Checkpoint Metric", checkpoint_metric),
        (
            "Best Checkpoint Score",
            checkpoint_score_text(checkpoint_metric, run_info.get("best_checkpoint_score", "")),
        ),
    ]
    if "best_epoch" in run_info:
        rows.append(("Best Epoch", run_info["best_epoch"]))
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<h4>Checkpoint Selection</h4><table><tbody>{row_html}</tbody></table>"


def run_card(run_dir: Path, out_html: Path) -> str:
    links = [
        link(run_dir / "run_report.html", "run report", out_html),
        link(run_dir / "last_checkpoint.pt", "last checkpoint", out_html),
        link(run_dir / "predictions.csv", "predictions", out_html),
        link(run_dir / "prediction_summary.csv", "prediction summary", out_html),
        link(run_dir / "prediction_rankings.csv", "prediction rankings", out_html),
        link(run_dir / "human_review.csv", "human review", out_html),
        link(run_dir / "human_pair_review.csv", "pairwise review", out_html),
        link(run_dir / "human_pair_judgment_template.csv", "judgment template", out_html),
        link(run_dir / "embedding_report.html", "embedding report", out_html),
        link(run_dir / "embedding_projection.csv", "embedding CSV", out_html),
        link(run_dir / "attention_report.html", "attention report", out_html),
        link(run_dir / "attention_map.csv", "attention CSV", out_html),
        link(run_dir / "error_slices.csv", "error slices", out_html),
        link(run_dir / "cv_prediction_rankings.csv", "CV prediction rankings", out_html),
        link(run_dir / "cv_prediction_summary.csv", "CV prediction summary", out_html),
        link(run_dir / "cv_human_review.csv", "CV human review", out_html),
        link(run_dir / "cv_human_pair_judgment_template.csv", "CV judgment template", out_html),
        link(run_dir / "cv_error_slices.csv", "CV error slices", out_html),
        link(run_dir / "eval_prediction_summary.csv", "evaluation prediction summary", out_html),
    ]
    links_html = " ".join(item for item in links if item)
    images_html = "".join(
        [
            image(run_dir / "learning_curve.png", f"{run_dir.name} learning curve", out_html),
            image(run_dir / "prediction_scatter.png", f"{run_dir.name} prediction scatter", out_html),
            image(run_dir / "cv_prediction_scatter.png", f"{run_dir.name} CV prediction scatter", out_html),
            image(run_dir / "embedding_projection.png", f"{run_dir.name} embedding projection", out_html),
            image(run_dir / "attention_map.png", f"{run_dir.name} transformer attention map", out_html),
            image(run_dir / "feature_importance.png", f"{run_dir.name} feature importance", out_html),
        ]
    )
    return (
        f'<section class="run-card"><h3>{html.escape(run_dir.name)}</h3>'
        f"<p>{links_html}</p>"
        f"{run_verdict_html(run_dir)}"
        f"{checkpoint_selection_html(run_dir)}"
        f"{human_judgment_table(run_dir)}"
        f"{training_health_html(run_dir / 'history.csv', heading_level=4)}"
        f"{training_performance_html(run_dir / 'history.csv', heading_level=4)}"
        f"{worst_error_slices_html(run_dir / 'error_slices.csv', heading_level=4)}"
        f"{images_html}</section>"
    )


def write_dashboard(
    out_html: Path,
    *,
    run_dirs: list[Path],
    audit_dir: Path | None = None,
    forest_sweep_dir: Path | None = None,
    neural_sweep_dir: Path | None = None,
    comparison_html: Path | None = None,
    decision_summary_csv: Path | None = None,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(run_dir) for run_dir in run_dirs if Path(run_dir).exists()]
    if decision_summary_csv is None:
        decision_summary_csv = out_html.with_name("run_decision_summary.csv")
    write_run_decision_summary_csv(run_dirs, decision_summary_csv)
    comparison_link = link(comparison_html, "comparison report", out_html) if comparison_html else ""
    decision_summary_link = link(decision_summary_csv, "decision summary CSV", out_html)
    run_cards = "".join(run_card(run_dir, out_html) for run_dir in run_dirs)
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mania difficulty dashboard</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    section {{ margin: 0 0 32px; }}
    .run-card {{ border-top: 1px solid #d9e2ec; padding-top: 16px; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; margin: 8px 0; }}
    .table-wrap {{ overflow-x: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>osu!mania Difficulty Model Dashboard</h1>
  <p>{comparison_link}</p>
  {audit_section(audit_dir, out_html)}
  {best_params_section("Forest Sweep", forest_sweep_dir, "sweep_report.html", out_html)}
  {best_params_section("Neural Sweep", neural_sweep_dir, "neural_sweep_report.html", out_html)}
  <section>
    <h2>Run Decision Summary</h2>
    <p>{decision_summary_link}</p>
    {run_decision_table(run_dirs)}
  </section>
  <section>
    <h2>Run Metrics</h2>
    {metrics_table(run_dirs)}
  </section>
  <section>
    <h2>Run Reports</h2>
    {run_cards}
  </section>
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a single HTML dashboard from run artifacts.")
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--out-html", type=Path, default=Path("outputs/dashboard.html"))
    parser.add_argument("--audit-dir", type=Path, default=None)
    parser.add_argument("--forest-sweep-dir", type=Path, default=None)
    parser.add_argument("--neural-sweep-dir", type=Path, default=None)
    parser.add_argument("--comparison-html", type=Path, default=None)
    parser.add_argument("--decision-summary-csv", type=Path, default=None)
    args = parser.parse_args()

    write_dashboard(
        args.out_html,
        run_dirs=args.runs,
        audit_dir=args.audit_dir,
        forest_sweep_dir=args.forest_sweep_dir,
        neural_sweep_dir=args.neural_sweep_dir,
        comparison_html=args.comparison_html,
        decision_summary_csv=args.decision_summary_csv,
    )
    print(f"Wrote dashboard to {args.out_html}")


if __name__ == "__main__":
    main()
