from __future__ import annotations

import html
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(history_csv: Path, out_path: Path) -> None:
    history = pd.read_csv(history_csv)
    has_validation_metrics = (
        "val_mean_mae" in history.columns
        or "val_mean_pairwise_order_accuracy" in history.columns
    )
    if has_validation_metrics:
        fig, axes = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        loss_ax = axes[0]
        metric_ax = axes[1]
    else:
        fig, loss_ax = plt.subplots(figsize=(8, 5))
        metric_ax = None

    loss_ax.plot(history["epoch"], history["train_loss"], label="train")
    loss_ax.plot(history["epoch"], history["val_loss"], label="validation")
    loss_ax.set_ylabel("Loss")
    loss_ax.set_title("Learning Curve")
    loss_ax.grid(True, alpha=0.25)
    loss_ax.legend()

    if metric_ax is not None:
        if "val_mean_mae" in history.columns:
            val_mean_mae = pd.to_numeric(history["val_mean_mae"], errors="coerce")
            metric_ax.plot(history["epoch"], val_mean_mae, label="validation mean MAE")
            metric_ax.set_ylabel("MAE")
        if "val_mean_pairwise_order_accuracy" in history.columns:
            pairwise_ax = metric_ax.twinx()
            pairwise_order = pd.to_numeric(
                history["val_mean_pairwise_order_accuracy"],
                errors="coerce",
            )
            pairwise_ax.plot(
                history["epoch"],
                pairwise_order,
                color="#c2410c",
                label="validation pairwise order",
            )
            pairwise_ax.set_ylabel("Pairwise order")
            pairwise_ax.set_ylim(0, 1)
            lines, labels = metric_ax.get_legend_handles_labels()
            pair_lines, pair_labels = pairwise_ax.get_legend_handles_labels()
            metric_ax.legend(lines + pair_lines, labels + pair_labels, loc="best")
        else:
            metric_ax.legend(loc="best")
        metric_ax.set_xlabel("Epoch")
        metric_ax.grid(True, alpha=0.25)
    else:
        loss_ax.set_xlabel("Epoch")

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_prediction_scatter(
    predictions_csv: Path,
    target_columns: list[str],
    out_path: Path,
) -> None:
    predictions = pd.read_csv(predictions_csv)
    fig, axes = plt.subplots(1, len(target_columns), figsize=(5 * len(target_columns), 5))
    if len(target_columns) == 1:
        axes = [axes]

    for ax, column in zip(axes, target_columns):
        actual = predictions[f"actual_{column}"]
        predicted = predictions[f"pred_{column}"]
        ax.scatter(actual, predicted, alpha=0.75, s=24)
        low = min(actual.min(), predicted.min())
        high = max(actual.max(), predicted.max())
        ax.plot([low, high], [low, high], color="black", linewidth=1)
        ax.set_xlabel(f"Observed top100 proxy {column}")
        ax.set_ylabel(f"Predicted top100 proxy {column}")
        ax.set_title(column)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_feature_importance(importances_csv: Path, out_path: Path, *, top_n: int = 20) -> None:
    importances = pd.read_csv(importances_csv).head(top_n)
    fig_height = max(4, 0.32 * len(importances) + 1.2)
    fig, ax = plt.subplots(figsize=(8, fig_height))
    ax.barh(importances["feature"], importances["importance"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances")
    ax.grid(True, axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def training_health_summary(history_csv: Path) -> dict[str, object]:
    if not history_csv.exists():
        return {}
    try:
        history = pd.read_csv(history_csv)
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    required_columns = {"epoch", "train_loss", "val_loss"}
    if not required_columns.issubset(history.columns):
        return {}
    history = history.dropna(subset=["epoch", "train_loss", "val_loss"])
    if history.empty:
        return {}

    best_row = history.loc[history["val_loss"].idxmin()]
    final_row = history.iloc[-1]
    best_val_loss = float(best_row["val_loss"])
    final_train_loss = float(final_row["train_loss"])
    final_val_loss = float(final_row["val_loss"])
    val_loss_regression = final_val_loss - best_val_loss
    generalization_gap = final_val_loss - final_train_loss
    regression_threshold = max(0.01, abs(best_val_loss) * 0.05)
    overfit_signal = (
        "Possible"
        if val_loss_regression > regression_threshold and generalization_gap > 0
        else "No obvious"
    )
    summary: dict[str, object] = {
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": best_val_loss,
        "final_epoch": int(final_row["epoch"]),
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "generalization_gap": generalization_gap,
        "val_loss_regression": val_loss_regression,
        "overfit_signal": overfit_signal,
    }
    if "val_mean_mae" in history.columns:
        val_mean_mae = pd.to_numeric(history["val_mean_mae"], errors="coerce")
        metric_history = history.assign(_val_mean_mae=val_mean_mae).dropna(subset=["_val_mean_mae"])
        if not metric_history.empty:
            best_metric_row = metric_history.loc[metric_history["_val_mean_mae"].idxmin()]
            final_metric_row = metric_history.iloc[-1]
            summary["best_val_mean_mae_epoch"] = int(best_metric_row["epoch"])
            summary["best_val_mean_mae"] = float(best_metric_row["_val_mean_mae"])
            summary["final_val_mean_mae"] = float(final_metric_row["_val_mean_mae"])
            summary["val_mean_mae_since_best"] = (
                summary["final_val_mean_mae"] - summary["best_val_mean_mae"]
            )
    if "val_mean_pairwise_order_accuracy" in history.columns:
        pairwise_order = pd.to_numeric(history["val_mean_pairwise_order_accuracy"], errors="coerce")
        metric_history = history.assign(_val_pairwise_order=pairwise_order).dropna(
            subset=["_val_pairwise_order"]
        )
        if not metric_history.empty:
            best_metric_row = metric_history.loc[metric_history["_val_pairwise_order"].idxmax()]
            final_metric_row = metric_history.iloc[-1]
            summary["best_val_pairwise_order_epoch"] = int(best_metric_row["epoch"])
            summary["best_val_pairwise_order"] = float(best_metric_row["_val_pairwise_order"])
            summary["final_val_pairwise_order"] = float(final_metric_row["_val_pairwise_order"])
    return summary


def training_health_html(history_csv: Path, *, heading_level: int = 2) -> str:
    summary = training_health_summary(history_csv)
    if not summary:
        return ""
    heading_tag = f"h{heading_level}"
    rows = [
        ("Best Epoch", summary["best_epoch"]),
        ("Best Val Loss", f"{summary['best_val_loss']:.6f}"),
        ("Final Epoch", summary["final_epoch"]),
        ("Final Train Loss", f"{summary['final_train_loss']:.6f}"),
        ("Final Val Loss", f"{summary['final_val_loss']:.6f}"),
        ("Generalization Gap", f"{summary['generalization_gap']:.6f}"),
        ("Val Loss Since Best", f"{summary['val_loss_regression']:.6f}"),
        ("Overfit Signal", summary["overfit_signal"]),
    ]
    if "best_val_mean_mae" in summary:
        rows.extend(
            [
                (
                    "Best Val MAE",
                    f"{summary['best_val_mean_mae']:.6f} @ epoch {summary['best_val_mean_mae_epoch']}",
                ),
                ("Final Val MAE", f"{summary['final_val_mean_mae']:.6f}"),
                ("Val MAE Since Best", f"{summary['val_mean_mae_since_best']:.6f}"),
            ]
        )
    if "best_val_pairwise_order" in summary:
        rows.extend(
            [
                (
                    "Best Val Pairwise Order",
                    f"{summary['best_val_pairwise_order']:.2%} @ epoch {summary['best_val_pairwise_order_epoch']}",
                ),
                ("Final Val Pairwise Order", f"{summary['final_val_pairwise_order']:.2%}"),
            ]
        )
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<{heading_tag}>Training Health</{heading_tag}><table><tbody>{row_html}</tbody></table>"


def training_performance_summary(history_csv: Path) -> dict[str, object]:
    if not history_csv.exists():
        return {}
    try:
        history = pd.read_csv(history_csv)
    except (pd.errors.EmptyDataError, ValueError):
        return {}
    if "epoch_seconds" not in history.columns:
        return {}

    epoch_seconds = pd.to_numeric(history["epoch_seconds"], errors="coerce").dropna()
    if epoch_seconds.empty:
        return {}

    summary: dict[str, object] = {
        "epoch_count": int(epoch_seconds.shape[0]),
        "total_epoch_seconds": float(epoch_seconds.sum()),
        "average_epoch_seconds": float(epoch_seconds.mean()),
    }
    if "lr" in history.columns:
        lr_values = pd.to_numeric(history["lr"], errors="coerce").dropna()
        if not lr_values.empty:
            summary["final_lr"] = float(lr_values.iloc[-1])
    if "cuda_max_memory_mb" in history.columns:
        memory_values = pd.to_numeric(history["cuda_max_memory_mb"], errors="coerce").dropna()
        if not memory_values.empty:
            summary["peak_cuda_memory_mb"] = float(memory_values.max())
    return summary


def training_performance_html(history_csv: Path, *, heading_level: int = 2) -> str:
    summary = training_performance_summary(history_csv)
    if not summary:
        return ""
    heading_tag = f"h{heading_level}"
    rows = [
        ("Epoch Count", summary["epoch_count"]),
        ("Total Epoch Seconds", f"{summary['total_epoch_seconds']:.2f}"),
        ("Average Epoch Seconds", f"{summary['average_epoch_seconds']:.2f}"),
    ]
    if "final_lr" in summary:
        rows.append(("Final LR", f"{summary['final_lr']:.6g}"))
    if "peak_cuda_memory_mb" in summary:
        rows.append(("Peak CUDA Memory MB", f"{summary['peak_cuda_memory_mb']:.1f}"))
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<{heading_tag}>Training Performance</{heading_tag}><table><tbody>{row_html}</tbody></table>"


def metrics_table_html(metrics: dict, target_columns: list[str]) -> str:
    rows = []
    for target in target_columns:
        target_metrics = metrics.get(target, {})
        baseline_mae = target_metrics.get("baseline_mae", float("nan"))
        improvement_pct = target_metrics.get("mae_improvement_pct", float("nan"))
        rows.append(
            "<tr>"
            f"<td>{html.escape(target)}</td>"
            f"<td>{target_metrics.get('mae', float('nan')):.6f}</td>"
            f"<td>{target_metrics.get('r2', float('nan')):.4f}</td>"
            f"<td>{target_metrics.get('spearman', float('nan')):.4f}</td>"
            f"<td>{target_metrics.get('pairwise_order_accuracy', float('nan')):.2%}</td>"
            f"<td>{baseline_mae:.6f}</td>"
            f"<td>{improvement_pct * 100:.2f}%</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr><th>Target</th><th>MAE</th><th>R2</th>"
        "<th>Spearman</th><th>Pairwise Order</th>"
        "<th>Baseline MAE</th><th>Improvement</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def metric_float(value: object) -> float | None:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def metric_targets(metrics: dict, target_columns: list[str] | None = None) -> list[str]:
    if target_columns:
        return [target for target in target_columns if isinstance(metrics.get(target), dict)]
    return [
        target
        for target, values in metrics.items()
        if not target.startswith("_") and isinstance(values, dict)
    ]


def model_verdict_summary(metrics: dict, target_columns: list[str] | None = None) -> dict[str, object]:
    targets = metric_targets(metrics, target_columns)
    if not targets:
        return {}

    mae_values: list[tuple[str, float]] = []
    pairwise_values: list[float] = []
    spearman_values: list[float] = []
    improvement_values: list[float] = []
    baseline_targets = 0
    beating_baseline = 0
    for target in targets:
        values = metrics.get(target, {})
        mae = metric_float(values.get("mae"))
        if mae is not None:
            mae_values.append((target, mae))
        pairwise = metric_float(values.get("pairwise_order_accuracy"))
        if pairwise is not None:
            pairwise_values.append(pairwise)
        spearman = metric_float(values.get("spearman"))
        if spearman is not None:
            spearman_values.append(spearman)
        improvement = metric_float(values.get("mae_improvement_pct"))
        if improvement is not None:
            baseline_targets += 1
            improvement_values.append(improvement)
            if improvement > 0:
                beating_baseline += 1

    weakest_target = max(mae_values, key=lambda item: item[1])[0] if mae_values else ""
    summary: dict[str, object] = {
        "target_count": len(targets),
        "baseline_target_count": baseline_targets,
        "targets_beating_baseline": beating_baseline,
        "weakest_target": weakest_target,
    }
    if mae_values:
        summary["mean_mae"] = sum(value for _, value in mae_values) / len(mae_values)
    if pairwise_values:
        summary["mean_pairwise_order_accuracy"] = sum(pairwise_values) / len(pairwise_values)
    if spearman_values:
        summary["mean_spearman"] = sum(spearman_values) / len(spearman_values)
    if improvement_values:
        summary["mean_improvement_pct"] = sum(improvement_values) / len(improvement_values)
    return summary


def model_verdict_html(
    metrics: dict,
    target_columns: list[str] | None = None,
    *,
    heading_level: int = 2,
) -> str:
    summary = model_verdict_summary(metrics, target_columns)
    if not summary:
        return ""
    heading_tag = f"h{heading_level}"
    rows = [
        ("Targets", summary["target_count"]),
        (
            "Targets Beating Baseline",
            f"{summary['targets_beating_baseline']} / {summary['baseline_target_count']}",
        ),
    ]
    if "mean_mae" in summary:
        rows.append(("Mean MAE", f"{summary['mean_mae']:.6f}"))
    if "mean_pairwise_order_accuracy" in summary:
        rows.append(("Mean Pairwise Order", f"{summary['mean_pairwise_order_accuracy']:.2%}"))
    if "mean_spearman" in summary:
        rows.append(("Mean Spearman", f"{summary['mean_spearman']:.4f}"))
    if "mean_improvement_pct" in summary:
        rows.append(("Mean Baseline Improvement", f"{summary['mean_improvement_pct'] * 100:.2f}%"))
    rows.append(("Weakest Target", summary["weakest_target"]))
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<{heading_tag}>Model Verdict</{heading_tag}><table><tbody>{row_html}</tbody></table>"


def checkpoint_score_text(metric: object, score: object) -> str:
    parsed = metric_float(score)
    if parsed is None:
        return str(score)
    if str(metric) == "val_mean_pairwise_order_accuracy":
        return f"{parsed:.2%}"
    return f"{parsed:.6f}"


def worst_error_slice_rows(path: Path, *, top_n: int = 5) -> list[dict[str, object]]:
    if not path.exists():
        return []
    try:
        frame = pd.read_csv(path)
    except (pd.errors.EmptyDataError, ValueError):
        return []
    required_columns = {"slice_column", "slice_value", "count", "mae", "bias", "max_abs_error"}
    if frame.empty or not required_columns.issubset(frame.columns):
        return []

    working = frame.copy()
    for column in ["count", "mae", "bias", "max_abs_error"]:
        working[column] = pd.to_numeric(working[column], errors="coerce")
    working = working.dropna(subset=["mae"])
    if working.empty:
        return []

    non_overall = working[working["slice_column"].astype(str) != "overall"]
    if not non_overall.empty:
        working = non_overall
    working = working.sort_values(["mae", "max_abs_error"], ascending=[False, False]).head(top_n)
    rows = []
    for _, row in working.iterrows():
        rows.append(
            {
                "slice": f"{row['slice_column']}: {row['slice_value']}",
                "count": int(row["count"]) if pd.notna(row["count"]) else "",
                "mae": float(row["mae"]),
                "bias": float(row["bias"]) if pd.notna(row["bias"]) else None,
                "max_abs_error": (
                    float(row["max_abs_error"]) if pd.notna(row["max_abs_error"]) else None
                ),
            }
        )
    return rows


def worst_error_slices_html(path: Path, *, heading_level: int = 2, top_n: int = 5) -> str:
    rows = worst_error_slice_rows(path, top_n=top_n)
    if not rows:
        return ""
    heading_tag = f"h{heading_level}"

    def format_optional_float(value: object) -> str:
        parsed = metric_float(value)
        return "" if parsed is None else f"{parsed:.6f}"

    row_html = "".join(
        "<tr>"
        f"<td>{html.escape(str(row['slice']))}</td>"
        f"<td>{html.escape(str(row['count']))}</td>"
        f"<td>{row['mae']:.6f}</td>"
        f"<td>{html.escape(format_optional_float(row['bias']))}</td>"
        f"<td>{html.escape(format_optional_float(row['max_abs_error']))}</td>"
        "</tr>"
        for row in rows
    )
    return (
        f"<{heading_tag}>Worst Error Slices</{heading_tag}>"
        "<p>Highest-MAE metadata bins, excluding the overall row when slice rows exist.</p>"
        "<div class=\"table-wrap\"><table><thead><tr>"
        "<th>Slice</th><th>Count</th><th>MAE</th><th>Bias</th><th>Max Abs Error</th>"
        f"</tr></thead><tbody>{row_html}</tbody></table></div>"
    )


def csv_preview_html(path: Path, title: str, description: str, *, max_rows: int = 20) -> str:
    if not path.exists():
        return ""
    frame = pd.read_csv(path)
    if frame.empty:
        return (
            f"<h2>{html.escape(title)}</h2>"
            f"<p>{html.escape(description)}</p>"
            f"<p><code>{html.escape(path.name)}</code> has no rows for this run.</p>"
        )
    table = frame.head(max_rows).to_html(index=False, float_format=lambda value: f"{value:.6f}")
    return (
        f"<h2>{html.escape(title)}</h2>"
        f"<p>{html.escape(description)}</p>"
        f'<div class="table-wrap">{table}</div>'
    )


def review_sections_html(run_dir: Path) -> str:
    sections = [
        csv_preview_html(
            run_dir / "prediction_rankings.csv",
            "Prediction Rankings",
            "Model outputs sorted into predicted hardest, predicted easiest, and largest-error maps for quick inspection.",
        ),
        csv_preview_html(
            run_dir / "human_review.csv",
            "Human Review",
            "Maps that are useful to inspect by hand: lowest predicted accuracy, lowest observed accuracy, and largest disagreement.",
        ),
        csv_preview_html(
            run_dir / "human_pair_review.csv",
            "Pairwise Human Review",
            "Pairs where the model and the top100 proxy disagree about which map is harder.",
        ),
        csv_preview_html(
            run_dir / "human_pair_judgment_template.csv",
            "Human Pair Judgment Template",
            "Fill human_harder_beatmap_id with the beatmap ID that feels harder after comparing the pair.",
        ),
        csv_preview_html(
            run_dir / "error_slices.csv",
            "Error Slices",
            "Mean-accuracy error grouped by metadata bins such as score count and note count.",
        ),
        csv_preview_html(
            run_dir / "cv_human_review.csv",
            "Cross-Validation Human Review",
            "Out-of-fold maps worth checking by hand.",
        ),
        csv_preview_html(
            run_dir / "cv_prediction_rankings.csv",
            "Cross-Validation Prediction Rankings",
            "Out-of-fold model outputs sorted into hardest, easiest, and largest-error maps.",
        ),
        csv_preview_html(
            run_dir / "cv_human_pair_review.csv",
            "Cross-Validation Pairwise Human Review",
            "Out-of-fold map pairs where the model and proxy rank difficulty in opposite directions.",
        ),
        csv_preview_html(
            run_dir / "cv_human_pair_judgment_template.csv",
            "Cross-Validation Human Pair Judgment Template",
            "Fill human_harder_beatmap_id to score out-of-fold model agreement with human judgment.",
        ),
        csv_preview_html(
            run_dir / "cv_error_slices.csv",
            "Cross-Validation Error Slices",
            "Out-of-fold mean-accuracy error grouped by metadata bins.",
        ),
        csv_preview_html(
            run_dir / "eval_human_review.csv",
            "Evaluation Human Review",
            "Maps worth checking by hand for this checkpoint evaluation.",
        ),
        csv_preview_html(
            run_dir / "eval_prediction_rankings.csv",
            "Evaluation Prediction Rankings",
            "Checkpoint model outputs sorted into hardest, easiest, and largest-error maps.",
        ),
        csv_preview_html(
            run_dir / "eval_human_pair_review.csv",
            "Evaluation Pairwise Human Review",
            "Checkpoint evaluation pairs where the model and proxy rank difficulty in opposite directions.",
        ),
        csv_preview_html(
            run_dir / "eval_human_pair_judgment_template.csv",
            "Evaluation Human Pair Judgment Template",
            "Fill human_harder_beatmap_id to score checkpoint agreement with human judgment.",
        ),
        csv_preview_html(
            run_dir / "eval_error_slices.csv",
            "Evaluation Error Slices",
            "Checkpoint evaluation mean-accuracy error grouped by metadata bins.",
        ),
    ]
    return "".join(section for section in sections if section)


def write_run_report(
    run_dir: Path,
    *,
    target_columns: list[str],
    metrics_path: Path | None = None,
    learning_curve_name: str = "learning_curve.png",
    prediction_scatter_name: str = "prediction_scatter.png",
    feature_importance_name: str = "feature_importance.png",
) -> None:
    metrics_html = "<p>No metrics yet.</p>"
    verdict_html = ""
    run_info_html = ""
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_html = metrics_table_html(metrics, target_columns)
        verdict_html = model_verdict_html(metrics, target_columns)
        run_info = metrics.get("_run", {})
        if run_info:
            checkpoint_metric = run_info.get("checkpoint_metric", "")
            best_checkpoint_score = run_info.get("best_checkpoint_score", "")
            model_config = run_info.get("model_config", "")
            if isinstance(model_config, dict):
                model_config = json.dumps(model_config, ensure_ascii=False)
            run_info_html = (
                "<table><tbody>"
                f"<tr><th>Model</th><td>{html.escape(str(run_info.get('model_name', '')))}</td></tr>"
                f"<tr><th>Model Config</th><td><code>{html.escape(str(model_config))}</code></td></tr>"
                f"<tr><th>Evaluation</th><td>{html.escape(str(run_info.get('evaluation', '')))}</td></tr>"
                f"<tr><th>Feature Set</th><td>{html.escape(str(run_info.get('feature_set', '')))}</td></tr>"
                f"<tr><th>AMP</th><td>{html.escape(str(run_info.get('amp', '')))}</td></tr>"
                f"<tr><th>AMP Enabled</th><td>{html.escape(str(run_info.get('amp_enabled', '')))}</td></tr>"
                f"<tr><th>Batch Size</th><td>{html.escape(str(run_info.get('batch_size', '')))}</td></tr>"
                f"<tr><th>Grad Accum Steps</th><td>{html.escape(str(run_info.get('grad_accum_steps', '')))}</td></tr>"
                f"<tr><th>Effective Batch Size</th><td>{html.escape(str(run_info.get('effective_batch_size', '')))}</td></tr>"
                f"<tr><th>Checkpoint Metric</th><td>{html.escape(str(checkpoint_metric))}</td></tr>"
                f"<tr><th>Best Checkpoint Score</th><td>{html.escape(checkpoint_score_text(checkpoint_metric, best_checkpoint_score))}</td></tr>"
                f"<tr><th>Checkpoint Best Epoch</th><td>{html.escape(str(run_info.get('best_epoch', '')))}</td></tr>"
                f"<tr><th>Resume</th><td>{html.escape(str(run_info.get('resume', '')))}</td></tr>"
                f"<tr><th>Resumed From Epoch</th><td>{html.escape(str(run_info.get('resumed_from_epoch', '')))}</td></tr>"
                f"<tr><th>Last Checkpoint</th><td>{html.escape(str(run_info.get('last_checkpoint', '')))}</td></tr>"
                f"<tr><th>Checkpoint Backup Dir</th><td>{html.escape(str(run_info.get('checkpoint_backup_dir', '')))}</td></tr>"
                f"<tr><th>Restored From Backup</th><td>{html.escape(str(run_info.get('restored_from_backup', '')))}</td></tr>"
                f"<tr><th>Split</th><td>{html.escape(str(run_info.get('split_strategy', '')))}</td></tr>"
                f"<tr><th>Group Column</th><td>{html.escape(str(run_info.get('group_column', '')))}</td></tr>"
                f"<tr><th>Group Count</th><td>{html.escape(str(run_info.get('group_count', '')))}</td></tr>"
                "</tbody></table>"
            )

    cv_html = ""
    cv_metrics_path = run_dir / "cv_metrics.json"
    if cv_metrics_path.exists():
        cv_metrics = json.loads(cv_metrics_path.read_text(encoding="utf-8"))
        cv_html = (
            "<h2>Cross-Validation Metrics</h2>"
            "<p>Out-of-fold metrics use every map as validation once, which is steadier for small datasets.</p>"
            f"{metrics_table_html(cv_metrics, target_columns)}"
        )
        cv_scatter = run_dir / "cv_prediction_scatter.png"
        if cv_scatter.exists():
            cv_html += '<p><img src="cv_prediction_scatter.png" alt="Cross-validation prediction scatter"></p>'

    learning_curve_html = (
        f'<p><img src="{html.escape(learning_curve_name)}" alt="Learning curve"></p>'
        if learning_curve_name and (run_dir / learning_curve_name).exists()
        else "<p>No learning curve image in this report.</p>"
    )
    scatter_html = (
        f'<p><img src="{html.escape(prediction_scatter_name)}" alt="Prediction scatter"></p>'
        if prediction_scatter_name and (run_dir / prediction_scatter_name).exists()
        else "<p>No prediction scatter image in this report.</p>"
    )
    feature_importance_html = (
        f'<p><img src="{html.escape(feature_importance_name)}" alt="Feature importance"></p>'
        if feature_importance_name and (run_dir / feature_importance_name).exists()
        else ""
    )
    health_html = training_health_html(run_dir / "history.csv")
    performance_html = training_performance_html(run_dir / "history.csv")
    worst_error_html = worst_error_slices_html(run_dir / "error_slices.csv")
    embedding_html = ""
    embedding_png = run_dir / "embedding_projection.png"
    embedding_report = run_dir / "embedding_report.html"
    if embedding_png.exists() or embedding_report.exists():
        report_link = (
            '<p><a href="embedding_report.html">embedding_report.html</a></p>'
            if embedding_report.exists()
            else ""
        )
        image_html = (
            '<p><img src="embedding_projection.png" alt="Embedding projection"></p>'
            if embedding_png.exists()
            else ""
        )
        embedding_html = f"<h2>Embedding Projection</h2>{report_link}{image_html}"
    attention_html = ""
    attention_png = run_dir / "attention_map.png"
    attention_report = run_dir / "attention_report.html"
    if attention_png.exists() or attention_report.exists():
        report_link = (
            '<p><a href="attention_report.html">attention_report.html</a></p>'
            if attention_report.exists()
            else ""
        )
        image_html = (
            '<p><img src="attention_map.png" alt="Transformer attention map"></p>'
            if attention_png.exists()
            else ""
        )
        attention_html = f"<h2>Transformer Attention Map</h2>{report_link}{image_html}"
    review_html = review_sections_html(run_dir)

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mania difficulty run report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    .table-wrap {{ overflow-x: auto; max-width: 100%; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>osu!mania Difficulty Model Run</h1>
  <p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
  {run_info_html}
  <h2>Metrics</h2>
  {verdict_html}
  {metrics_html}
  {cv_html}
  {health_html}
  {performance_html}
  {worst_error_html}
  <h2>Learning Curve</h2>
  {learning_curve_html}
  <h2>Predicted vs Observed Proxy</h2>
  {scatter_html}
  {f"<h2>Feature Importance</h2>{feature_importance_html}" if feature_importance_html else ""}
  {embedding_html}
  {attention_html}
  {review_html}
  <h2>Files</h2>
  <p>Open <code>predictions.csv</code> to inspect the model output map by map.</p>
  <p>Open <code>prediction_rankings.csv</code> to inspect predicted hardest, easiest, and largest-error maps.</p>
  <p>Open <code>human_review.csv</code> for maps that need human judgment.</p>
  <p>Open <code>human_pair_review.csv</code> to compare map pairs where the ranking disagrees.</p>
  <p>Fill <code>human_pair_judgment_template.csv</code>, then score it with <code>python -m mania_difficulty.tools.human_judgments score</code>.</p>
  <p>Open <code>error_slices.csv</code> to see where metadata bins have larger errors.</p>
  <p>Open <code>embedding_report.html</code> to inspect whether model embeddings cluster into meaningful map groups.</p>
  <p>Open <code>attention_report.html</code> to inspect Transformer note-level attention for one selected map.</p>
</body>
</html>
"""
    (run_dir / "run_report.html").write_text(report, encoding="utf-8")
