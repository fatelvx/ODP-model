from __future__ import annotations

import html
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def plot_learning_curve(history_csv: Path, out_path: Path) -> None:
    history = pd.read_csv(history_csv)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(history["epoch"], history["train_loss"], label="train")
    ax.plot(history["epoch"], history["val_loss"], label="validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curve")
    ax.grid(True, alpha=0.25)
    ax.legend()
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
    run_info_html = ""
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics_html = metrics_table_html(metrics, target_columns)
        run_info = metrics.get("_run", {})
        if run_info:
            model_config = run_info.get("model_config", "")
            if isinstance(model_config, dict):
                model_config = json.dumps(model_config, ensure_ascii=False)
            run_info_html = (
                "<table><tbody>"
                f"<tr><th>Model</th><td>{html.escape(str(run_info.get('model_name', '')))}</td></tr>"
                f"<tr><th>Model Config</th><td><code>{html.escape(str(model_config))}</code></td></tr>"
                f"<tr><th>Evaluation</th><td>{html.escape(str(run_info.get('evaluation', '')))}</td></tr>"
                f"<tr><th>Feature Set</th><td>{html.escape(str(run_info.get('feature_set', '')))}</td></tr>"
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
  {metrics_html}
  {cv_html}
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
