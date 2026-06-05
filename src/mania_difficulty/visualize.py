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
        ax.set_xlabel(f"Actual {column}")
        ax.set_ylabel(f"Predicted {column}")
        ax.set_title(column)
        ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_run_report(
    run_dir: Path,
    *,
    target_columns: list[str],
    metrics_path: Path | None = None,
    learning_curve_name: str = "learning_curve.png",
    prediction_scatter_name: str = "prediction_scatter.png",
) -> None:
    metrics_html = "<p>No metrics yet.</p>"
    if metrics_path and metrics_path.exists():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows = []
        for target in target_columns:
            target_metrics = metrics.get(target, {})
            rows.append(
                "<tr>"
                f"<td>{html.escape(target)}</td>"
                f"<td>{target_metrics.get('mae', float('nan')):.6f}</td>"
                f"<td>{target_metrics.get('r2', float('nan')):.4f}</td>"
                "</tr>"
            )
        metrics_html = (
            "<table><thead><tr><th>Target</th><th>MAE</th><th>R2</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
        )

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

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>mania difficulty run report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    h1, h2 {{ margin-bottom: 8px; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>osu!mania Difficulty Model Run</h1>
  <p>Run directory: <code>{html.escape(str(run_dir))}</code></p>
  <h2>Metrics</h2>
  {metrics_html}
  <h2>Learning Curve</h2>
  {learning_curve_html}
  <h2>Predicted vs Actual</h2>
  {scatter_html}
  <h2>Files</h2>
  <p>Open <code>predictions.csv</code> to inspect the model output map by map.</p>
</body>
</html>
"""
    (run_dir / "run_report.html").write_text(report, encoding="utf-8")
