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
from mania_difficulty.visualize import training_health_html, training_performance_html


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
    ]
    keep_columns = [column for column in keep_columns if column in frame.columns]
    return frame[keep_columns].sort_values(["evaluation", "target", "mae", "run"]).to_html(
        index=False,
        float_format=lambda value: f"{value:.6f}",
    )


def human_judgment_rows(run_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for evaluation, filename in [
        ("holdout", "human_pair_judgment_template.csv"),
        ("cv_oof", "cv_human_pair_judgment_template.csv"),
        ("checkpoint_eval", "eval_human_pair_judgment_template.csv"),
    ]:
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


def run_card(run_dir: Path, out_html: Path) -> str:
    links = [
        link(run_dir / "run_report.html", "run report", out_html),
        link(run_dir / "last_checkpoint.pt", "last checkpoint", out_html),
        link(run_dir / "predictions.csv", "predictions", out_html),
        link(run_dir / "human_review.csv", "human review", out_html),
        link(run_dir / "human_pair_review.csv", "pairwise review", out_html),
        link(run_dir / "human_pair_judgment_template.csv", "judgment template", out_html),
        link(run_dir / "embedding_report.html", "embedding report", out_html),
        link(run_dir / "embedding_projection.csv", "embedding CSV", out_html),
        link(run_dir / "attention_report.html", "attention report", out_html),
        link(run_dir / "attention_map.csv", "attention CSV", out_html),
        link(run_dir / "error_slices.csv", "error slices", out_html),
        link(run_dir / "cv_human_review.csv", "CV human review", out_html),
        link(run_dir / "cv_human_pair_judgment_template.csv", "CV judgment template", out_html),
        link(run_dir / "cv_error_slices.csv", "CV error slices", out_html),
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
        f"{human_judgment_table(run_dir)}"
        f"{training_health_html(run_dir / 'history.csv', heading_level=4)}"
        f"{training_performance_html(run_dir / 'history.csv', heading_level=4)}"
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
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    run_dirs = [Path(run_dir) for run_dir in run_dirs if Path(run_dir).exists()]
    comparison_link = link(comparison_html, "comparison report", out_html) if comparison_html else ""
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
    args = parser.parse_args()

    write_dashboard(
        args.out_html,
        run_dirs=args.runs,
        audit_dir=args.audit_dir,
        forest_sweep_dir=args.forest_sweep_dir,
        neural_sweep_dir=args.neural_sweep_dir,
        comparison_html=args.comparison_html,
    )
    print(f"Wrote dashboard to {args.out_html}")


if __name__ == "__main__":
    main()
