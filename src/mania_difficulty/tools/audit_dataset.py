from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mania_difficulty.data.dataset import DEFAULT_TARGET_COLUMNS


def numeric_summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "min": 0.0, "median": 0.0, "mean": 0.0, "max": 0.0, "std": 0.0}
    array = np.asarray(values, dtype=float)
    return {
        "count": int(len(array)),
        "min": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
    }


def sequence_length(path: Path) -> int:
    array = np.load(path, mmap_mode="r")
    return int(array.shape[0])


def audit_dataset(
    labels_csv: Path,
    sequences_dir: Path,
    *,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
) -> tuple[dict[str, Any], list[dict[str, object]]]:
    labels = pd.read_csv(labels_csv)
    sequences_dir = Path(sequences_dir)
    sequence_files = {path.stem for path in sequences_dir.glob("*.npy")}
    target_missing = [column for column in target_columns if column not in labels.columns]

    missing_rows: list[dict[str, object]] = []
    usable_indices = []
    sequence_lengths = []
    seen_ids: set[str] = set()
    duplicate_count = 0

    for row_index, row in labels.reset_index(drop=True).iterrows():
        beatmap_id = str(int(row["beatmap_id"]))
        if beatmap_id in seen_ids:
            duplicate_count += 1
        seen_ids.add(beatmap_id)
        sequence_path = sequences_dir / f"{beatmap_id}.npy"
        if not sequence_path.exists():
            missing_rows.append(
                {
                    "beatmap_id": int(row["beatmap_id"]),
                    "title": row.get("title", ""),
                    "artist": row.get("artist", ""),
                    "mapper": row.get("mapper", ""),
                    "version": row.get("version", ""),
                    "missing_path": str(sequence_path),
                }
            )
            continue
        usable_indices.append(row_index)
        sequence_lengths.append(sequence_length(sequence_path))

    usable = labels.iloc[usable_indices].copy() if usable_indices else pd.DataFrame()
    target_stats = {
        column: numeric_summary(pd.to_numeric(usable[column], errors="coerce").dropna().tolist())
        for column in target_columns
        if column in usable.columns
    }
    extra_sequence_count = len(sequence_files - {str(int(value)) for value in labels["beatmap_id"]})
    group_count = (
        int(labels["beatmapset_id"].nunique(dropna=True)) if "beatmapset_id" in labels.columns else 0
    )
    score_summary = (
        numeric_summary(pd.to_numeric(usable["score_count"], errors="coerce").dropna().tolist())
        if "score_count" in usable.columns
        else {}
    )

    summary: dict[str, Any] = {
        "label_rows": int(len(labels)),
        "usable_rows": int(len(usable)),
        "missing_sequence_count": int(len(missing_rows)),
        "extra_sequence_count": int(extra_sequence_count),
        "duplicate_beatmap_id_count": int(duplicate_count),
        "target_missing": target_missing,
        "group_column": "beatmapset_id" if "beatmapset_id" in labels.columns else "",
        "group_count": group_count,
        "sequence_length": numeric_summary(sequence_lengths),
        "score_count": score_summary,
        "targets": target_stats,
    }
    return summary, missing_rows


def write_missing_sequences(path: Path, missing_rows: list[dict[str, object]]) -> None:
    fieldnames = ["beatmap_id", "title", "artist", "mapper", "version", "missing_path"]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missing_rows)


def plot_dataset_distributions(
    labels_csv: Path,
    sequences_dir: Path,
    target_columns: tuple[str, ...],
    out_path: Path,
) -> None:
    labels = pd.read_csv(labels_csv)
    usable_rows = []
    lengths = []
    for _, row in labels.iterrows():
        sequence_path = sequences_dir / f"{int(row['beatmap_id'])}.npy"
        if sequence_path.exists():
            usable_rows.append(row)
            lengths.append(sequence_length(sequence_path))
    usable = pd.DataFrame(usable_rows)

    plot_columns = [column for column in target_columns if column in usable.columns]
    extra_panels = 1 + (1 if "score_count" in usable.columns else 0)
    panel_count = max(1, len(plot_columns) + extra_panels)
    fig, axes = plt.subplots(panel_count, 1, figsize=(8, max(4, panel_count * 2.4)))
    if panel_count == 1:
        axes = [axes]

    axis_index = 0
    for column in plot_columns:
        axes[axis_index].hist(pd.to_numeric(usable[column], errors="coerce").dropna(), bins=20)
        axes[axis_index].set_title(column)
        axes[axis_index].grid(True, alpha=0.25)
        axis_index += 1
    axes[axis_index].hist(lengths, bins=20)
    axes[axis_index].set_title("sequence note count")
    axes[axis_index].grid(True, alpha=0.25)
    axis_index += 1
    if "score_count" in usable.columns:
        axes[axis_index].hist(pd.to_numeric(usable["score_count"], errors="coerce").dropna(), bins=20)
        axes[axis_index].set_title("score_count")
        axes[axis_index].grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def summary_table(summary: dict[str, Any]) -> str:
    rows = [
        ("Label rows", summary["label_rows"]),
        ("Usable rows", summary["usable_rows"]),
        ("Missing sequences", summary["missing_sequence_count"]),
        ("Extra sequence files", summary["extra_sequence_count"]),
        ("Duplicate beatmap IDs", summary["duplicate_beatmap_id_count"]),
        ("Group count", summary["group_count"]),
    ]
    return "<table><tbody>" + "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    ) + "</tbody></table>"


def stats_table(title: str, stats: dict[str, Any]) -> str:
    if not stats:
        return ""
    rows = []
    for name, values in stats.items():
        if isinstance(values, dict):
            rows.append(
                "<tr>"
                f"<td>{html.escape(str(name))}</td>"
                f"<td>{values.get('count', 0)}</td>"
                f"<td>{values.get('min', 0):.6f}</td>"
                f"<td>{values.get('median', 0):.6f}</td>"
                f"<td>{values.get('mean', 0):.6f}</td>"
                f"<td>{values.get('max', 0):.6f}</td>"
                f"<td>{values.get('std', 0):.6f}</td>"
                "</tr>"
            )
    if not rows:
        return ""
    return (
        f"<h2>{html.escape(title)}</h2>"
        "<table><thead><tr><th>Name</th><th>Count</th><th>Min</th><th>Median</th>"
        "<th>Mean</th><th>Max</th><th>Std</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def write_html_report(summary: dict[str, Any], out_dir: Path) -> None:
    target_missing = ", ".join(summary["target_missing"]) if summary["target_missing"] else "none"
    sequence_stats = {"sequence_length": summary["sequence_length"]}
    score_stats = {"score_count": summary["score_count"]} if summary.get("score_count") else {}
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>dataset audit</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Dataset Audit</h1>
  <p>Target columns missing: <code>{html.escape(target_missing)}</code></p>
  {summary_table(summary)}
  {stats_table("Target Distributions", summary["targets"])}
  {stats_table("Sequence Length", sequence_stats)}
  {stats_table("Score Count", score_stats)}
  <h2>Distributions</h2>
  <p><img src="dataset_distributions.png" alt="Dataset distributions"></p>
  <h2>Files</h2>
  <p>Open <code>dataset_audit.json</code> for machine-readable summary.</p>
  <p>Open <code>missing_sequences.csv</code> for labels that cannot be trained because their .npy sequence is missing.</p>
</body>
</html>
"""
    (out_dir / "dataset_audit.html").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit labels and parsed sequence coverage before training.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/dataset_audit"))
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    args = parser.parse_args()

    target_columns = tuple(column.strip() for column in args.targets.split(",") if column.strip())
    summary, missing_rows = audit_dataset(args.labels, args.sequences, target_columns=target_columns)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "dataset_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_missing_sequences(args.out_dir / "missing_sequences.csv", missing_rows)
    plot_dataset_distributions(
        args.labels,
        args.sequences,
        target_columns,
        args.out_dir / "dataset_distributions.png",
    )
    write_html_report(summary, args.out_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote dataset audit to {args.out_dir}")


if __name__ == "__main__":
    main()
