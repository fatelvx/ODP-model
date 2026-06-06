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
from mania_difficulty.data.parse_notes import read_osu_metadata


MIN_USABLE_ROWS_FOR_REAL_TRAINING = 100
MIN_FULL_TOP100_RATE = 0.8
MAX_LOW_SCORE_COUNT_RATE = 0.25
MAX_SEQUENCE_TRUNCATION_RATE = 0.05
DIFFICULTY_METADATA_COLUMNS = ("hp_drain_rate", "overall_difficulty", "approach_rate")


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


def label_reliability_summary(
    usable: pd.DataFrame,
    *,
    low_score_count_threshold: int = 80,
) -> dict[str, Any]:
    if "score_count" not in usable.columns:
        return {"score_count_available": False}
    scores = pd.to_numeric(usable["score_count"], errors="coerce").dropna()
    if scores.empty:
        return {"score_count_available": False}
    row_count = int(len(scores))
    low_rows = int((scores < low_score_count_threshold).sum())
    full_top100_rows = int((scores >= 100).sum())
    return {
        "score_count_available": True,
        "usable_score_count_rows": row_count,
        "low_score_count_threshold": low_score_count_threshold,
        "low_score_count_rows": low_rows,
        "low_score_count_rate": low_rows / row_count if row_count else 0.0,
        "full_top100_rows": full_top100_rows,
        "full_top100_rate": full_top100_rows / row_count if row_count else 0.0,
        "min_score_count": float(scores.min()),
        "median_score_count": float(scores.median()),
    }


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series([np.nan] * len(frame), index=frame.index, dtype="float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _metadata_mismatch(label_value: Any, osu_value: Any) -> bool:
    if label_value in (None, "") or osu_value in (None, ""):
        return False
    try:
        return not bool(np.isclose(float(label_value), float(osu_value), atol=1e-6))
    except (TypeError, ValueError):
        return str(label_value) != str(osu_value)


def source_integrity_summary(
    labels: pd.DataFrame,
    *,
    expected_mode: int = 3,
    expected_keys: int = 4,
    osu_dir: Path | None = None,
) -> dict[str, Any]:
    mode = _numeric_column(labels, "mode")
    key_column = "keys" if "keys" in labels.columns else "circle_size"
    keys = _numeric_column(labels, key_column)
    summary: dict[str, Any] = {
        "checked_rows": int(len(labels)),
        "expected_mode": int(expected_mode),
        "expected_keys": int(expected_keys),
        "mode_column_available": "mode" in labels.columns,
        "mode_missing_rows": int(mode.isna().sum()),
        "mode_mismatch_rows": int(((mode.notna()) & (mode != expected_mode)).sum()),
        "keys_column": key_column if key_column in labels.columns else "",
        "keys_column_available": key_column in labels.columns,
        "keys_missing_rows": int(keys.isna().sum()),
        "keys_mismatch_rows": int(((keys.notna()) & (keys != expected_keys)).sum()),
    }
    for column in DIFFICULTY_METADATA_COLUMNS:
        values = _numeric_column(labels, column)
        summary[f"{column}_available"] = column in labels.columns
        summary[f"{column}_missing_rows"] = int(values.isna().sum())

    summary.update(
        {
            "osu_dir_checked": osu_dir is not None,
            "osu_files_checked": 0,
            "osu_missing_files": 0,
            "osu_mode_mismatch_rows": 0,
            "osu_keys_mismatch_rows": 0,
            "osu_metadata_label_mismatch_rows": 0,
            "osu_metadata_label_mismatch_examples": [],
        }
    )
    if osu_dir is None:
        return summary

    examples: list[dict[str, Any]] = []
    for _, row in labels.iterrows():
        beatmap_id = str(int(row["beatmap_id"]))
        osu_path = osu_dir / f"{beatmap_id}.osu"
        if not osu_path.exists():
            summary["osu_missing_files"] += 1
            continue
        summary["osu_files_checked"] += 1
        metadata = read_osu_metadata(osu_path)
        if metadata.get("mode") != expected_mode:
            summary["osu_mode_mismatch_rows"] += 1
        if metadata.get("keys") != expected_keys:
            summary["osu_keys_mismatch_rows"] += 1

        mismatched_columns = [
            column
            for column in ("mode", "keys", *DIFFICULTY_METADATA_COLUMNS)
            if column in labels.columns and _metadata_mismatch(row.get(column), metadata.get(column))
        ]
        if mismatched_columns:
            summary["osu_metadata_label_mismatch_rows"] += 1
            if len(examples) < 10:
                examples.append(
                    {
                        "beatmap_id": int(row["beatmap_id"]),
                        "columns": ",".join(mismatched_columns),
                    }
                )
    summary["osu_metadata_label_mismatch_examples"] = examples
    return summary


def sequence_truncation_summary(sequence_lengths: list[int], *, max_notes: int) -> dict[str, Any]:
    if not sequence_lengths:
        return {
            "max_notes": int(max_notes),
            "truncated_rows": 0,
            "truncated_rate": 0.0,
            "max_notes_over_limit": 0,
        }
    lengths = np.asarray(sequence_lengths, dtype=int)
    over_limit = np.maximum(lengths - int(max_notes), 0)
    truncated_rows = int((over_limit > 0).sum())
    return {
        "max_notes": int(max_notes),
        "truncated_rows": truncated_rows,
        "truncated_rate": truncated_rows / int(len(lengths)),
        "max_notes_over_limit": int(over_limit.max()),
    }


def quality_warning(code: str, message: str, severity: str = "warning") -> dict[str, str]:
    return {"code": code, "severity": severity, "message": message}


def dataset_quality_warnings(summary: dict[str, Any]) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    missing_sequence_count = int(summary.get("missing_sequence_count", 0) or 0)
    if missing_sequence_count:
        warnings.append(
            quality_warning(
                "missing_sequences",
                f"{missing_sequence_count} labels are missing parsed note sequences and cannot be trained.",
            )
        )

    target_missing = summary.get("target_missing", [])
    if target_missing:
        warnings.append(
            quality_warning(
                "missing_targets",
                f"Missing target columns: {', '.join(str(column) for column in target_missing)}.",
                severity="error",
            )
        )

    usable_rows = int(summary.get("usable_rows", 0) or 0)
    if usable_rows < MIN_USABLE_ROWS_FOR_REAL_TRAINING:
        warnings.append(
            quality_warning(
                "small_usable_dataset",
                (
                    f"Only {usable_rows} usable maps are available; use this for smoke tests, "
                    "not final model comparison."
                ),
            )
        )

    duplicate_count = int(summary.get("duplicate_beatmap_id_count", 0) or 0)
    if duplicate_count:
        warnings.append(
            quality_warning(
                "duplicate_beatmap_ids",
                f"{duplicate_count} duplicate beatmap_id rows were found; deduplicate labels before training.",
            )
        )

    truncation = summary.get("sequence_truncation", {})
    if isinstance(truncation, dict):
        truncated_rate = float(truncation.get("truncated_rate", 0.0) or 0.0)
        if truncated_rate > MAX_SEQUENCE_TRUNCATION_RATE:
            warnings.append(
                quality_warning(
                    "sequence_truncation",
                    (
                        f"{truncated_rate:.2%} of usable maps exceed max_notes "
                        f"{truncation.get('max_notes', '')}; raise --max-notes or inspect GPU memory."
                    ),
                )
            )

    reliability = summary.get("label_reliability", {})
    if isinstance(reliability, dict) and reliability.get("score_count_available"):
        full_top100_rate = float(reliability.get("full_top100_rate", 0.0) or 0.0)
        if full_top100_rate < MIN_FULL_TOP100_RATE:
            warnings.append(
                quality_warning(
                    "low_full_top100_rate",
                    (
                        f"Only {full_top100_rate:.2%} of usable maps have full top100 score coverage; "
                        "keep sample weighting on and verify with human judgments."
                    ),
                )
            )
        low_score_count_rate = float(reliability.get("low_score_count_rate", 0.0) or 0.0)
        if low_score_count_rate > MAX_LOW_SCORE_COUNT_RATE:
            threshold = reliability.get("low_score_count_threshold", "")
            warnings.append(
                quality_warning(
                    "high_low_score_count_rate",
                    (
                        f"{low_score_count_rate:.2%} of usable maps are below score_count {threshold}; "
                        "labels may be noisy."
                    ),
                )
            )

    source = summary.get("source_integrity", {})
    if isinstance(source, dict) and source:
        if not source.get("mode_column_available"):
            warnings.append(
                quality_warning(
                    "missing_mode_metadata",
                    "Labels do not include a mode column; cannot prove rows are osu!mania.",
                    severity="error",
                )
            )
        mode_mismatch_rows = int(source.get("mode_mismatch_rows", 0) or 0)
        if mode_mismatch_rows:
            warnings.append(
                quality_warning(
                    "non_mania_rows",
                    f"{mode_mismatch_rows} label rows do not match expected osu!mania mode 3.",
                    severity="error",
                )
            )
        if not source.get("keys_column_available"):
            warnings.append(
                quality_warning(
                    "missing_key_metadata",
                    "Labels do not include keys/circle_size; cannot prove the key mode.",
                    severity="error",
                )
            )
        key_mismatch_rows = int(source.get("keys_mismatch_rows", 0) or 0)
        if key_mismatch_rows:
            warnings.append(
                quality_warning(
                    "wrong_key_mode_rows",
                    f"{key_mismatch_rows} label rows do not match expected {source.get('expected_keys')}K mania.",
                    severity="error",
                )
            )
        missing_difficulty_rows = max(
            int(source.get(f"{column}_missing_rows", 0) or 0)
            for column in DIFFICULTY_METADATA_COLUMNS
        )
        if missing_difficulty_rows:
            warnings.append(
                quality_warning(
                    "missing_difficulty_metadata",
                    f"Up to {missing_difficulty_rows} rows are missing HP/OD/AR metadata.",
                )
            )
        if source.get("osu_dir_checked"):
            osu_missing_files = int(source.get("osu_missing_files", 0) or 0)
            if osu_missing_files:
                warnings.append(
                    quality_warning(
                        "missing_osu_files",
                        f"{osu_missing_files} labels do not have a matching raw .osu file for source verification.",
                    )
                )
            osu_mode_mismatch_rows = int(source.get("osu_mode_mismatch_rows", 0) or 0)
            if osu_mode_mismatch_rows:
                warnings.append(
                    quality_warning(
                        "osu_non_mania_files",
                        f"{osu_mode_mismatch_rows} raw .osu files do not declare mode 3.",
                        severity="error",
                    )
                )
            osu_key_mismatch_rows = int(source.get("osu_keys_mismatch_rows", 0) or 0)
            if osu_key_mismatch_rows:
                warnings.append(
                    quality_warning(
                        "osu_wrong_key_mode_files",
                        f"{osu_key_mismatch_rows} raw .osu files do not declare {source.get('expected_keys')}K.",
                        severity="error",
                    )
                )
            label_mismatch_rows = int(source.get("osu_metadata_label_mismatch_rows", 0) or 0)
            if label_mismatch_rows:
                warnings.append(
                    quality_warning(
                        "osu_label_metadata_mismatch",
                        f"{label_mismatch_rows} raw .osu files disagree with label metadata.",
                        severity="error",
                    )
                )

    return warnings


def audit_dataset(
    labels_csv: Path,
    sequences_dir: Path,
    *,
    target_columns: tuple[str, ...] = DEFAULT_TARGET_COLUMNS,
    max_notes: int = 3000,
    expected_mode: int = 3,
    expected_keys: int = 4,
    osu_dir: Path | None = None,
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
        "sequence_truncation": sequence_truncation_summary(sequence_lengths, max_notes=max_notes),
        "score_count": score_summary,
        "label_reliability": label_reliability_summary(usable),
        "source_integrity": source_integrity_summary(
            labels,
            expected_mode=expected_mode,
            expected_keys=expected_keys,
            osu_dir=osu_dir,
        ),
        "targets": target_stats,
    }
    summary["quality_warnings"] = dataset_quality_warnings(summary)
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


def label_reliability_table(summary: dict[str, Any]) -> str:
    reliability = summary.get("label_reliability", {})
    if not reliability or not reliability.get("score_count_available"):
        return ""
    rows = [
        ("Usable Score Count Rows", reliability.get("usable_score_count_rows", 0)),
        ("Low Score Count Threshold", reliability.get("low_score_count_threshold", "")),
        ("Low Score Count Rows", reliability.get("low_score_count_rows", 0)),
        ("Low Score Count Rate", f"{float(reliability.get('low_score_count_rate', 0.0)):.2%}"),
        ("Full Top100 Rows", reliability.get("full_top100_rows", 0)),
        ("Full Top100 Rate", f"{float(reliability.get('full_top100_rate', 0.0)):.2%}"),
        ("Min Score Count", f"{float(reliability.get('min_score_count', 0.0)):.0f}"),
        ("Median Score Count", f"{float(reliability.get('median_score_count', 0.0)):.0f}"),
    ]
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<h2>Label Reliability</h2><table><tbody>{row_html}</tbody></table>"


def source_integrity_table(summary: dict[str, Any]) -> str:
    source = summary.get("source_integrity", {})
    if not isinstance(source, dict) or not source:
        return ""
    rows = [
        ("Expected Mode", source.get("expected_mode", "")),
        ("Expected Keys", source.get("expected_keys", "")),
        ("Mode Column Available", source.get("mode_column_available", "")),
        ("Mode Missing Rows", source.get("mode_missing_rows", "")),
        ("Mode Mismatch Rows", source.get("mode_mismatch_rows", "")),
        ("Keys Column", source.get("keys_column", "")),
        ("Keys Missing Rows", source.get("keys_missing_rows", "")),
        ("Keys Mismatch Rows", source.get("keys_mismatch_rows", "")),
        ("HP Missing Rows", source.get("hp_drain_rate_missing_rows", "")),
        ("OD Missing Rows", source.get("overall_difficulty_missing_rows", "")),
        ("AR Missing Rows", source.get("approach_rate_missing_rows", "")),
        ("Raw .osu Checked", source.get("osu_dir_checked", "")),
        ("Raw .osu Files Checked", source.get("osu_files_checked", "")),
        ("Raw .osu Missing Files", source.get("osu_missing_files", "")),
        ("Raw .osu Mode Mismatch Rows", source.get("osu_mode_mismatch_rows", "")),
        ("Raw .osu Keys Mismatch Rows", source.get("osu_keys_mismatch_rows", "")),
        ("Raw .osu / Label Metadata Mismatch Rows", source.get("osu_metadata_label_mismatch_rows", "")),
    ]
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<h2>Source Integrity</h2><table><tbody>{row_html}</tbody></table>"


def sequence_truncation_table(summary: dict[str, Any]) -> str:
    truncation = summary.get("sequence_truncation", {})
    if not isinstance(truncation, dict) or not truncation:
        return ""
    rows = [
        ("Max Notes", truncation.get("max_notes", "")),
        ("Truncated Rows", truncation.get("truncated_rows", "")),
        ("Truncated Rate", f"{float(truncation.get('truncated_rate', 0.0)):.2%}"),
        ("Max Notes Over Limit", truncation.get("max_notes_over_limit", "")),
    ]
    row_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in rows
    )
    return f"<h2>Sequence Truncation</h2><table><tbody>{row_html}</tbody></table>"


def quality_warnings_table(summary: dict[str, Any]) -> str:
    warnings = summary.get("quality_warnings", [])
    if not isinstance(warnings, list) or not warnings:
        return ""
    rows = []
    for warning in warnings:
        if not isinstance(warning, dict):
            continue
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(warning.get('severity', 'warning')))}</td>"
            f"<td><code>{html.escape(str(warning.get('code', '')))}</code></td>"
            f"<td>{html.escape(str(warning.get('message', '')))}</td>"
            "</tr>"
        )
    if not rows:
        return ""
    return (
        "<h2>Dataset Quality Warnings</h2>"
        "<table><thead><tr><th>Severity</th><th>Code</th><th>Message</th></tr></thead>"
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
  {quality_warnings_table(summary)}
  {source_integrity_table(summary)}
  {sequence_truncation_table(summary)}
  {label_reliability_table(summary)}
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
    parser.add_argument(
        "--max-notes",
        type=int,
        default=3000,
        help="Training max-notes value used to estimate sequence truncation risk.",
    )
    parser.add_argument("--expected-mode", type=int, default=3, help="Expected osu! mode id; mania is 3.")
    parser.add_argument("--expected-keys", type=int, default=4, help="Expected mania key mode.")
    parser.add_argument("--osu-dir", type=Path, default=None, help="Optional raw .osu directory to cross-check.")
    args = parser.parse_args()

    target_columns = tuple(column.strip() for column in args.targets.split(",") if column.strip())
    summary, missing_rows = audit_dataset(
        args.labels,
        args.sequences,
        target_columns=target_columns,
        max_notes=args.max_notes,
        expected_mode=args.expected_mode,
        expected_keys=args.expected_keys,
        osu_dir=args.osu_dir,
    )
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
