from __future__ import annotations

import csv
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


JUDGMENT_COLUMNS = ["human_harder_beatmap_id", "human_confidence", "human_notes"]


def write_pair_judgment_template(path: Path, pair_review_csv: Path) -> None:
    review = pd.read_csv(pair_review_csv)
    for column in JUDGMENT_COLUMNS:
        if column not in review.columns:
            review[column] = ""
    review.to_csv(path, index=False, encoding="utf-8")


def normalize_human_choice(value: object) -> int | None:
    if pd.isna(value) or str(value).strip() == "":
        return None
    text = str(value).strip().lower()
    if text in {"tie", "same", "equal", "skip", "unknown"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def normalize_human_confidence(value: object) -> float:
    if pd.isna(value) or str(value).strip() == "":
        return 1.0
    try:
        confidence = float(str(value).strip())
    except ValueError:
        return 1.0
    if confidence <= 0:
        return 1.0
    return confidence


def score_pair_judgments(judgments_csv: Path) -> dict[str, Any]:
    judgments = pd.read_csv(judgments_csv)
    model_agree = 0
    proxy_agree = 0
    invalid = 0
    unjudged = 0
    judged = 0
    weighted_model_agree = 0.0
    weighted_proxy_agree = 0.0
    weighted_judged = 0.0

    for _, row in judgments.iterrows():
        human_choice = normalize_human_choice(row.get("human_harder_beatmap_id", ""))
        if human_choice is None:
            unjudged += 1
            continue
        confidence = normalize_human_confidence(row.get("human_confidence", ""))
        model_harder = int(row["model_harder_beatmap_id"])
        observed_harder = int(row["observed_harder_beatmap_id"])
        if human_choice == model_harder:
            model_agree += 1
            judged += 1
            weighted_model_agree += confidence
            weighted_judged += confidence
        elif human_choice == observed_harder:
            proxy_agree += 1
            judged += 1
            weighted_proxy_agree += confidence
            weighted_judged += confidence
        else:
            invalid += 1

    row_count = int(len(judgments))
    model_rate = model_agree / judged if judged else 0.0
    proxy_rate = proxy_agree / judged if judged else 0.0

    return {
        "row_count": row_count,
        "judged_count": judged,
        "unjudged_count": unjudged,
        "invalid_choice_count": invalid,
        "model_agree_count": model_agree,
        "proxy_agree_count": proxy_agree,
        "judgment_coverage_rate": judged / row_count if row_count else 0.0,
        "model_agreement_rate": model_rate,
        "proxy_agreement_rate": proxy_rate,
        "model_vs_proxy_agreement_delta": model_rate - proxy_rate,
        "confidence_weighted_judged_count": weighted_judged,
        "confidence_weighted_model_agree_count": weighted_model_agree,
        "confidence_weighted_proxy_agree_count": weighted_proxy_agree,
        "confidence_weighted_model_agreement_rate": (
            weighted_model_agree / weighted_judged if weighted_judged else 0.0
        ),
        "confidence_weighted_proxy_agreement_rate": (
            weighted_proxy_agree / weighted_judged if weighted_judged else 0.0
        ),
        "confidence_weighted_model_vs_proxy_delta": (
            (weighted_model_agree - weighted_proxy_agree) / weighted_judged
            if weighted_judged
            else 0.0
        ),
    }


def write_score_json(path: Path, score: dict[str, Any]) -> None:
    path.write_text(json.dumps(score, indent=2, ensure_ascii=False), encoding="utf-8")


def write_score_html(path: Path, score: dict[str, Any], judgments_csv: Path) -> None:
    rows = []
    for key, value in score.items():
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(str(value))}</td></tr>")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>human judgment score</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Human Pair Judgment Score</h1>
  <p>Judgments file: <code>{html.escape(str(judgments_csv))}</code></p>
  <table><tbody>{''.join(rows)}</tbody></table>
</body>
</html>
"""
    path.write_text(report, encoding="utf-8")


def write_score_csv(path: Path, score: dict[str, Any]) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(score.keys()))
        writer.writeheader()
        writer.writerow(score)
