from __future__ import annotations

import argparse
from pathlib import Path

from mania_difficulty.human_judgments import (
    score_pair_judgments,
    write_pair_judgment_template,
    write_score_csv,
    write_score_html,
    write_score_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create and score human pairwise difficulty judgments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    template_parser = subparsers.add_parser("template", help="Create a fillable judgment CSV.")
    template_parser.add_argument("--pair-review", type=Path, required=True)
    template_parser.add_argument("--out", type=Path, required=True)

    score_parser = subparsers.add_parser("score", help="Score filled human judgment CSV.")
    score_parser.add_argument("--judgments", type=Path, required=True)
    score_parser.add_argument("--out-json", type=Path, default=Path("outputs/human_judgment_score.json"))
    score_parser.add_argument("--out-html", type=Path, default=Path("outputs/human_judgment_score.html"))
    score_parser.add_argument("--out-csv", type=Path, default=Path("outputs/human_judgment_score.csv"))
    args = parser.parse_args()

    if args.command == "template":
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_pair_judgment_template(args.out, args.pair_review)
        print(f"Wrote human judgment template to {args.out}")
        return

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_html.parent.mkdir(parents=True, exist_ok=True)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    score = score_pair_judgments(args.judgments)
    write_score_json(args.out_json, score)
    write_score_html(args.out_html, score, args.judgments)
    write_score_csv(args.out_csv, score)
    print(score)
    print(f"Wrote {args.out_json}, {args.out_html}, and {args.out_csv}")


if __name__ == "__main__":
    main()
