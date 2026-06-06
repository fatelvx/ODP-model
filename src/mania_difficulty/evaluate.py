from __future__ import annotations

import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from mania_difficulty.data.dataset import ManiaDifficultyDataset, collate_batch
from mania_difficulty.error_analysis import write_error_slices
from mania_difficulty.human_judgments import write_pair_judgment_template
from mania_difficulty.metrics import regression_report
from mania_difficulty.models.factory import create_model
from mania_difficulty.train import (
    dataloader_options,
    evaluate_loader,
    mixed_precision_enabled,
    ranking_target_column,
    write_human_review,
    write_pairwise_review,
    write_prediction_rankings,
    write_prediction_summary,
    write_predictions,
)
from mania_difficulty.visualize import plot_prediction_scatter, write_run_report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a saved checkpoint on a dataset.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--pin-memory", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--loader-prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp_active = mixed_precision_enabled(args, device)
    checkpoint = torch.load(args.checkpoint, map_location=device)
    target_columns = list(checkpoint["target_columns"])
    target_mean_np = np.asarray(checkpoint["target_mean"], dtype="float32")
    target_std_np = np.asarray(checkpoint["target_std"], dtype="float32")
    target_mean_t = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std_np, dtype=torch.float32, device=device)

    dataset = ManiaDifficultyDataset(args.labels, args.sequences, target_columns=target_columns)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, max_notes=int(checkpoint.get("max_notes", 3000))),
        **dataloader_options(args, device),
    )

    model = create_model(
        checkpoint.get("model_name", "lstm"),
        output_dim=len(target_columns),
        config=checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    _, pred, actual, beatmap_ids = evaluate_loader(
        model,
        loader,
        device,
        target_mean_t,
        target_std_t,
        target_mean_np,
        target_std_np,
        amp_active,
    )

    out_dir = args.out_dir or args.checkpoint.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = out_dir / "eval_predictions.csv"
    write_predictions(predictions_csv, beatmap_ids, actual, pred, target_columns)
    write_prediction_summary(out_dir / "eval_prediction_summary.csv", predictions_csv, target_columns)
    write_prediction_rankings(
        out_dir / "eval_prediction_rankings.csv",
        args.labels,
        predictions_csv,
        target_column=ranking_target_column(target_columns),
    )
    write_human_review(out_dir / "eval_human_review.csv", args.labels, predictions_csv, target_columns)
    write_pairwise_review(out_dir / "eval_human_pair_review.csv", args.labels, predictions_csv)
    write_pair_judgment_template(
        out_dir / "eval_human_pair_judgment_template.csv",
        out_dir / "eval_human_pair_review.csv",
    )
    write_error_slices(out_dir / "eval_error_slices.csv", args.labels, predictions_csv)
    metrics = regression_report(actual, pred, target_columns)
    metrics_path = out_dir / "eval_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    plot_prediction_scatter(predictions_csv, target_columns, out_dir / "eval_prediction_scatter.png")
    write_run_report(
        out_dir,
        target_columns=target_columns,
        metrics_path=metrics_path,
        learning_curve_name="",
        prediction_scatter_name="eval_prediction_scatter.png",
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
