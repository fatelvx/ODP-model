from __future__ import annotations

import argparse
import html
import json
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from mania_difficulty.data.dataset import DEFAULT_TARGET_COLUMNS
from mania_difficulty.train import (
    CHECKPOINT_METRICS,
    positive_float,
    positive_int,
    sample_weight_min_float,
    train,
)
from mania_difficulty.tools.sweep_selection import (
    parse_selection_metric,
    selection_sort_ascending,
    selection_sort_value,
)


def parse_int_list(value: str) -> list[int]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    parsed = [int(item) for item in items]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return parsed


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    parsed = [float(item) for item in items]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return parsed


def parse_model_list(value: str) -> list[str]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    allowed = {"summary", "lstm", "transformer"}
    invalid = [item for item in items if item not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(f"Unknown neural models: {invalid}")
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one model.")
    return items


def value_token(value: object) -> str:
    return str(value).replace("-", "m").replace(".", "p")


def model_size_score(candidate: dict[str, Any]) -> int:
    if candidate["model"] == "summary":
        hidden_dim = int(candidate["summary_hidden_dim"])
        return hidden_dim + hidden_dim // 2
    if candidate["model"] == "transformer":
        embed_dim = int(candidate["transformer_embed_dim"])
        layers = int(candidate["transformer_layers"])
        ff_dim = int(candidate["transformer_ff_dim"])
        return embed_dim * max(1, layers) + ff_dim * max(1, layers)
    embed_dim = int(candidate["lstm_embed_dim"])
    hidden_dim = int(candidate["lstm_hidden_dim"])
    layers = int(candidate["lstm_layers"])
    return embed_dim * hidden_dim * max(1, layers)


def neural_grid(
    *,
    models: list[str],
    lrs: list[float],
    weight_decays: list[float],
    batch_sizes: list[int],
    huber_deltas: list[float],
    summary_hidden_dims: list[int],
    summary_dropouts: list[float],
    lstm_embed_dims: list[int],
    lstm_hidden_dims: list[int],
    lstm_layers: list[int],
    lstm_dropouts: list[float],
    lstm_head_dropouts: list[float],
    transformer_embed_dims: list[int] | None = None,
    transformer_heads: list[int] | None = None,
    transformer_layers: list[int] | None = None,
    transformer_ff_dims: list[int] | None = None,
    transformer_dropouts: list[float] | None = None,
    transformer_head_dropouts: list[float] | None = None,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    shared_options = list(product(lrs, weight_decays, batch_sizes, huber_deltas))
    transformer_embed_dims = transformer_embed_dims or [64]
    transformer_heads = transformer_heads or [4]
    transformer_layers = transformer_layers or [2]
    transformer_ff_dims = transformer_ff_dims or [256]
    transformer_dropouts = transformer_dropouts or [0.1]
    transformer_head_dropouts = transformer_head_dropouts or [0.2]
    for model_name in models:
        if model_name == "summary":
            for lr, weight_decay, batch_size, huber_delta in shared_options:
                for hidden_dim, dropout in product(summary_hidden_dims, summary_dropouts):
                    candidate = {
                        "candidate_id": (
                            f"summary_h{hidden_dim}_do{value_token(dropout)}_"
                            f"lr{value_token(lr)}_wd{value_token(weight_decay)}_"
                            f"bs{batch_size}_hd{value_token(huber_delta)}"
                        ),
                        "model": "summary",
                        "summary_hidden_dim": hidden_dim,
                        "summary_dropout": dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "huber_delta": huber_delta,
                    }
                    candidate["model_size_score"] = model_size_score(candidate)
                    candidates.append(candidate)
        elif model_name == "lstm":
            for lr, weight_decay, batch_size, huber_delta in shared_options:
                for embed_dim, hidden_dim, layers, dropout, head_dropout in product(
                    lstm_embed_dims,
                    lstm_hidden_dims,
                    lstm_layers,
                    lstm_dropouts,
                    lstm_head_dropouts,
                ):
                    candidate = {
                        "candidate_id": (
                            f"lstm_e{embed_dim}_h{hidden_dim}_l{layers}_do{value_token(dropout)}_"
                            f"head{value_token(head_dropout)}_lr{value_token(lr)}_"
                            f"wd{value_token(weight_decay)}_bs{batch_size}_"
                            f"hd{value_token(huber_delta)}"
                        ),
                        "model": "lstm",
                        "lstm_embed_dim": embed_dim,
                        "lstm_hidden_dim": hidden_dim,
                        "lstm_layers": layers,
                        "lstm_dropout": dropout,
                        "lstm_head_dropout": head_dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "huber_delta": huber_delta,
                    }
                    candidate["model_size_score"] = model_size_score(candidate)
                    candidates.append(candidate)
        elif model_name == "transformer":
            for lr, weight_decay, batch_size, huber_delta in shared_options:
                for embed_dim, heads, layers, ff_dim, dropout, head_dropout in product(
                    transformer_embed_dims,
                    transformer_heads,
                    transformer_layers,
                    transformer_ff_dims,
                    transformer_dropouts,
                    transformer_head_dropouts,
                ):
                    candidate = {
                        "candidate_id": (
                            f"transformer_e{embed_dim}_h{heads}_l{layers}_ff{ff_dim}_"
                            f"do{value_token(dropout)}_head{value_token(head_dropout)}_"
                            f"lr{value_token(lr)}_wd{value_token(weight_decay)}_"
                            f"bs{batch_size}_hd{value_token(huber_delta)}"
                        ),
                        "model": "transformer",
                        "transformer_embed_dim": embed_dim,
                        "transformer_heads": heads,
                        "transformer_layers": layers,
                        "transformer_ff_dim": ff_dim,
                        "transformer_dropout": dropout,
                        "transformer_head_dropout": head_dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "batch_size": batch_size,
                        "huber_delta": huber_delta,
                    }
                    candidate["model_size_score"] = model_size_score(candidate)
                    candidates.append(candidate)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    return candidates


def choose_best_candidate(
    summary_rows: list[dict[str, Any]],
    *,
    selection_metric: str = "mean_mae",
) -> dict[str, Any]:
    if not summary_rows:
        raise ValueError("No candidate rows to choose from.")
    return sorted(
        summary_rows,
        key=lambda row: (
            selection_sort_value(row, selection_metric),
            int(row.get("model_size_score", 0)),
            float(row.get("best_val_loss", float("inf"))),
            str(row.get("candidate_id", "")),
        ),
    )[0]


def candidate_train_args(base_args: argparse.Namespace, candidate: dict[str, Any]) -> SimpleNamespace:
    return SimpleNamespace(
        labels=base_args.labels,
        sequences=base_args.sequences,
        run_name=f"{base_args.run_prefix}_{candidate['candidate_id']}",
        targets=base_args.targets,
        epochs=base_args.epochs,
        batch_size=int(candidate["batch_size"]),
        lr=float(candidate["lr"]),
        weight_decay=float(candidate["weight_decay"]),
        patience=base_args.patience,
        max_notes=base_args.max_notes,
        group_column=base_args.group_column,
        seed=base_args.seed,
        device=base_args.device,
        model=candidate["model"],
        loss_weights=base_args.loss_weights,
        huber_delta=float(candidate.get("huber_delta", getattr(base_args, "huber_delta", 1.0))),
        sample_weight_column=getattr(base_args, "sample_weight_column", ""),
        sample_weight_min=getattr(base_args, "sample_weight_min", 0.25),
        sample_weight_max_value=getattr(base_args, "sample_weight_max_value", 100.0),
        forest_trees=400,
        forest_min_samples_leaf=2,
        forest_max_features="sqrt",
        feature_set="core",
        cv_folds=0,
        workers=base_args.workers,
        loader_workers=base_args.loader_workers,
        pin_memory=base_args.pin_memory,
        loader_prefetch_factor=base_args.loader_prefetch_factor,
        amp=base_args.amp,
        grad_accum_steps=getattr(base_args, "grad_accum_steps", 1),
        grad_clip_norm=getattr(base_args, "grad_clip_norm", 1.0),
        checkpoint_metric=getattr(base_args, "checkpoint_metric", "val_loss"),
        checkpoint_backup_dir=getattr(base_args, "checkpoint_backup_dir", None),
        lstm_embed_dim=int(candidate.get("lstm_embed_dim", base_args.lstm_embed_dims[0])),
        lstm_hidden_dim=int(candidate.get("lstm_hidden_dim", base_args.lstm_hidden_dims[0])),
        lstm_layers=int(candidate.get("lstm_layers", base_args.lstm_layers[0])),
        lstm_dropout=float(candidate.get("lstm_dropout", base_args.lstm_dropouts[0])),
        lstm_head_dropout=float(
            candidate.get("lstm_head_dropout", base_args.lstm_head_dropouts[0])
        ),
        summary_hidden_dim=int(candidate.get("summary_hidden_dim", base_args.summary_hidden_dims[0])),
        summary_dropout=float(candidate.get("summary_dropout", base_args.summary_dropouts[0])),
        transformer_embed_dim=int(
            candidate.get("transformer_embed_dim", getattr(base_args, "transformer_embed_dims", [64])[0])
        ),
        transformer_heads=int(
            candidate.get("transformer_heads", getattr(base_args, "transformer_heads", [4])[0])
        ),
        transformer_layers=int(
            candidate.get("transformer_layers", getattr(base_args, "transformer_layers", [2])[0])
        ),
        transformer_ff_dim=int(
            candidate.get("transformer_ff_dim", getattr(base_args, "transformer_ff_dims", [256])[0])
        ),
        transformer_dropout=float(
            candidate.get("transformer_dropout", getattr(base_args, "transformer_dropouts", [0.1])[0])
        ),
        transformer_head_dropout=float(
            candidate.get(
                "transformer_head_dropout",
                getattr(base_args, "transformer_head_dropouts", [0.2])[0],
            )
        ),
    )


def target_metric_rows(candidate: dict[str, Any], metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for target, values in metrics.items():
        if target.startswith("_") or not isinstance(values, dict):
            continue
        rows.append(
            {
                **candidate,
                "target": target,
                "mae": values.get("mae"),
                "r2": values.get("r2"),
                "spearman": values.get("spearman"),
                "pairwise_order_accuracy": values.get("pairwise_order_accuracy"),
                "baseline_mae": values.get("baseline_mae"),
                "mae_improvement_vs_baseline": values.get("mae_improvement_vs_baseline"),
                "mae_improvement_pct": values.get("mae_improvement_pct"),
            }
        )
    return rows


def summarize_run(
    candidate: dict[str, Any],
    metrics: dict[str, Any],
    run_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = target_metric_rows(candidate, metrics)
    run_info = metrics.get("_run", {})
    summary = {
        **candidate,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "mean_mae": float(np.mean([row["mae"] for row in rows])),
        "mean_r2": float(np.mean([row["r2"] for row in rows])),
        "mean_spearman": float(np.mean([row["spearman"] for row in rows])),
        "mean_pairwise_order_accuracy": float(
            np.mean([row["pairwise_order_accuracy"] for row in rows])
        ),
        "mean_baseline_mae": float(np.mean([row["baseline_mae"] for row in rows])),
        "mean_improvement_pct": float(np.mean([row["mae_improvement_pct"] for row in rows])),
        "best_epoch": run_info.get("best_epoch"),
        "best_val_loss": run_info.get("best_val_loss"),
        "checkpoint_metric": run_info.get("checkpoint_metric"),
        "best_checkpoint_score": run_info.get("best_checkpoint_score"),
        "test_loss": run_info.get("test_loss"),
        "split_strategy": run_info.get("split_strategy"),
        "group_column": run_info.get("group_column"),
        "group_count": run_info.get("group_count"),
        "train_size": run_info.get("train_size"),
        "val_size": run_info.get("val_size"),
        "test_size": run_info.get("test_size"),
        "grad_accum_steps": run_info.get("grad_accum_steps"),
        "grad_clip_norm": run_info.get("grad_clip_norm"),
        "huber_delta": run_info.get("huber_delta", candidate.get("huber_delta")),
        "effective_batch_size": run_info.get("effective_batch_size"),
        "model_config": json.dumps(run_info.get("model_config", {}), ensure_ascii=False),
    }
    return summary, rows


def evaluate_candidate(
    base_args: argparse.Namespace,
    candidate: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run_dir = train(candidate_train_args(base_args, candidate))
    metrics_path = run_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    return summarize_run(candidate, metrics, run_dir)


def write_html_report(
    summary_rows: list[dict[str, Any]],
    best: dict[str, Any],
    out_html: Path,
    *,
    selection_metric: str = "mean_mae",
) -> None:
    frame = pd.DataFrame(summary_rows).sort_values(
        [selection_metric, "model_size_score"],
        ascending=[selection_sort_ascending(selection_metric), True],
    )
    table = frame.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>neural sweep report</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin-top: 16px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
  </style>
</head>
<body>
  <h1>Neural Parameter Sweep</h1>
  <p>Best candidate: <code>{html.escape(str(best["candidate_id"]))}</code></p>
  <p>Selection metric: <code>{html.escape(selection_metric)}</code>. Lower MAE/loss is better; higher R2, Spearman, pairwise order, and improvement are better.</p>
  {table}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sweep summary/LSTM/Transformer neural parameters.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/neural_sweep"))
    parser.add_argument("--run-prefix", default="neural_sweep")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", dest="batch_sizes", type=parse_int_list, default=[32])
    parser.add_argument("--lrs", type=parse_float_list, default=[1e-3])
    parser.add_argument("--weight-decays", type=parse_float_list, default=[1e-4])
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-notes", type=int, default=3000)
    parser.add_argument("--group-column", default="beatmapset_id")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument("--models", type=parse_model_list, default=["summary"])
    parser.add_argument("--loss-weights", type=float, nargs="*", default=[1.0, 0.5, 0.5])
    parser.add_argument("--huber-deltas", type=parse_float_list, default=[1.0])
    parser.add_argument("--sample-weight-column", default="")
    parser.add_argument("--sample-weight-min", type=sample_weight_min_float, default=0.25)
    parser.add_argument("--sample-weight-max-value", type=positive_float, default=100.0)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--pin-memory", choices=["auto", "on", "off"], default="auto")
    parser.add_argument("--loader-prefetch-factor", type=int, default=2)
    parser.add_argument("--amp", choices=["auto", "on", "off"], default="auto")
    parser.add_argument(
        "--grad-accum-steps",
        type=positive_int,
        default=1,
        help="Accumulate gradients across N micro-batches for every candidate.",
    )
    parser.add_argument(
        "--grad-clip-norm",
        type=float,
        default=1.0,
        help="Clip neural gradient norm for every candidate. Set 0 to disable clipping.",
    )
    parser.add_argument(
        "--checkpoint-backup-dir",
        type=Path,
        default=None,
        help="Optional base directory for backing up each candidate's neural checkpoints.",
    )
    parser.add_argument(
        "--checkpoint-metric",
        choices=sorted(CHECKPOINT_METRICS),
        default="val_loss",
        help="Validation metric used inside each candidate run for best checkpoint and early stopping.",
    )
    parser.add_argument("--summary-hidden-dims", type=parse_int_list, default=[96, 128, 192])
    parser.add_argument("--summary-dropouts", type=parse_float_list, default=[0.1, 0.2, 0.35])
    parser.add_argument("--lstm-embed-dims", type=parse_int_list, default=[32, 64])
    parser.add_argument("--lstm-hidden-dims", type=parse_int_list, default=[64, 128])
    parser.add_argument("--lstm-layers", type=parse_int_list, default=[1, 2])
    parser.add_argument("--lstm-dropouts", type=parse_float_list, default=[0.1, 0.2])
    parser.add_argument("--lstm-head-dropouts", type=parse_float_list, default=[0.2, 0.3])
    parser.add_argument("--transformer-embed-dims", type=parse_int_list, default=[32, 64])
    parser.add_argument("--transformer-heads", type=parse_int_list, default=[4])
    parser.add_argument("--transformer-layers", type=parse_int_list, default=[1, 2])
    parser.add_argument("--transformer-ff-dims", type=parse_int_list, default=[128, 256])
    parser.add_argument("--transformer-dropouts", type=parse_float_list, default=[0.1])
    parser.add_argument("--transformer-head-dropouts", type=parse_float_list, default=[0.2])
    parser.add_argument("--selection-metric", type=parse_selection_metric, default="mean_mae")
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Run only the first N candidates. Use 0 for the full grid.",
    )
    args = parser.parse_args()

    candidates = neural_grid(
        models=args.models,
        lrs=args.lrs,
        weight_decays=args.weight_decays,
        batch_sizes=args.batch_sizes,
        huber_deltas=args.huber_deltas,
        summary_hidden_dims=args.summary_hidden_dims,
        summary_dropouts=args.summary_dropouts,
        lstm_embed_dims=args.lstm_embed_dims,
        lstm_hidden_dims=args.lstm_hidden_dims,
        lstm_layers=args.lstm_layers,
        lstm_dropouts=args.lstm_dropouts,
        lstm_head_dropouts=args.lstm_head_dropouts,
        transformer_embed_dims=args.transformer_embed_dims,
        transformer_heads=args.transformer_heads,
        transformer_layers=args.transformer_layers,
        transformer_ff_dims=args.transformer_ff_dims,
        transformer_dropouts=args.transformer_dropouts,
        transformer_head_dropouts=args.transformer_head_dropouts,
    )
    if args.max_candidates > 0:
        candidates = candidates[: args.max_candidates]
    if not candidates:
        raise RuntimeError("No neural sweep candidates were generated.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    detail_rows = []
    for candidate in candidates:
        print(f"Evaluating {candidate['candidate_id']}")
        summary, details = evaluate_candidate(args, candidate)
        summary_rows.append(summary)
        detail_rows.extend(details)

    best = choose_best_candidate(summary_rows, selection_metric=args.selection_metric)
    best_params = {
        **best,
        "selection_metric": args.selection_metric,
        "evaluation": "holdout",
        "candidate_count": len(candidates),
    }

    summary_csv = args.out_dir / "neural_sweep_summary.csv"
    detail_csv = args.out_dir / "neural_sweep_details.csv"
    pd.DataFrame(summary_rows).sort_values(
        [args.selection_metric, "model_size_score"],
        ascending=[selection_sort_ascending(args.selection_metric), True],
    ).to_csv(
        summary_csv,
        index=False,
        encoding="utf-8",
    )
    pd.DataFrame(detail_rows).to_csv(detail_csv, index=False, encoding="utf-8")
    (args.out_dir / "best_params.json").write_text(
        json.dumps(best_params, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (args.out_dir / "sweep_config.json").write_text(
        json.dumps(vars(args), indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    write_html_report(
        summary_rows,
        best,
        args.out_dir / "neural_sweep_report.html",
        selection_metric=args.selection_metric,
    )
    print(json.dumps(best_params, indent=2, ensure_ascii=False))
    print(f"Wrote {summary_csv}, {detail_csv}, and {args.out_dir / 'neural_sweep_report.html'}")


if __name__ == "__main__":
    main()
