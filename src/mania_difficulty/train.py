from __future__ import annotations

import argparse
import csv
import json
import random
from functools import partial
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from mania_difficulty.data.dataset import (
    DEFAULT_TARGET_COLUMNS,
    ManiaDifficultyDataset,
    collate_batch,
)
from mania_difficulty.metrics import regression_report
from mania_difficulty.models.factory import create_model
from mania_difficulty.models.tabular import SUMMARY_FEATURE_NAMES, summarize_sequence
from mania_difficulty.visualize import (
    plot_feature_importance,
    plot_learning_curve,
    plot_prediction_scatter,
    write_run_report,
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def split_indices(size: int, seed: int) -> tuple[list[int], list[int], list[int]]:
    indices = list(range(size))
    random.Random(seed).shuffle(indices)
    train_end = int(size * 0.8)
    val_end = int(size * 0.9)
    return indices[:train_end], indices[train_end:val_end], indices[val_end:]


def target_stats(dataset: ManiaDifficultyDataset, indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    targets = []
    for index in indices:
        targets.append(dataset[index]["targets"])
    array = np.stack(targets).astype("float32")
    mean = array.mean(axis=0)
    std = array.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std


def transform_targets(targets: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (targets - mean) / std


def inverse_transform(array: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return array * std + mean


def parse_max_features(value: str) -> str | float:
    if value in {"sqrt", "log2"}:
        return value
    try:
        parsed = float(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use sqrt, log2, or a float in (0, 1].") from error
    if not 0 < parsed <= 1:
        raise argparse.ArgumentTypeError("Float max_features must be in (0, 1].")
    return parsed


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    return (loss * weights).mean()


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean_t: torch.Tensor,
    target_std_t: torch.Tensor,
    target_mean_np: np.ndarray,
    target_std_np: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray, list[int]]:
    model.eval()
    losses = []
    pred_chunks = []
    actual_chunks = []
    beatmap_ids: list[int] = []
    weights = torch.ones(target_mean_t.shape[0], device=device)

    for batch in loader:
        features = batch.features.to(device)
        lengths = batch.lengths.to(device)
        targets = batch.targets.to(device)
        normalized_targets = transform_targets(targets, target_mean_t, target_std_t)
        normalized_pred = model(features, lengths)
        loss = weighted_huber_loss(normalized_pred, normalized_targets, weights)
        losses.append(float(loss.item()))

        pred = inverse_transform(normalized_pred.cpu().numpy(), target_mean_np, target_std_np)
        actual = targets.cpu().numpy()
        pred_chunks.append(pred)
        actual_chunks.append(actual)
        beatmap_ids.extend(batch.beatmap_ids)

    return (
        float(np.mean(losses)) if losses else 0.0,
        np.concatenate(pred_chunks, axis=0),
        np.concatenate(actual_chunks, axis=0),
        beatmap_ids,
    )


def write_history_header(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()


def append_history(path: Path, epoch: int, train_loss: float, val_loss: float) -> None:
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writerow({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})


def write_predictions(
    path: Path,
    beatmap_ids: list[int],
    actual: np.ndarray,
    pred: np.ndarray,
    target_columns: list[str],
) -> None:
    rows = []
    for index, beatmap_id in enumerate(beatmap_ids):
        row = {"beatmap_id": beatmap_id}
        for target_index, column in enumerate(target_columns):
            row[f"actual_{column}"] = float(actual[index, target_index])
            row[f"pred_{column}"] = float(pred[index, target_index])
            row[f"error_{column}"] = float(pred[index, target_index] - actual[index, target_index])
        rows.append(row)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def repeat_baseline(targets: np.ndarray, baseline_values: np.ndarray) -> np.ndarray:
    return np.repeat(baseline_values.reshape(1, -1), len(targets), axis=0).astype("float32")


def write_human_review(
    path: Path,
    labels_csv: Path,
    predictions_csv: Path,
    target_columns: list[str],
    *,
    top_n: int = 10,
) -> None:
    labels = pd.read_csv(labels_csv)
    predictions = pd.read_csv(predictions_csv)
    merged = predictions.merge(labels, on="beatmap_id", how="left", suffixes=("", "_label"))
    if merged.empty:
        return

    rows = []

    def add_candidates(frame: pd.DataFrame, reason: str) -> None:
        for _, row in frame.iterrows():
            output = {
                "review_reason": reason,
                "beatmap_id": int(row["beatmap_id"]),
                "title": row.get("title", ""),
                "artist": row.get("artist", ""),
                "mapper": row.get("mapper", ""),
                "version": row.get("version", ""),
            }
            for column in target_columns:
                output[f"observed_{column}"] = row.get(f"actual_{column}")
                output[f"predicted_{column}"] = row.get(f"pred_{column}")
                output[f"error_{column}"] = row.get(f"error_{column}")
            rows.append(output)

    if "pred_mean_acc" in merged:
        add_candidates(
            merged.sort_values("pred_mean_acc", ascending=True).head(top_n),
            "model_lowest_predicted_mean_acc",
        )
    if "actual_mean_acc" in merged:
        add_candidates(
            merged.sort_values("actual_mean_acc", ascending=True).head(top_n),
            "observed_lowest_mean_acc",
        )
        add_candidates(
            merged.reindex(merged["error_mean_acc"].abs().sort_values(ascending=False).index).head(top_n),
            "largest_mean_acc_disagreement",
        )

    review = pd.DataFrame(rows).drop_duplicates(["review_reason", "beatmap_id"])
    review.to_csv(path, index=False, encoding="utf-8")


def tabular_arrays(
    dataset: ManiaDifficultyDataset,
    indices: list[int],
    *,
    max_notes: int,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    features = []
    targets = []
    beatmap_ids = []
    for index in indices:
        sample = dataset[index]
        sequence = np.asarray(sample["features"], dtype="float32")[:max_notes]
        features.append(summarize_sequence(sequence))
        targets.append(np.asarray(sample["targets"], dtype="float32"))
        beatmap_ids.append(int(sample["beatmap_id"]))
    return (
        np.stack(features).astype("float32"),
        np.stack(targets).astype("float32"),
        beatmap_ids,
    )


def write_feature_importance(path: Path, importances: np.ndarray) -> None:
    rows = sorted(
        zip(SUMMARY_FEATURE_NAMES, importances, strict=True),
        key=lambda item: float(item[1]),
        reverse=True,
    )
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["feature", "importance"])
        writer.writeheader()
        for feature, importance in rows:
            writer.writerow({"feature": feature, "importance": float(importance)})


def create_tabular_forest_model(args: argparse.Namespace, *, seed: int) -> TransformedTargetRegressor:
    regressor = ExtraTreesRegressor(
        n_estimators=args.forest_trees,
        min_samples_leaf=args.forest_min_samples_leaf,
        max_features=args.forest_max_features,
        random_state=seed,
        n_jobs=args.workers,
    )
    return TransformedTargetRegressor(regressor=regressor, transformer=StandardScaler())


def write_cv_fold_metrics(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "fold",
        "target",
        "mae",
        "r2",
        "baseline_mae",
        "mae_improvement_vs_baseline",
        "mae_improvement_pct",
        "val_size",
    ]
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_tabular_cross_validation(
    args: argparse.Namespace,
    dataset: ManiaDifficultyDataset,
    target_columns: list[str],
    run_dir: Path,
) -> None:
    if args.cv_folds < 2:
        return
    if len(dataset) < args.cv_folds:
        raise RuntimeError(f"Need at least {args.cv_folds} maps for cross-validation.")

    all_indices = list(range(len(dataset)))
    x_all, y_all, beatmap_ids = tabular_arrays(dataset, all_indices, max_notes=args.max_notes)
    oof_pred = np.zeros_like(y_all, dtype="float32")
    oof_baseline = np.zeros_like(y_all, dtype="float32")
    fold_rows: list[dict[str, object]] = []

    splitter = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    for fold_index, (train_idx, val_idx) in enumerate(splitter.split(x_all), start=1):
        model = create_tabular_forest_model(args, seed=args.seed + fold_index)
        model.fit(x_all[train_idx], y_all[train_idx])
        fold_pred = model.predict(x_all[val_idx]).astype("float32")
        fold_baseline = repeat_baseline(y_all[val_idx], y_all[train_idx].mean(axis=0))
        oof_pred[val_idx] = fold_pred
        oof_baseline[val_idx] = fold_baseline

        fold_report = regression_report(
            y_all[val_idx],
            fold_pred,
            target_columns,
            baseline_pred=fold_baseline,
        )
        for target in target_columns:
            fold_rows.append(
                {
                    "fold": fold_index,
                    "target": target,
                    "val_size": len(val_idx),
                    **fold_report[target],
                }
            )

    cv_predictions_csv = run_dir / "cv_predictions.csv"
    write_predictions(cv_predictions_csv, beatmap_ids, y_all, oof_pred, target_columns)
    write_human_review(
        run_dir / "cv_human_review.csv",
        args.labels,
        cv_predictions_csv,
        target_columns,
        top_n=20,
    )

    cv_metrics = regression_report(y_all, oof_pred, target_columns, baseline_pred=oof_baseline)
    cv_metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "cv_oof",
        "cv_folds": args.cv_folds,
        "sample_size": len(dataset),
    }
    (run_dir / "cv_metrics.json").write_text(json.dumps(cv_metrics, indent=2), encoding="utf-8")
    write_cv_fold_metrics(run_dir / "cv_fold_metrics.csv", fold_rows)
    plot_prediction_scatter(cv_predictions_csv, target_columns, run_dir / "cv_prediction_scatter.png")


def train_tabular_forest(
    args: argparse.Namespace,
    dataset: ManiaDifficultyDataset,
    target_columns: list[str],
    run_dir: Path,
    train_indices: list[int],
    val_indices: list[int],
    test_indices: list[int],
) -> Path:
    x_train, y_train, _ = tabular_arrays(dataset, train_indices, max_notes=args.max_notes)
    x_val, y_val, _ = tabular_arrays(dataset, val_indices, max_notes=args.max_notes)
    x_test, y_test, beatmap_ids = tabular_arrays(dataset, test_indices, max_notes=args.max_notes)

    model = create_tabular_forest_model(args, seed=args.seed)
    model.fit(x_train, y_train)

    train_pred = model.predict(x_train)
    val_pred = model.predict(x_val)
    test_pred = model.predict(x_test)
    train_loss = float(np.mean(np.abs(train_pred - y_train)))
    val_loss = float(np.mean(np.abs(val_pred - y_val)))
    test_loss = float(np.mean(np.abs(test_pred - y_test)))

    history_csv = run_dir / "history.csv"
    write_history_header(history_csv)
    append_history(history_csv, 1, train_loss, val_loss)

    checkpoint_path = run_dir / "best_model.joblib"
    joblib.dump(
        {
            "model": model,
            "model_name": args.model,
            "target_columns": target_columns,
            "feature_names": SUMMARY_FEATURE_NAMES,
            "max_notes": args.max_notes,
        },
        checkpoint_path,
    )

    predictions_csv = run_dir / "predictions.csv"
    write_predictions(predictions_csv, beatmap_ids, y_test, test_pred, target_columns)
    write_human_review(run_dir / "human_review.csv", args.labels, predictions_csv, target_columns)

    baseline_pred = repeat_baseline(y_test, y_train.mean(axis=0))
    metrics = regression_report(y_test, test_pred, target_columns, baseline_pred=baseline_pred)
    metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "holdout",
        "best_epoch": 1,
        "best_val_loss": val_loss,
        "test_loss": test_loss,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
        "checkpoint": checkpoint_path.name,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    feature_importance_csv = run_dir / "feature_importance.csv"
    write_feature_importance(feature_importance_csv, model.regressor_.feature_importances_)
    plot_feature_importance(feature_importance_csv, run_dir / "feature_importance.png")
    plot_learning_curve(history_csv, run_dir / "learning_curve.png")
    plot_prediction_scatter(predictions_csv, target_columns, run_dir / "prediction_scatter.png")
    write_tabular_cross_validation(args, dataset, target_columns, run_dir)
    write_run_report(run_dir, target_columns=target_columns, metrics_path=metrics_path)

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Run artifacts written to {run_dir}")
    return run_dir


def train(args: argparse.Namespace) -> Path:
    seed_everything(args.seed)
    target_columns = [column.strip() for column in args.targets.split(",") if column.strip()]
    run_dir = Path("outputs/runs") / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    dataset = ManiaDifficultyDataset(args.labels, args.sequences, target_columns=target_columns)
    if len(dataset) < 10:
        raise RuntimeError("Need at least 10 maps for train/val/test split.")

    train_indices, val_indices, test_indices = split_indices(len(dataset), args.seed)
    if args.model == "tabular_forest":
        return train_tabular_forest(
            args,
            dataset,
            target_columns,
            run_dir,
            train_indices,
            val_indices,
            test_indices,
        )

    target_mean_np, target_std_np = target_stats(dataset, train_indices)

    collate = partial(collate_batch, max_notes=args.max_notes)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
    )

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = create_model(args.model, output_dim=len(target_columns)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    target_mean_t = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    loss_weights = torch.tensor(args.loss_weights, dtype=torch.float32, device=device)
    if loss_weights.numel() != len(target_columns):
        loss_weights = torch.ones(len(target_columns), dtype=torch.float32, device=device)

    history_csv = run_dir / "history.csv"
    write_history_header(history_csv)

    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    checkpoint_path = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            features = batch.features.to(device)
            lengths = batch.lengths.to(device)
            targets = batch.targets.to(device)
            normalized_targets = transform_targets(targets, target_mean_t, target_std_t)

            optimizer.zero_grad(set_to_none=True)
            pred = model(features, lengths)
            loss = weighted_huber_loss(pred, normalized_targets, loss_weights)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_losses.append(float(loss.item()))

        scheduler.step()
        train_loss = float(np.mean(train_losses))
        val_loss, _, _, _ = evaluate_loader(
            model,
            val_loader,
            device,
            target_mean_t,
            target_std_t,
            target_mean_np,
            target_std_np,
        )
        append_history(history_csv, epoch, train_loss, val_loss)
        print(f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            patience_left = args.patience
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "model_name": args.model,
                    "model_config": model.config,
                    "target_columns": target_columns,
                    "target_mean": target_mean_np.tolist(),
                    "target_std": target_std_np.tolist(),
                    "max_notes": args.max_notes,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}; best epoch was {best_epoch}.")
                break

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, pred, actual, beatmap_ids = evaluate_loader(
        model,
        test_loader,
        device,
        target_mean_t,
        target_std_t,
        target_mean_np,
        target_std_np,
    )
    predictions_csv = run_dir / "predictions.csv"
    write_predictions(predictions_csv, beatmap_ids, actual, pred, target_columns)
    write_human_review(run_dir / "human_review.csv", args.labels, predictions_csv, target_columns)

    baseline_pred = repeat_baseline(actual, target_mean_np)
    metrics = regression_report(actual, pred, target_columns, baseline_pred=baseline_pred)
    metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "holdout",
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "test_size": len(test_indices),
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    plot_learning_curve(history_csv, run_dir / "learning_curve.png")
    plot_prediction_scatter(predictions_csv, target_columns, run_dir / "prediction_scatter.png")
    write_run_report(run_dir, target_columns=target_columns, metrics_path=metrics_path)

    (run_dir / "config.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    print(f"Run artifacts written to {run_dir}")
    return run_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the LSTM mania difficulty baseline.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--run-name", default="lstm_baseline")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-notes", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--model",
        choices=["summary", "lstm", "tabular_forest"],
        default="lstm",
        help="Use tabular_forest for small-data baseline, summary for fast neural CPU pilots, lstm for sequence GPU runs.",
    )
    parser.add_argument(
        "--loss-weights",
        type=float,
        nargs="*",
        default=[1.0, 0.5, 0.5],
        help="Weights matching --targets. Falls back to all ones if the count differs.",
    )
    parser.add_argument("--forest-trees", type=int, default=400)
    parser.add_argument("--forest-min-samples-leaf", type=int, default=2)
    parser.add_argument("--forest-max-features", type=parse_max_features, default="sqrt")
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="For tabular_forest, also write K-fold out-of-fold metrics when set to 2 or higher.",
    )
    parser.add_argument("--workers", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
