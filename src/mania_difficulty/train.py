from __future__ import annotations

import argparse
import csv
import json
import random
import time
from contextlib import nullcontext
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
from sklearn.preprocessing import StandardScaler

from mania_difficulty.data.dataset import (
    DEFAULT_TARGET_COLUMNS,
    ManiaDifficultyDataset,
    collate_batch,
)
from mania_difficulty.error_analysis import write_error_slices
from mania_difficulty.human_judgments import write_pair_judgment_template
from mania_difficulty.metrics import regression_report
from mania_difficulty.models.factory import create_model
from mania_difficulty.models.tabular import feature_names_for_set, summarize_sequence
from mania_difficulty.visualize import (
    plot_feature_importance,
    plot_learning_curve,
    plot_prediction_scatter,
    write_run_report,
)


HISTORY_COLUMNS = [
    "epoch",
    "train_loss",
    "val_loss",
    "epoch_seconds",
    "lr",
    "cuda_max_memory_mb",
]


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


def split_indices_by_group(groups: list[str], seed: int) -> tuple[list[int], list[int], list[int]]:
    unique_groups = sorted(set(groups))
    if len(unique_groups) < 3:
        raise RuntimeError("Need at least 3 unique groups for grouped train/val/test split.")

    random.Random(seed).shuffle(unique_groups)
    group_count = len(unique_groups)
    test_count = max(1, round(group_count * 0.1))
    val_count = max(1, round(group_count * 0.1))
    if test_count + val_count >= group_count:
        test_count = 1
        val_count = 1
    train_count = group_count - val_count - test_count

    train_groups = set(unique_groups[:train_count])
    val_groups = set(unique_groups[train_count : train_count + val_count])
    test_groups = set(unique_groups[train_count + val_count :])
    return (
        [index for index, group in enumerate(groups) if group in train_groups],
        [index for index, group in enumerate(groups) if group in val_groups],
        [index for index, group in enumerate(groups) if group in test_groups],
    )


def cross_validation_splits(
    size: int,
    *,
    groups: list[str] | None,
    folds: int,
    seed: int,
) -> list[tuple[list[int], list[int]]]:
    if folds < 2:
        return []
    indices = list(range(size))
    if groups and len(set(groups)) >= folds:
        unique_groups = sorted(set(groups))
        random.Random(seed).shuffle(unique_groups)
        group_sizes = {group: groups.count(group) for group in unique_groups}
        unique_groups.sort(key=lambda group: group_sizes[group], reverse=True)
        fold_group_sets = [set() for _ in range(folds)]
        fold_sizes = [0 for _ in range(folds)]
        for group in unique_groups:
            fold_index = min(range(folds), key=lambda index: fold_sizes[index])
            fold_group_sets[fold_index].add(group)
            fold_sizes[fold_index] += group_sizes[group]
        return [
            (
                [index for index in indices if groups[index] not in val_groups],
                [index for index in indices if groups[index] in val_groups],
            )
            for val_groups in fold_group_sets
        ]

    random.Random(seed).shuffle(indices)
    shuffled_splits = np.array_split(np.asarray(indices), folds)
    return [
        (
            [index for index in indices if index not in set(val_indices.tolist())],
            val_indices.tolist(),
        )
        for val_indices in shuffled_splits
    ]


def dataset_groups(dataset: ManiaDifficultyDataset, group_column: str) -> list[str] | None:
    if not group_column or group_column not in dataset.labels.columns:
        return None

    groups = []
    for row_index, row in dataset.labels.reset_index(drop=True).iterrows():
        value = row.get(group_column)
        if pd.isna(value) or value == "":
            value = f"__row_{row_index}"
        groups.append(str(value))
    return groups if len(set(groups)) >= 3 else None


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


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a positive integer.") from error
    if parsed < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer.")
    return parsed


def gradient_accumulation_steps(args: argparse.Namespace) -> int:
    return max(1, int(getattr(args, "grad_accum_steps", 1)))


def latest_checkpoint_path(run_dir: Path | str) -> Path:
    return Path(run_dir) / "last_checkpoint.pt"


def model_config_from_args(args: argparse.Namespace) -> dict[str, object]:
    if args.model == "lstm":
        return {
            "embed_dim": args.lstm_embed_dim,
            "hidden_dim": args.lstm_hidden_dim,
            "num_layers": args.lstm_layers,
            "dropout": args.lstm_dropout,
            "head_dropout": args.lstm_head_dropout,
        }
    if args.model == "summary":
        return {
            "hidden_dim": args.summary_hidden_dim,
            "dropout": args.summary_dropout,
        }
    if args.model == "transformer":
        return {
            "embed_dim": args.transformer_embed_dim,
            "num_heads": args.transformer_heads,
            "num_layers": args.transformer_layers,
            "ff_dim": args.transformer_ff_dim,
            "dropout": args.transformer_dropout,
            "head_dropout": args.transformer_head_dropout,
            "max_positions": args.max_notes,
        }
    return {}


def weighted_huber_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = nn.functional.huber_loss(pred, target, reduction="none", delta=1.0)
    return (loss * weights).mean()


def dataloader_options(args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    worker_count = max(0, int(getattr(args, "loader_workers", 0)))
    pin_memory_arg = getattr(args, "pin_memory", "auto")
    if pin_memory_arg == "auto":
        pin_memory = device.type == "cuda"
    else:
        pin_memory = pin_memory_arg == "on"

    options: dict[str, object] = {
        "num_workers": worker_count,
        "pin_memory": pin_memory,
    }
    if worker_count > 0:
        options["persistent_workers"] = True
        options["prefetch_factor"] = max(1, int(getattr(args, "loader_prefetch_factor", 2)))
    return options


def mixed_precision_enabled(args: argparse.Namespace, device: torch.device) -> bool:
    amp_mode = getattr(args, "amp", "auto")
    if amp_mode == "off":
        return False
    if amp_mode == "auto":
        return device.type == "cuda"
    if amp_mode == "on":
        if device.type != "cuda":
            raise RuntimeError("--amp on requires a CUDA device. Use --amp auto or --amp off on CPU.")
        return True
    raise ValueError(f"Unknown amp mode: {amp_mode}")


def autocast_context(device: torch.device, enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.float16, enabled=enabled)


def make_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


@torch.no_grad()
def evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    target_mean_t: torch.Tensor,
    target_std_t: torch.Tensor,
    target_mean_np: np.ndarray,
    target_std_np: np.ndarray,
    amp_enabled: bool = False,
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
        with autocast_context(device, amp_enabled):
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
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()


def ensure_history_columns(path: Path) -> None:
    if not path.exists():
        write_history_header(path)
        return

    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames or []
        if fieldnames == HISTORY_COLUMNS:
            return
        rows = list(reader)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in HISTORY_COLUMNS})


def append_history(
    path: Path,
    epoch: int,
    train_loss: float,
    val_loss: float,
    *,
    epoch_seconds: float | None = None,
    lr: float | None = None,
    cuda_max_memory_mb: float | None = None,
) -> None:
    ensure_history_columns(path)
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=HISTORY_COLUMNS)
        writer.writerow(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "epoch_seconds": "" if epoch_seconds is None else epoch_seconds,
                "lr": "" if lr is None else lr,
                "cuda_max_memory_mb": "" if cuda_max_memory_mb is None else cuda_max_memory_mb,
            }
        )


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


def clean_review_value(value: object) -> object:
    if pd.isna(value):
        return ""
    return value


def pair_review_fields(target_column: str) -> list[str]:
    return [
        "review_reason",
        "model_harder_beatmap_id",
        "model_harder_title",
        "model_harder_artist",
        "model_harder_mapper",
        "model_harder_version",
        f"model_harder_pred_{target_column}",
        f"model_harder_observed_{target_column}",
        "observed_harder_beatmap_id",
        "observed_harder_title",
        "observed_harder_artist",
        "observed_harder_mapper",
        "observed_harder_version",
        f"observed_harder_pred_{target_column}",
        f"observed_harder_observed_{target_column}",
        "predicted_acc_gap_model_minus_observed",
        "observed_acc_gap_model_minus_observed",
        "disagreement_strength",
    ]


def pair_review_side(row: pd.Series, prefix: str, target_column: str) -> dict[str, object]:
    return {
        f"{prefix}_beatmap_id": int(row["beatmap_id"]),
        f"{prefix}_title": clean_review_value(row.get("title", "")),
        f"{prefix}_artist": clean_review_value(row.get("artist", "")),
        f"{prefix}_mapper": clean_review_value(row.get("mapper", "")),
        f"{prefix}_version": clean_review_value(row.get("version", "")),
        f"{prefix}_pred_{target_column}": float(row[f"pred_{target_column}"]),
        f"{prefix}_observed_{target_column}": float(row[f"actual_{target_column}"]),
    }


def write_pairwise_review(
    path: Path,
    labels_csv: Path,
    predictions_csv: Path,
    *,
    target_column: str = "mean_acc",
    top_n: int = 20,
    min_abs_pred_gap: float = 0.0,
    min_abs_actual_gap: float = 0.0,
) -> None:
    actual_column = f"actual_{target_column}"
    pred_column = f"pred_{target_column}"
    predictions = pd.read_csv(predictions_csv)
    fieldnames = pair_review_fields(target_column)
    if actual_column not in predictions.columns or pred_column not in predictions.columns:
        pd.DataFrame(columns=fieldnames).to_csv(path, index=False, encoding="utf-8")
        return

    labels = pd.read_csv(labels_csv)
    merged = predictions.merge(labels, on="beatmap_id", how="left", suffixes=("", "_label"))
    rows = []
    for left_index in range(len(merged)):
        for right_index in range(left_index + 1, len(merged)):
            left = merged.iloc[left_index]
            right = merged.iloc[right_index]
            pred_gap = float(left[pred_column] - right[pred_column])
            actual_gap = float(left[actual_column] - right[actual_column])
            if not np.isfinite(pred_gap) or not np.isfinite(actual_gap):
                continue
            if abs(pred_gap) < min_abs_pred_gap or abs(actual_gap) < min_abs_actual_gap:
                continue
            if pred_gap == 0.0 or actual_gap == 0.0 or np.sign(pred_gap) == np.sign(actual_gap):
                continue

            if pred_gap < 0:
                model_harder = left
                observed_harder = right
            else:
                model_harder = right
                observed_harder = left

            model_pred_gap = float(model_harder[pred_column] - observed_harder[pred_column])
            model_actual_gap = float(model_harder[actual_column] - observed_harder[actual_column])
            rows.append(
                {
                    "review_reason": "pairwise_rank_disagreement",
                    **pair_review_side(model_harder, "model_harder", target_column),
                    **pair_review_side(observed_harder, "observed_harder", target_column),
                    "predicted_acc_gap_model_minus_observed": model_pred_gap,
                    "observed_acc_gap_model_minus_observed": model_actual_gap,
                    "disagreement_strength": abs(model_pred_gap) + abs(model_actual_gap),
                }
            )

    frame = pd.DataFrame(rows, columns=fieldnames)
    if not frame.empty:
        frame = frame.sort_values("disagreement_strength", ascending=False).head(top_n)
    frame.to_csv(path, index=False, encoding="utf-8")


def tabular_arrays(
    dataset: ManiaDifficultyDataset,
    indices: list[int],
    *,
    max_notes: int,
    feature_set: str,
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    features = []
    targets = []
    beatmap_ids = []
    for index in indices:
        sample = dataset[index]
        sequence = np.asarray(sample["features"], dtype="float32")[:max_notes]
        features.append(summarize_sequence(sequence, feature_set=feature_set))
        targets.append(np.asarray(sample["targets"], dtype="float32"))
        beatmap_ids.append(int(sample["beatmap_id"]))
    return (
        np.stack(features).astype("float32"),
        np.stack(targets).astype("float32"),
        beatmap_ids,
    )


def write_feature_importance(path: Path, importances: np.ndarray, feature_names: list[str]) -> None:
    rows = sorted(
        zip(feature_names, importances, strict=True),
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
        "spearman",
        "pairwise_order_accuracy",
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
    groups: list[str] | None,
) -> None:
    if args.cv_folds < 2:
        return
    if len(dataset) < args.cv_folds:
        raise RuntimeError(f"Need at least {args.cv_folds} maps for cross-validation.")

    all_indices = list(range(len(dataset)))
    x_all, y_all, beatmap_ids = tabular_arrays(
        dataset,
        all_indices,
        max_notes=args.max_notes,
        feature_set=args.feature_set,
    )
    oof_pred = np.zeros_like(y_all, dtype="float32")
    oof_baseline = np.zeros_like(y_all, dtype="float32")
    fold_rows: list[dict[str, object]] = []

    split_strategy = (
        f"group:{args.group_column}"
        if groups and len(set(groups)) >= args.cv_folds
        else "random_map"
    )
    for fold_index, (train_idx, val_idx) in enumerate(
        cross_validation_splits(len(dataset), groups=groups, folds=args.cv_folds, seed=args.seed),
        start=1,
    ):
        train_idx_array = np.asarray(train_idx, dtype=int)
        val_idx_array = np.asarray(val_idx, dtype=int)
        model = create_tabular_forest_model(args, seed=args.seed + fold_index)
        model.fit(x_all[train_idx_array], y_all[train_idx_array])
        fold_pred = model.predict(x_all[val_idx_array]).astype("float32")
        fold_baseline = repeat_baseline(y_all[val_idx_array], y_all[train_idx_array].mean(axis=0))
        oof_pred[val_idx_array] = fold_pred
        oof_baseline[val_idx_array] = fold_baseline

        fold_report = regression_report(
            y_all[val_idx_array],
            fold_pred,
            target_columns,
            baseline_pred=fold_baseline,
        )
        for target in target_columns:
            fold_rows.append(
                {
                    "fold": fold_index,
                    "target": target,
                    "val_size": len(val_idx_array),
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
    write_pairwise_review(
        run_dir / "cv_human_pair_review.csv",
        args.labels,
        cv_predictions_csv,
        target_column="mean_acc",
        top_n=30,
    )
    write_pair_judgment_template(
        run_dir / "cv_human_pair_judgment_template.csv",
        run_dir / "cv_human_pair_review.csv",
    )
    write_error_slices(
        run_dir / "cv_error_slices.csv",
        args.labels,
        cv_predictions_csv,
        target_column="mean_acc",
    )

    cv_metrics = regression_report(y_all, oof_pred, target_columns, baseline_pred=oof_baseline)
    cv_metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "cv_oof",
        "feature_set": args.feature_set,
        "cv_folds": args.cv_folds,
        "split_strategy": split_strategy,
        "group_column": args.group_column if split_strategy.startswith("group:") else "",
        "group_count": len(set(groups)) if groups else 0,
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
    split_metadata: dict[str, object],
    groups: list[str] | None,
) -> Path:
    feature_names = feature_names_for_set(args.feature_set)
    x_train, y_train, _ = tabular_arrays(
        dataset,
        train_indices,
        max_notes=args.max_notes,
        feature_set=args.feature_set,
    )
    x_val, y_val, _ = tabular_arrays(
        dataset,
        val_indices,
        max_notes=args.max_notes,
        feature_set=args.feature_set,
    )
    x_test, y_test, beatmap_ids = tabular_arrays(
        dataset,
        test_indices,
        max_notes=args.max_notes,
        feature_set=args.feature_set,
    )

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
            "feature_names": feature_names,
            "feature_set": args.feature_set,
            "max_notes": args.max_notes,
        },
        checkpoint_path,
    )

    predictions_csv = run_dir / "predictions.csv"
    write_predictions(predictions_csv, beatmap_ids, y_test, test_pred, target_columns)
    write_human_review(run_dir / "human_review.csv", args.labels, predictions_csv, target_columns)
    write_pairwise_review(run_dir / "human_pair_review.csv", args.labels, predictions_csv)
    write_pair_judgment_template(
        run_dir / "human_pair_judgment_template.csv",
        run_dir / "human_pair_review.csv",
    )
    write_error_slices(run_dir / "error_slices.csv", args.labels, predictions_csv, target_column="mean_acc")

    baseline_pred = repeat_baseline(y_test, y_train.mean(axis=0))
    metrics = regression_report(y_test, test_pred, target_columns, baseline_pred=baseline_pred)
    metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "holdout",
        "feature_set": args.feature_set,
        **split_metadata,
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
    write_feature_importance(feature_importance_csv, model.regressor_.feature_importances_, feature_names)
    plot_feature_importance(feature_importance_csv, run_dir / "feature_importance.png")
    plot_learning_curve(history_csv, run_dir / "learning_curve.png")
    plot_prediction_scatter(predictions_csv, target_columns, run_dir / "prediction_scatter.png")
    write_tabular_cross_validation(args, dataset, target_columns, run_dir, groups)
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

    groups = dataset_groups(dataset, args.group_column)
    if groups:
        train_indices, val_indices, test_indices = split_indices_by_group(groups, args.seed)
        split_metadata: dict[str, object] = {
            "split_strategy": f"group:{args.group_column}",
            "group_column": args.group_column,
            "group_count": len(set(groups)),
        }
    else:
        train_indices, val_indices, test_indices = split_indices(len(dataset), args.seed)
        split_metadata = {
            "split_strategy": "random_map",
            "group_column": "",
            "group_count": 0,
        }
    if args.model == "tabular_forest":
        return train_tabular_forest(
            args,
            dataset,
            target_columns,
            run_dir,
            train_indices,
            val_indices,
            test_indices,
            split_metadata,
            groups,
        )

    target_mean_np, target_std_np = target_stats(dataset, train_indices)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    loader_kwargs = dataloader_options(args, device)
    amp_active = mixed_precision_enabled(args, device)
    grad_accum_steps = gradient_accumulation_steps(args)
    effective_batch_size = args.batch_size * grad_accum_steps

    collate = partial(collate_batch, max_notes=args.max_notes)
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        **loader_kwargs,
    )
    val_loader = DataLoader(
        Subset(dataset, val_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )
    test_loader = DataLoader(
        Subset(dataset, test_indices),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate,
        **loader_kwargs,
    )

    model = create_model(args.model, output_dim=len(target_columns), config=model_config_from_args(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))
    scaler = make_grad_scaler(amp_active)
    target_mean_t = torch.tensor(target_mean_np, dtype=torch.float32, device=device)
    target_std_t = torch.tensor(target_std_np, dtype=torch.float32, device=device)
    loss_weights = torch.tensor(args.loss_weights, dtype=torch.float32, device=device)
    if loss_weights.numel() != len(target_columns):
        loss_weights = torch.ones(len(target_columns), dtype=torch.float32, device=device)

    history_csv = run_dir / "history.csv"
    best_val_loss = float("inf")
    best_epoch = 0
    patience_left = args.patience
    checkpoint_path = run_dir / "best_model.pt"
    last_checkpoint_path = latest_checkpoint_path(run_dir)
    resume_requested = bool(getattr(args, "resume", False))
    resumed_from_epoch = 0
    start_epoch = 1

    if resume_requested and last_checkpoint_path.exists():
        resume_checkpoint = torch.load(last_checkpoint_path, map_location=device)
        model.load_state_dict(resume_checkpoint["model_state_dict"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(resume_checkpoint["scheduler_state_dict"])
        scaler_state = resume_checkpoint.get("scaler_state_dict")
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        best_val_loss = float(resume_checkpoint.get("best_val_loss", best_val_loss))
        best_epoch = int(resume_checkpoint.get("best_epoch", best_epoch))
        patience_left = int(resume_checkpoint.get("patience_left", patience_left))
        resumed_from_epoch = int(resume_checkpoint.get("epoch", 0))
        start_epoch = resumed_from_epoch + 1
        if not history_csv.exists():
            write_history_header(history_csv)
        print(f"Resuming from epoch {resumed_from_epoch}; next epoch is {start_epoch}.")
    else:
        write_history_header(history_csv)
        if resume_requested:
            print(f"--resume was set but {last_checkpoint_path} was not found; starting fresh.")

    for epoch in range(start_epoch, args.epochs + 1):
        epoch_start_time = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_lr = float(optimizer.param_groups[0]["lr"])
        model.train()
        train_losses = []
        optimizer.zero_grad(set_to_none=True)
        train_batch_count = len(train_loader)
        for batch_index, batch in enumerate(train_loader, start=1):
            features = batch.features.to(device)
            lengths = batch.lengths.to(device)
            targets = batch.targets.to(device)
            normalized_targets = transform_targets(targets, target_mean_t, target_std_t)

            with autocast_context(device, amp_active):
                pred = model(features, lengths)
                loss = weighted_huber_loss(pred, normalized_targets, loss_weights)
            group_start = ((batch_index - 1) // grad_accum_steps) * grad_accum_steps + 1
            accumulation_divisor = min(grad_accum_steps, train_batch_count - group_start + 1)
            scaler.scale(loss / accumulation_divisor).backward()
            should_step = batch_index % grad_accum_steps == 0 or batch_index == len(train_loader)
            if should_step:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
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
            amp_active,
        )
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            cuda_max_memory_mb = torch.cuda.max_memory_allocated(device) / (1024 * 1024)
        else:
            cuda_max_memory_mb = None
        epoch_seconds = time.perf_counter() - epoch_start_time
        append_history(
            history_csv,
            epoch,
            train_loss,
            val_loss,
            epoch_seconds=epoch_seconds,
            lr=epoch_lr,
            cuda_max_memory_mb=cuda_max_memory_mb,
        )
        memory_text = (
            f" cuda_peak_mb={cuda_max_memory_mb:.1f}"
            if cuda_max_memory_mb is not None
            else ""
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.5f} val_loss={val_loss:.5f} "
            f"epoch_seconds={epoch_seconds:.2f} lr={epoch_lr:.6g}{memory_text}"
        )

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
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "model_name": args.model,
                "model_config": model.config,
                "target_columns": target_columns,
                "target_mean": target_mean_np.tolist(),
                "target_std": target_std_np.tolist(),
                "max_notes": args.max_notes,
                "best_epoch": best_epoch,
                "best_val_loss": best_val_loss,
                "patience_left": patience_left,
            },
            last_checkpoint_path,
        )
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
        amp_active,
    )
    predictions_csv = run_dir / "predictions.csv"
    write_predictions(predictions_csv, beatmap_ids, actual, pred, target_columns)
    write_human_review(run_dir / "human_review.csv", args.labels, predictions_csv, target_columns)
    write_pairwise_review(run_dir / "human_pair_review.csv", args.labels, predictions_csv)
    write_pair_judgment_template(
        run_dir / "human_pair_judgment_template.csv",
        run_dir / "human_pair_review.csv",
    )
    write_error_slices(run_dir / "error_slices.csv", args.labels, predictions_csv, target_column="mean_acc")

    baseline_pred = repeat_baseline(actual, target_mean_np)
    metrics = regression_report(actual, pred, target_columns, baseline_pred=baseline_pred)
    metrics["_run"] = {
        "model_name": args.model,
        "seed": args.seed,
        "evaluation": "holdout",
        "model_config": model.config,
        **split_metadata,
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "test_loss": test_loss,
        "amp": getattr(args, "amp", "auto"),
        "amp_enabled": amp_active,
        "batch_size": args.batch_size,
        "grad_accum_steps": grad_accum_steps,
        "effective_batch_size": effective_batch_size,
        "resume": resume_requested,
        "resumed_from_epoch": resumed_from_epoch,
        "last_checkpoint": last_checkpoint_path.name if last_checkpoint_path.exists() else "",
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
    parser = argparse.ArgumentParser(description="Train an osu!mania difficulty model.")
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--run-name", default="lstm_baseline")
    parser.add_argument("--targets", default=",".join(DEFAULT_TARGET_COLUMNS))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--grad-accum-steps",
        type=positive_int,
        default=1,
        help="Accumulate gradients across N micro-batches before stepping. Effective batch size is batch-size * N.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="For neural models, resume from last_checkpoint.pt in the run directory when it exists.",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--max-notes", type=int, default=3000)
    parser.add_argument(
        "--group-column",
        default="beatmapset_id",
        help="Group-aware splitting column. Defaults to beatmapset_id to avoid same-set leakage.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="")
    parser.add_argument(
        "--model",
        choices=["summary", "lstm", "transformer", "tabular_forest"],
        default="lstm",
        help="Use tabular_forest for small-data baseline, summary for fast neural CPU pilots, lstm/transformer for GPU sequence runs.",
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
        "--feature-set",
        choices=["core", "burst"],
        default="core",
        help="Tabular feature set. core preserves the stable baseline; burst adds density/jack/chord-burst features.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=0,
        help="For tabular_forest, also write K-fold out-of-fold metrics when set to 2 or higher.",
    )
    parser.add_argument("--workers", type=int, default=-1)
    parser.add_argument(
        "--loader-workers",
        type=int,
        default=0,
        help="PyTorch DataLoader workers for summary/lstm. Keep 0 on Windows; use 2+ in Colab.",
    )
    parser.add_argument(
        "--pin-memory",
        choices=["auto", "on", "off"],
        default="auto",
        help="Pin DataLoader memory for GPU transfer. auto enables it on CUDA.",
    )
    parser.add_argument(
        "--loader-prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor when --loader-workers is greater than 0.",
    )
    parser.add_argument(
        "--amp",
        choices=["auto", "on", "off"],
        default="auto",
        help="Mixed precision for neural models. auto enables it on CUDA and disables it on CPU.",
    )
    parser.add_argument("--lstm-embed-dim", type=int, default=64)
    parser.add_argument("--lstm-hidden-dim", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=2)
    parser.add_argument("--lstm-dropout", type=float, default=0.2)
    parser.add_argument("--lstm-head-dropout", type=float, default=0.3)
    parser.add_argument("--summary-hidden-dim", type=int, default=128)
    parser.add_argument("--summary-dropout", type=float, default=0.2)
    parser.add_argument("--transformer-embed-dim", type=int, default=64)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-ff-dim", type=int, default=256)
    parser.add_argument("--transformer-dropout", type=float, default=0.1)
    parser.add_argument("--transformer-head-dropout", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
