from __future__ import annotations

import argparse
import html
import json
import os
from functools import partial
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from mania_difficulty.data.dataset import ManiaDifficultyDataset, collate_batch
from mania_difficulty.models.factory import create_model
from mania_difficulty.models.tabular import summarize_sequence


def scaled_embeddings(embeddings: np.ndarray) -> np.ndarray:
    mean = embeddings.mean(axis=0, keepdims=True)
    std = embeddings.std(axis=0, keepdims=True)
    return (embeddings - mean) / np.where(std > 0, std, 1.0)


def project_embedding_matrix(
    embeddings: np.ndarray,
    *,
    method: str = "pca",
    seed: int = 42,
    perplexity: float = 30.0,
) -> np.ndarray:
    if embeddings.ndim != 2:
        raise ValueError("embeddings must be a 2D array")
    if len(embeddings) == 0:
        raise ValueError("cannot project an empty embedding matrix")
    if len(embeddings) == 1:
        return np.zeros((1, 2), dtype="float32")

    scaled = scaled_embeddings(embeddings.astype("float32"))
    if method == "tsne" and len(scaled) >= 3:
        actual_perplexity = min(float(perplexity), float(len(scaled) - 1))
        coords = TSNE(
            n_components=2,
            perplexity=max(1.0, actual_perplexity),
            init="random",
            learning_rate="auto",
            random_state=seed,
        ).fit_transform(scaled)
    else:
        component_count = min(2, scaled.shape[0], scaled.shape[1])
        coords = PCA(n_components=component_count, random_state=seed).fit_transform(scaled)

    if coords.shape[1] == 1:
        coords = np.column_stack([coords[:, 0], np.zeros(len(coords), dtype=coords.dtype)])
    return coords.astype("float32")


def label_rows_by_beatmap(dataset: ManiaDifficultyDataset) -> dict[int, dict[str, Any]]:
    rows = {}
    for row in dataset.labels.to_dict(orient="records"):
        rows[int(row["beatmap_id"])] = row
    return rows


def collect_neural_embeddings(
    *,
    checkpoint_path: Path,
    labels_csv: Path,
    sequences_dir: Path,
    batch_size: int,
    device_name: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    target_columns = list(checkpoint["target_columns"])
    dataset = ManiaDifficultyDataset(labels_csv, sequences_dir, target_columns=target_columns)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, max_notes=int(checkpoint.get("max_notes", 3000))),
    )
    model = create_model(
        checkpoint.get("model_name", "lstm"),
        output_dim=len(target_columns),
        config=checkpoint["model_config"],
    ).to(device)
    if not hasattr(model, "encode"):
        raise RuntimeError(f"{checkpoint.get('model_name', 'model')} does not expose encode()")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    target_mean = np.asarray(checkpoint["target_mean"], dtype="float32")
    target_std = np.asarray(checkpoint["target_std"], dtype="float32")
    label_rows = label_rows_by_beatmap(dataset)
    rows: list[dict[str, Any]] = []
    embeddings: list[np.ndarray] = []

    with torch.no_grad():
        for batch in loader:
            features = batch.features.to(device)
            lengths = batch.lengths.to(device)
            batch_embeddings = model.encode(features, lengths).detach().cpu().numpy()
            pred_norm = model(features, lengths).detach().cpu().numpy()
            pred = pred_norm * target_std + target_mean
            embeddings.append(batch_embeddings)
            for row_index, beatmap_id in enumerate(batch.beatmap_ids):
                row = dict(label_rows[int(beatmap_id)])
                row["model_name"] = checkpoint.get("model_name", "lstm")
                row["embedding_source"] = "neural_encode"
                for target_index, target in enumerate(target_columns):
                    row[f"pred_{target}"] = float(pred[row_index, target_index])
                rows.append(row)

    return pd.DataFrame(rows), np.vstack(embeddings).astype("float32")


def collect_tabular_embeddings(
    *,
    checkpoint_path: Path,
    labels_csv: Path,
    sequences_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray]:
    checkpoint = joblib.load(checkpoint_path)
    target_columns = list(checkpoint["target_columns"])
    dataset = ManiaDifficultyDataset(labels_csv, sequences_dir, target_columns=target_columns)
    feature_set = checkpoint.get("feature_set", "core")
    max_notes = int(checkpoint.get("max_notes", 3000))
    label_rows = label_rows_by_beatmap(dataset)
    rows: list[dict[str, Any]] = []
    embeddings = []

    for sample in dataset:
        beatmap_id = int(sample["beatmap_id"])
        features = np.asarray(sample["features"], dtype="float32")[:max_notes]
        summary = summarize_sequence(features, feature_set=feature_set)
        embeddings.append(summary)
        row = dict(label_rows[beatmap_id])
        row["model_name"] = checkpoint.get("model_name", "tabular_forest")
        row["feature_set"] = feature_set
        row["embedding_source"] = "tabular_summary"
        rows.append(row)

    embedding_array = np.vstack(embeddings).astype("float32")
    pred = np.asarray(checkpoint["model"].predict(embedding_array))
    if pred.ndim == 1:
        pred = pred.reshape(-1, 1)
    for row_index, row in enumerate(rows):
        for target_index, target in enumerate(target_columns):
            row[f"pred_{target}"] = float(pred[row_index, target_index])
    return pd.DataFrame(rows), embedding_array


def write_projection_plot(frame: pd.DataFrame, out_png: Path, *, color_target: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    color_values = None
    if color_target in frame.columns and pd.api.types.is_numeric_dtype(frame[color_target]):
        color_values = frame[color_target]
    scatter = ax.scatter(
        frame["projection_x"],
        frame["projection_y"],
        c=color_values,
        cmap="viridis" if color_values is not None else None,
        s=42,
        alpha=0.85,
        edgecolors="#1f2933",
        linewidths=0.4,
    )
    if color_values is not None:
        fig.colorbar(scatter, ax=ax, label=color_target)
    ax.set_title("Embedding Projection")
    ax.set_xlabel("projection_x")
    ax.set_ylabel("projection_y")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def write_projection_html(
    out_html: Path,
    *,
    out_csv: Path,
    out_png: Path,
    frame: pd.DataFrame,
    method: str,
    color_target: str,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    summary_rows = [
        ("Rows", len(frame)),
        ("Method", method),
        ("Color target", color_target if color_target in frame.columns else "embedding_norm"),
    ]
    summary_html = "".join(
        f"<tr><th>{html.escape(str(label))}</th><td>{html.escape(str(value))}</td></tr>"
        for label, value in summary_rows
    )
    preview_columns = [
        column
        for column in [
            "beatmap_id",
            "title",
            "artist",
            "mapper",
            "mean_acc",
            "acc_std",
            "skill_gap",
            "pred_mean_acc",
            "projection_x",
            "projection_y",
            "embedding_norm",
        ]
        if column in frame.columns
    ]
    preview = frame[preview_columns].head(20).to_html(index=False, float_format=lambda value: f"{value:.6f}")
    csv_href = os.path.relpath(out_csv, start=out_html.parent).replace("\\", "/")
    png_href = os.path.relpath(out_png, start=out_html.parent).replace("\\", "/")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>embedding projection</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>Embedding Projection</h1>
  <table><tbody>{summary_html}</tbody></table>
  <p><a href="{html.escape(csv_href)}">embedding_projection.csv</a></p>
  <p><img src="{html.escape(png_href)}" alt="Embedding projection"></p>
  <h2>Preview</h2>
  {preview}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def write_embedding_projection(
    *,
    checkpoint_path: Path,
    labels_csv: Path,
    sequences_dir: Path,
    out_csv: Path,
    out_png: Path,
    out_html: Path,
    method: str = "pca",
    color_target: str = "mean_acc",
    batch_size: int = 32,
    device_name: str = "",
    seed: int = 42,
    perplexity: float = 30.0,
) -> pd.DataFrame:
    if checkpoint_path.suffix == ".joblib":
        frame, embeddings = collect_tabular_embeddings(
            checkpoint_path=checkpoint_path,
            labels_csv=labels_csv,
            sequences_dir=sequences_dir,
        )
    else:
        frame, embeddings = collect_neural_embeddings(
            checkpoint_path=checkpoint_path,
            labels_csv=labels_csv,
            sequences_dir=sequences_dir,
            batch_size=batch_size,
            device_name=device_name,
        )

    coords = project_embedding_matrix(embeddings, method=method, seed=seed, perplexity=perplexity)
    frame.insert(0, "projection_y", coords[:, 1])
    frame.insert(0, "projection_x", coords[:, 0])
    frame["embedding_norm"] = np.linalg.norm(embeddings, axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    write_projection_plot(frame, out_png, color_target=color_target)
    write_projection_html(
        out_html,
        out_csv=out_csv,
        out_png=out_png,
        frame=frame,
        method=method,
        color_target=color_target,
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Project model embeddings to 2D for cluster inspection.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--method", choices=["pca", "tsne"], default="pca")
    parser.add_argument("--color-target", default="mean_acc")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--perplexity", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = args.checkpoint.parent
    out_csv = args.out_csv or (run_dir / "embedding_projection.csv")
    out_png = args.out_png or (run_dir / "embedding_projection.png")
    out_html = args.out_html or (run_dir / "embedding_report.html")
    frame = write_embedding_projection(
        checkpoint_path=args.checkpoint,
        labels_csv=args.labels,
        sequences_dir=args.sequences,
        out_csv=out_csv,
        out_png=out_png,
        out_html=out_html,
        method=args.method,
        color_target=args.color_target,
        batch_size=args.batch_size,
        device_name=args.device,
        seed=args.seed,
        perplexity=args.perplexity,
    )
    summary = {
        "rows": int(len(frame)),
        "csv": str(out_csv),
        "png": str(out_png),
        "html": str(out_html),
        "method": args.method,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
