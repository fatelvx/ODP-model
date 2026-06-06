from __future__ import annotations

import argparse
import html
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from mania_difficulty.models.factory import create_model
from mania_difficulty.visualize import write_run_report


def attention_frame(features: np.ndarray, attention: np.ndarray, *, keys: int = 4) -> pd.DataFrame:
    rows = []
    for index, row in enumerate(features):
        rows.append(
            {
                "note_index": index,
                "time_fraction": float(row[0]),
                "time_delta_sec": float(row[1]),
                "column": int(round(float(row[2]) * max(1, keys - 1))),
                "column_fraction": float(row[2]),
                "is_ln": int(round(float(row[3]))),
                "ln_length_sec": float(row[4]),
                "chord_size": float(row[5]) * keys,
                "attention": float(attention[index]),
            }
        )
    return pd.DataFrame(rows)


def write_attention_plot(frame: pd.DataFrame, out_png: Path) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 9))
    size = 36 + 520 * frame["attention"].clip(lower=0)
    scatter = ax.scatter(
        frame["column"],
        frame["time_fraction"],
        c=frame["attention"],
        s=size,
        cmap="magma",
        alpha=0.86,
        edgecolors="#1f2933",
        linewidths=0.35,
    )
    ln_rows = frame[frame["is_ln"] > 0]
    if not ln_rows.empty:
        ax.scatter(
            ln_rows["column"],
            ln_rows["time_fraction"],
            marker="s",
            s=28,
            facecolors="none",
            edgecolors="#00bcd4",
            linewidths=1.0,
            label="LN",
        )
        ax.legend(loc="upper right")
    fig.colorbar(scatter, ax=ax, label="attention")
    ax.set_title("Transformer Attention Map")
    ax.set_xlabel("Column")
    ax.set_ylabel("Time through map")
    ax.set_xticks(sorted(frame["column"].unique()))
    ax.invert_yaxis()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def write_attention_html(
    out_html: Path,
    *,
    out_csv: Path,
    out_png: Path,
    frame: pd.DataFrame,
    beatmap_id: int,
) -> None:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    csv_href = os.path.relpath(out_csv, start=out_html.parent).replace("\\", "/")
    png_href = os.path.relpath(out_png, start=out_html.parent).replace("\\", "/")
    top_notes = frame.sort_values("attention", ascending=False).head(20)
    preview = top_notes.to_html(index=False, float_format=lambda value: f"{value:.6f}")
    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>transformer attention map</title>
  <style>
    body {{ font-family: Arial, sans-serif; margin: 32px; color: #1f2933; }}
    table {{ border-collapse: collapse; margin: 12px 0 24px; }}
    th, td {{ border: 1px solid #bcccdc; padding: 8px 12px; text-align: right; }}
    th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
    img {{ max-width: 100%; border: 1px solid #d9e2ec; }}
    code {{ background: #f0f4f8; padding: 2px 4px; }}
    a {{ margin-right: 12px; }}
  </style>
</head>
<body>
  <h1>Transformer Attention Map</h1>
  <p>Beatmap ID: <code>{html.escape(str(beatmap_id))}</code></p>
  <p><a href="{html.escape(csv_href)}">attention_map.csv</a></p>
  <p><img src="{html.escape(png_href)}" alt="Transformer attention map"></p>
  <h2>Highest Attention Notes</h2>
  {preview}
</body>
</html>
"""
    out_html.write_text(report, encoding="utf-8")


def write_attention_map(
    *,
    checkpoint_path: Path,
    beatmap_id: int,
    sequences_dir: Path,
    out_csv: Path,
    out_png: Path,
    out_html: Path,
    device_name: str = "",
    keys: int = 4,
) -> pd.DataFrame:
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if checkpoint.get("model_name") != "transformer":
        raise RuntimeError("attention maps require a transformer checkpoint")
    model = create_model(
        "transformer",
        output_dim=len(checkpoint["target_columns"]),
        config=checkpoint["model_config"],
    ).to(device)
    if not hasattr(model, "attention_importance"):
        raise RuntimeError("checkpoint model does not expose attention_importance()")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    sequence_path = sequences_dir / f"{int(beatmap_id)}.npy"
    features = np.load(sequence_path).astype("float32")
    max_notes = int(checkpoint.get("max_notes", len(features)))
    features = features[:max_notes]
    if len(features) == 0:
        raise RuntimeError(f"No notes found in {sequence_path}")

    with torch.no_grad():
        attention = (
            model.attention_importance(
                torch.from_numpy(features[None, :, :]).to(device),
                torch.tensor([len(features)], dtype=torch.long, device=device),
            )
            .cpu()
            .numpy()[0]
        )

    frame = attention_frame(features, attention, keys=keys)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False, encoding="utf-8")
    write_attention_plot(frame, out_png)
    write_attention_html(
        out_html,
        out_csv=out_csv,
        out_png=out_png,
        frame=frame,
        beatmap_id=beatmap_id,
    )
    if out_html.parent.resolve() == checkpoint_path.parent.resolve():
        write_run_report(
            checkpoint_path.parent,
            target_columns=list(checkpoint["target_columns"]),
            metrics_path=checkpoint_path.parent / "metrics.json",
        )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize note-level Transformer attention for one map.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--beatmap-id", type=int, required=True)
    parser.add_argument("--sequences", type=Path, required=True)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--out-png", type=Path, default=None)
    parser.add_argument("--out-html", type=Path, default=None)
    parser.add_argument("--device", default="")
    parser.add_argument("--keys", type=int, default=4)
    args = parser.parse_args()

    run_dir = args.checkpoint.parent
    out_csv = args.out_csv or (run_dir / "attention_map.csv")
    out_png = args.out_png or (run_dir / "attention_map.png")
    out_html = args.out_html or (run_dir / "attention_report.html")
    frame = write_attention_map(
        checkpoint_path=args.checkpoint,
        beatmap_id=args.beatmap_id,
        sequences_dir=args.sequences,
        out_csv=out_csv,
        out_png=out_png,
        out_html=out_html,
        device_name=args.device,
        keys=args.keys,
    )
    summary = {
        "beatmap_id": int(args.beatmap_id),
        "rows": int(len(frame)),
        "csv": str(out_csv),
        "png": str(out_png),
        "html": str(out_html),
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
