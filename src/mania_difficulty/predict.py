from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch

from mania_difficulty.data.parse_notes import parse_osu_file
from mania_difficulty.models.factory import create_model
from mania_difficulty.models.tabular import summarize_sequence


def predict_tabular(checkpoint_path: Path, features: np.ndarray, osu_path: Path) -> dict[str, object]:
    checkpoint = joblib.load(checkpoint_path)
    max_notes = int(checkpoint.get("max_notes", 3000))
    clipped = features[:max_notes]
    summary = summarize_sequence(clipped)
    pred = checkpoint["model"].predict(summary[None, :])[0]
    return {
        "osu_file": str(osu_path),
        "model_name": checkpoint.get("model_name", "tabular_forest"),
        "num_notes_used": int(len(clipped)),
        "predictions": {
            column: float(pred[index])
            for index, column in enumerate(checkpoint["target_columns"])
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict difficulty descriptors for one .osu file.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--osu", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", default="")
    args = parser.parse_args()

    features = np.asarray(parse_osu_file(args.osu), dtype="float32")
    if len(features) == 0:
        raise RuntimeError(f"No notes found in {args.osu}")

    if args.checkpoint.suffix == ".joblib":
        result = predict_tabular(args.checkpoint, features, args.osu)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = create_model(
        checkpoint.get("model_name", "lstm"),
        output_dim=len(checkpoint["target_columns"]),
        config=checkpoint["model_config"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    max_notes = int(checkpoint.get("max_notes", 3000))
    features = features[:max_notes]

    with torch.no_grad():
        pred_norm = model(
            torch.from_numpy(features[None, :, :]).to(device),
            torch.tensor([len(features)], dtype=torch.long, device=device),
        ).cpu().numpy()[0]

    target_mean = np.asarray(checkpoint["target_mean"], dtype="float32")
    target_std = np.asarray(checkpoint["target_std"], dtype="float32")
    pred = pred_norm * target_std + target_mean
    result = {
        "osu_file": str(args.osu),
        "model_name": checkpoint.get("model_name", "lstm"),
        "num_notes_used": int(len(features)),
        "predictions": {
            column: float(pred[index])
            for index, column in enumerate(checkpoint["target_columns"])
        },
    }

    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
