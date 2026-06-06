from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


DEFAULT_TARGET_COLUMNS = ("mean_acc", "acc_std", "skill_gap")


@dataclass(frozen=True)
class Batch:
    features: torch.Tensor
    lengths: torch.Tensor
    targets: torch.Tensor
    sample_weights: torch.Tensor
    beatmap_ids: list[int]


class ManiaDifficultyDataset(Dataset):
    def __init__(
        self,
        labels_csv: Path,
        sequences_dir: Path,
        *,
        target_columns: Sequence[str] = DEFAULT_TARGET_COLUMNS,
        sample_weight_column: str = "",
        sample_weight_min: float = 0.25,
        sample_weight_max_value: float = 100.0,
    ) -> None:
        self.labels = pd.read_csv(labels_csv)
        self.sequences_dir = Path(sequences_dir)
        self.target_columns = tuple(target_columns)
        self.sample_weight_column = sample_weight_column
        self.sample_weight_min = float(sample_weight_min)
        self.sample_weight_max_value = float(sample_weight_max_value)

        missing = [column for column in self.target_columns if column not in self.labels.columns]
        if missing:
            raise ValueError(f"Missing target columns in labels.csv: {missing}")
        if self.sample_weight_column and self.sample_weight_column not in self.labels.columns:
            raise ValueError(
                f"Missing sample weight column in labels.csv: {self.sample_weight_column}"
            )

        keep_rows = []
        for _, row in self.labels.iterrows():
            if (self.sequences_dir / f"{int(row['beatmap_id'])}.npy").exists():
                keep_rows.append(row)
        self.labels = pd.DataFrame(keep_rows).reset_index(drop=True)
        if self.labels.empty:
            raise RuntimeError("No labels have matching .npy sequence files.")

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> dict[str, object]:
        row = self.labels.iloc[index]
        beatmap_id = int(row["beatmap_id"])
        features = np.load(self.sequences_dir / f"{beatmap_id}.npy").astype("float32")
        targets = row.loc[list(self.target_columns)].to_numpy(dtype="float32")
        sample_weight = 1.0
        if self.sample_weight_column:
            sample_weight = label_sample_weight(
                row[self.sample_weight_column],
                min_weight=self.sample_weight_min,
                max_value=self.sample_weight_max_value,
            )
        return {
            "beatmap_id": beatmap_id,
            "features": features,
            "targets": targets,
            "sample_weight": sample_weight,
        }


def label_sample_weight(value: object, *, min_weight: float, max_value: float) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return float(min_weight)
    if not np.isfinite(numeric):
        return float(min_weight)
    if max_value <= 0:
        return 1.0
    return float(np.clip(numeric / max_value, min_weight, 1.0))


def collate_batch(samples: list[dict[str, object]], max_notes: int = 3000) -> Batch:
    clipped = []
    lengths = []
    for sample in samples:
        features = np.asarray(sample["features"], dtype="float32")[:max_notes]
        clipped.append(features)
        lengths.append(len(features))

    batch_size = len(samples)
    feature_dim = clipped[0].shape[1]
    max_len = max(lengths)
    padded = np.zeros((batch_size, max_len, feature_dim), dtype="float32")
    for index, features in enumerate(clipped):
        padded[index, : len(features)] = features

    targets = np.stack([np.asarray(sample["targets"], dtype="float32") for sample in samples])
    sample_weights = np.asarray(
        [float(sample.get("sample_weight", 1.0)) for sample in samples],
        dtype="float32",
    )
    return Batch(
        features=torch.from_numpy(padded),
        lengths=torch.tensor(lengths, dtype=torch.long),
        targets=torch.from_numpy(targets),
        sample_weights=torch.from_numpy(sample_weights),
        beatmap_ids=[int(sample["beatmap_id"]) for sample in samples],
    )
