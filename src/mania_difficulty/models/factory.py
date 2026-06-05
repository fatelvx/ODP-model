from __future__ import annotations

from torch import nn

from mania_difficulty.models.lstm import LSTMDifficultyModel
from mania_difficulty.models.summary import SummaryDifficultyModel


def create_model(model_name: str, *, output_dim: int, config: dict | None = None) -> nn.Module:
    config = dict(config or {})
    if model_name == "lstm":
        config.setdefault("output_dim", output_dim)
        return LSTMDifficultyModel(**config)
    if model_name == "summary":
        config.setdefault("output_dim", output_dim)
        return SummaryDifficultyModel(**config)
    raise ValueError(f"Unknown model: {model_name}")
