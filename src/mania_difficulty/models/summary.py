from __future__ import annotations

import torch
from torch import nn


class SummaryDifficultyModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,
        hidden_dim: int = 128,
        output_dim: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "output_dim": output_dim,
            "dropout": dropout,
        }
        summary_dim = input_dim * 3 + 1
        self.net = nn.Sequential(
            nn.Linear(summary_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def summarize(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        mask = torch.arange(features.size(1), device=features.device)[None, :] < lengths[:, None].to(
            features.device
        )
        mask_f = mask.unsqueeze(-1)
        safe_lengths = lengths.to(features.device).clamp_min(1).float().unsqueeze(-1)

        masked = features * mask_f
        means = masked.sum(dim=1) / safe_lengths

        centered = (features - means.unsqueeze(1)) * mask_f
        stds = torch.sqrt((centered.square().sum(dim=1) / safe_lengths).clamp_min(1e-8))

        very_negative = torch.full_like(features, -1e9)
        maxes = torch.where(mask_f, features, very_negative).max(dim=1).values
        note_count = torch.log1p(lengths.to(features.device).float()).unsqueeze(-1) / 10.0

        return torch.cat([means, stds, maxes, note_count], dim=1)

    def encode(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded = self.summarize(features, lengths)
        for layer in list(self.net.children())[:-1]:
            encoded = layer(encoded)
        return encoded

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.net(self.summarize(features, lengths))
