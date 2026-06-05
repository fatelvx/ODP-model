from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LSTMDifficultyModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,
        embed_dim: int = 64,
        hidden_dim: int = 128,
        num_layers: int = 2,
        output_dim: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.config = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "num_layers": num_layers,
            "output_dim": output_dim,
            "dropout": dropout,
        }
        self.note_embedding = nn.Linear(input_dim, embed_dim)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, output_dim),
        )

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        embedded = self.note_embedding(features)
        packed = pack_padded_sequence(
            embedded,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        packed_output, _ = self.lstm(packed)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)

        mask = torch.arange(output.size(1), device=output.device)[None, :] < lengths[:, None].to(
            output.device
        )
        masked_output = output * mask.unsqueeze(-1)
        pooled = masked_output.sum(dim=1) / lengths.to(output.device).clamp_min(1).unsqueeze(-1)
        return self.head(pooled)
