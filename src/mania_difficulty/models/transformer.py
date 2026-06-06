from __future__ import annotations

import torch
from torch import nn


class TransformerBlock(nn.Module):
    def __init__(self, *, embed_dim: int, num_heads: int, ff_dim: int, dropout: float) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.attention_norm(x)
        attn_out, attn_weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=key_padding_mask,
            need_weights=return_attention,
            average_attn_weights=False,
        )
        x = x + self.dropout(attn_out)
        x = x + self.dropout(self.feedforward(self.feedforward_norm(x)))
        return x, attn_weights if return_attention else None


class TransformerDifficultyModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int = 6,
        embed_dim: int = 64,
        num_heads: int = 4,
        num_layers: int = 3,
        ff_dim: int = 256,
        output_dim: int = 3,
        dropout: float = 0.1,
        head_dropout: float = 0.2,
        max_positions: int = 3000,
    ) -> None:
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError("transformer embed_dim must be divisible by num_heads")
        self.config = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "num_heads": num_heads,
            "num_layers": num_layers,
            "ff_dim": ff_dim,
            "output_dim": output_dim,
            "dropout": dropout,
            "head_dropout": head_dropout,
            "max_positions": max_positions,
        }
        self.note_embedding = nn.Linear(input_dim, embed_dim)
        self.position_embedding = nn.Embedding(max_positions, embed_dim)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    ff_dim=ff_dim,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(128, output_dim),
        )

    def padding_mask(self, lengths: torch.Tensor, max_len: int) -> torch.Tensor:
        positions = torch.arange(max_len, device=lengths.device)[None, :]
        return positions >= lengths[:, None]

    def sequence_encoding(
        self,
        features: torch.Tensor,
        lengths: torch.Tensor,
        *,
        return_attention: bool = False,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        seq_len = features.size(1)
        if seq_len > int(self.config["max_positions"]):
            raise ValueError(
                f"sequence length {seq_len} exceeds transformer max_positions={self.config['max_positions']}"
            )
        positions = torch.arange(seq_len, device=features.device)
        x = self.note_embedding(features) + self.position_embedding(positions)[None, :, :]
        mask = self.padding_mask(lengths.to(features.device), seq_len)
        attentions = []
        for layer in self.layers:
            x, attention = layer(x, key_padding_mask=mask, return_attention=return_attention)
            if attention is not None:
                attentions.append(attention)
        return self.final_norm(x), attentions

    def encode(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.sequence_encoding(features, lengths, return_attention=False)
        valid = (~self.padding_mask(lengths.to(features.device), encoded.size(1))).unsqueeze(-1)
        pooled = (encoded * valid).sum(dim=1) / lengths.to(features.device).clamp_min(1).unsqueeze(-1)
        return pooled

    def attention_importance(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        _, attentions = self.sequence_encoding(features, lengths, return_attention=True)
        if not attentions:
            return torch.zeros(features.shape[:2], dtype=features.dtype, device=features.device)
        valid = ~self.padding_mask(lengths.to(features.device), features.size(1))
        layer_scores = []
        for attention in attentions:
            query_weight = valid[:, None, :, None].to(attention.dtype)
            query_count = valid.sum(dim=1).clamp_min(1).to(attention.dtype)[:, None, None]
            score = (attention * query_weight).sum(dim=2) / query_count
            score = score.mean(dim=1)
            score = score * valid
            score = score / score.sum(dim=1, keepdim=True).clamp_min(1e-8)
            layer_scores.append(score)
        return torch.stack(layer_scores, dim=0).mean(dim=0) * valid

    def forward(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(features, lengths))
