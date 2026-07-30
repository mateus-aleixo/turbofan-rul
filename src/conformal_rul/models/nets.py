"""PyTorch sequence models: LSTM and Transformer encoder regressors."""

from __future__ import annotations

import math

import torch
from torch import nn

from . import OUTPUT_DIM


def _head(width: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(width, width // 2),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(width // 2, OUTPUT_DIM),
    )


class LSTMRegressor(nn.Module):
    def __init__(self, n_features: int, hidden: int = 128, layers: int = 2, dropout: float = 0.25):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features,
            hidden,
            num_layers=layers,
            batch_first=True,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.head = _head(hidden, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1])


class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class TransformerRegressor(nn.Module):
    """Pre-LN encoder (norm_first) — stable without a warmup schedule, which
    matters at this model scale where a tuned schedule would be overkill."""

    def __init__(
        self,
        n_features: int,
        d_model: int = 96,
        layers: int = 3,
        heads: int = 4,
        ff_mult: int = 4,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.proj = nn.Linear(n_features, d_model)
        self.pe = SinusoidalPE(d_model)
        layer = nn.TransformerEncoderLayer(
            d_model,
            heads,
            dim_feedforward=d_model * ff_mult,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.head = _head(d_model, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(self.pe(self.proj(x)))
        return self.head(h.mean(dim=1))


def build_net(model: str, n_features: int, cfg) -> nn.Module:
    if model == "lstm":
        return LSTMRegressor(n_features, cfg.hidden, cfg.layers, cfg.dropout)
    if model == "transformer":
        return TransformerRegressor(
            n_features, cfg.hidden, cfg.layers, cfg.heads, cfg.ff_mult, cfg.dropout
        )
    raise ValueError(f"unknown net: {model}")
