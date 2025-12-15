"""Surrogate brain model predicting future ROI-band envelopes."""
from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from .blocks import GRUEncoder


class SurrogateModel(nn.Module):
    """Single shared surrogate model across all subjects."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        model_type: Literal["gru"] = "gru",
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if model_type != "gru":
            raise ValueError("Only GRU model_type is implemented in this template.")
        self.encoder = GRUEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.regressor = nn.Linear(hidden_dim * (2 if bidirectional else 1), output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, input_dim)
        h = self.encoder(x)
        return self.regressor(h)
