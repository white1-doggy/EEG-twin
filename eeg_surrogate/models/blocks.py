"""Model building blocks."""
from __future__ import annotations

import torch
from torch import nn


class GRUEncoder(nn.Module):
    """GRU-based temporal encoder returning the final hidden state."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
        dropout: float = 0.0,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        self.num_directions = 2 if bidirectional else 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, input_dim)
        _, h_n = self.gru(x)
        # h_n: (num_layers * num_directions, batch, hidden_dim)
        last = h_n[-self.num_directions :].transpose(0, 1).reshape(x.size(0), -1)
        return last
