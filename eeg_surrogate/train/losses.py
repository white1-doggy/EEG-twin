"""Loss utilities for surrogate brain training."""
from __future__ import annotations

import torch
from torch import nn


def get_loss(name: str) -> nn.Module:
    name = name.lower()
    if name == "mse":
        return nn.MSELoss()
    raise ValueError(f"Unsupported loss type: {name}")
