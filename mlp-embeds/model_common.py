"""The shared MLP head used by every network.

All four models end in the same 3-layer (2-hidden) ReLU MLP with dropout to a
single output — only the encoder that produces `in_dim` features differs. The
ReLU activations make it a nonlinear regressor; dropout reduces overfitting.
"""

from __future__ import annotations

from torch import nn


def mlp_head(in_dim: int, hidden_dim: int = 256, dropout: float = 0.2) -> nn.Sequential:
    """input -> hidden -> hidden -> 1, with ReLU + dropout between layers."""
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),      # layer 1: features -> hidden
        nn.ReLU(),                          # nonlinearity
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),  # layer 2: hidden -> hidden
        nn.ReLU(),                          # nonlinearity
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, 1),           # layer 3: hidden -> 1 (log half-life)
    )
