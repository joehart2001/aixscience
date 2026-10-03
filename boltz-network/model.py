"""A small MLP that predicts half-life from a Boltz2 embedding.

Deliberately the **same level of complexity** as direct-network's
`DirectAffinityNet`: a 3-layer (2-hidden) ReLU MLP with dropout to a single
output. The only difference is the input — here the allele-peptide system is
already a continuous embedding vector, so there are no embedding tables and no
flattening; the vector feeds the MLP directly.
"""

from __future__ import annotations

import torch
from torch import nn


class BoltzAffinityNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Same MLP shape as direct-network: input -> 256 -> 256 -> 1, with ReLU
        # nonlinearities (so it's a nonlinear regressor) and dropout.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # layer 1: embedding -> 256
            nn.ReLU(),                         # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # layer 2: 256 -> 256
            nn.ReLU(),                          # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # layer 3: 256 -> 1 (predicted log half-life)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, input_dim)
        # Squeeze the trailing size-1 dim so output is (batch,).
        return self.mlp(x).squeeze(-1)
