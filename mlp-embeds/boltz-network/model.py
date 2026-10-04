"""A small MLP that predicts half-life from a Boltz2 embedding.

Deliberately the **same level of complexity** as the sequence networks: the
shared 3-layer MLP head. The only difference is the input — the allele-peptide
system is already a continuous embedding vector, so there are no embedding tables
and no flattening; the vector feeds the head directly.
"""

from __future__ import annotations

import torch
from torch import nn

from model_common import mlp_head


class BoltzAffinityNet(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.mlp = mlp_head(input_dim, hidden_dim, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, input_dim)
        # Squeeze the trailing size-1 dim so output is (batch,).
        return self.mlp(x).squeeze(-1)
