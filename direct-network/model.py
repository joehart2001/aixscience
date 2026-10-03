"""A small, readable PyTorch model that directly predicts peptide-MHC half-life.

The idea is intentionally simple:

  1. Embed each amino acid as a small learned vector (one shared embedding table
     for both the peptide and the HLA sequence).
  2. Flatten those per-residue vectors into one fixed-size vector per sequence.
  3. Embed the allele name as its own learned vector.
  4. Concatenate [peptide, hla, allele] and pass through a plain MLP that
     outputs a single number: the predicted log1p(thalf_hours).
"""

from __future__ import annotations

import torch
from torch import nn

from data import AA_VOCAB_SIZE, PAD_IDX


class DirectAffinityNet(nn.Module):
    def __init__(
        self,
        peptide_len: int,
        hla_len: int,
        num_alleles: int,
        aa_embed_dim: int = 16,
        allele_embed_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
    ):
        super().__init__()

        # Shared amino-acid embedding. padding_idx keeps the pad/unknown vector
        # at zero so it contributes nothing.
        self.aa_embed = nn.Embedding(
            AA_VOCAB_SIZE, aa_embed_dim, padding_idx=PAD_IDX
        )
        # Allele embedding (index 0 is the "unknown allele" slot).
        self.allele_embed = nn.Embedding(num_alleles, allele_embed_dim)

        # After flattening, each sequence becomes length * aa_embed_dim features.
        input_dim = (
            peptide_len * aa_embed_dim
            + hla_len * aa_embed_dim
            + allele_embed_dim
        )

        # A 3-layer multilayer perceptron; the ReLU activations make it a
        # *nonlinear* regressor (without them it would collapse to plain linear
        # regression on the embeddings). Dropout randomly zeros units to reduce
        # overfitting.
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),  # layer 1: input features -> 256
            nn.ReLU(),                         # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),  # layer 2: 256 -> 256
            nn.ReLU(),                          # nonlinearity
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),  # layer 3: 256 -> 1 (predicted log half-life)
        )

    def forward(
        self,
        peptide: torch.Tensor,  # (batch, peptide_len) int indices
        hla: torch.Tensor,      # (batch, hla_len) int indices
        allele: torch.Tensor,   # (batch,) int indices
    ) -> torch.Tensor:
        batch = peptide.shape[0]

        # Look up each residue's embedding, then flatten to one vector per example.
        # (batch, len, aa_embed_dim) -> (batch, len * aa_embed_dim)
        pep_vec = self.aa_embed(peptide).reshape(batch, -1)
        hla_vec = self.aa_embed(hla).reshape(batch, -1)
        allele_vec = self.allele_embed(allele)  # (batch, allele_embed_dim)

        # Join all three inputs into one feature vector for the MLP.
        features = torch.cat([pep_vec, hla_vec, allele_vec], dim=1)
        # Squeeze the trailing size-1 dim so output is (batch,).
        return self.mlp(features).squeeze(-1)
