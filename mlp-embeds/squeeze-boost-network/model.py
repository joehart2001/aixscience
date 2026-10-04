"""Squeeze-and-Excitation recalibration over naive embeddings, feeding a head.

The embedding front end is identical to direct-network: one shared amino-acid
table encodes the peptide and the HLA pseudosequence, those are flattened, and
an allele embedding is concatenated. The new part sits between that feature
vector and the MLP:

  Squeeze     The feature vector is already 1-D (flattening pooled it), so
              there is nothing to pool a second time. What a 2-D SE block gets
              from global average pooling — a cheap summary of each field — we
              instead get by computing per-field **mean and variance** across
              the three groups (peptide / HLA / allele) and appending them to
              the gate's input. That is the "grouped field" squeeze; set
              `squeeze_stats=False` to use the plain form where the gate reads
              X alone.

  Excitation  A bottleneck gate, s = sigma(W2 . ReLU(W1 . X)), with
              W1: (C/r, C) and W2: (C, C/r) at reduction ratio r = 8.

  Scale       X~ = s * X, elementwise.

The point is that s depends on the example, so the same channel can be
suppressed for one HLA molecule and amplified for another — damping noisy
pocket interactions and emphasising the anchor positions that matter for *this*
groove. A plain MLP has to settle on one fixed weight per channel instead.

`features()` exposes X~ so the XGBoost stage in boost.py can be trained on the
recalibrated features rather than the raw ones — the "boost" half of the name.
"""

from __future__ import annotations

import torch
from torch import nn

from data_seq import AA_VOCAB_SIZE, PAD_IDX
from model_common import mlp_head


class SqueezeExcite1d(nn.Module):
    """Channel gating for a 1-D feature vector (see the module docstring).

    `field_sizes` gives the width of each contiguous group of channels, which
    is what lets the squeeze compute per-field statistics. They must sum to
    `channels`.
    """

    def __init__(
        self,
        channels: int,
        field_sizes: tuple[int, ...],
        reduction: int = 8,
        squeeze_stats: bool = True,
    ):
        super().__init__()
        if sum(field_sizes) != channels:
            raise ValueError(
                f"field_sizes {field_sizes} sum to {sum(field_sizes)}, "
                f"expected {channels}"
            )
        self.field_sizes = field_sizes
        self.squeeze_stats = squeeze_stats

        # Two stats (mean, variance) per field get appended to the gate input.
        gate_in = channels + (2 * len(field_sizes) if squeeze_stats else 0)
        # Bottleneck width C/r; keep at least 1 channel for tiny inputs.
        bottleneck = max(1, channels // reduction)

        self.gate = nn.Sequential(
            nn.Linear(gate_in, bottleneck),  # W1: (C/r, C)
            nn.ReLU(),
            nn.Linear(bottleneck, channels),  # W2: (C, C/r)
            nn.Sigmoid(),                     # s in (0, 1) per channel
        )

    def _squeeze(self, x: torch.Tensor) -> torch.Tensor:
        """Per-field mean and variance, concatenated onto x."""
        if not self.squeeze_stats:
            return x
        stats = []
        start = 0
        for size in self.field_sizes:
            field = x[:, start : start + size]
            # keepdim so these stay (batch, 1) and concatenate cleanly.
            stats.append(field.mean(dim=1, keepdim=True))
            # unbiased=False: this is a descriptor of the vector, not a sample
            # estimate, and it must not divide by zero when a field has width 1.
            stats.append(field.var(dim=1, unbiased=False, keepdim=True))
            start += size
        return torch.cat([x, *stats], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (batch, channels)
        s = self.gate(self._squeeze(x))
        return s * x  # scale

    @torch.no_grad()
    def gate_values(self, x: torch.Tensor) -> torch.Tensor:
        """The gate s itself — useful for inspecting which channels get damped."""
        return self.gate(self._squeeze(x))


class SqueezeBoostNet(nn.Module):
    """Naive embeddings -> SE recalibration -> MLP head.

    Same constructor contract as direct-network's `DirectAffinityNet` (plus SE
    options), so the shared train.py/compare.py drive it unchanged.
    """

    def __init__(
        self,
        peptide_len: int,
        hla_len: int,
        num_alleles: int,
        aa_embed_dim: int = 16,
        allele_embed_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.2,
        reduction: int = 8,
        squeeze_stats: bool = True,
    ):
        super().__init__()

        # --- identical front end to direct-network ---
        self.aa_embed = nn.Embedding(
            AA_VOCAB_SIZE, aa_embed_dim, padding_idx=PAD_IDX
        )
        self.allele_embed = nn.Embedding(num_alleles, allele_embed_dim)

        # The three contiguous fields of the flattened vector. Keeping them
        # explicit is what makes the grouped squeeze possible.
        self.field_sizes = (
            peptide_len * aa_embed_dim,
            hla_len * aa_embed_dim,
            allele_embed_dim,
        )
        input_dim = sum(self.field_sizes)

        self.se = SqueezeExcite1d(
            input_dim,
            self.field_sizes,
            reduction=reduction,
            squeeze_stats=squeeze_stats,
        )

        # Same shared head as every other network, so any gain is attributable
        # to the SE block rather than to extra head capacity.
        self.mlp = mlp_head(input_dim, hidden_dim, dropout)

    def features(
        self,
        peptide: torch.Tensor,
        hla: torch.Tensor,
        allele: torch.Tensor,
    ) -> torch.Tensor:
        """The recalibrated feature vector X~ — the input XGBoost is fitted on."""
        batch = peptide.shape[0]
        pep_vec = self.aa_embed(peptide).reshape(batch, -1)
        hla_vec = self.aa_embed(hla).reshape(batch, -1)
        allele_vec = self.allele_embed(allele)
        x = torch.cat([pep_vec, hla_vec, allele_vec], dim=1)
        return self.se(x)

    def forward(
        self,
        peptide: torch.Tensor,  # (batch, peptide_len) int indices
        hla: torch.Tensor,      # (batch, hla_len) int indices
        allele: torch.Tensor,   # (batch,) int indices
    ) -> torch.Tensor:
        # Squeeze the trailing size-1 dim so output is (batch,).
        return self.mlp(self.features(peptide, hla, allele)).squeeze(-1)


# train.py imports this name; alias so the shared engine needs no edit.
DirectAffinityNet = SqueezeBoostNet
