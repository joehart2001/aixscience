"""A transformer encoder that predicts peptide-MHC half-life from sequence.

Same task and same inputs as direct-network (integer amino-acid indices for the
peptide, the HLA sequence, and the allele name) — only the encoder changed.

direct-network *flattens* the per-residue embeddings, so position i of the
peptide only ever meets position j of the groove through the first Linear
layer's weights: a fixed, position-specific interaction learned independently
for every (i, j) pair. Here the two sequences are concatenated into one token
stream and run through self-attention, so each residue's representation is
built from whichever other residues matter for *that* example — the peptide's
anchor residues can attend to the groove pockets they actually sit in. That is
the "richer representation" this network is testing; it is a brute-force bet
that attention recovers structure that the MLP has to memorise per position.

Layout of the token stream (lengths are from the training vocab):

    [CLS] p1 p2 ... p9  h1 h2 ... h34
      |   \\_____________/  \\___________/
      |      segment 0        segment 1
      pooled representation -> head

Three embeddings are summed per token: the shared amino-acid embedding (so a
leucine is the same vector in the peptide and in the groove), a learned
positional embedding (attention is order-blind without it), and a segment
embedding (which of the two sequences this residue came from). The allele
embedding is concatenated onto the pooled [CLS] vector rather than added as a
token, mirroring how direct-network feeds it to the head.
"""

from __future__ import annotations

import torch
from torch import nn

from data_seq import AA_VOCAB_SIZE, PAD_IDX
from model_common import mlp_head


class DirectAffinityNet(nn.Module):
    """Transformer encoder + MLP head. Name matches direct-network's model so
    the shared train.py/compare.py can drive either network unchanged."""

    def __init__(
        self,
        peptide_len: int,
        hla_len: int,
        num_alleles: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        allele_embed_dim: int = 16,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.peptide_len = peptide_len
        self.hla_len = hla_len
        # +1 for the [CLS] token we prepend.
        n_tokens = 1 + peptide_len + hla_len

        # Shared amino-acid embedding, as in direct-network. padding_idx keeps
        # the pad/unknown vector at zero (and unlearned); padded positions are
        # also masked out of attention below, so they contribute nothing.
        self.aa_embed = nn.Embedding(AA_VOCAB_SIZE, d_model, padding_idx=PAD_IDX)
        # Learned absolute positions. Self-attention is permutation-invariant,
        # so without this the model could not tell P2 from P9 — and anchor
        # position is most of what determines binding.
        self.pos_embed = nn.Embedding(n_tokens, d_model)
        # Which sequence a token belongs to: 0 = [CLS], 1 = peptide, 2 = HLA.
        self.segment_embed = nn.Embedding(3, d_model)
        # A learned token whose final state is the pooled summary of the pair.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Allele embedding (index 0 is the "unknown allele" slot), kept out of
        # the token stream and concatenated at the head.
        self.allele_embed = nn.Embedding(num_alleles, allele_embed_dim)

        self.input_norm = nn.LayerNorm(d_model)
        self.input_dropout = nn.Dropout(dropout)

        # norm_first (pre-LN) trains more stably than the post-LN original on a
        # dataset this small, where a warmup schedule would otherwise be needed.
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="relu",
            batch_first=True,
            norm_first=True,
        )
        # enable_nested_tensor is incompatible with norm_first; say so
        # explicitly rather than letting torch warn about it every run.
        self.encoder = nn.TransformerEncoder(
            layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.encoder_norm = nn.LayerNorm(d_model)

        # Same shared head as every other network (2 hidden ReLU layers -> 1),
        # so a difference in results is attributable to the encoder, not the head.
        self.mlp = mlp_head(d_model + allele_embed_dim, hidden_dim, dropout)

        # Small init for the embeddings keeps the pre-LN residual stream tame.
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed.weight, std=0.02)
        nn.init.normal_(self.segment_embed.weight, std=0.02)

    def forward(
        self,
        peptide: torch.Tensor,  # (batch, peptide_len) int indices
        hla: torch.Tensor,      # (batch, hla_len) int indices
        allele: torch.Tensor,   # (batch,) int indices
    ) -> torch.Tensor:
        batch = peptide.shape[0]
        device = peptide.device

        # One token stream per example: [CLS] + peptide residues + HLA residues.
        tokens = torch.cat([peptide, hla], dim=1)          # (batch, pep+hla)
        x = self.aa_embed(tokens)                          # (batch, pep+hla, d)
        cls = self.cls_token.expand(batch, -1, -1)         # (batch, 1, d)
        x = torch.cat([cls, x], dim=1)                     # (batch, n_tokens, d)

        # Segment ids: 0 for [CLS], 1 for the peptide block, 2 for the HLA block.
        segments = torch.cat(
            [
                torch.zeros(1, dtype=torch.long, device=device),
                torch.ones(self.peptide_len, dtype=torch.long, device=device),
                torch.full((self.hla_len,), 2, dtype=torch.long, device=device),
            ]
        )
        positions = torch.arange(x.shape[1], device=device)
        x = x + self.pos_embed(positions) + self.segment_embed(segments)
        x = self.input_dropout(self.input_norm(x))

        # Mask padded residues out of attention. [CLS] is never masked, and a
        # row is all-real apart from padding, so no example is fully masked.
        pad_mask = torch.cat(
            [
                torch.zeros(batch, 1, dtype=torch.bool, device=device),
                tokens == PAD_IDX,
            ],
            dim=1,
        )  # True marks positions to ignore

        x = self.encoder(x, src_key_padding_mask=pad_mask)
        pooled = self.encoder_norm(x[:, 0])  # the [CLS] position

        features = torch.cat([pooled, self.allele_embed(allele)], dim=1)
        # Squeeze the trailing size-1 dim so output is (batch,).
        return self.mlp(features).squeeze(-1)
