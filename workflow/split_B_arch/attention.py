"""Cross-attention from the peptide onto the groove it sits in.

Motivation. Stability is an off-rate, so the physically relevant object is the
peptide-receptor interface, and the gradient-boosted tree importances say the
same thing empirically: the three cross_z blocks carry 58% of the gain from a
quarter of the features, 3-7x more per feature than the peptide's own pooled
embedding. Pooling averages that interface away. This keeps it.

Each of the 9 peptide tokens attends over the K nearest receptor tokens, so the
model can learn which groove residues each peptide position is answering to,
rather than being handed a mean.

    peptide   (9, 384) -> Linear -> d_model, plus a LEARNED POSITION EMBEDDING
                         for the 9 positions. Position identity matters here
                         (P2 and P9 are anchors), so nothing is shared across
                         positions and no translation equivariance is assumed.
    receptor  (K, 384) -> Linear -> d_model

    n_layers x [ cross-attention(peptide <- receptor) + residual + LayerNorm
                 feed-forward(d_model -> 2*d_model -> d_model) + residual + LN ]

    attention logits get a DISTANCE BIAS: the alpha-carbon distance between the
    peptide token and the receptor token is expanded in 16 Gaussian radial basis
    functions (centres 2-20 A) and mapped linearly to one bias per head. The
    model can therefore learn its own distance dependence instead of having a
    hard cutoff imposed, and attention stays geometry-aware.

    attention pooling over the 9 peptide tokens -> head -> prediction

Same training recipe, same frozen rows, same seeds as every other architecture.
"""

import numpy as np
import torch
import torch.nn as nn

from trainer import get_data, run_family

TOKENS = "tokens.npz"
RBF_CENTRES = torch.linspace(2.0, 20.0, 16)
RBF_WIDTH = 1.5


class CrossBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.h = heads
        self.dk = d // heads
        self.q = nn.Linear(d, d)
        self.k = nn.Linear(d, d)
        self.v = nn.Linear(d, d)
        self.o = nn.Linear(d, d)
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(2 * d, d))
        self.drop = nn.Dropout(dropout)

    def forward(self, x, r, bias):
        B, L, _ = x.shape
        K = r.shape[1]
        q = self.q(x).view(B, L, self.h, self.dk).transpose(1, 2)
        k = self.k(r).view(B, K, self.h, self.dk).transpose(1, 2)
        v = self.v(r).view(B, K, self.h, self.dk).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / self.dk ** 0.5 + bias
        a = self.drop(att.softmax(-1)) @ v
        a = a.transpose(1, 2).reshape(B, L, -1)
        x = self.n1(x + self.drop(self.o(a)))
        return self.n2(x + self.drop(self.ff(x)))


class CrossAttention(nn.Module):
    def __init__(self, d_in=384, d=128, heads=4, layers=2, n_pos=9,
                 n_extra=0, dropout=0.3):
        super().__init__()
        self.pep = nn.Linear(d_in, d)
        self.rec = nn.Linear(d_in, d)
        self.pos = nn.Parameter(torch.zeros(1, n_pos, d))
        nn.init.normal_(self.pos, std=0.02)
        self.rbf_to_bias = nn.Linear(len(RBF_CENTRES), heads)
        self.blocks = nn.ModuleList(
            [CrossBlock(d, heads, dropout) for _ in range(layers)])
        self.pool = nn.Linear(d, 1)                 # attention pooling weights
        self.head = nn.Sequential(
            nn.Linear(d + n_extra, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1))
        self.register_buffer("centres", RBF_CENTRES.clone())

    def forward(self, P, R, Dm, extra=None):
        x = self.pep(P) + self.pos
        r = self.rec(R)
        rbf = torch.exp(-((Dm[..., None] - self.centres) / RBF_WIDTH) ** 2)
        bias = self.rbf_to_bias(rbf).permute(0, 3, 1, 2)   # B, heads, L, K
        for b in self.blocks:
            x = b(x, r, bias)
        w = self.pool(x).softmax(1)
        h = (w * x).sum(1)
        if extra is not None:
            h = torch.cat([h, extra], dim=-1)
        return self.head(h).squeeze(-1)


def text(d, heads, layers, extra=0):
    return (f"peptide Linear(384,{d}) + learned position embedding (9); "
            f"receptor Linear(384,{d}); {layers} x [cross-attention {heads} "
            f"heads, 16-Gaussian distance bias on the logits, residual+LayerNorm; "
            f"FFN {d}->{2*d}->{d}, residual+LayerNorm]; attention pooling over "
            f"the 9 peptide tokens; head Linear({d}{f'+{extra}' if extra else ''}"
            f",128) -> GELU -> Dropout(0.3) -> Linear(128,32) -> GELU -> "
            f"Dropout(0.3) -> Linear(32,1)")


if __name__ == "__main__":
    D = get_data()
    z = np.load(TOKENS, allow_pickle=True)
    assert list(z["ids"]) == list(D["df"].sample_id), "token file is out of step"
    P, R, Dm = z["P"], z["R"], z["D"]
    tr = D["tr"]
    # standardise the token features on the train fold, as everywhere else
    mu, sd = P[tr].reshape(-1, P.shape[-1]).mean(0), P[tr].reshape(-1, P.shape[-1]).std(0)
    P = ((P - mu) / (sd + 1e-6)).astype(np.float32)
    mu, sd = R[tr].reshape(-1, R.shape[-1]).mean(0), R[tr].reshape(-1, R.shape[-1]).std(0)
    R = ((R - mu) / (sd + 1e-6)).astype(np.float32)

    std, F = D["std"], D["F"]
    pooled = std(F["pooled"])
    seq = std(D["oh"])
    charge = np.hstack([std(F["charge_seq"]), std(F["charge_struct"])])
    extra = np.hstack([seq, pooled, charge]).astype(np.float32)

    models = [
        ("attn_d64", "cross-attention, d_model 64, 4 heads, 2 layers",
         (lambda: CrossAttention(d=64, heads=4, layers=2)), (P, R, Dm),
         text(64, 4, 2), {"d_model": 64, "heads": 4, "layers": 2, "K": int(z["K"])}),
        ("attn_d128", "cross-attention, d_model 128, 4 heads, 2 layers",
         (lambda: CrossAttention(d=128, heads=4, layers=2)), (P, R, Dm),
         text(128, 4, 2), {"d_model": 128, "heads": 4, "layers": 2, "K": int(z["K"])}),
        ("attn_d128_L1", "cross-attention, d_model 128, 4 heads, 1 layer",
         (lambda: CrossAttention(d=128, heads=4, layers=1)), (P, R, Dm),
         text(128, 4, 1), {"d_model": 128, "heads": 4, "layers": 1, "K": int(z["K"])}),
        ("attn_d128+seq+pooled+charge",
         "cross-attention with sequence, pooled Boltz and charge in the head",
         (lambda: CrossAttention(d=128, heads=4, layers=2,
                                 n_extra=extra.shape[1])),
         (P, R, Dm, extra), text(128, 4, 2, extra.shape[1]),
         {"d_model": 128, "heads": 4, "layers": 2, "K": int(z["K"]),
          "extra": "seq + boltz_pooled + charge"}),
    ]
    run_family(models, D)
