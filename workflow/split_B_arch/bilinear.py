"""Low-rank bilinear interaction between the peptide and the groove.

Motivation. The strongest result so far is that the signal is INTERACTION, not
either side alone: peptide one-hot +0.447, pseudosequence one-hot +0.529, the
two concatenated +0.747. A plain perceptron fed a concatenation has to discover
that interaction from scratch through its first linear layer. A bilinear form
encodes it directly.

The full bilinear map p^T W g would need 180 x 680 = 122,400 weights per output
channel, which is hopeless at n = 8,691. So W is factorised to rank r:

    u = U p   (r,)        v = V g   (r,)        interaction = u * v   (r,)

which is the rank-r approximation of the bilinear form, and is what a
factorisation machine does. The head sees [u, v, u*v] so it can still use each
side alone; only the third block is the interaction, and comparing against the
perceptron on the same inputs isolates what the inductive bias buys.

    Linear(d_pep, r) and Linear(d_groove, r), no bias
    -> concat[u, v, u*v]  (3r)  (+ any extra features)
    -> Linear(3r + extra, 128) -> GELU -> Dropout(0.3)
    -> Linear(128, 32) -> GELU -> Dropout(0.3) -> Linear(32, 1)

Ranks 16, 64 and 128 are tried on the sequence inputs to see whether the rank
is binding. Everything else -- optimiser, schedule, seeds, early stopping --
matches the perceptron runs exactly.
"""

import numpy as np
import torch
import torch.nn as nn

from trainer import get_data, run_family


class Bilinear(nn.Module):
    def __init__(self, d_pep, d_grv, rank=64, n_extra=0, dropout=0.3):
        super().__init__()
        self.U = nn.Linear(d_pep, rank, bias=False)
        self.V = nn.Linear(d_grv, rank, bias=False)
        self.head = nn.Sequential(
            nn.Linear(3 * rank + n_extra, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, p, g, extra=None):
        u, v = self.U(p), self.V(g)
        h = torch.cat([u, v, u * v], dim=-1)
        if extra is not None:
            h = torch.cat([h, extra], dim=-1)
        return self.head(h).squeeze(-1)


def text(r, extra=0):
    return (f"u=Linear(d_pep,{r},bias=False); v=Linear(d_groove,{r},bias=False); "
            f"concat[u,v,u*v]{f' + {extra} extra' if extra else ''} -> "
            f"Linear({3*r + extra},128) -> GELU -> Dropout(0.3) -> "
            f"Linear(128,32) -> GELU -> Dropout(0.3) -> Linear(32,1)")


if __name__ == "__main__":
    D = get_data()
    std, F = D["std"], D["F"]
    pep_oh, grv_oh = std(D["oh_pep"]), std(D["oh_hla"])
    pep_bz, grv_bz = std(F["pmean"]), std(F["recep"])
    pooled = std(F["pooled"])
    charge = np.hstack([std(F["charge_seq"]), std(F["charge_struct"])])

    models = []
    for r in (16, 64, 128):
        models.append((
            f"bilinear_seq_r{r}",
            f"rank-{r} bilinear between peptide one-hot and pseudosequence one-hot",
            (lambda r=r: Bilinear(pep_oh.shape[1], grv_oh.shape[1], r)),
            (pep_oh, grv_oh), text(r),
            {"rank": r, "peptide_input": "peptide one-hot",
             "groove_input": "pseudosequence one-hot"}))

    models.append((
        "bilinear_boltz_r64",
        "rank-64 bilinear between the Boltz peptide mean and the contacting-receptor mean",
        (lambda: Bilinear(pep_bz.shape[1], grv_bz.shape[1], 64)),
        (pep_bz, grv_bz), text(64),
        {"rank": 64, "peptide_input": "peptide_s_mean",
         "groove_input": "contact_receptor_s_mean"}))

    models.append((
        "bilinear_seq_r64+boltz_pooled",
        "rank-64 sequence bilinear, with the pooled Boltz blocks in the head",
        (lambda: Bilinear(pep_oh.shape[1], grv_oh.shape[1], 64,
                          n_extra=pooled.shape[1])),
        (pep_oh, grv_oh, pooled), text(64, pooled.shape[1]),
        {"rank": 64, "extra": "boltz_pooled"}))

    models.append((
        "bilinear_seq_r64+boltz_pooled+charge",
        "rank-64 sequence bilinear, with pooled Boltz and both charge blocks in the head",
        (lambda: Bilinear(pep_oh.shape[1], grv_oh.shape[1], 64,
                          n_extra=pooled.shape[1] + charge.shape[1])),
        (pep_oh, grv_oh, np.hstack([pooled, charge])),
        text(64, pooled.shape[1] + charge.shape[1]),
        {"rank": 64, "extra": "boltz_pooled + charge"}))

    run_family(models, D)
