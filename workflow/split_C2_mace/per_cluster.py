"""Per-cluster Spearman for the boosted trees trained on Boltz alone (split_C2).

Both held-out folds are reported: the 5 test clusters and the 5 validation
clusters, 10 unseen groove families in all. Predictions are the seed mean from
runs/gbm_boltz, whose row order is the fold index order, so clusters can be
attached by re-deriving the same frozen split.

Within a single cluster every complex shares a very similar groove, so the
allele-identity contribution is held fixed and the within-cluster Spearman
measures PEPTIDE discrimination only -- the thing the overall number mixes with
the model's ability to rank one allele above another.
"""

import glob
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

import trainer
from data_split import load
from mace_blocks import have

D = load("split_C2")
ids = D["df"].sample_id.to_numpy()
frozen = [l.strip() for l in open(trainer.FROZEN) if l.strip()]
D = trainer._subset(D, np.isin(ids, frozen))
df, idx, ly = D["df"], D["idx"], D["ly"]

pred = {}
for fold in ("val", "test"):
    P = [np.load(f)[f"pred_{fold}"] for f in
         sorted(glob.glob("runs/gbm_boltz/seed*/predictions.npz"))]
    pred[fold] = np.mean(P, 0)
    print(f"{fold}: {len(P)} seeds, {len(pred[fold]):,} complexes")

rows = []
for fold in ("val", "test"):
    ix = idx[fold]
    sub = df.iloc[ix]
    y = ly[ix]
    p = pred[fold]
    rows.append(dict(fold=fold, cluster="ALL", allele="-", n=len(ix),
                     n_alleles=sub.allele.nunique(),
                     rho=spearmanr(p, y).statistic,
                     zero_frac=(sub.thalf_hours == 0).mean()))
    for c, g in sub.groupby("cluster"):
        m = sub.cluster.to_numpy() == c
        # compute wherever it is defined at all; n is printed so thin clusters are
        # visible rather than hidden. Spearman needs >=3 points and some spread.
        r = (spearmanr(p[m], y[m]).statistic
             if m.sum() >= 3 and np.ptp(y[m]) > 0 else np.nan)
        rows.append(dict(fold=fold, cluster=str(c),
                         allele=", ".join(sorted(g.allele.unique()))[:60],
                         n=int(m.sum()), n_alleles=g.allele.nunique(), rho=r,
                         zero_frac=(g.thalf_hours == 0).mean()))

res = pd.DataFrame(rows)
res.to_csv("per_cluster_gbm_boltz.csv", index=False)
print()
print(f"{'fold':<5} {'cluster':>8} {'n':>6} {'alleles':>8} {'zero%':>6} "
      f"{'rho':>7}   alleles")
for r in res.itertuples():
    mark = "  " if r.cluster != "ALL" else "* "
    print(f"{mark}{r.fold:<5} {r.cluster:>6} {r.n:>6,} {r.n_alleles:>8} "
          f"{r.zero_frac*100:>5.0f}% {r.rho:>+7.3f}   {r.allele}")
