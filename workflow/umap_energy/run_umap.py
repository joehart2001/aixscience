"""UMAP of the MACE site energies, coloured by pseudosequence cluster.

The question behind this: the energy block scores +0.704 on split_B, where
alleles are shared, but -0.171 on split_C2, where they are not. An allele-mean
predictor that ignores the peptide entirely already reaches +0.536 on split_B.
That pattern says the energies are largely encoding WHICH GROOVE this is rather
than how well the peptide fits it.

If that reading is right, the embedding should separate cleanly by allele and by
pseudosequence cluster, and show little structure in half-life. Panels:

    1  coloured by cluster      -- is the layout groove identity?
    2  coloured by allele       -- the finer version of the same question
    3  coloured by log half-life -- is the TARGET laid out anywhere?
    4  coloured by split_C2 fold -- do the held-out clusters sit apart from
                                    training, which would explain why a model
                                    fitted on train cannot speak about them?

Standardised on the training fold only, as everywhere else in this project.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import umap

from data_split import load
from mace_blocks import have, load_blocks
import trainer

HERE = os.path.dirname(os.path.abspath(__file__))
SEED = 0

D = load("split_C2")
ids = D["df"].sample_id.to_numpy()
keep = np.array([have(s) for s in ids])
D = trainer._subset(D, keep)
df, idx, tr = D["df"], D["idx"], D["tr"]
print(f"{len(df):,} complexes, {df.allele.nunique()} alleles, "
      f"{df.cluster.nunique()} clusters", flush=True)

print("reading energies ...", flush=True)
E = load_blocks(list(df.sample_id))["energy"]
mu, sd = E[tr].mean(0), E[tr].std(0)
X = ((E - mu) / (sd + 1e-6)).astype(np.float32)
print(f"energy block {X.shape}", flush=True)

print("fitting UMAP ...", flush=True)
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, metric="euclidean",
                random_state=SEED).fit_transform(X)
np.savez(os.path.join(HERE, "umap_energy.npz"), emb=emb,
         sample_id=df.sample_id.to_numpy(), cluster=df.cluster.to_numpy(),
         allele=df.allele.to_numpy(), ly=D["ly"], split=df["split_C2"].to_numpy())

fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))

a = ax[0][0]
cl = df.cluster.to_numpy()
s = a.scatter(emb[:, 0], emb[:, 1], c=cl, cmap="tab20", s=3, alpha=0.6, lw=0)
for c in np.unique(cl):                       # label each cluster at its centre
    m = cl == c
    a.annotate(str(c), emb[m].mean(0), fontsize=9, fontweight="bold",
               ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0))
a.set_title(f"pseudosequence cluster ({len(np.unique(cl))})", fontsize=11)

a = ax[0][1]
al = df.allele.to_numpy()
codes = {v: i for i, v in enumerate(sorted(set(al)))}
a.scatter(emb[:, 0], emb[:, 1], c=[codes[v] for v in al], cmap="gist_ncar",
          s=3, alpha=0.6, lw=0)
a.set_title(f"allele ({len(codes)})", fontsize=11)

a = ax[1][0]
s = a.scatter(emb[:, 0], emb[:, 1], c=D["ly"], cmap="viridis", s=3,
              alpha=0.6, lw=0)
fig.colorbar(s, ax=a, label="log10 half-life (h)")
a.set_title("log10 half-life -- the target", fontsize=11)

a = ax[1][1]
for name, col in (("train", "#cccccc"), ("val", "#1f77b4"), ("test", "#d62728")):
    m = df["split_C2"].to_numpy() == name
    a.scatter(emb[m, 0], emb[m, 1], c=col, s=3, alpha=0.6, lw=0, label=name)
a.legend(markerscale=4, fontsize=9)
a.set_title("split_C2 fold (clusters held out)", fontsize=11)

for a in ax.ravel():
    a.set_xticks([])
    a.set_yticks([])
fig.suptitle(f"UMAP of MACE site energies  |  {len(df):,} complexes, "
             f"{X.shape[1]} features  |  n_neighbors=30, min_dist=0.1",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_umap_energy.png"), dpi=150)
print("wrote fig_umap_energy.png")
