"""UMAP of any feature block, with the diagnostics that make it interpretable.

    python umap_block.py energy|boltz|node|edge|seq

Four panels: pseudosequence cluster, allele, log half-life, and split_C2 fold.
Then three numbers that say what the picture means:

  neighbour purity      fraction of a point's 10 nearest neighbours sharing its
                        allele / cluster. High means the block recovers groove
                        identity, which is the failure mode this project keeps
                        hitting -- identity does not transfer to unseen alleles.
  variance split        how much of log half-life lies BETWEEN alleles versus
                        WITHIN them. Only the within part is peptide fit.
  extrapolation ratio   median embedding distance from a TEST complex to its
                        nearest TRAINING complex, over the typical train-train
                        spacing. Large means a model asked about held-out
                        clusters is extrapolating, not interpolating.

Standardised on the training fold only. Seeded, so single-threaded by design.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors

from data_split import load
from mace_blocks import have, load_blocks
import trainer

HERE = os.path.dirname(os.path.abspath(__file__))
BLOCK = sys.argv[1] if len(sys.argv) > 1 else "energy"
# Which split to colour the fold panel by, and to measure extrapolation against.
# split_C2 holds out whole pseudosequence clusters; split_B holds out peptides,
# which is the relevant one for a peptide-only representation.
SPLIT = sys.argv[2] if len(sys.argv) > 2 else "split_C2"
TAG = BLOCK if SPLIT == "split_C2" else f"{BLOCK}_{SPLIT}"
if os.environ.get("FORCE_PCA"):
    TAG += "_pca50"
SEED = 0

D = load(SPLIT)
ids = D["df"].sample_id.to_numpy()
D = trainer._subset(D, np.array([have(s) for s in ids]))
df, idx, tr = D["df"], D["idx"], D["tr"]
print(f"{len(df):,} complexes, {df.allele.nunique()} alleles, "
      f"{df.cluster.nunique()} clusters", flush=True)

if BLOCK in ("energy", "node", "edge"):
    print(f"reading MACE {BLOCK} ...", flush=True)
    E = load_blocks(list(df.sample_id))[BLOCK]
elif BLOCK == "node_pooled":
    # mean over the 45 (residue, slot) positions -> one 256-dim descriptor
    E = load_blocks(list(df.sample_id))["node"].reshape(len(df), 9, 5, 256)
    E = E.mean(axis=(1, 2))
elif BLOCK == "boltz":
    E = D["F"]["pooled"]
elif BLOCK == "peptide":
    E = D["oh_pep"]
elif BLOCK == "pseudoseq":
    E = D["oh_hla"]
elif BLOCK == "seq":
    E = D["oh"]
else:
    raise SystemExit(f"unknown block {BLOCK}")

mu, sd = E[tr].mean(0), E[tr].std(0)
X = ((E - mu) / (sd + 1e-6)).astype(np.float32)
note = ""
if X.shape[1] > 2000 or os.environ.get("FORCE_PCA"):
    # Euclidean neighbourhoods are poorly conditioned at this width, and these
    # features are heavily correlated (256-dim descriptors repeated over 9
    # residues x 5 slots), so reduce first -- UMAP's own guidance.
    pca = PCA(n_components=50, random_state=SEED).fit(X[tr])
    X = pca.transform(X).astype(np.float32)
    ev = pca.explained_variance_ratio_.sum()
    note = f" (PCA 50 first, {ev*100:.1f}% variance)"
    print(f"  reduced to {X.shape} {note}", flush=True)
print(f"{BLOCK} block {X.shape}{note}; fitting UMAP ...", flush=True)
emb = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=SEED).fit_transform(X)
np.savez(os.path.join(HERE, f"umap_{TAG}.npz"), emb=emb,
         sample_id=df.sample_id.to_numpy(), cluster=df.cluster.to_numpy(),
         allele=df.allele.to_numpy(), ly=D["ly"], split=df[SPLIT].to_numpy())

ly, al, cl = D["ly"], df.allele.to_numpy(), df.cluster.to_numpy()
sp = df[SPLIT].to_numpy()
fig, ax = plt.subplots(2, 2, figsize=(13, 11.5))

a = ax[0][0]
a.scatter(emb[:, 0], emb[:, 1], c=cl, cmap="tab20", s=3, alpha=0.6, lw=0)
for c in np.unique(cl):
    a.annotate(str(c), emb[cl == c].mean(0), fontsize=9, fontweight="bold",
               ha="center", va="center",
               bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.75, lw=0))
a.set_title(f"pseudosequence cluster ({len(np.unique(cl))})", fontsize=11)

a = ax[0][1]
codes = {v: i for i, v in enumerate(sorted(set(al)))}
a.scatter(emb[:, 0], emb[:, 1], c=[codes[v] for v in al], cmap="gist_ncar",
          s=3, alpha=0.6, lw=0)
a.set_title(f"allele ({len(codes)})", fontsize=11)

a = ax[1][0]
s = a.scatter(emb[:, 0], emb[:, 1], c=ly, cmap="viridis", s=3, alpha=0.6, lw=0)
fig.colorbar(s, ax=a, label="log10 half-life (h)")
a.set_title("log10 half-life -- the target", fontsize=11)

a = ax[1][1]
for name, col in (("train", "#cccccc"), ("val", "#1f77b4"), ("test", "#d62728")):
    m = sp == name
    a.scatter(emb[m, 0], emb[m, 1], c=col, s=3, alpha=0.6, lw=0, label=name)
a.legend(markerscale=4, fontsize=9)
a.set_title(f"{SPLIT} fold "
            + ("(clusters held out)" if SPLIT == "split_C2"
               else "(peptides held out)"), fontsize=11)
for a in ax.ravel():
    a.set_xticks([]); a.set_yticks([])
fig.suptitle(f"UMAP of the {BLOCK} block  |  {len(df):,} complexes, "
             f"{E.shape[1]:,} features{note}  |  n_neighbors=30, min_dist=0.1",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, f"fig_umap_{TAG}.png"), dpi=150)

_, ix = NearestNeighbors(n_neighbors=11).fit(emb).kneighbors(emb)
pa = np.mean([al[ix[i, 1:]] == al[i] for i in range(len(al))])
pc = np.mean([cl[ix[i, 1:]] == cl[i] for i in range(len(cl))])
gm = pd.DataFrame({"al": al, "ly": ly}).groupby("al").ly.transform("mean")
btw = np.var(gm) / np.var(ly)
trm, tem = sp == "train", sp == "test"
dte, _ = NearestNeighbors(n_neighbors=1).fit(emb[trm]).kneighbors(emb[tem])
dtr, _ = NearestNeighbors(n_neighbors=2).fit(emb[trm]).kneighbors(emb[trm])
print(f"\n{TAG}:")
print(f"  neighbour purity   allele {pa*100:5.1f}%   cluster {pc*100:5.1f}%")
print(f"  half-life variance between alleles {btw*100:5.1f}%  "
      f"within {100-btw*100:5.1f}%")
print(f"  extrapolation      test->train {np.median(dte):.3f} vs "
      f"train->train {np.median(dtr[:,1]):.3f}  "
      f"({np.median(dte)/max(np.median(dtr[:,1]),1e-9):.0f}x)")
