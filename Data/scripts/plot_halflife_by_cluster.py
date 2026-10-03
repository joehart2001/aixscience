"""Half-life distributions per HLA pseudosequence cluster.

Alleles are grouped by single-linkage clustering on the 34-residue NetMHCpan
pseudosequence at >= IDENTITY_MIN / 34 identity. The pseudosequence is the set
of peptide-contacting positions, so two alleles in one cluster present a nearly
identical binding groove and must never be split across train/test.

Writes:
  fig5_halflife_by_cluster.png  - small multiples, one panel per cluster
  pseudoseq_clusters.csv        - allele -> cluster membership, for making splits
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)                 # Data/
CSV = os.path.join(DATA, "rasmussen_et_al_dataset.csv")
PLOTS = os.path.join(DATA, "plots")
WEIGHTS = os.path.join(DATA, "weights")
SUBSETS = os.path.join(DATA, "subsets")
IDENTITY_MIN = 30          # of 34 pseudosequence positions

df = pd.read_csv(CSV)

INK, MUTED = "#1f2328", "#6b7280"
BAR, ZERO = "#4a7fb5", "#9aa5b1"
plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8,
    "axes.edgecolor": "#d0d7de", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

# ---- cluster alleles on pseudosequence identity (single linkage, union-find)
al = df.drop_duplicates("allele")[["allele", "hla_pseudoseq"]].reset_index(drop=True)
P = np.frombuffer("".join(al.hla_pseudoseq).encode(), dtype="S1").reshape(len(al), 34)
sim = (P[:, None, :] == P[None, :, :]).sum(-1)

parent = list(range(len(al)))
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x
def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

for i in range(len(al)):
    for j in range(i + 1, len(al)):
        if sim[i, j] >= IDENTITY_MIN:
            union(i, j)

al["root"] = [find(i) for i in range(len(al))]
roots = {r: k for k, r in enumerate(sorted(al.root.unique()))}
al["cluster"] = al.root.map(roots)

df = df.merge(al[["allele", "cluster"]], on="allele", how="left")

# ---- summarise and persist membership -----------------------------------
memb = (al.groupby("cluster").allele.apply(lambda s: ",".join(sorted(s)))
        .rename("alleles").to_frame())
memb["n_alleles"] = al.groupby("cluster").size()
memb["n_rows"] = df.groupby("cluster").size()
memb["median_thalf"] = df.groupby("cluster").thalf_hours.median()
memb["pct_zero"] = (100 * df.groupby("cluster").thalf_hours.apply(lambda s: (s == 0).mean())).round(1)
memb["pct_of_data"] = (100 * memb.n_rows / len(df)).round(2)
memb.sort_values("median_thalf").to_csv(os.path.join(SUBSETS, "pseudoseq_clusters.csv"))

# ---- plot ----------------------------------------------------------------
t_all = df["thalf_hours"]
nz_all = t_all[t_all > 0]
LO, HI = np.log10(nz_all.min()), np.log10(nz_all.max())
BINS = np.linspace(LO, HI, 30)
ZERO_X, ZERO_W = LO - 0.45, 0.30

order = df.groupby("cluster").thalf_hours.median().sort_values().index.tolist()
NC = 5
NR = int(np.ceil(len(order) / NC))
fig, axes = plt.subplots(NR, NC, figsize=(17, 2.7 * NR), sharex=True, sharey=True)
axes = axes.ravel()

for ax, c in zip(axes, order):
    sub = df.loc[df.cluster == c, "thalf_hours"].to_numpy()
    n = len(sub)
    nz, nzero = sub[sub > 0], int((sub == 0).sum())
    if len(nz):
        ax.hist(np.log10(nz), bins=BINS, weights=np.full(len(nz), 1 / n),
                color=BAR, edgecolor="white", linewidth=0.3)
    if nzero:
        ax.bar(ZERO_X, nzero / n, width=ZERO_W, color=ZERO,
               edgecolor="white", linewidth=0.3)

    names = memb.alleles[c].split(",")
    shown = ", ".join(a.replace("HLA-", "") for a in names[:3])
    if len(names) > 3:
        shown += f", +{len(names)-3} more"
    ax.set_title(f"cluster {c}  ({memb.n_alleles[c]} allele"
                 f"{'s' if memb.n_alleles[c] > 1 else ''})",
                 fontsize=8.5, pad=4, weight="bold")
    ax.text(0.5, 1.005, shown, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6, color=MUTED)
    ax.text(0.98, 0.95, f"n={n:,}  ({memb.pct_of_data[c]:.1f}% of data)\n"
                        f"med {memb.median_thalf[c]:.1f} h\n"
                        f"{memb.pct_zero[c]:.0f}% zero",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.5,
            color=MUTED, linespacing=1.3)
    ax.tick_params(labelsize=6.5, length=2)
    ax.grid(axis="y", color="#eaeef2", linewidth=0.6)
    ax.set_axisbelow(True)

for ax in axes[len(order):]:
    ax.axis("off")

ticks = [ZERO_X] + list(range(-1, 3))
labels = ["0"] + [f"$10^{{{d}}}$" for d in range(-1, 3)]
for ax in axes[:len(order)]:
    ax.set_xlim(ZERO_X - 0.35, HI + 0.1)
    ax.set_xticks(ticks, labels)

fig.suptitle(f"Half-life distribution per HLA pseudosequence cluster "
             f"(single linkage at >= {IDENTITY_MIN}/34 identity) — "
             f"ordered by median, censored zeros as the grey bar",
             fontsize=12.5, weight="bold", y=0.997)
fig.supxlabel("complex half-life, $t_{1/2}$ (hours)", fontsize=10, y=0.004)
fig.supylabel("fraction of that cluster's measurements", fontsize=10, x=0.007)
fig.tight_layout(rect=[0.014, 0.014, 1, 0.982])
fig.savefig(os.path.join(PLOTS, "fig5_halflife_by_cluster.png"),
            bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"{len(al)} alleles -> {len(order)} clusters at >= {IDENTITY_MIN}/34 identity\n")
print(f"{'clu':>3} {'alle':>4} {'rows':>6} {'%data':>6} {'med h':>6} {'%zero':>6}  alleles")
print("-" * 92)
for c in order:
    nm = memb.alleles[c].split(",")
    s = ", ".join(a.replace("HLA-", "") for a in nm[:4]) + (f", +{len(nm)-4}" if len(nm) > 4 else "")
    print(f"{c:>3} {memb.n_alleles[c]:>4} {memb.n_rows[c]:>6,} {memb.pct_of_data[c]:>5.1f}% "
          f"{memb.median_thalf[c]:>6.1f} {memb.pct_zero[c]:>5.1f}%  {s}")
print("-" * 92)
print("wrote fig5_halflife_by_cluster.png and pseudoseq_clusters.csv")
