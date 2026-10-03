"""Per-allele half-life distributions as small multiples.

Panels are ordered by median half-life, so the grid reads as a gradient from
alleles that bind almost nothing to alleles that hold peptides for days.

Each panel shows, on a shared log x-axis, the distribution of non-zero
half-lives for that allele, plus a separate grey bar for the censored zeros
(which cannot go on a log scale). The y-axis is the fraction of that allele's
own measurements, so panels are comparable despite counts ranging 7 to 1070.
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
df = pd.read_csv(CSV)

INK, MUTED = "#1f2328", "#6b7280"
BAR, ZERO = "#4a7fb5", "#9aa5b1"

plt.rcParams.update({
    "figure.dpi": 130, "savefig.dpi": 130, "font.size": 8,
    "axes.edgecolor": "#d0d7de", "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})

t_all = df["thalf_hours"]
nz_all = t_all[t_all > 0]
LO, HI = np.log10(nz_all.min()), np.log10(nz_all.max())
BINS = np.linspace(LO, HI, 26)
ZERO_X = LO - 0.45          # sentinel slot for the censored zeros
ZERO_W = 0.30

# Order panels by median half-life (zeros included, so heavy-zero alleles sink).
order = df.groupby("allele").thalf_hours.median().sort_values().index.tolist()
stats = df.groupby("allele").thalf_hours.agg(
    n="size", med="median", zero=lambda s: (s == 0).mean())

NC = 8
NR = int(np.ceil(len(order) / NC))
fig, axes = plt.subplots(NR, NC, figsize=(19, 2.05 * NR), sharex=True, sharey=True)
axes = axes.ravel()

for ax, allele in zip(axes, order):
    sub = df.loc[df.allele == allele, "thalf_hours"].to_numpy()
    n = len(sub)
    nz, nzero = sub[sub > 0], int((sub == 0).sum())

    if len(nz):
        ax.hist(np.log10(nz), bins=BINS, weights=np.full(len(nz), 1 / n),
                color=BAR, edgecolor="white", linewidth=0.3)
    if nzero:
        ax.bar(ZERO_X, nzero / n, width=ZERO_W, color=ZERO,
               edgecolor="white", linewidth=0.3)

    med = stats.med[allele]
    ax.set_title(f"{allele}", fontsize=7.5, pad=3, weight="bold")
    ax.text(0.98, 0.93, f"n={n:,}\nmed {med:.1f} h\n{100*stats.zero[allele]:.0f}% zero",
            transform=ax.transAxes, ha="right", va="top", fontsize=6, color=MUTED,
            linespacing=1.25)
    ax.tick_params(labelsize=6, length=2)
    ax.grid(axis="y", color="#eaeef2", linewidth=0.6)
    ax.set_axisbelow(True)

for ax in axes[len(order):]:
    ax.axis("off")

# One shared, readable x-axis: decade ticks plus the zero slot.
ticks = [ZERO_X] + [d for d in range(-1, 3)]
labels = ["0"] + [f"$10^{{{d}}}$" for d in range(-1, 3)]
for ax in axes[:len(order)]:
    ax.set_xlim(ZERO_X - 0.35, HI + 0.1)
    ax.set_xticks(ticks, labels)

fig.suptitle(
    "Half-life distribution per HLA allele — ordered by median, log scale, "
    "censored zeros as the grey bar at left",
    fontsize=13, weight="bold", y=0.998)
fig.supxlabel("complex half-life, $t_{1/2}$ (hours)", fontsize=10, y=0.004)
fig.supylabel("fraction of that allele's measurements", fontsize=10, x=0.006)
fig.tight_layout(rect=[0.012, 0.012, 1, 0.985])
out = os.path.join(PLOTS, "fig4_halflife_by_allele.png")
fig.savefig(out, bbox_inches="tight", facecolor="white")
plt.close(fig)

print(f"{len(order)} alleles plotted, ordered by median half-life")
print(f"  lowest  median: {order[0]} ({stats.med[order[0]]:.2f} h, "
      f"{100*stats.zero[order[0]]:.0f}% zeros)")
print(f"  highest median: {order[-1]} ({stats.med[order[-1]]:.2f} h, "
      f"{100*stats.zero[order[-1]]:.0f}% zeros)")
print(f"\nwrote {out}")
