"""Exploratory plots for the Rasmussen et al. peptide-HLA stability dataset.

Produces:
  fig1_allele_composition.png  - pie (top 15 + Other) and sorted bar of allele counts
  fig2_halflife_distribution.png - histogram of thalf_hours with cumulative density overlay
  fig3_halflife_log.png        - log-scale view, which the skew really demands
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

OUT = PLOTS

INK, MUTED = "#1f2328", "#6b7280"
BAR, LINE = "#4a7fb5", "#d97706"       # blue bars, amber cumulative line

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.size": 9, "axes.labelcolor": INK, "text.color": INK,
    "axes.edgecolor": "#d0d7de", "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#eaeef2", "grid.linewidth": 0.8,
    "axes.axisbelow": True,
})

df = pd.read_csv(CSV)
t = df["thalf_hours"].to_numpy()
n = len(df)

# ---------------------------------------------------------------- figure 1
# 75 alleles is far too many for a pie, so collapse the tail into "Other".
# Slices are ordered by size and coloured with a sequential ramp: the colour
# encodes magnitude (which is ordered), not identity (which would need 75 hues).
TOP = 15
vc = df["allele"].value_counts()
top, other = vc.head(TOP), vc.iloc[TOP:].sum()
labels = list(top.index) + [f"Other ({len(vc) - TOP} alleles)"]
sizes = list(top.values) + [other]
colors = [plt.cm.Blues(x) for x in np.linspace(0.80, 0.30, TOP)] + ["#c9d1d9"]

fig, (axp, axb) = plt.subplots(1, 2, figsize=(14, 7.5))

wedges, _ = axp.pie(
    sizes, colors=colors, startangle=90, counterclock=False,
    wedgeprops={"edgecolor": "white", "linewidth": 1.5},
)
axp.legend(
    wedges, [f"{l}  ({s:,} | {100*s/n:.1f}%)" for l, s in zip(labels, sizes)],
    loc="center left", bbox_to_anchor=(-0.28, 0.5), frameon=False, fontsize=8,
)
axp.set_title(f"Allele composition — {len(vc)} alleles, top {TOP} shown",
              fontsize=11, weight="bold", pad=16)

# The honest version of the same data.
ally = vc.head(25).iloc[::-1]
axb.barh(range(len(ally)), ally.values, color=BAR, height=0.72)
axb.set_yticks(range(len(ally)), ally.index, fontsize=7.5)
axb.set_xlabel("measurements")
axb.set_title("Top 25 alleles by count", fontsize=11, weight="bold", pad=16)
for i, v in enumerate(ally.values):
    axb.text(v + 12, i, f"{v:,}", va="center", fontsize=7, color=MUTED)
axb.set_xlim(0, ally.max() * 1.12)

fig.suptitle(
    f"Rasmussen et al. peptide-HLA stability dataset — {n:,} measurements",
    fontsize=13, weight="bold", y=0.985)
fig.tight_layout(rect=[0, 0.02, 1, 0.96])
fig.savefig(f"{OUT}/fig1_allele_composition.png", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure 2
# Histogram + cumulative density. The distribution is extremely right-skewed
# (median 1.1 h, max 257 h), so the x-axis is clipped at the 99th percentile
# and everything beyond is pooled into a labelled overflow bin.
XMAX = 60.0
BW = 1.0
n_over = int((t > XMAX).sum())

fig, ax = plt.subplots(figsize=(11, 6))
bins = np.arange(0, XMAX + BW, BW)
counts, edges, patches = ax.hist(np.clip(t, 0, XMAX), bins=bins, color=BAR,
                                 edgecolor="white", linewidth=0.4)
patches[-1].set_facecolor("#9aa5b1")       # overflow bin reads as different

ax.set_xlabel("complex half-life, $t_{1/2}$ (hours)")
ax.set_ylabel("measurements", color=BAR)
ax.tick_params(axis="y", colors=BAR)
ax.set_xlim(0, XMAX)

# Cumulative density, computed on the FULL data so the integral is honest.
xs = np.sort(t)
cdf = np.arange(1, n + 1) / n
ax2 = ax.twinx()
ax2.plot(xs, cdf, color=LINE, linewidth=2.2, zorder=5)
ax2.set_ylabel("cumulative fraction of dataset", color=LINE)
ax2.tick_params(axis="y", colors=LINE)
ax2.set_ylim(0, 1.02)
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(LINE)

# Mark the quartiles where the cumulative line crosses them.
for q, lbl, off in [(0.5, "median", (12, -16)), (0.9, "90th pct", (12, -20))]:
    xq = float(np.quantile(t, q))
    ax2.plot([xq], [q], "o", color=LINE, ms=7, mec="white", mew=1.5, zorder=6)
    ax2.annotate(f"{lbl} {xq:.1f} h", (xq, q), textcoords="offset points",
                 xytext=off, fontsize=8, color=LINE)

# Placed in the empty mid-right of the panel: the bars hug the left edge and
# the cumulative line runs along the top, so this is the one clear region.
frac0 = (t == 0).mean()
ax.annotate(
    f"{(t == 0).sum():,} values ({100*frac0:.0f}%) are exactly 0 h\n"
    f"— non-binders / below detection limit",
    xy=(0.5, counts[0] * 0.55), xytext=(24, 6400), fontsize=8.5,
    color=INK, ha="left",
    arrowprops=dict(arrowstyle="->", color=MUTED, lw=1,
                    connectionstyle="arc3,rad=0.15"))
ax.annotate(f"final bin pools {n_over:,} values > {XMAX:.0f} h (max {t.max():.0f} h)",
            xy=(XMAX - 0.5, counts[-1]), xytext=(-12, 60),
            textcoords="offset points", ha="right", fontsize=8, color=MUTED,
            arrowprops=dict(arrowstyle="->", color=MUTED, lw=1))

ax.set_title("Distribution of peptide-HLA complex half-life, with cumulative density",
             fontsize=12, weight="bold", pad=14)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_halflife_distribution.png", bbox_inches="tight",
            facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- figure 3
# The skew is severe enough that a log axis shows the structure far better.
# Zeros cannot go on a log axis, so they get their own bar, kept visually apart.
nz = t[t > 0]
fig, ax = plt.subplots(figsize=(11, 6))
lb = np.logspace(np.log10(nz.min()), np.log10(nz.max()), 45)
ax.hist(nz, bins=lb, color=BAR, edgecolor="white", linewidth=0.4)
ax.set_xscale("log")
ax.set_xlabel("complex half-life, $t_{1/2}$ (hours, log scale)")
ax.set_ylabel("measurements", color=BAR)
ax.tick_params(axis="y", colors=BAR)

ax2 = ax.twinx()
xs_nz = np.sort(nz)
ax2.plot(xs_nz, np.arange(1, len(nz) + 1) / len(nz), color=LINE, linewidth=2.2)
ax2.set_ylabel("cumulative fraction (non-zero only)", color=LINE)
ax2.tick_params(axis="y", colors=LINE)
ax2.set_ylim(0, 1.02)
ax2.grid(False)
ax2.spines["right"].set_visible(True)
ax2.spines["right"].set_color(LINE)

ax.set_title(f"Half-life on a log scale — {len(nz):,} non-zero values "
             f"({(t == 0).sum():,} zeros excluded)",
             fontsize=12, weight="bold", pad=14)
fig.tight_layout()
fig.savefig(f"{OUT}/fig3_halflife_log.png", bbox_inches="tight", facecolor="white")
plt.close(fig)

# ---------------------------------------------------------------- summary
print(f"rows                {n:,}")
print(f"alleles             {len(vc)}")
print(f"top-15 share        {100*top.sum()/n:.1f}%")
print(f"zeros               {(t == 0).sum():,} ({100*frac0:.1f}%)")
print(f"median / mean       {np.median(t):.2f} h / {t.mean():.2f} h")
print(f"max                 {t.max():.1f} h")
print(f"> {XMAX:.0f} h (overflow)   {n_over:,} ({100*n_over/n:.2f}%)")
print("\nwrote fig1_allele_composition.png, fig2_halflife_distribution.png, "
      "fig3_halflife_log.png")
