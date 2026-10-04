"""Static version of the held-out-allele bar chart, for the write-up.

Same data, ordering and colour rule as the interactive page: all 31 combinations
of the five feature blocks, scored by gradient-boosted trees on split_C2, where
whole pseudosequence clusters are held out so no allele in the test fold appears
in training. Bars are coloured by whether the combination contains the Boltz-2
block, because that single binary separates the 31 results with no overlap.

Codes are alphabetical. Atomic energy takes E and edge features take G, since
both words start with the same letter.

Writes fig_gbm_bars.png and fig_gbm_bars.pdf.
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
WITH_B, NO_B = "#2a78d6", "#eb6834"
INK, INK2, GRID = "#16181c", "#55585f", "#e4e3dd"

d = json.load(open(os.path.join(HERE, "gbm_bar_data.json")))
rows = sorted(d["rows"], key=lambda r: -r["test"])
codes = [r["code"] for r in rows]
test = np.array([r["test"] for r in rows])
sd = np.array([r["sd"] for r in rows])
hasB = np.array([r["hasB"] for r in rows])
cols = [WITH_B if b else NO_B for b in hasB]

fig, ax = plt.subplots(figsize=(12.4, 5.2))
x = np.arange(len(rows))
ax.bar(x, test, color=cols, width=0.78, zorder=3)
ax.errorbar(x, test, yerr=sd, fmt="none", ecolor=INK2, elinewidth=1.0,
            capsize=2.2, alpha=0.75, zorder=4)

ax.axhline(0, color=INK2, lw=1.1, zorder=2)
ax.set_yticks(np.arange(-0.3, 0.71, 0.1))
ax.set_ylim(-0.30, 0.70)
ax.set_xlim(-0.8, len(rows) - 0.2)
ax.set_xticks(x)
ax.set_xticklabels(codes, fontsize=8.5, family="monospace", color=INK2)
ax.set_ylabel("Spearman $\\rho$, held-out test fold", fontsize=11, color=INK)
ax.set_xlabel("Feature-block combination (alphabetical code), ordered by test "
              r"$\rho$", fontsize=10.5, color=INK2, labelpad=8)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=9)

for i in (0, len(rows) - 1):                      # selective direct labels only
    v = test[i]
    ax.annotate(f"{v:+.3f}", (i, v), textcoords="offset points",
                xytext=(0, 7 if v > 0 else -15), ha="center",
                fontsize=9.5, family="monospace", weight="bold", color=INK)

from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=WITH_B, label="Includes Boltz-2 embeddings"),
                   Patch(facecolor=NO_B, label="No Boltz-2 embeddings")],
          loc="upper right", frameon=False, fontsize=10.5, labelcolor=INK2)

key = ("B = Boltz-2 embeddings   S = Sequence one-hot   "
       "E = Atomic energy   N = Node features   G = Edge features")
fig.text(0.012, 0.015, key, fontsize=9, color=INK2, family="monospace")
ax.set_title("Only combinations containing Boltz-2 survive a held-out allele\n"
             "14,997 complexes, 75 alleles, 22 clusters · split_C2 · "
             "gradient-boosted trees, mean of 3 seeds · whiskers ±1 s.d.",
             fontsize=12, color=INK, loc="left", pad=12)
fig.tight_layout(rect=[0, 0.045, 1, 1])
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(HERE, f"fig_gbm_bars.{ext}"), dpi=200,
                facecolor="white")
print("wrote fig_gbm_bars.png and fig_gbm_bars.pdf")
print(f"  {len(rows)} bars, {hasB.sum()} with Boltz, {(~hasB).sum()} without")
print(f"  range {test.min():+.3f} to {test.max():+.3f}")
