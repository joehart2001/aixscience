"""Every molecular-mechanics feature against half-life, with Spearman.

DESCRIPTIVE ONLY. These are in-sample rank correlations over 24 features on
n = 3,134, computed with no split and no model. Screening features this way is
what produced this project's retracted "P2 anchor" result (in sample
rho = -0.233, p = 0.020; cross-validated +0.015, p = 0.882). Nothing here is
evidence that anything predicts stability; it is a look at the raw relationships
before any model is fitted.

Three families are shown, separated by panel colour:
  Hessian      curvature of the energy against rigid peptide translation, now
               at a genuine stationary point (the peptide was relaxed in
               translation first, 3 Newton steps, residual force ~0.002 eV/A)
  relaxation   how far the peptide had to slide to get there, which measures
               how far the predicted pose was from an MM minimum
  GBSA         the Generalized Born decomposition: vacuum interaction energy,
               solvation penalty, and the screened total

Axis limits come from the 0.5-99.5 percentiles; points outside are clipped to
the edge and drawn as hollow red squares with the count stated, so the three
steric-clash outliers (0.1% of the set) cannot flatten a panel.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
       "008-mace-all75/mmgbsa")
FAMILY = {**{k: "Hessian" for k in
             ["H_ee", "H_ee_inv", "cos_soft", "cos_stiff", "lam_soft",
              "lam_mid", "lam_stiff", "anisotropy", "trace", "det", "log_det",
              "n_negative", "depth", "burial"]},
          **{k: "relaxation" for k in
             ["slide_mag", "slide_e", "force_before_relax", "net_force_mag",
              "newton_iters"]},
          **{k: "GBSA" for k in
             ["dE_gb", "dE_vac", "dG_solv", "E_bound_gb", "E_sep_gb"]}}
COLOUR = {"Hessian": "#1f77b4", "relaxation": "#7f7f7f", "GBSA": "#2ca02c"}

z = np.load(os.path.join(HERE, "mm_features.npz"), allow_pickle=True)
X, sids, fn = z["X"], [str(s) for s in z["names"]], [str(s) for s in z["feature_names"]]

# the mmgbsa files do not carry the target, so take it from the manifest
MAN = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
       "006-modal-boltz/subset_manifest.csv")
man = pd.read_csv(MAN).set_index("sample_id")
keep = [i for i, s in enumerate(sids) if s in man.index]
if len(keep) != len(sids):
    print(f"{len(sids)-len(keep)} complexes not in the manifest, dropped")
    X = X[keep]
    sids = [sids[i] for i in keep]
sub = man.loc[sids]
y = sub.thalf_hours.to_numpy(dtype=float)
alleles = sub.allele.tolist()
ly = np.log10(np.where(y > 0, y, y[y > 0].min() / 2))
print(f"n = {len(y):,}   alleles {len(set(alleles))}   "
      f"zero half-life {(y == 0).mean()*100:.0f}%")

order = sorted(range(len(fn)), key=lambda i: (FAMILY.get(fn[i], "z"), fn[i]))
ncol = 5
nrow = int(np.ceil(len(fn) / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.95 * nrow))
rows = []
for a, i in zip(ax.ravel(), order):
    v, name = X[:, i], fn[i]
    fam = FAMILY.get(name, "other")
    r = spearmanr(v, ly).statistic
    rows.append((name, fam, r))
    lo, hi = np.percentile(v, [0.5, 99.5])
    if hi - lo < 1e-9:
        lo, hi = v.min() - 0.5, v.max() + 0.5
    pad = 0.05 * (hi - lo)
    lim = (lo - pad, hi + pad)
    off = (v < lim[0]) | (v > lim[1])
    a.scatter(v[~off], ly[~off], s=6, alpha=0.3, lw=0, color=COLOUR.get(fam, "k"))
    if off.any():
        a.scatter(np.clip(v[off], *lim), ly[off], s=16, facecolors="none",
                  edgecolors="red", lw=0.7)
    a.set_xlim(*lim)
    a.set_xlabel(f"{name}  [{fam}]", fontsize=8)
    a.set_ylabel("log10 t1/2 (h)", fontsize=7)
    a.set_title(rf"$\rho$ = {r:+.3f}" + (f"   ({off.sum()} off)" if off.any() else ""),
                fontsize=9)
    a.tick_params(labelsize=6)
    a.grid(alpha=0.2, lw=0.4)
for a in ax.ravel()[len(fn):]:
    a.axis("off")
fig.suptitle(f"Molecular-mechanics features vs half-life  |  n = {len(y):,}, "
             f"{len(set(alleles))} alleles  |  IN-SAMPLE, descriptive only",
             fontsize=12)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scatter_mm.png"), dpi=150)

t = pd.DataFrame(rows, columns=["feature", "family", "rho"])
t["abs"] = t.rho.abs()
t = t.sort_values("abs", ascending=False).drop(columns="abs")
t.to_csv(os.path.join(HERE, "scatter_spearman.csv"), index=False)
print("\nin-sample Spearman, strongest first:")
print(t.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))
print(f"\nwrote fig_scatter_mm.png  ({len(fn)} panels)")
