"""Hessian features against half-life: scatters, then cross-validated models.

TWO SEPARATE THINGS, kept apart on purpose.

1. SCATTERS. One panel per feature, with the in-sample Spearman printed. This is
   DESCRIPTIVE ONLY. Screening 16 features in sample on n = 129 is exactly the
   procedure that produced this project's retracted "P2 anchor" finding
   (rho = -0.233, p = 0.020 in sample; +0.015, p = 0.882 cross-validated).
   Nothing is concluded from these numbers.

2. MODELS. Ridge regression and gradient-boosted trees, scored on OUT-OF-FOLD
   predictions only, with peptides grouped so a peptide never appears in both
   the fitting and the scoring fold. 5 folds x 3 repeats. Feature scaling is
   fitted inside each fold. This is the number that may be believed, within the
   limits of n = 129.

With only three alleles present, the pseudosequence one-hot carries almost no
information -- it takes three distinct values and acts as an allele indicator --
so "seq" here is effectively the peptide one-hot.
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
HESS = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
        "008-mace-all75/hess_c16_hmin")
EMB = ("/share/ijp30/hackathons/aixscience2026/"
       "boltz_inputs_and_predictions/embeddings/all75")
EMB_OLD = ("/share/jh2536/hackathons/aixscience2026/"
           "boltz_inputs_and_predictions/embeddings/a0201")
CSV = "/share/ijp30/hackathons/aixscience/Data/rasmussen_et_al_dataset.csv"
AA = "ACDEFGHIKLMNPQRSTVWY"
REPEATS, FOLDS = 5, 5


def onehot(seqs, n):
    out = np.zeros((len(seqs), n * 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:n]):
            if c in AA:
                out[i, j * 20 + AA.index(c)] = 1
    return out


z = np.load(os.path.join(HERE, "escape.npz"), allow_pickle=True)
sids, Xh, fnames = list(z["names"]), z["X"], [str(s) for s in z["feature_names"]]

import pandas as pd
src = pd.read_csv(CSV)
pseudo = dict(zip(zip(src.allele, src.peptide), src.hla_pseudoseq))

y, alleles, peptides, boltz = [], [], [], []
for s in sids:
    d = np.load(os.path.join(HESS, f"{s}.npz"), allow_pickle=True)
    y.append(float(d["thalf_hours"]))
    alleles.append(str(d["allele"]))
    peptides.append(str(d["peptide"]))
    for base in (EMB, EMB_OLD):
        f = os.path.join(base, f"{s}.npz")
        if os.path.exists(f):
            boltz.append(np.load(f, allow_pickle=True)["features"])
            break
    else:
        raise SystemExit(f"no Boltz embedding for {s}")

y = np.asarray(y)
ly = np.log10(np.where(y > 0, y, y[y > 0].min() / 2))
Xb = np.asarray(boltz, dtype=np.float32)
Xs = np.hstack([onehot(peptides, 9),
                onehot([pseudo[(a, p)] for a, p in zip(alleles, peptides)], 34)])
groups = np.asarray(peptides)
print(f"n = {len(y)}   alleles {sorted(set(alleles))}   "
      f"unique peptides {len(set(peptides))}   zero half-life "
      f"{(y == 0).mean()*100:.0f}%")

# ---- 1. scatters ----------------------------------------------------------
ncol = 4
nrow = int(np.ceil(len(fnames) / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(3.3 * ncol, 2.9 * nrow))
for a, (i, n) in zip(ax.ravel(), enumerate(fnames)):
    r = spearmanr(Xh[:, i], ly).statistic
    a.scatter(Xh[:, i], ly, s=11, alpha=0.55, lw=0)
    a.set_xlabel(n, fontsize=8)
    a.set_ylabel("log10 t1/2 (h)", fontsize=7)
    a.set_title(rf"$\rho$ = {r:+.3f}", fontsize=9)
    a.tick_params(labelsize=6)
    a.grid(alpha=0.2, lw=0.4)
for a in ax.ravel()[len(fnames):]:
    a.axis("off")
fig.suptitle(f"Hessian features vs half-life  |  n = {len(y)}, "
             f"{len(set(alleles))} alleles  |  IN-SAMPLE, descriptive only",
             fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(HERE, "fig_scatter.png"), dpi=150)
print(f"wrote fig_scatter.png")

# ---- 2. cross-validated models --------------------------------------------
REPS = {"hess": Xh, "seq": Xs, "boltz": Xb,
        "hess+seq": np.hstack([Xh, Xs]),
        "hess+boltz": np.hstack([Xh, Xb]),
        "seq+boltz": np.hstack([Xs, Xb])}
MODELS = {
    "ridge": lambda: make_pipeline(StandardScaler(),
                                   RidgeCV(alphas=np.logspace(-2, 4, 25))),
    "gbm": lambda: HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
        min_samples_leaf=10, l2_regularization=1.0, early_stopping=False),
}

oof_store = {}
print(f"\nout-of-fold Spearman, peptide-grouped {FOLDS}-fold x {REPEATS} repeats")
print(f"{'representation':<12} {'dim':>6}  {'ridge':>18}  {'gbm':>18}")
for name, X in REPS.items():
    line = f"{name:<12} {X.shape[1]:>6}"
    for mname, make in MODELS.items():
        rhos, oofs = [], []
        for rep in range(REPEATS):
            # GroupKFold sorts groups internally, so repeats with it are
            # identical. Shuffle the GROUP LABELS instead, which genuinely
            # reassigns peptides to folds.
            rng = np.random.default_rng(rep)
            uniq = np.unique(groups)
            fold_of = dict(zip(uniq, rng.integers(0, FOLDS, len(uniq))))
            fold = np.array([fold_of[g] for g in groups])
            oof = np.zeros(len(y))
            for f in range(FOLDS):
                te = np.flatnonzero(fold == f)
                tr = np.flatnonzero(fold != f)
                if not len(te):
                    continue
                m = make()
                m.fit(X[tr], ly[tr])
                oof[te] = m.predict(X[te])
            rhos.append(spearmanr(oof, ly).statistic)
            oofs.append(oof)
        oof_store[(name, mname)] = np.mean(oofs, 0)
        line += f"  {np.mean(rhos):+.3f} +/- {np.std(rhos):.3f}"
    print(line, flush=True)

np.savez(os.path.join(HERE, "oof_predictions.npz"), y=y, ly=ly,
         **{f"{a}__{b}": v for (a, b), v in oof_store.items()})
print("\nwrote oof_predictions.npz")
