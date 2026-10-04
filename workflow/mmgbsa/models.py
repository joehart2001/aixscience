"""Molecular-mechanics features against the sequence and Boltz baselines.

Held-out peptides (split_B): a test 9-mer appears nowhere in training, under any
allele. The alleles themselves ARE shared, so this cannot test transfer to a new
groove -- every one of these 6,493 complexes falls in the split_C2 TRAINING fold,
so the held-out-allele question is out of reach for this subset.

Blocks
  hess   3x3 rigid-translation Hessian at a relaxed stationary point, its
         invariants, and the escape-coordinate contractions
  gbsa   Generalized Born decomposition: vacuum interaction, solvation, total
  relax  how far the peptide slid to reach the stationary point
  mm     all three together
  seq    peptide one-hot + pseudosequence one-hot
  boltz  Boltz-2 pooled trunk blocks

Ridge and gradient-boosted trees, 5 seeds for the trees. Scaling is fitted on
the training fold only. The comparison that matters is whether mm adds anything
to boltz, by paired bootstrap of the test Spearman over 2,000 resamples.
"""

import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
MAN = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
       "006-modal-boltz/subset_manifest.csv")
CSV = "/share/ijp30/hackathons/aixscience/Data/rasmussen_et_al_dataset.csv"
EMB = ("/share/ijp30/hackathons/aixscience2026/"
       "boltz_inputs_and_predictions/embeddings/all75")
EMB_OLD = ("/share/jh2536/hackathons/aixscience2026/"
           "boltz_inputs_and_predictions/embeddings/a0201")
AA = "ACDEFGHIKLMNPQRSTVWY"
SEEDS = [0, 1, 2, 3, 4]
NBOOT = 2000

HESS = ["H_ee", "H_ee_inv", "cos_soft", "cos_stiff", "lam_soft", "lam_mid",
        "lam_stiff", "anisotropy", "trace", "det", "log_det", "n_negative",
        "depth", "burial"]
RELAX = ["slide_mag", "slide_e", "force_before_relax", "net_force_mag",
         "newton_iters"]
GBSA = ["dE_gb", "dE_vac", "dG_solv", "E_bound_gb", "E_sep_gb"]


def onehot(seqs, n):
    out = np.zeros((len(seqs), n * 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:n]):
            if c in AA:
                out[i, j * 20 + AA.index(c)] = 1
    return out


z = np.load(os.path.join(HERE, "mm_features.npz"), allow_pickle=True)
sids = [str(s) for s in z["names"]]
fn = [str(s) for s in z["feature_names"]]
Xmm = z["X"]
man = pd.read_csv(MAN).set_index("sample_id")
keep = [i for i, s in enumerate(sids) if s in man.index]
Xmm, sids = Xmm[keep], [sids[i] for i in keep]
sub = man.loc[sids]

boltz = []
for s in sids:
    for b in (EMB, EMB_OLD):
        f = os.path.join(b, f"{s}.npz")
        if os.path.exists(f):
            boltz.append(np.load(f, allow_pickle=True)["features"])
            break
    else:
        raise SystemExit(f"no embedding for {s}")
Xb = np.asarray(boltz, dtype=np.float32)

src = pd.read_csv(CSV)
pseudo = dict(zip(zip(src.allele, src.peptide), src.hla_pseudoseq))
Xs = np.hstack([onehot(sub.peptide.tolist(), 9),
                onehot([pseudo[(a, p)] for a, p in
                        zip(sub.allele, sub.peptide)], 34)])

y = sub.thalf_hours.to_numpy(float)
ly = np.log10(np.where(y > 0, y, y[y > 0].min() / 2))
where = sub.split_B.to_numpy()
idx = {k: np.flatnonzero(where == k) for k in ("train", "val", "test")}
tr, te = idx["train"], idx["test"]
print(f"n = {len(y):,}   alleles {sub.allele.nunique()}   "
      f"train {len(tr):,}  val {len(idx['val']):,}  test {len(te):,}")

col = {n: i for i, n in enumerate(fn)}
take = lambda names: Xmm[:, [col[n] for n in names]]
REPS = {"hess": take(HESS), "gbsa": take(GBSA), "relax": take(RELAX),
        "mm": Xmm, "seq": Xs, "boltz": Xb}
REPS["mm+seq"] = np.hstack([Xmm, Xs])
REPS["mm+boltz"] = np.hstack([Xmm, Xb])
REPS["seq+boltz"] = np.hstack([Xs, Xb])
REPS["mm+seq+boltz"] = np.hstack([Xmm, Xs, Xb])

MODELS = {
    "ridge": lambda s: make_pipeline(
        StandardScaler(), RidgeCV(alphas=np.logspace(-2, 5, 30))),
    "gbm": lambda s: HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
        min_samples_leaf=20, l2_regularization=1.0, random_state=s,
        early_stopping=True, validation_fraction=0.15),
}

pred, rows = {}, []
for name, X in REPS.items():
    X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    r = {"rep": name, "dim": X.shape[1]}
    for mname, make in MODELS.items():
        seeds = SEEDS if mname == "gbm" else [0]
        ps = []
        for s in seeds:
            m = make(s)
            m.fit(X[tr], ly[tr])
            ps.append(m.predict(X[te]))
        p = np.mean(ps, 0)
        pred[(name, mname)] = p
        r[mname] = spearmanr(p, ly[te]).statistic
    rows.append(r)
    print(f"{name:<14} dim {r['dim']:>5}   ridge {r['ridge']:+.3f}   "
          f"gbm {r['gbm']:+.3f}", flush=True)

rng = np.random.default_rng(0)
boot = rng.integers(0, len(te), size=(NBOOT, len(te)))
print("\npaired bootstrap of the test Spearman, 2,000 resamples")
for mname in MODELS:
    base = pred[("boltz", mname)]
    print(f"  {mname}:")
    for name in ("mm", "mm+boltz", "seq+boltz", "mm+seq+boltz"):
        d = np.array([spearmanr(pred[(name, mname)][b], ly[te][b]).statistic
                      - spearmanr(base[b], ly[te][b]).statistic for b in boot])
        lo, hi = np.percentile(d, [2.5, 97.5])
        flag = "YES" if lo > 0 else ("WORSE" if hi < 0 else "ns")
        print(f"    {name:<14} vs boltz  {d.mean():+.4f}  "
              f"[{lo:+.4f}, {hi:+.4f}]  {flag}")

pd.DataFrame(rows).to_csv(os.path.join(HERE, "model_results.csv"), index=False)
np.savez_compressed(os.path.join(HERE, "test_predictions.npz"),
                    y_test=ly[te], **{f"{a}__{b}": v for (a, b), v in pred.items()})
print("\nwrote model_results.csv and test_predictions.npz")
