"""Collect every run folder into one parity grid and one Spearman table.

Reads only what the runs already wrote (predictions.npz, run_details.json), so
it can be re-run at any time without retraining.

    runs/fig_parity_all.png   one row per representation, one column per fold
    runs/summary_table.txt    train / val / test Spearman, per representation
    runs/summary_table.csv    the same, machine readable
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FOLDS = ("train", "val", "test")
BASELINE = "seq"

# ---- gather ---------------------------------------------------------------
reps = {}
for d in sorted(glob.glob(os.path.join(RUNS, "*", "seed*"))):
    f = os.path.join(d, "predictions.npz")
    j = os.path.join(d, "run_details.json")
    if not (os.path.exists(f) and os.path.exists(j)):
        continue
    det = json.load(open(j))
    z = np.load(f)
    reps.setdefault(det["representation"], {"det": det, "seeds": []})
    reps[det["representation"]]["seeds"].append(
        {"pred": {k: z[f"pred_{k}"] for k in FOLDS},
         "true": {k: z[f"true_{k}"] for k in FOLDS},
         "rho": det["spearman"], "epochs": det["epochs_run"],
         "restored": det["restored_epoch"], "seed": det["seed"]})

order = sorted(reps, key=lambda n: -np.mean([s["rho"]["test"]
                                             for s in reps[n]["seeds"]]))

# ---- table ----------------------------------------------------------------
rows = []
for n in order:
    R = reps[n]
    g = lambda k: np.array([s["rho"][k] for s in R["seeds"]])
    rows.append(dict(
        rep=n, dim=R["det"]["input_dim"], params=R["det"]["n_parameters"],
        n_seeds=len(R["seeds"]),
        epochs=float(np.mean([s["epochs"] for s in R["seeds"]])),
        restored=float(np.mean([s["restored"] for s in R["seeds"]])),
        rho_train=g("train").mean(), rho_train_sd=g("train").std(),
        rho_val=g("val").mean(), rho_val_sd=g("val").std(),
        rho_test=g("test").mean(), rho_test_sd=g("test").std(),
        gap_train_test=g("train").mean() - g("test").mean(),
        **{f"rho_{k}_ens": float(spearmanr(
            np.mean([s["pred"][k] for s in R["seeds"]], 0),
            R["seeds"][0]["true"][k]).statistic) for k in FOLDS},
        description=R["det"]["description"]))
tab = pd.DataFrame(rows)
tab.to_csv(os.path.join(RUNS, "summary_table.csv"), index=False)

det0 = reps[order[0]]["det"]
ds = det0["dataset"]
head = (
    f"# split_A comparison, {ds['n_alleles']} alleles, {ds['n_clusters']} clusters\n"
    f"# n = {ds['n_total']:,}   train {ds['n_train']:,}  val {ds['n_val']:,}  "
    f"test {ds['n_test']:,}   ({ds['n_not_yet_predicted']:,} not yet predicted)\n"
    f"# {det0['architecture']}\n"
    f"# Adam lr={det0['lr']} weight_decay={det0['weight_decay']} "
    f"batch={det0['batch_size']} dropout={det0['dropout']}, "
    f"early stopping patience {det0['early_stopping_patience']}\n"
    f"# Spearman rank correlation, mean +/- s.d. over seeds; "
    f"'ep' = epochs run, 'best' = restored epoch\n"
    f"# test(ens) averages the 5 seeds' PREDICTIONS first, then correlates: an\n"
    f"#   ensemble, so it is systematically higher than the mean of the single\n"
    f"#   seed values. The parity grid plots the ensemble; compare like with like.\n"
    f"# WARNING: {ds['split_note']}\n#\n")

lines = [head]
w = max(len(n) for n in order)
lines.append(f"{'representation':<{w}}  {'dim':>5} {'params':>9} {'ep':>5} "
             f"{'best':>5}   {'train':>15} {'val':>15} {'test':>15}"
             f" {'test(ens)':>10}\n")
for r in tab.itertuples():
    lines.append(
        f"{r.rep:<{w}}  {r.dim:>5} {r.params:>9,} {r.epochs:>5.0f} "
        f"{r.restored:>5.0f}   "
        f"{r.rho_train:>+8.3f} +/-{r.rho_train_sd:5.3f} "
        f"{r.rho_val:>+8.3f} +/-{r.rho_val_sd:5.3f} "
        f"{r.rho_test:>+8.3f} +/-{r.rho_test_sd:5.3f}"
        f" {r.rho_test_ens:>+10.3f}\n")

bs = os.path.join(RUNS, "_summary", "results.csv")
if os.path.exists(bs):
    b = pd.read_csv(bs).set_index("rep")
    lines.append(f"\n# paired bootstrap of the test Spearman against "
                 f"'{BASELINE}', 2,000 resamples of complexes\n")
    lines.append(f"{'representation':<{w}}  {'delta':>8} {'95% interval':>20}"
                 f"   beats baseline\n")
    for r in tab.itertuples():
        if r.rep not in b.index:
            continue
        q = b.loc[r.rep]
        lines.append(f"{r.rep:<{w}}  {q.delta_vs_seq:>+8.3f} "
                     f"{'[%+.3f, %+.3f]' % (q.ci_lo, q.ci_hi):>20}   "
                     f"{'YES' if q.beats_sequence else 'no'}\n")

txt = "".join(lines)
open(os.path.join(RUNS, "summary_table.txt"), "w").write(txt)
print(txt)

# ---- one parity grid: representations down, folds across -------------------
nrow, ncol = len(order), len(FOLDS)
fig, ax = plt.subplots(nrow, ncol, figsize=(3.05 * ncol, 2.95 * nrow),
                       squeeze=False)
allv = np.concatenate([s["true"]["test"] for s in reps[order[0]]["seeds"][:1]]
                      + [s["true"]["train"] for s in reps[order[0]]["seeds"][:1]]
                      + [s["true"]["val"] for s in reps[order[0]]["seeds"][:1]])
lo, hi = np.log10(allv[allv > 0].min()), np.log10(allv.max())
pad = 0.1 * (hi - lo)
lim = (lo - pad, hi + pad)

for i, n in enumerate(order):
    S = reps[n]["seeds"]
    for j, k in enumerate(FOLDS):
        a = ax[i][j]
        p = np.mean([s["pred"][k] for s in S], 0)     # seed-averaged
        y = S[0]["true"][k]
        t = np.log10(np.where(y > 0, y, 10 ** lo))
        off = (p < lim[0]) | (p > lim[1])
        a.plot(lim, lim, "k--", lw=0.8, zorder=1)
        a.scatter(t[~off], p[~off], s=3.5, alpha=0.22, lw=0, zorder=2)
        if off.any():
            a.scatter(t[off], np.clip(p[off], *lim), s=14, facecolors="none",
                      edgecolors="red", lw=0.6, zorder=3)
        rho = spearmanr(p, y).statistic
        a.set_xlim(*lim); a.set_ylim(*lim)
        a.set_aspect("equal")
        a.grid(alpha=0.18, lw=0.4)
        a.tick_params(labelsize=6)
        a.set_title(rf"$\rho_{{ens}}$ = {rho:+.3f}" +
                    (f"  ({off.sum()} off)" if off.any() else ""), fontsize=8)
        if i == 0:
            a.text(0.5, 1.30, k, transform=a.transAxes, ha="center",
                   fontsize=11, fontweight="bold")
        if j == 0:
            a.set_ylabel(f"{n}\n({reps[n]['det']['input_dim']} inputs)"
                         "\npredicted log10 t1/2", fontsize=7)
        if i == nrow - 1:
            a.set_xlabel("measured log10 t1/2 (h)", fontsize=7)
fig.suptitle(f"split_A (random, leaky) | {ds['n_alleles']} alleles | "
             f"train {ds['n_train']:,} / val {ds['n_val']:,} / "
             f"test {ds['n_test']:,} | 5-seed ENSEMBLE predictions, "
             f"ordered by single-seed mean test rho",
             fontsize=11, y=0.998)
fig.tight_layout(rect=[0, 0, 1, 0.985])
fig.savefig(os.path.join(RUNS, "fig_parity_all.png"), dpi=140)
print(f"wrote {RUNS}/fig_parity_all.png  ({nrow} x {ncol} panels)")
