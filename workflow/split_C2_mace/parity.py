"""Per-run parity plots, drawn from the predictions each run folder saved.

One panel per fold, so the gap between a memorised training fold and the held
out test fold is visible in the same figure rather than inferred from a table.
Axis limits come from the measured range; a prediction outside it is clipped to
the edge and drawn as a hollow red square with the count stated, so one wild
value cannot silently flatten the scale.

    python parity.py            # (re)draw for every run folder under runs/
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

FOLDS = ("train", "val", "test")


def parity_figure(path, pred, true, title):
    """pred/true: dicts keyed by fold name, half-life in hours (pred is log10)."""
    allv = np.concatenate([t[t > 0] for t in true.values()])
    lo, hi = np.log10(allv.min()), np.log10(allv.max())
    pad = 0.1 * (hi - lo)
    lim = (lo - pad, hi + pad)

    fig, ax = plt.subplots(1, len(FOLDS), figsize=(4.0 * len(FOLDS), 4.1))
    for a, k in zip(np.atleast_1d(ax), FOLDS):
        p, y = pred[k], true[k]
        t = np.log10(np.where(y > 0, y, 10 ** lo))
        off = (p < lim[0]) | (p > lim[1])
        a.plot(lim, lim, "k--", lw=0.8, zorder=1)
        a.scatter(t[~off], p[~off], s=5, alpha=0.25, lw=0, zorder=2)
        if off.any():
            a.scatter(t[off], np.clip(p[off], *lim), s=16, facecolors="none",
                      edgecolors="red", lw=0.7, zorder=3)
        rho = spearmanr(p, y).statistic
        a.set_xlim(*lim)
        a.set_ylim(*lim)
        a.set_aspect("equal")
        a.grid(alpha=0.2, lw=0.5)
        a.set_xlabel("measured log10 half-life (h)", fontsize=8)
        a.set_ylabel("predicted log10 half-life (h)", fontsize=8)
        a.set_title(f"{k}   n = {len(y):,}\n" + rf"$\rho$ = {rho:+.3f}" +
                    (f"   ({off.sum()} off-scale)" if off.any() else ""),
                    fontsize=9)
        a.tick_params(labelsize=7)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def draw_run(d):
    z = np.load(os.path.join(d, "predictions.npz"))
    pred = {k: z[f"pred_{k}"] for k in FOLDS}
    true = {k: z[f"true_{k}"] for k in FOLDS}
    name = seed = None
    det = os.path.join(d, "run_details.json")
    if os.path.exists(det):
        j = json.load(open(det))
        name, seed = j["representation"], j["seed"]
    else:
        name, seed = os.path.basename(os.path.dirname(d)), os.path.basename(d)
    parity_figure(os.path.join(d, "fig_parity.png"), pred, true,
                  f"{name}   seed {seed}")
    return f"{name} seed {seed}"


if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    ds = sorted(glob.glob(os.path.join(HERE, "runs", "*", "seed*")))
    ds = [d for d in ds if os.path.exists(os.path.join(d, "predictions.npz"))]
    for d in ds:
        print("  " + draw_run(d), flush=True)
    print(f"drew {len(ds)} per-run parity plots")
