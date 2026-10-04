"""split_A (random) comparison: Boltz representations vs a sequence baseline.

Architecture, identical for every representation so only the input differs:

    Linear(n_in, 128) -> GELU -> Dropout(0.3)
    Linear(128,   32) -> GELU -> Dropout(0.3)
    Linear(32,     1)

3 weight layers, 128 and 32 hidden nodes, GELU activation. Adam, lr 1e-3,
weight decay 1e-4, batch 32, MSE on standardised log10 half-life, at most 300
epochs with early stopping on validation loss (patience 40), best-validation
weights restored. 5 seeds per representation; reported as mean +/- s.d.

Target: log10 half-life, zeros floored at half the smallest positive value.
Metric: Spearman rank correlation, which is what the brief asks for.
Comparison: paired bootstrap of the test Spearman difference against the
sequence baseline, resampling complexes (not predictions) 2,000 times.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from data_split_a import load

HERE = os.path.dirname(os.path.abspath(__file__))
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS, PATIENCE, BATCH, LR, WD, DROPOUT = 300, 40, 32, 1e-3, 1e-4, 0.3
SEEDS = [0, 1, 2, 3, 4]
NBOOT = 2000

D = load("split_A")
idx, tr, y, yz, ym, ys = D["idx"], D["tr"], D["y"], D["yz"], D["ym"], D["ys"]
std, F = D["std"], D["F"]

REPS = {
    "seq_peptide":      std(D["oh_pep"]),
    "seq_hla_pseudo":   std(D["oh_hla"]),
    "seq":              std(D["oh"]),
    "boltz_conf":       std(F["conf"]),
    "boltz_pep_mean":   std(F["pmean"]),
    "boltz_receptor":   std(F["recep"]),
    "boltz_pooled":     std(F["pooled"]),
    "boltz_pep_tokens": std(F["tok"]),
}
REPS["seq+boltz_pooled"] = np.hstack([REPS["seq"], REPS["boltz_pooled"]])
REPS["seq+boltz_pep_tokens"] = np.hstack([REPS["seq"], REPS["boltz_pep_tokens"]])
BASELINE = "seq"


class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def run(X, seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = nn.MSELoss()
    T = {k: torch.as_tensor(X[idx[k]]).to(DEV) for k in idx}
    Y = {k: torch.as_tensor(yz[idx[k]]).to(DEV) for k in idx}

    best, best_ep, bad, hist, bestw = np.inf, 0, 0, [], None
    for ep in range(1, MAX_EPOCHS + 1):
        model.train()
        perm = torch.randperm(len(tr), device=DEV)
        tot = 0.0
        for b in range(0, len(perm), BATCH):
            j = perm[b:b + BATCH]
            opt.zero_grad()
            l = lossfn(model(T["train"][j]), Y["train"][j])
            l.backward()
            opt.step()
            tot += l.item() * len(j)
        model.eval()
        with torch.no_grad():
            trl = lossfn(model(T["train"]), Y["train"]).item()
            vl = lossfn(model(T["val"]), Y["val"]).item()
        hist.append((tot / len(perm), trl, vl))
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            bestw = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break

    model.load_state_dict(bestw)
    model.eval()
    with torch.no_grad():
        pred = {k: model(T[k]).cpu().numpy() * ys + ym for k in idx}
    rho = {k: spearmanr(pred[k], y[idx[k]]).statistic for k in idx}
    return dict(epochs=len(hist), best_epoch=best_ep,
                params=sum(p.numel() for p in model.parameters()),
                hist=np.array(hist), pred=pred, rho=rho)


out, t0 = {}, time.time()
for name, X in REPS.items():
    rs = [run(X, s) for s in SEEDS]
    out[name] = rs
    r = lambda k: np.array([x["rho"][k] for x in rs])
    print(f"{name:<22} dim {X.shape[1]:>5}  params {rs[0]['params']:>9,}  "
          f"ep {np.mean([x['epochs'] for x in rs]):5.0f}  "
          f"train {r('train').mean():+.3f}  val {r('val').mean():+.3f}  "
          f"test {r('test').mean():+.3f} +/- {r('test').std():.3f}"
          f"   [{time.time()-t0:5.0f}s]", flush=True)

# ---- paired bootstrap against the sequence baseline ------------------------
te = idx["test"]
yte = y[te]
mean_pred = {n: np.mean([x["pred"]["test"] for x in rs], 0) for n, rs in out.items()}
rng = np.random.default_rng(0)
boot = rng.integers(0, len(te), size=(NBOOT, len(te)))
base = mean_pred[BASELINE]
rows = []
for n, p in mean_pred.items():
    d = np.array([spearmanr(p[b], yte[b]).statistic
                  - spearmanr(base[b], yte[b]).statistic for b in boot])
    lo, hi = np.percentile(d, [2.5, 97.5])
    r = lambda k: np.array([x["rho"][k] for x in out[n]])
    rows.append(dict(rep=n, dim=REPS[n].shape[1], params=out[n][0]["params"],
                     epochs=float(np.mean([x["epochs"] for x in out[n]])),
                     rho_train=r("train").mean(), rho_val=r("val").mean(),
                     rho_test=r("test").mean(), rho_test_sd=r("test").std(),
                     delta=d.mean(), lo=lo, hi=hi,
                     beats_sequence=bool(lo > 0)))

import pandas as pd
res = pd.DataFrame(rows).sort_values("rho_test", ascending=False)
res.to_csv(f"{HERE}/results.csv", index=False)
print()
print(res.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ---- parity plots, one panel per representation ----------------------------
ncol = 5
nrow = int(np.ceil(len(REPS) / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.2 * nrow))
lo_a, hi_a = np.log10(max(yte[yte > 0].min(), 1e-3)), np.log10(yte.max())
pad = 0.1 * (hi_a - lo_a)
lim = (lo_a - pad, hi_a + pad)
for a, (n, p) in zip(ax.ravel(), mean_pred.items()):
    t = np.log10(np.where(yte > 0, yte, 10 ** lo_a))
    off = (p < lim[0]) | (p > lim[1])
    a.plot(lim, lim, "k--", lw=0.8, zorder=1)
    a.scatter(t[~off], p[~off], s=4, alpha=0.25, lw=0, zorder=2)
    if off.any():
        a.scatter(t[off], np.clip(p[off], *lim), s=14, facecolors="none",
                  edgecolors="red", lw=0.7, zorder=3)
    rho = np.mean([x["rho"]["test"] for x in out[n]])
    a.set_title(f"{n}\n" + rf"$\rho$ = {rho:+.3f}" +
                (f"  ({off.sum()} off-scale)" if off.any() else ""), fontsize=8)
    a.set_xlim(*lim)
    a.set_ylim(*lim)
    a.set_xlabel("measured log10 t1/2 (h)", fontsize=7)
    a.set_ylabel("predicted", fontsize=7)
    a.tick_params(labelsize=6)
for a in ax.ravel()[len(REPS):]:
    a.axis("off")
fig.suptitle(f"split_A (random, leaky) | n_test = {len(te):,} | "
             f"{D['df'].allele.nunique()} alleles | mean of {len(SEEDS)} seeds",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_parity.png", dpi=150)

# ---- training curves -------------------------------------------------------
fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.8 * nrow), sharex=True)
for a, (n, rs) in zip(ax.ravel(), out.items()):
    h = rs[0]["hist"]
    a.plot(h[:, 1], label="train", lw=1)
    a.plot(h[:, 2], label="val", lw=1)
    a.axvline(rs[0]["best_epoch"] - 1, color="k", ls=":", lw=0.8)
    a.set_title(f"{n}  ({rs[0]['epochs']} ep)", fontsize=8)
    a.set_yscale("log")
    a.tick_params(labelsize=6)
    a.set_xlabel("epoch", fontsize=7)
    a.set_ylabel("MSE", fontsize=7)
ax.ravel()[0].legend(fontsize=6)
for a in ax.ravel()[len(REPS):]:
    a.axis("off")
fig.suptitle("training curves, seed 0 (dotted = restored best-validation epoch)",
             fontsize=10)
fig.tight_layout()
fig.savefig(f"{HERE}/fig_training.png", dpi=150)

json.dump({"n": len(D["df"]), "n_alleles": int(D["df"].allele.nunique()),
           "n_train": len(tr), "n_val": len(idx["val"]), "n_test": len(te),
           "not_yet_predicted": D["n_missing"], "split": "split_A",
           "seeds": SEEDS, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
           "batch": BATCH, "lr": LR, "weight_decay": WD, "dropout": DROPOUT,
           "arch": "Linear(n,128) GELU Drop Linear(128,32) GELU Drop Linear(32,1)",
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(f"{HERE}/run_config.json", "w"), indent=2)
np.savez_compressed(f"{HERE}/test_predictions.npz",
                    y_test=yte, **{n: p for n, p in mean_pred.items()})
print(f"\nwrote results.csv, fig_parity.png, fig_training.png in "
      f"{(time.time()-t0)/60:.1f} min")
