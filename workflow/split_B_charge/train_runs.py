"""split_B with charge descriptors added to the pooled Boltz representation.

Layout, where a "run" is one representation trained with one seed:

    runs/<representation>/seed<k>/
        train.log          per-epoch train and validation loss, human readable
        epoch_log.csv      the same numbers for plotting
        model.pt           restored best-validation weights + architecture
        run_details.json   model, optimiser, dataset and split details
        fig_loss.png       train and validation loss against epoch
        predictions.npz    train/val/test predictions and measured values
    runs/<representation>/
        summary.json       across seeds
        fig_loss_seeds.png all seeds on one axis
        fig_parity.png     seed-averaged test parity
    runs/_summary/
        results.csv        every representation, with the paired bootstrap
        fig_parity_all.png
        fig_loss_all.png

Architecture, identical for every representation so only the input differs:

    Linear(n_in, 128) -> GELU -> Dropout(0.3)
    Linear(128,   32) -> GELU -> Dropout(0.3)
    Linear(32,     1)

Adam, lr 1e-3, weight decay 1e-4, batch 32, MSE on standardised log10 half-life,
at most 300 epochs, early stopping on validation loss with patience 40, the
best-validation weights restored before any prediction is made.
"""

import json
import os
import platform
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from data_split import load
from parity import parity_figure

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS, PATIENCE, BATCH, LR, WD, DROPOUT = 300, 40, 32, 1e-3, 1e-4, 0.3
SEEDS = [0, 1, 2, 3, 4]
NBOOT = 2000
SPLIT = "split_B"

D = load(SPLIT)
idx, tr, y, yz, ym, ys = D["idx"], D["tr"], D["y"], D["yz"], D["ym"], D["ys"]
std, F, df = D["std"], D["F"], D["df"]

REPS = {
    "seq_peptide":      (std(D["oh_pep"]), "peptide one-hot, 9 x 20"),
    "seq_hla_pseudo":   (std(D["oh_hla"]), "NetMHCpan pseudosequence one-hot, 34 x 20"),
    "seq":              (std(D["oh"]),     "peptide + pseudosequence one-hot (BASELINE)"),
    "boltz_conf":       (std(F["conf"]),   "Boltz confidence metrics (ipTM, pTM, pLDDT, ...)"),
    "boltz_pep_mean":   (std(F["pmean"]),  "Boltz trunk single rep, mean over peptide tokens"),
    "boltz_receptor":   (std(F["recep"]),  "Boltz trunk single rep, mean over contacting receptor tokens"),
    "boltz_pooled":     (std(F["pooled"]), "Boltz pooled blocks, the hook's own feature vector"),
    "boltz_pep_tokens": (std(F["tok"]),    "Boltz trunk single rep, all 9 peptide tokens, flattened"),
}
REPS["charge_seq"] = (std(F["charge_seq"]),
    "sequence-derived charge: peptide per-position + summary, groove "
    "pseudosequence per-position + summary, and pair complementarity terms")
REPS["charge_struct"] = (std(F["charge_struct"]),
    "structure-derived charge: Coulomb proxy over charged centres, salt "
    "bridges, receptor charge within 8/12 A, per-position charge environment")
REPS["charge_all"] = (np.hstack([REPS["charge_seq"][0], REPS["charge_struct"][0]]),
    "both charge blocks")
REPS["seq+boltz_pooled"] = (
    np.hstack([REPS["seq"][0], REPS["boltz_pooled"][0]]),
    "sequence baseline concatenated with the Boltz pooled blocks")
REPS["seq+charge"] = (np.hstack([REPS["seq"][0], REPS["charge_all"][0]]),
    "sequence baseline plus both charge blocks: does charge add to sequence?")
REPS["boltz_pooled+charge"] = (
    np.hstack([REPS["boltz_pooled"][0], REPS["charge_all"][0]]),
    "pooled Boltz plus both charge blocks")
REPS["boltz_pooled+charge_struct"] = (
    np.hstack([REPS["boltz_pooled"][0], REPS["charge_struct"][0]]),
    "pooled Boltz plus the structure-derived charges only")
REPS["seq+boltz_pooled+charge"] = (
    np.hstack([REPS["seq"][0], REPS["boltz_pooled"][0], REPS["charge_all"][0]]),
    "the full stack: sequence, pooled Boltz, and both charge blocks")
BASELINE = "seq"

ARCH = ("Linear(n_in,128) -> GELU -> Dropout(0.3) -> "
        "Linear(128,32) -> GELU -> Dropout(0.3) -> Linear(32,1)")


class MLP(nn.Module):
    def __init__(self, nin):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 128), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(DROPOUT),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def loss_figure(path, curves, title):
    """curves: list of (label, epochs, train_loss, val_loss, best_epoch)."""
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    for lab, ep, trl, vl, be in curves:
        l, = ax[0].plot(ep, trl, lw=1.1, label=lab)
        ax[1].plot(ep, vl, lw=1.1, color=l.get_color(), label=lab)
        ax[1].axvline(be, color=l.get_color(), ls=":", lw=0.8)
    for a, t in zip(ax, ("training loss", "validation loss")):
        a.set_xlabel("epoch")
        a.set_ylabel("mean squared error (standardised log10 t1/2)")
        a.set_title(t, fontsize=10)
        a.set_yscale("log")
        a.grid(alpha=0.25, lw=0.5)
    if len(curves) <= 12:
        ax[0].legend(fontsize=7)
    ax[1].set_title("validation loss (dotted = restored best epoch)", fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def train_one(name, X, desc, seed):
    d = os.path.join(RUNS, name, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = MLP(X.shape[1]).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = nn.MSELoss()
    T = {k: torch.as_tensor(X[idx[k]]).to(DEV) for k in idx}
    Y = {k: torch.as_tensor(yz[idx[k]]).to(DEV) for k in idx}

    log = open(os.path.join(d, "train.log"), "w")
    log.write(f"# {name} seed {seed}\n# {desc}\n"
              f"# input dim {X.shape[1]}, {sum(p.numel() for p in model.parameters()):,} parameters\n"
              f"# {ARCH}\n"
              f"# Adam lr={LR} weight_decay={WD} batch={BATCH} dropout={DROPOUT}\n"
              f"# loss = MSE on standardised log10 half-life\n"
              f"# early stopping on validation loss, patience {PATIENCE}, "
              f"max {MAX_EPOCHS} epochs\n#\n"
              f"{'epoch':>6} {'train_batch':>12} {'train':>12} {'val':>12}  note\n")
    rows, best, best_ep, bad, bestw = [], np.inf, 0, 0, None
    t0 = time.time()
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
        note = ""
        if vl < best - 1e-5:
            best, best_ep, bad = vl, ep, 0
            bestw = {k: v.detach().clone() for k, v in model.state_dict().items()}
            note = "* best"
        else:
            bad += 1
        rows.append((ep, tot / len(perm), trl, vl))
        log.write(f"{ep:>6} {tot/len(perm):>12.6f} {trl:>12.6f} {vl:>12.6f}  {note}\n")
        log.flush()
        if bad >= PATIENCE:
            log.write(f"# stopped at epoch {ep}: no improvement for {PATIENCE} epochs\n")
            break
    secs = time.time() - t0

    model.load_state_dict(bestw)
    model.eval()
    with torch.no_grad():
        pred = {k: (model(T[k]).cpu().numpy() * ys + ym).astype(np.float64)
                for k in idx}
    rho = {k: float(spearmanr(pred[k], y[idx[k]]).statistic) for k in idx}
    log.write(f"# restored epoch {best_ep} (val MSE {best:.6f})\n")
    log.write("# Spearman  " + "  ".join(f"{k} {v:+.4f}" for k, v in rho.items()) + "\n")
    log.close()

    hist = pd.DataFrame(rows, columns=["epoch", "train_batch_loss",
                                       "train_loss", "val_loss"])
    hist.to_csv(os.path.join(d, "epoch_log.csv"), index=False)
    torch.save({"state_dict": bestw, "input_dim": X.shape[1],
                "architecture": ARCH, "dropout": DROPOUT,
                "representation": name, "seed": seed,
                "target": "standardised log10 half-life",
                "y_mean_log10": float(ym), "y_std_log10": float(ys),
                "note": "prediction = model(x) * y_std_log10 + y_mean_log10, "
                        "giving log10 half-life in hours"},
               os.path.join(d, "model.pt"))
    np.savez_compressed(os.path.join(d, "predictions.npz"),
                        **{f"pred_{k}": pred[k] for k in idx},
                        **{f"true_{k}": y[idx[k]] for k in idx})
    parity_figure(os.path.join(d, "fig_parity.png"), pred,
                  {k: y[idx[k]] for k in idx}, f"{name}   seed {seed}")
    loss_figure(os.path.join(d, "fig_loss.png"),
                [("seed %d" % seed, hist.epoch, hist.train_loss,
                  hist.val_loss, best_ep)],
                f"{name}  seed {seed}  (restored epoch {best_ep})")

    det = {
        "representation": name, "description": desc, "seed": seed,
        "input_dim": int(X.shape[1]),
        "n_parameters": int(sum(p.numel() for p in model.parameters())),
        "architecture": ARCH, "activation": "GELU", "dropout": DROPOUT,
        "hidden_layers": [128, 32], "n_weight_layers": 3,
        "optimiser": "Adam", "lr": LR, "weight_decay": WD, "batch_size": BATCH,
        "loss": "MSE on standardised log10 half-life",
        "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
        "epochs_run": int(len(rows)), "restored_epoch": int(best_ep),
        "best_val_mse": float(best), "seconds": round(secs, 1), "device": DEV,
        "spearman": rho,
        "dataset": {
            "split_column": SPLIT,
            "split_note": "peptides held out: a test 9-mer appears nowhere in "
                          "train, under any allele. A real generalisation "
                          "estimate for a NEW PEPTIDE on a SEEN allele; it does "
                          "not test a new allele (that is split_C2)",
            "n_total": int(len(df)), "n_train": int(len(idx["train"])),
            "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
            "n_alleles": int(df.allele.nunique()),
            "n_clusters": int(df.cluster.nunique()),
            "n_not_yet_predicted": int(D["n_missing"]),
            "target": "thalf_hours, log10, zeros floored at half the smallest "
                      "positive value",
            "y_mean_log10_train": float(ym), "y_std_log10_train": float(ys),
            "standardisation": "feature means and s.d. from the TRAIN fold only",
            "source": "Boltz-2 embeddings, /share/ijp30/hackathons/aixscience2026"
                      "/boltz_inputs_and_predictions/embeddings/all75",
        },
        "environment": {"torch": torch.__version__,
                        "python": platform.python_version(),
                        "host": platform.node(),
                        "gpu": torch.cuda.get_device_name(0)
                        if DEV == "cuda" else None},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    json.dump(det, open(os.path.join(d, "run_details.json"), "w"), indent=2)
    return dict(hist=hist, pred=pred, rho=rho, best_epoch=best_ep,
                params=det["n_parameters"], seconds=secs)


os.makedirs(os.path.join(RUNS, "_summary"), exist_ok=True)
out, t0 = {}, time.time()
for name, (X, desc) in REPS.items():
    rs = [train_one(name, X, desc, s) for s in SEEDS]
    out[name] = rs
    r = lambda k: np.array([x["rho"][k] for x in rs])
    loss_figure(os.path.join(RUNS, name, "fig_loss_seeds.png"),
                [(f"seed {s}", x["hist"].epoch, x["hist"].train_loss,
                  x["hist"].val_loss, x["best_epoch"])
                 for s, x in zip(SEEDS, rs)],
                f"{name}  ({X.shape[1]} inputs, all {len(SEEDS)} seeds)")
    json.dump({"representation": name, "description": desc,
               "input_dim": int(X.shape[1]), "n_parameters": rs[0]["params"],
               "seeds": SEEDS,
               "epochs_mean": float(np.mean([len(x["hist"]) for x in rs])),
               "spearman_train_mean": float(r("train").mean()),
               "spearman_val_mean": float(r("val").mean()),
               "spearman_test_mean": float(r("test").mean()),
               "spearman_test_sd": float(r("test").std()),
               "spearman_test_per_seed": [float(v) for v in r("test")]},
              open(os.path.join(RUNS, name, "summary.json"), "w"), indent=2)
    print(f"{name:<22} dim {X.shape[1]:>5}  params {rs[0]['params']:>9,}  "
          f"ep {np.mean([len(x['hist']) for x in rs]):5.0f}  "
          f"train {r('train').mean():+.3f}  val {r('val').mean():+.3f}  "
          f"test {r('test').mean():+.3f} +/- {r('test').std():.3f}"
          f"   [{time.time()-t0:5.0f}s]", flush=True)

# ---- paired bootstrap against the sequence baseline ------------------------
te, yte = idx["test"], y[idx["test"]]
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
    rows.append(dict(rep=n, dim=REPS[n][0].shape[1], params=out[n][0]["params"],
                     epochs=float(np.mean([len(x["hist"]) for x in out[n]])),
                     rho_train=r("train").mean(), rho_val=r("val").mean(),
                     rho_test=r("test").mean(), rho_test_sd=r("test").std(),
                     delta_vs_seq=d.mean(), ci_lo=lo, ci_hi=hi,
                     beats_sequence=bool(lo > 0)))
res = pd.DataFrame(rows).sort_values("rho_test", ascending=False)
res.to_csv(os.path.join(RUNS, "_summary", "results.csv"), index=False)
print()
print(res.to_string(index=False, float_format=lambda v: f"{v:+.3f}"))

# ---- combined figures ------------------------------------------------------
loss_figure(os.path.join(RUNS, "_summary", "fig_loss_all.png"),
            [(n, rs[0]["hist"].epoch, rs[0]["hist"].train_loss,
              rs[0]["hist"].val_loss, rs[0]["best_epoch"])
             for n, rs in out.items()],
            "every representation, seed 0")

ncol = 5
nrow = int(np.ceil(len(REPS) / ncol))
fig, ax = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 3.2 * nrow))
lo_a, hi_a = np.log10(yte[yte > 0].min()), np.log10(yte.max())
pad = 0.1 * (hi_a - lo_a)
lim = (lo_a - pad, hi_a + pad)
t = np.log10(np.where(yte > 0, yte, 10 ** lo_a))
for a, (n, p) in zip(ax.ravel(), mean_pred.items()):
    off = (p < lim[0]) | (p > lim[1])
    a.plot(lim, lim, "k--", lw=0.8, zorder=1)
    a.scatter(t[~off], p[~off], s=4, alpha=0.25, lw=0, zorder=2)
    if off.any():
        a.scatter(t[off], np.clip(p[off], *lim), s=14, facecolors="none",
                  edgecolors="red", lw=0.7, zorder=3)
    rho = np.mean([x["rho"]["test"] for x in out[n]])
    a.set_title(f"{n}\n" + rf"$\rho$ = {rho:+.3f}" +
                (f"  ({off.sum()} off-scale)" if off.any() else ""), fontsize=8)
    a.set_xlim(*lim); a.set_ylim(*lim)
    a.set_xlabel("measured log10 t1/2 (h)", fontsize=7)
    a.set_ylabel("predicted", fontsize=7)
    a.tick_params(labelsize=6)
for a in ax.ravel()[len(REPS):]:
    a.axis("off")
fig.suptitle(f"{SPLIT} (held-out peptides) | n_test = {len(te):,} | "
             f"{df.allele.nunique()} alleles | mean of {len(SEEDS)} seeds",
             fontsize=10)
fig.tight_layout()
fig.savefig(os.path.join(RUNS, "_summary", "fig_parity_all.png"), dpi=150)

for n, p in mean_pred.items():
    f2, a2 = plt.subplots(figsize=(4.2, 4.2))
    off = (p < lim[0]) | (p > lim[1])
    a2.plot(lim, lim, "k--", lw=0.8)
    a2.scatter(t[~off], p[~off], s=6, alpha=0.3, lw=0)
    if off.any():
        a2.scatter(t[off], np.clip(p[off], *lim), s=16, facecolors="none",
                   edgecolors="red", lw=0.7)
    a2.set_xlim(*lim); a2.set_ylim(*lim)
    a2.set_xlabel("measured log10 half-life (h)")
    a2.set_ylabel("predicted log10 half-life (h)")
    a2.set_title(f"{n}\n" +
                 rf"test $\rho$ = {np.mean([x['rho']['test'] for x in out[n]]):+.3f}",
                 fontsize=10)
    f2.tight_layout()
    f2.savefig(os.path.join(RUNS, n, "fig_parity.png"), dpi=150)
    plt.close(f2)

json.dump({"split": SPLIT, "baseline": BASELINE, "seeds": SEEDS,
           "n_bootstrap": NBOOT, "n_total": int(len(df)),
           "n_train": int(len(idx["train"])), "n_val": int(len(idx["val"])),
           "n_test": int(len(te)), "n_alleles": int(df.allele.nunique()),
           "n_clusters": int(df.cluster.nunique()),
           "n_not_yet_predicted": int(D["n_missing"]),
           "minutes": round((time.time() - t0) / 60, 1),
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())},
          open(os.path.join(RUNS, "_summary", "run_config.json"), "w"), indent=2)
np.savez_compressed(os.path.join(RUNS, "_summary", "test_predictions.npz"),
                    y_test=yte, **mean_pred)
print(f"\nwrote {RUNS} in {(time.time()-t0)/60:.1f} min")
