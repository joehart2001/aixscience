"""Gradient-boosted trees on the same representations and the same rows.

At n = 8,691 with one to two thousand tabular features, boosted trees are the
natural competitor to a small perceptron, and the pooled Boltz blocks are
exactly the tabular regime they are built for. If trees beat every network in
the table, that is a finding about our modelling rather than about Boltz, and
it has to be known before anything is reported.

Same frozen rows, same split, same five seeds, early stopping on the validation
fold. The per-iteration training and validation loss is written to train.log in
the same format the networks use, so the folders stay interchangeable.

Trees need no feature standardisation -- they split on order, not scale -- so
the raw blocks are passed through. Importances are written per representation,
which the networks cannot give and which says WHICH pooled block carries the
signal.
"""

import json
import os
import platform
import time

import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from parity import parity_figure
from trainer import FROZEN, RUNS, SEEDS, SPLIT, get_data

os.makedirs(os.path.join(RUNS, "_summary"), exist_ok=True)

PARAMS = dict(objective="l2", learning_rate=0.05, num_leaves=31,
              min_data_in_leaf=20, feature_fraction=0.3, bagging_fraction=0.8,
              bagging_freq=1, lambda_l2=1.0, verbose=-1, num_threads=16)
ROUNDS, STOP = 3000, 100
TEXT = (f"LightGBM gradient-boosted trees, objective l2, lr "
        f"{PARAMS['learning_rate']}, num_leaves {PARAMS['num_leaves']}, "
        f"min_data_in_leaf {PARAMS['min_data_in_leaf']}, feature_fraction "
        f"{PARAMS['feature_fraction']}, bagging {PARAMS['bagging_fraction']}, "
        f"lambda_l2 {PARAMS['lambda_l2']}, up to {ROUNDS} rounds, early "
        f"stopping {STOP} on validation l2")


def run(name, desc, X, D, seed, blocks=None):
    idx, y, yz, ym, ys, df = D["idx"], D["y"], D["yz"], D["ym"], D["ys"], D["df"]
    d = os.path.join(RUNS, name, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    p = dict(PARAMS, seed=seed, bagging_seed=seed, feature_fraction_seed=seed)
    dtr = lgb.Dataset(X[idx["train"]], yz[idx["train"]], free_raw_data=False)
    dva = lgb.Dataset(X[idx["val"]], yz[idx["val"]], reference=dtr,
                      free_raw_data=False)
    hist = {}
    t0 = time.time()
    booster = lgb.train(p, dtr, num_boost_round=ROUNDS,
                        valid_sets=[dtr, dva], valid_names=["train", "val"],
                        callbacks=[lgb.early_stopping(STOP, verbose=False),
                                   lgb.record_evaluation(hist)])
    secs = time.time() - t0
    best = booster.best_iteration
    pred = {k: (booster.predict(X[idx[k]], num_iteration=best) * ys + ym)
            for k in idx}
    rho = {k: float(spearmanr(pred[k], y[idx[k]]).statistic) for k in idx}

    tr_l, va_l = hist["train"]["l2"], hist["val"]["l2"]
    with open(os.path.join(d, "train.log"), "w") as log:
        log.write(f"# {name} seed {seed}\n# {desc}\n"
                  f"# inputs {X.shape[1]}, {booster.num_trees():,} trees\n"
                  f"# {TEXT}\n# loss = l2 on standardised log10 half-life\n#\n"
                  f"{'round':>6} {'train':>12} {'val':>12}  note\n")
        for i, (a, b) in enumerate(zip(tr_l, va_l), 1):
            log.write(f"{i:>6} {a:>12.6f} {b:>12.6f}  "
                      f"{'* best' if i == best else ''}\n")
        log.write(f"# best iteration {best} (val l2 {va_l[best-1]:.6f})\n")
        log.write("# Spearman  " + "  ".join(f"{k} {v:+.4f}"
                                             for k, v in rho.items()) + "\n")
    pd.DataFrame({"epoch": np.arange(1, len(tr_l) + 1),
                  "train_loss": tr_l, "val_loss": va_l}).to_csv(
        os.path.join(d, "epoch_log.csv"), index=False)
    booster.save_model(os.path.join(d, "model.txt"), num_iteration=best)
    np.savez_compressed(os.path.join(d, "predictions.npz"),
                        **{f"pred_{k}": pred[k] for k in idx},
                        **{f"true_{k}": y[idx[k]] for k in idx})
    parity_figure(os.path.join(d, "fig_parity.png"), pred,
                  {k: y[idx[k]] for k in idx}, f"{name}   seed {seed}")

    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ax.plot(np.arange(1, len(tr_l) + 1), tr_l, lw=1.1, label="train")
    ax.plot(np.arange(1, len(va_l) + 1), va_l, lw=1.1, label="val")
    ax.axvline(best, color="k", ls=":", lw=0.8)
    ax.set_xlabel("boosting round")
    ax.set_ylabel("l2 (standardised log10 t1/2)")
    ax.set_yscale("log")
    ax.grid(alpha=0.25, lw=0.5)
    ax.legend(fontsize=8)
    ax.set_title(f"{name}  seed {seed}  (best round {best})", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "fig_loss.png"), dpi=150)
    plt.close(fig)

    det = {"representation": name, "description": desc, "seed": seed,
           "input_dim": int(X.shape[1]), "model_family": "gradient boosted trees",
           "n_parameters": int(booster.num_trees()),
           "n_trees": int(booster.num_trees()), "architecture": TEXT,
           "lightgbm": lgb.__version__, "params": p,
           "max_epochs": ROUNDS, "early_stopping_patience": STOP,
           "epochs_run": int(len(tr_l)), "restored_epoch": int(best),
           "best_val_mse": float(va_l[best - 1]), "seconds": round(secs, 1),
           "device": "cpu", "spearman": rho, "dropout": None,
           "optimiser": "gradient boosting", "lr": PARAMS["learning_rate"],
           "weight_decay": PARAMS["lambda_l2"], "batch_size": None,
           "loss": "l2 on standardised log10 half-life",
           "dataset": {"split_column": SPLIT, "frozen_row_set":
                       os.path.basename(FROZEN),
                       "split_note": "peptides held out",
                       "n_total": int(len(df)),
                       "n_train": int(len(idx["train"])),
                       "n_val": int(len(idx["val"])),
                       "n_test": int(len(idx["test"])),
                       "n_alleles": int(df.allele.nunique()),
                       "n_clusters": int(df.cluster.nunique()),
                       "target": "thalf_hours, log10, zeros floored",
                       "standardisation": "none needed; trees split on order"},
           "environment": {"python": platform.python_version(),
                           "host": platform.node()},
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    json.dump(det, open(os.path.join(d, "run_details.json"), "w"), indent=2)

    gain = booster.feature_importance("gain")
    return dict(rho=rho, pred=pred, best=best, rounds=len(tr_l), gain=gain,
                params=int(booster.num_trees()), seconds=secs)


if __name__ == "__main__":
    D = get_data()
    F = D["F"]
    charge = np.hstack([F["charge_seq"], F["charge_struct"]])
    REPS = {
        "gbm_seq": (D["oh"], "peptide + pseudosequence one-hot"),
        "gbm_boltz_pooled": (F["pooled"], "Boltz pooled blocks"),
        "gbm_charge_all": (charge, "sequence and structure charge blocks"),
        "gbm_seq+boltz_pooled": (np.hstack([D["oh"], F["pooled"]]),
                                 "sequence with the pooled Boltz blocks"),
        "gbm_seq+boltz_pooled+charge":
            (np.hstack([D["oh"], F["pooled"], charge]),
             "sequence, pooled Boltz and both charge blocks"),
        "gbm_boltz_pep_tokens": (F["tok"], "Boltz peptide tokens, flattened"),
    }
    t0 = time.time()
    imp = {}
    for name, (X, desc) in REPS.items():
        X = np.ascontiguousarray(X, dtype=np.float32)
        rs = [run(name, desc, X, D, s) for s in SEEDS]
        r = lambda k: np.array([x["rho"][k] for x in rs])
        imp[name] = np.mean([x["gain"] for x in rs], 0)
        print(f"{name:<30} dim {X.shape[1]:>5}  trees "
              f"{np.mean([x['params'] for x in rs]):>6.0f}  "
              f"best {np.mean([x['best'] for x in rs]):>5.0f}  "
              f"train {r('train').mean():+.3f}  val {r('val').mean():+.3f}  "
              f"test {r('test').mean():+.3f} +/- {r('test').std():.3f}"
              f"   [{time.time()-t0:5.0f}s]", flush=True)

    # which pooled Boltz block carries the signal?
    import glob
    z = np.load(sorted(glob.glob("/share/ijp30/hackathons/aixscience2026/"
                                 "boltz_inputs_and_predictions/embeddings/"
                                 "all75/*.npz"))[0], allow_pickle=True)
    blocks, sizes = list(z["feature_blocks"]), list(z["feature_block_sizes"])
    g = imp["gbm_boltz_pooled"]
    edges = np.cumsum([0] + sizes)
    print("\ngain by pooled Boltz block (gbm_boltz_pooled):")
    tot = g.sum()
    for b, a, c in zip(blocks, edges[:-1], edges[1:]):
        print(f"  {b:<28} {g[a:c].sum()/tot*100:5.1f}%   "
              f"per-feature {g[a:c].mean()/tot*100:6.3f}%")
    np.savez_compressed(os.path.join(RUNS, "_summary", "tree_importances.npz"),
                        **imp)
