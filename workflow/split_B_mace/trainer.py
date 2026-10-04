"""Shared training recipe and output layout for every architecture tried.

Identical to the multilayer perceptron runs in workflow/split_B_charge: Adam,
lr 1e-3, weight decay 1e-4, batch 32, mean squared error on standardised log10
half-life, at most 300 epochs, early stopping on validation loss with patience
40, best-validation weights restored, 5 seeds. Only the model differs, so any
difference in the table is the architecture and not the recipe.

A model may take several feature tensors (a bilinear form takes the peptide and
the groove separately), so `feats` is a tuple and models receive *args.

Split is split_C2: whole NetMHCpan pseudosequence clusters are held out, so no
receptor in the test fold has been seen during training. This is the result the
project is actually about -- split_A leaks and split_B still shares alleles.

The row set is FROZEN to frozen_ids.txt, written on first use.
"""

import json
import os
import platform
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr

from data_split import load
from parity import parity_figure

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
FROZEN = os.path.join(HERE, "frozen_ids.txt")
DEV = "cuda" if torch.cuda.is_available() else "cpu"
MAX_EPOCHS, PATIENCE, BATCH, LR, WD, DROPOUT = 300, 40, 32, 1e-3, 1e-4, 0.3
SEEDS = [0, 1, 2]
SPLIT = "split_B"


def get_data():
    D = load(SPLIT)
    ids = D["df"].sample_id.to_numpy()
    if os.path.exists(FROZEN):
        keep_ids = [l.strip() for l in open(FROZEN) if l.strip()]
        m = np.isin(ids, keep_ids)
        if m.sum() != len(keep_ids):
            raise SystemExit(f"frozen set has {len(keep_ids)} ids, found {m.sum()}")
        if not m.all():
            D = _subset(D, m)
    else:
        with open(FROZEN, "w") as fh:
            fh.write("\n".join(ids) + "\n")
        print(f"froze {len(ids):,} complexes -> {FROZEN}")
    return D


def _subset(D, m):
    df = D["df"][m].reset_index(drop=True)
    F = {k: v[m] for k, v in D["F"].items()}
    y, ly = D["y"][m], D["ly"][m]
    where = df[SPLIT].to_numpy()
    idx = {k: np.flatnonzero(where == k) for k in ("train", "val", "test")}
    tr = idx["train"]
    ym, ys = ly[tr].mean(), ly[tr].std()

    def std(A):
        return ((A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)).astype(np.float32)

    return dict(df=df, F=F, oh_pep=D["oh_pep"][m], oh_hla=D["oh_hla"][m],
                oh=D["oh"][m], std=std, y=y, ly=ly,
                yz=((ly - ym) / ys).astype(np.float32), ym=ym, ys=ys,
                idx=idx, tr=tr, pseudoseq=[q for q, k in zip(D["pseudoseq"], m) if k],
                n_missing=D["n_missing"], n_bad=D["n_bad"])


def train(name, desc, build, feats, D, seed, arch_text, extra_details=None):
    idx, tr, y, yz = D["idx"], D["tr"], D["y"], D["yz"]
    ym, ys, df = D["ym"], D["ys"], D["df"]
    d = os.path.join(RUNS, name, f"seed{seed}")
    os.makedirs(d, exist_ok=True)
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build().to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lossfn = nn.MSELoss()
    T = {k: tuple(torch.as_tensor(f[idx[k]]).to(DEV) for f in feats) for k in idx}
    Y = {k: torch.as_tensor(yz[idx[k]]).to(DEV) for k in idx}
    npar = sum(p.numel() for p in model.parameters())

    log = open(os.path.join(d, "train.log"), "w")
    log.write(f"# {name} seed {seed}\n# {desc}\n"
              f"# inputs {[f.shape[1:] for f in feats]}, {npar:,} parameters\n"
              f"# {arch_text}\n"
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
            l = lossfn(model(*(t[j] for t in T["train"])), Y["train"][j])
            l.backward()
            opt.step()
            tot += l.item() * len(j)
        model.eval()
        with torch.no_grad():
            trl = lossfn(model(*T["train"]), Y["train"]).item()
            vl = lossfn(model(*T["val"]), Y["val"]).item()
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
        pred = {k: (model(*T[k]).cpu().numpy() * ys + ym).astype(np.float64)
                for k in idx}
    rho = {k: float(spearmanr(pred[k], y[idx[k]]).statistic) for k in idx}
    log.write(f"# restored epoch {best_ep} (val MSE {best:.6f})\n")
    log.write("# Spearman  " + "  ".join(f"{k} {v:+.4f}" for k, v in rho.items()) + "\n")
    log.close()

    hist = pd.DataFrame(rows, columns=["epoch", "train_batch_loss",
                                       "train_loss", "val_loss"])
    hist.to_csv(os.path.join(d, "epoch_log.csv"), index=False)
    torch.save({"state_dict": bestw, "architecture": arch_text,
                "representation": name, "seed": seed,
                "y_mean_log10": float(ym), "y_std_log10": float(ys),
                "note": "prediction = model(x) * y_std_log10 + y_mean_log10"},
               os.path.join(d, "model.pt"))
    np.savez_compressed(os.path.join(d, "predictions.npz"),
                        **{f"pred_{k}": pred[k] for k in idx},
                        **{f"true_{k}": y[idx[k]] for k in idx})
    parity_figure(os.path.join(d, "fig_parity.png"), pred,
                  {k: y[idx[k]] for k in idx}, f"{name}   seed {seed}")
    from train_mlp import loss_figure
    loss_figure(os.path.join(d, "fig_loss.png"),
                [(f"seed {seed}", hist.epoch, hist.train_loss,
                  hist.val_loss, best_ep)],
                f"{name}  seed {seed}  (restored epoch {best_ep})")

    det = {"representation": name, "description": desc, "seed": seed,
           "input_dim": int(sum(int(np.prod(f.shape[1:])) for f in feats)),
           "input_shapes": [list(f.shape[1:]) for f in feats],
           "n_parameters": npar, "architecture": arch_text, "dropout": DROPOUT,
           "optimiser": "Adam", "lr": LR, "weight_decay": WD,
           "batch_size": BATCH, "loss": "MSE on standardised log10 half-life",
           "max_epochs": MAX_EPOCHS, "early_stopping_patience": PATIENCE,
           "epochs_run": int(len(rows)), "restored_epoch": int(best_ep),
           "best_val_mse": float(best), "seconds": round(secs, 1), "device": DEV,
           "spearman": rho,
           "dataset": {"split_column": SPLIT,
                       "split_note": "peptides held out: a test 9-mer appears "
                                     "nowhere in train, under any allele. The "
                                     "alleles themselves ARE seen in training",
                       "frozen_row_set": os.path.basename(FROZEN),
                       "n_total": int(len(df)), "n_train": int(len(idx["train"])),
                       "n_val": int(len(idx["val"])), "n_test": int(len(idx["test"])),
                       "n_alleles": int(df.allele.nunique()),
                       "n_clusters": int(df.cluster.nunique()),
                       "target": "thalf_hours, log10, zeros floored at half the "
                                 "smallest positive value",
                       "standardisation": "train fold only"},
           "environment": {"torch": torch.__version__,
                           "python": platform.python_version(),
                           "host": platform.node(),
                           "gpu": torch.cuda.get_device_name(0) if DEV == "cuda" else None},
           "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    if extra_details:
        det.update(extra_details)
    json.dump(det, open(os.path.join(d, "run_details.json"), "w"), indent=2)
    return dict(hist=hist, pred=pred, rho=rho, best_epoch=best_ep,
                params=npar, seconds=secs)


def run_family(models, D):
    """models: list of (name, desc, build, feats, arch_text, extra)."""
    out = {}
    t0 = time.time()
    for name, desc, build, feats, arch_text, extra in models:
        rs = [train(name, desc, build, feats, D, s, arch_text, extra)
              for s in SEEDS]
        out[name] = rs
        r = lambda k: np.array([x["rho"][k] for x in rs])
        from train_mlp import loss_figure
        loss_figure(os.path.join(RUNS, name, "fig_loss_seeds.png"),
                    [(f"seed {s}", x["hist"].epoch, x["hist"].train_loss,
                      x["hist"].val_loss, x["best_epoch"])
                     for s, x in zip(SEEDS, rs)], f"{name}  (all seeds)")
        print(f"{name:<28} params {rs[0]['params']:>9,}  "
              f"ep {np.mean([len(x['hist']) for x in rs]):5.0f}  "
              f"train {r('train').mean():+.3f}  val {r('val').mean():+.3f}  "
              f"test {r('test').mean():+.3f} +/- {r('test').std():.3f}"
              f"   [{time.time()-t0:5.0f}s]", flush=True)
    return out
