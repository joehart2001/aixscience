"""Multilayer perceptron baselines, re-run on the frozen row set.

These repeat the workflow/split_B_charge results so every architecture in this
folder is scored on exactly the same complexes. The shared loss_figure lives
here because trainer.py imports it.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn

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




class MLP(nn.Module):
    """Linear(n,128) -> GELU -> Dropout -> Linear(128,32) -> GELU -> Dropout -> Linear(32,1)."""

    def __init__(self, nin, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, 128), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(128, 32), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(32, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


MLP_TEXT = ("Linear(n_in,128) -> GELU -> Dropout(0.3) -> Linear(128,32) -> "
            "GELU -> Dropout(0.3) -> Linear(32,1)")

if __name__ == "__main__":
    from trainer import get_data, run_family

    D = get_data()
    std, F = D["std"], D["F"]
    reps = {
        "mlp_seq": (std(D["oh"]), "peptide + pseudosequence one-hot (BASELINE)"),
        "mlp_boltz_pooled": (std(F["pooled"]), "Boltz pooled blocks"),
    }
    reps["mlp_seq+boltz_pooled"] = (
        np.hstack([reps["mlp_seq"][0], reps["mlp_boltz_pooled"][0]]),
        "sequence baseline concatenated with the Boltz pooled blocks")

    models = [(n, d, (lambda X=X: MLP(X.shape[1])), (X,), MLP_TEXT, None)
              for n, (X, d) in reps.items()]
    run_family(models, D)
