"""Shared training engine: metrics, evaluation, loaders, and the epoch loop.

Every network's `train.py` builds its own dataset + model, then calls
`run_training(...)` with a `forward_fn` that supplies the one line that differs
between architectures (how the model is called on a batch). The metric helpers
(`metrics_from_log`, `_spearman`, `within_allele_spearman`) are used identically
everywhere, including by boost.py.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from device_utils import move_batch

# Default batch forward for the sequence networks (indices for peptide/hla/allele).
# The boltz network passes its own `forward_fn` since its input is a single vector.
def _seq_forward(model, batch):
    return model(batch["peptide"], batch["hla"], batch["allele"])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation = Pearson of the ranks (no scipy dependency)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def within_allele_spearman(
    preds_log: np.ndarray,
    targets_log: np.ndarray,
    alleles: np.ndarray,
    min_count: int = 5,
) -> float:
    """Mean Spearman ρ computed *within* each allele (unweighted over alleles).

    Only alleles with >= `min_count` rows and non-constant predictions/targets
    contribute. A model that just predicts each allele's baseline stability scores
    near 0 here, even if its global correlation looks high. NaN if no allele
    qualifies (e.g. a single peptide per allele).
    """
    alleles = np.asarray(alleles)
    rhos = []
    for a in np.unique(alleles):
        m = alleles == a
        if int(m.sum()) < min_count:
            continue
        p, t = preds_log[m], targets_log[m]
        if np.std(p) == 0 or np.std(t) == 0:
            continue
        rhos.append(_spearman(p, t))
    return float(np.mean(rhos)) if rhos else float("nan")


def metrics_from_log(
    preds_log: np.ndarray,
    targets_log: np.ndarray,
    alleles: np.ndarray | None = None,
) -> dict:
    """Error metrics + correlations from predictions in log1p space.

    Split out so anything holding raw predictions — notably the XGBoost head in
    boost.py — is scored by exactly the same code as a network's own eval.

    When `alleles` (one allele label per row) is given, we also compute the
    **within-allele Spearman** — the mean per-allele rank correlation. Global
    Pearson/Spearman are inflated because much of the variance is just the
    allele's baseline stability; within-allele ρ measures the harder, clinically
    relevant task of ranking peptides *for one allele*.
    """
    # Back to hours for interpretable error numbers.
    preds_hours = np.expm1(preds_log)
    targets_hours = np.expm1(targets_log)
    out = {
        "mae_hours": float(np.mean(np.abs(preds_hours - targets_hours))),
        "rmse_hours": float(np.sqrt(np.mean((preds_hours - targets_hours) ** 2))),
        "pearson": float(np.corrcoef(preds_log, targets_log)[0, 1]),
        "spearman": _spearman(preds_log, targets_log),
        "preds_hours": preds_hours,
        "targets_hours": targets_hours,
    }
    if alleles is not None:
        out["within_allele_spearman"] = within_allele_spearman(
            preds_log, targets_log, alleles
        )
    return out


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    alleles: np.ndarray | None = None,
    forward_fn=_seq_forward,
) -> dict[str, float]:
    """Return log-space loss plus hour-space error metrics and correlations.

    `forward_fn(model, batch)` is the one architecture-specific line. `alleles`
    (one label per row, in loader order — so the loader must be unshuffled)
    enables the within-allele Spearman metric.
    """
    model.eval()
    total_loss, n = 0.0, 0
    preds_log, targets_log = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["target"]

        pred = forward_fn(model, batch)
        total_loss += loss_fn(pred, target).item() * len(target)
        n += len(target)
        # .cpu() is required before numpy when tensors live on the GPU.
        preds_log.append(pred.cpu().numpy())
        targets_log.append(target.cpu().numpy())

    return {
        "loss": total_loss / n,
        **metrics_from_log(
            np.concatenate(preds_log), np.concatenate(targets_log), alleles
        ),
    }


def make_loaders(train_ds, val_ds, test_ds, batch_size, num_workers, device):
    """Build the three DataLoaders (only train is shuffled; test may be None)."""
    # pin_memory speeds up host->GPU copies; it's pointless on CPU so gate on it.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        pin_memory=pin, num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        pin_memory=pin, num_workers=num_workers,
    )
    test_loader = (
        DataLoader(
            test_ds, batch_size=batch_size, shuffle=False,
            pin_memory=pin, num_workers=num_workers,
        )
        if test_ds is not None and len(test_ds) > 0
        else None
    )
    return train_loader, val_loader, test_loader


def run_training(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader | None,
    val_alleles: np.ndarray,
    test_alleles: np.ndarray,
    *,
    forward_fn=_seq_forward,
    epochs: int = 30,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    device: torch.device,
    prefix: str = "",
    out: str | None = None,
    checkpoint=None,
) -> tuple[dict[str, list[float]], dict, dict | None]:
    """The epoch loop shared by every network. Returns (history, best_eval, test_eval).

    `forward_fn(model, batch)` supplies the model call. `checkpoint(model)` returns
    the payload torch.save'd when `out` is set (each net decides what to store).
    The best (by val loss) weights are snapshotted and reloaded for the single
    held-out test evaluation.
    """
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = float("inf")
    best_eval = None  # keep the best epoch's val predictions for the scatter plot
    best_state = None  # snapshot of the best-val weights, for the final test eval
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_hours": [],
        "val_pearson": [],
        "val_spearman": [],
        "val_within_rho": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            target = batch["target"]

            optimizer.zero_grad()           # clear gradients from the last step
            pred = forward_fn(model, batch)  # forward
            loss = loss_fn(pred, target)    # mean squared error in log space
            loss.backward()                 # backprop the gradients
            optimizer.step()                # update the weights

            running += loss.item() * len(target)
            n += len(target)

        train_loss = running / n
        val = evaluate(model, val_loader, loss_fn, device, val_alleles, forward_fn)
        print(
            f"{prefix} epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | val_MAE {val['mae_hours']:.2f} h | "
            f"val_pearson {val['pearson']:.3f} | "
            f"val_within_rho {val['within_allele_spearman']:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mae_hours"].append(val["mae_hours"])
        history["val_pearson"].append(val["pearson"])
        history["val_spearman"].append(val["spearman"])
        history["val_within_rho"].append(val["within_allele_spearman"])

        if val["loss"] < best_val:
            best_val = val["loss"]
            best_eval = val
            # Snapshot the weights (copied to CPU) so test uses this exact model.
            best_state = copy.deepcopy(
                {k: v.cpu() for k, v in model.state_dict().items()}
            )
            if out is not None and checkpoint is not None:
                torch.save(checkpoint(model), out)
                print(f"{prefix}          ↳ saved best model to {out}")

    print(f"{prefix} done. best val_loss = {best_val:.4f}")

    # Final test evaluation with the best-val model (held-out until now).
    test_eval = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_eval = evaluate(model, test_loader, loss_fn, device, test_alleles, forward_fn)
        print(
            f"{prefix} TEST | loss {test_eval['loss']:.4f} | "
            f"MAE {test_eval['mae_hours']:.2f} h | "
            f"RMSE {test_eval['rmse_hours']:.2f} h | "
            f"pearson {test_eval['pearson']:.3f} | "
            f"spearman {test_eval['spearman']:.3f} | "
            f"within_rho {test_eval['within_allele_spearman']:.3f}"
        )

    return history, best_eval, test_eval
