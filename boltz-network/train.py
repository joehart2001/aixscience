"""Train the Boltz-embedding half-life network from a YAML config.

Example:
    python train.py templates/split_A.yaml

Same model complexity as direct-network, but the allele-peptide system is
represented by a precomputed Boltz2 embedding instead of integer amino-acid
indices. The config file holds every parameter (embeddings dir, feature key,
split, hyperparameters, device, outputs); see templates/*.yaml. To compare
several splits on shared plots, use compare.py instead.

The script:
  1. Loads the chosen split, restricted to rows with embeddings.
  2. Fits a feature scaler on the training split.
  3. Trains an MLP with MSE loss on log1p(thalf_hours).
  4. Reports metrics each epoch, saves the best checkpoint, evaluates on test.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import build_results, load_config, print_summary, train_from_config
from data import BoltzDataset, Scaler, load_splits
from device_utils import move_batch, resolve_device, seed_everything
from model import BoltzAffinityNet
from plots import make_all_plots, make_test_plots, save_run_data


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one model from a YAML config (see templates/)."
    )
    p.add_argument("config", help="Path to a YAML config file")
    return p.parse_args()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    alleles: np.ndarray | None = None,
) -> dict[str, float]:
    """Return log-space loss plus hour-space error metrics and correlations.

    `alleles` (one label per row, in loader order — so the loader must be
    unshuffled) enables the within-allele Spearman metric.
    """
    model.eval()
    total_loss, n = 0.0, 0
    preds_log, targets_log = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["target"]

        pred = model(batch["x"])
        total_loss += loss_fn(pred, target).item() * len(target)
        n += len(target)
        # .cpu() is required before numpy when tensors live on the GPU.
        preds_log.append(pred.cpu().numpy())
        targets_log.append(target.cpu().numpy())

    preds_log = np.concatenate(preds_log)
    targets_log = np.concatenate(targets_log)

    # Back to hours for interpretable error numbers.
    preds_hours = np.expm1(preds_log)
    targets_hours = np.expm1(targets_log)
    out = {
        "loss": total_loss / n,
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
    qualifies. (With embeddings for a single allele, this equals global Spearman.)
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


def train_model(
    csv: str,
    splits_csv: str,
    split: str,
    embeddings_dir: str,
    feature_key: str = "features",
    allele: str = "HLA-A*02:01",
    train_value: str = "train",
    val_value: str = "val",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    device: str | None = None,
    num_workers: int = 0,
    test_value: str = "test",
    out: str | None = None,
    tag: str = "",
) -> tuple[dict[str, list[float]], dict, dict | None]:
    """Train one model and return (history, best_eval, test_eval).

    `split` selects which precomputed split to use (A/B from `splits_csv`);
    `embeddings_dir` holds the Boltz `.npz` files and `feature_key` picks which
    embedding vector to feed the model. `device` is auto-selected (GPU if
    available, else CPU) unless forced. `history` holds per-epoch metrics,
    `best_eval` holds the best (by val loss) epoch's validation predictions, and
    `test_eval` holds that same best model's predictions on the held-out test
    split (None if the split has no test rows).
    """
    device = resolve_device(device)
    seed_everything(seed, device)
    prefix = f"[{tag or split}]"
    print(f"{prefix} device = {device}")

    # 1. Load the chosen split, restricted to `allele` rows with embeddings.
    train_df, val_df, test_df = load_splits(
        csv,
        splits_csv=splits_csv,
        split=split,
        embeddings_dir=embeddings_dir,
        allele=allele,
        train_value=train_value,
        val_value=val_value,
        test_value=test_value,
    )
    print(
        f"{prefix} split={split} train rows = {len(train_df)}, "
        f"val rows = {len(val_df)}, test rows = {len(test_df)}"
    )

    # 2. Feature scaler is fit on the training split only (like Vocab before).
    scaler = Scaler.build(train_df, embeddings_dir, feature_key)
    print(f"{prefix} feature_key={feature_key} input_dim={scaler.dim}")

    train_ds = BoltzDataset(train_df, embeddings_dir, scaler)
    val_ds = BoltzDataset(val_df, embeddings_dir, scaler)
    # pin_memory speeds up host->GPU copies; it's pointless on CPU so gate on it.
    pin = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=pin,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        pin_memory=pin,
        num_workers=num_workers,
    )
    # The test split is evaluated once at the end with the best-val model.
    test_loader = (
        DataLoader(
            BoltzDataset(test_df, embeddings_dir, scaler),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin,
            num_workers=num_workers,
        )
        if len(test_df) > 0
        else None
    )

    # 3. Model, loss, optimizer.
    model = BoltzAffinityNet(input_dim=scaler.dim).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # Allele labels per row (loaders are unshuffled) for within-allele metrics.
    val_alleles = val_df["allele"].to_numpy()
    test_alleles = test_df["allele"].to_numpy()

    # 4. Training loop.
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
            pred = model(batch["x"])  # forward
            loss = loss_fn(pred, target)    # mean squared error in log space
            loss.backward()                 # backprop the gradients
            optimizer.step()                # update the weights

            running += loss.item() * len(target)
            n += len(target)

        train_loss = running / n
        val = evaluate(model, val_loader, loss_fn, device, alleles=val_alleles)
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
            if out is not None:  # only persist a checkpoint when asked to
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "scaler": scaler,
                        "config": {"input_dim": scaler.dim, "feature_key": feature_key},
                    },
                    out,
                )
                print(f"{prefix}          ↳ saved best model to {out}")

    print(f"{prefix} done. best val_loss = {best_val:.4f}")

    # 5. Final test evaluation with the best-val model (held-out until now).
    test_eval = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_eval = evaluate(model, test_loader, loss_fn, device, alleles=test_alleles)
        print(
            f"{prefix} TEST | loss {test_eval['loss']:.4f} | "
            f"MAE {test_eval['mae_hours']:.2f} h | "
            f"RMSE {test_eval['rmse_hours']:.2f} h | "
            f"pearson {test_eval['pearson']:.3f} | "
            f"spearman {test_eval['spearman']:.3f} | "
            f"within_rho {test_eval['within_allele_spearman']:.3f}"
        )

    return history, best_eval, test_eval


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label = cfg["label"]
    print(f"=== Training model: {label} (split {cfg['split']}) ===")

    history, best_eval, test_eval = train_from_config(cfg, train_model)

    # Persist run data + draw the single-model figures (val) and test suite.
    results = build_results({label: history}, {label: best_eval}, {label: test_eval})
    figs_dir = cfg["figs_dir"]
    save_run_data(results, figs_dir)
    make_all_plots(results, figs_dir)
    make_test_plots(results, figs_dir)
    print(f"[train] saved figures + data to {figs_dir}/")

    print_summary({label: history}, {label: test_eval})


if __name__ == "__main__":
    main()
