"""Train the direct peptide-MHC half-life network from a YAML config.

Example:
    python train.py templates/pseudoseq.yaml

The config file holds every parameter (dataset path, HLA column, hyperparameters,
device, outputs); see templates/*.yaml for the schema. To compare several models
on shared plots, use compare.py instead.

The script:
  1. Loads the CSV and separates the flagged train / validation splits.
  2. Builds the vocabulary (lengths + allele lookup) from the training split.
  3. Trains with MSE loss on log1p(thalf_hours).
  4. Reports metrics each epoch and saves the best checkpoint by val loss.
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from config import build_results, load_config, print_summary, train_from_config
from data import PeptideMHCDataset, Vocab, load_splits
from device_utils import move_batch, resolve_device, seed_everything
from model import DirectAffinityNet
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
) -> dict[str, float]:
    """Return log-space loss plus hour-space error metrics and correlations."""
    model.eval()
    total_loss, n = 0.0, 0
    preds_log, targets_log = [], []
    for batch in loader:
        batch = move_batch(batch, device)
        target = batch["target"]

        pred = model(batch["peptide"], batch["hla"], batch["allele"])
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
    mae_hours = float(np.mean(np.abs(preds_hours - targets_hours)))
    rmse_hours = float(np.sqrt(np.mean((preds_hours - targets_hours) ** 2)))
    pearson = float(np.corrcoef(preds_log, targets_log)[0, 1])
    spearman = _spearman(preds_log, targets_log)

    return {
        "loss": total_loss / n,
        "mae_hours": mae_hours,
        "rmse_hours": rmse_hours,
        "pearson": pearson,
        "spearman": spearman,
        "preds_hours": preds_hours,
        "targets_hours": targets_hours,
    }


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Rank correlation = Pearson of the ranks (no scipy dependency)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def train_model(
    csv: str,
    splits_csv: str,
    split: str,
    hla_col: str = "hla_seq",
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

    `split` selects which precomputed split to use (A/B/C/C2 from `splits_csv`);
    sequences are joined in from `csv`. `hla_col` selects the HLA representation
    ("hla_seq" or "hla_pseudoseq"), so the same code trains either variant.
    `device` is auto-selected (GPU if available, else CPU) unless forced.
    `history` holds per-epoch metrics, `best_eval` holds the best (by val loss)
    epoch's validation predictions, and `test_eval` holds that same best model's
    predictions on the held-out test split (None if the split has no test rows).
    """
    device = resolve_device(device)
    seed_everything(seed, device)
    prefix = f"[{tag or hla_col}]"
    print(f"{prefix} device = {device}")

    # 1. Load the chosen precomputed split (sequences joined in by row_id).
    train_df, val_df, test_df = load_splits(
        csv,
        splits_csv=splits_csv,
        split=split,
        train_value=train_value,
        val_value=val_value,
        test_value=test_value,
    )
    print(
        f"{prefix} split={split} train rows = {len(train_df)}, "
        f"val rows = {len(val_df)}, test rows = {len(test_df)}"
    )

    # 2. Vocabulary is built from the training split only, for this HLA column.
    vocab = Vocab.build(train_df, hla_col=hla_col)
    print(
        f"{prefix} hla_col={vocab.hla_col} peptide_len={vocab.peptide_len} "
        f"hla_len={vocab.hla_len} num_alleles={vocab.num_alleles}"
    )

    train_ds = PeptideMHCDataset(train_df, vocab)
    val_ds = PeptideMHCDataset(val_df, vocab)
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
            PeptideMHCDataset(test_df, vocab),
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin,
            num_workers=num_workers,
        )
        if len(test_df) > 0
        else None
    )

    # 3. Model, loss, optimizer.
    model = DirectAffinityNet(
        peptide_len=vocab.peptide_len,
        hla_len=vocab.hla_len,
        num_alleles=vocab.num_alleles,
    ).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )

    # 4. Training loop.
    best_val = float("inf")
    best_eval = None  # keep the best epoch's val predictions for the scatter plot
    best_state = None  # snapshot of the best-val weights, for the final test eval
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_mae_hours": [],
        "val_pearson": [],
    }
    for epoch in range(1, epochs + 1):
        model.train()
        running, n = 0.0, 0
        for batch in train_loader:
            batch = move_batch(batch, device)
            target = batch["target"]

            optimizer.zero_grad()           # clear gradients from the last step
            pred = model(batch["peptide"], batch["hla"], batch["allele"])  # forward
            loss = loss_fn(pred, target)    # mean squared error in log space
            loss.backward()                 # backprop the gradients
            optimizer.step()                # update the weights

            running += loss.item() * len(target)
            n += len(target)

        train_loss = running / n
        val = evaluate(model, val_loader, loss_fn, device)
        print(
            f"{prefix} epoch {epoch:3d} | train_loss {train_loss:.4f} | "
            f"val_loss {val['loss']:.4f} | val_MAE {val['mae_hours']:.2f} h | "
            f"val_pearson {val['pearson']:.3f}"
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val["loss"])
        history["val_mae_hours"].append(val["mae_hours"])
        history["val_pearson"].append(val["pearson"])

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
                        "vocab": vocab,
                        "config": {
                            "peptide_len": vocab.peptide_len,
                            "hla_len": vocab.hla_len,
                            "num_alleles": vocab.num_alleles,
                        },
                    },
                    out,
                )
                print(f"{prefix}          ↳ saved best model to {out}")

    print(f"{prefix} done. best val_loss = {best_val:.4f}")

    # 5. Final test evaluation with the best-val model (held-out until now).
    test_eval = None
    if test_loader is not None and best_state is not None:
        model.load_state_dict(best_state)
        test_eval = evaluate(model, test_loader, loss_fn, device)
        print(
            f"{prefix} TEST | loss {test_eval['loss']:.4f} | "
            f"MAE {test_eval['mae_hours']:.2f} h | "
            f"RMSE {test_eval['rmse_hours']:.2f} h | "
            f"pearson {test_eval['pearson']:.3f} | "
            f"spearman {test_eval['spearman']:.3f}"
        )

    return history, best_eval, test_eval


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label = cfg["label"]
    print(f"=== Training model: {label} ({cfg['hla_col']}) ===")

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
