"""Shared YAML-config handling for train.py and compare.py.

Kept import-light (only `yaml`) so both entry points can use it without circular
imports. `train_from_config` takes the training function as an argument rather
than importing it here, which is what keeps this module dependency-free.
"""

from __future__ import annotations

from typing import Callable

import yaml

# Defaults applied when a YAML omits a key, so configs can stay minimal.
DEFAULTS = {
    "splits_csv": "../Data/subsets/splits.csv",  # precomputed A/B/C/C2 splits
    "split": "A",          # which split strategy: A, B, C, or C2
    "hla_col": "hla_seq",
    "train_value": "train",
    "val_value": "val",
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.00001,
    "seed": 0,
    "device": "auto",      # CUDA if available (prod GPU), else CPU (local)
    "num_workers": 0,
    "figs_dir": "figs",
    "out": None,
}


def load_config(path: str) -> dict:
    """Read one YAML config and fill in any missing keys with the defaults."""
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "csv" not in cfg:
        raise ValueError(f"{path}: missing required key 'csv'")
    merged = {**DEFAULTS, **cfg}
    # Fall back to the file path as the label if none was given.
    merged.setdefault("label", path)
    return merged


def train_from_config(cfg: dict, train_fn: Callable) -> tuple[dict, dict]:
    """Call `train_fn` (i.e. train.train_model) with the parameters from `cfg`."""
    return train_fn(
        csv=cfg["csv"],
        splits_csv=cfg["splits_csv"],
        split=cfg["split"],
        hla_col=cfg["hla_col"],
        train_value=cfg["train_value"],
        val_value=cfg["val_value"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        device=cfg["device"],
        num_workers=cfg["num_workers"],
        out=cfg["out"],
        tag=cfg["label"],
    )


def build_results(
    histories: dict[str, dict],
    best_evals: dict[str, dict],
    test_evals: dict[str, dict] | None = None,
) -> dict:
    """Combine per-label histories + validation (and optional test) predictions.

    Validation predictions stay at the top level (unchanged); test predictions,
    when present, go in a nested `test` dict.
    """
    test_evals = test_evals or {}
    results = {}
    for label in histories:
        entry = {
            "history": histories[label],
            "preds_hours": best_evals[label]["preds_hours"],
            "targets_hours": best_evals[label]["targets_hours"],
            "pearson": best_evals[label]["pearson"],
        }
        test = test_evals.get(label)
        if test is not None:
            entry["test"] = {
                "preds_hours": test["preds_hours"],
                "targets_hours": test["targets_hours"],
                "loss": test["loss"],
                "mae_hours": test["mae_hours"],
                "rmse_hours": test["rmse_hours"],
                "pearson": test["pearson"],
                "spearman": test["spearman"],
            }
        results[label] = entry
    return results


def print_summary(
    histories: dict[str, dict], test_evals: dict[str, dict] | None = None
) -> None:
    """Print the final per-model validation (and test, if available) metrics."""
    print("\nmodel            | best val_loss | final val_MAE(h) | final Pearson")
    print("-" * 66)
    for label, h in histories.items():
        best_loss = min(h["val_loss"])
        print(
            f"{label:16s} | {best_loss:13.4f} | "
            f"{h['val_mae_hours'][-1]:16.2f} | {h['val_pearson'][-1]:13.3f}"
        )

    test_evals = {k: v for k, v in (test_evals or {}).items() if v is not None}
    if test_evals:
        print("\nmodel            | test_loss | test_MAE(h) | test_RMSE(h) | test_Pearson | test_Spearman")
        print("-" * 90)
        for label, t in test_evals.items():
            print(
                f"{label:16s} | {t['loss']:9.4f} | {t['mae_hours']:11.2f} | "
                f"{t['rmse_hours']:12.2f} | {t['pearson']:12.3f} | {t['spearman']:13.3f}"
            )
