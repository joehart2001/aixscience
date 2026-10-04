"""Shared YAML-config handling for every network's config.py.

Kept import-light (only `yaml`) so entry points can use it without circular
imports. Each network's `config.py` defines its own `DEFAULTS` + `train_from_config`
and imports `load_config`, `build_results`, `print_summary` from here.
"""

from __future__ import annotations

import yaml


def load_config(path: str, defaults: dict) -> dict:
    """Read one YAML config and fill in any missing keys with the caller's `defaults`.

    Each network's config.py wraps this with its own DEFAULTS so framework-specific
    keys apply. `csv` is required; `label` falls back to the file path.
    """
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "csv" not in cfg:
        raise ValueError(f"{path}: missing required key 'csv'")
    merged = {**defaults, **cfg}
    merged.setdefault("label", path)
    return merged


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
                "within_allele_spearman": test.get(
                    "within_allele_spearman", float("nan")
                ),
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
        print("\nmodel            | test_loss | test_MAE(h) | test_RMSE(h) | test_Pearson | test_Spearman | within-allele_rho")
        print("-" * 108)
        for label, t in test_evals.items():
            print(
                f"{label:16s} | {t['loss']:9.4f} | {t['mae_hours']:11.2f} | "
                f"{t['rmse_hours']:12.2f} | {t['pearson']:12.3f} | {t['spearman']:13.3f} | "
                f"{t.get('within_allele_spearman', float('nan')):17.3f}"
            )
