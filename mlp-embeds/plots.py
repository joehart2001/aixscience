"""Figures for the training process, saved to a figs/ folder.

Uses the non-interactive "Agg" backend so it works without a display.
"""

from __future__ import annotations

import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402


# --- Published reference performance, drawn as horizontal lines for context ----
# Directly-comparable SOTA for pMHC-I *stability* (half-life) regression under a
# leakage-controlled (similarity-aware) split — the same regime as our B/C/C2:
#   "Peptide:MHC Binding Stability Prediction Using Protein Language Models",
#   bioRxiv 2026, doi:10.64898/2026.06.28.735023. Best model (MINT Transfer) on
#   the NetMHCstabpan test set with 80%-identity peptide-cluster splits:
#   Pearson r = 0.76, Spearman ρ = 0.79.
# For reference, the classic NetMHCstabpan (Rasmussen et al., J Immunol 2016,
# doi:10.4049/jimmunol.1600582) pan-specific PCC = 0.676 (global rescaling); it
# scores ρ = 0.88 on the above test set but that is leakage-inflated.
# Keyed by metric so each panel gets the comparable line; set a value to None or
# remove the key to hide it.
REFERENCES = {
    "pearson": (0.693, r"NetMHCstabpan stability $r=0.76$ (Rasmussen et. al, 2016)"),
    # no directly-comparable published within-allele ρ, so none is drawn there
}


def _add_reference(metric: str, ax=None) -> None:
    """Draw the published-reference line for a given metric ('pearson'/'spearman')."""
    ref = REFERENCES.get(metric)
    if not ref:
        return
    value, label = ref
    (ax or plt).axhline(
        value, color="black", linestyle="--", linewidth=3, label=label
    )


def save_training_curves(history: dict[str, list[float]], figs_dir: str) -> None:
    """Plot train/val loss, val MAE, and val Pearson over epochs."""
    os.makedirs(figs_dir, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves (train vs validation).
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["train_loss"], label="train loss", marker="o")
    plt.plot(epochs, history["val_loss"], label="val loss (OOD)", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("MSE on log1p(thalf_hours)")
    plt.title("Training / validation loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "loss_curve.png"), dpi=150)
    plt.close()

    # Validation MAE in hours.
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_mae_hours"], color="C2", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("MAE (hours)")
    plt.title("Validation MAE (out-of-domain proteins)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "val_mae.png"), dpi=150)
    plt.close()

    # One validation-correlation curve per metric that's present in history:
    # global Pearson, global Spearman, and within-allele ρ.
    corr_panels = [
        ("val_pearson", "Pearson r (global)", "val_pearson.png", "C3", "pearson"),
        ("val_spearman", "Spearman ρ (global)", "val_spearman.png", "C4", "spearman"),
        ("val_within_rho", "within-allele ρ", "val_within_rho.png", "C5", None),
    ]
    for key, ylabel, fname, color, ref_metric in corr_panels:
        if not history.get(key):
            continue
        plt.figure(figsize=(7, 5))
        plt.plot(epochs, history[key], color=color, marker="o", label=ylabel)
        if ref_metric:
            _add_reference(ref_metric)  # SOTA stability reference
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"Validation {ylabel}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, fname), dpi=150)
        plt.close()


def save_prediction_scatter(
    preds_hours: np.ndarray,
    targets_hours: np.ndarray,
    figs_dir: str,
    pearson: float,
) -> None:
    """Scatter of predicted vs. actual half-life on the validation set."""
    os.makedirs(figs_dir, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(targets_hours, preds_hours, s=8, alpha=0.3)
    lo = 0.0
    hi = float(max(targets_hours.max(), preds_hours.max()))
    plt.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
    plt.xlabel("actual thalf_hours")
    plt.ylabel("predicted thalf_hours")
    plt.title(f"Validation predictions (Pearson r = {pearson:.3f})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "pred_vs_actual.png"), dpi=150)
    plt.close()


# --- Comparison plots: overlay several models on the same axes ---------------


def save_comparison_curves(
    histories: dict[str, dict[str, list[float]]], figs_dir: str
) -> None:
    """Overlay the per-epoch curves of several models (label -> history)."""
    os.makedirs(figs_dir, exist_ok=True)

    # (metric key, y-axis label, output filename) for each curve we compare.
    panels = [
        ("val_loss", "val MSE on log1p(thalf)", "compare_val_loss.png"),
        ("val_mae_hours", "val MAE (hours)", "compare_val_mae.png"),
        ("val_pearson", "val Pearson r (global)", "compare_val_pearson.png"),
        ("val_spearman", "val Spearman ρ (global)", "compare_val_spearman.png"),
        ("val_within_rho", "val within-allele ρ", "compare_val_within_rho.png"),
    ]
    ref_metric_of = {"val_pearson": "pearson", "val_spearman": "spearman"}
    for key, ylabel, fname in panels:
        # Skip a metric unless every model tracked it (older runs may not have it).
        if not all(h.get(key) for h in histories.values()):
            continue
        plt.figure(figsize=(7, 5))
        # Draw one line per model so they sit on identical axes.
        for label, history in histories.items():
            epochs = range(1, len(history[key]) + 1)
            plt.plot(epochs, history[key], marker="o", label=label)
        if key in ref_metric_of:
            _add_reference(ref_metric_of[key])  # SOTA stability reference
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.title(f"Model comparison: {ylabel}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(figs_dir, fname), dpi=150)
        plt.close()


def save_comparison_scatter(
    results: dict[str, dict], figs_dir: str
) -> None:
    """Side-by-side predicted-vs-actual scatter, one panel per model."""
    os.makedirs(figs_dir, exist_ok=True)

    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    for ax, (label, ev) in zip(axes.flatten(), results.items()):
        targets = ev["targets_hours"]
        preds = ev["preds_hours"]
        ax.scatter(targets, preds, s=8, alpha=0.3)
        lo = float(min(targets.min(), preds.min()))
        hi = float(max(targets.max(), preds.max()))
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
        ax.set_xlabel("actual thalf_hours")
        ax.set_ylabel("predicted thalf_hours")
        ax.set_title(f"{label} (Pearson r = {ev['pearson']:.3f})")
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(figs_dir, "compare_pred_vs_actual.png"), dpi=150)
    plt.close(fig)


# --- Persisting run data so plots can be regenerated without retraining -------

# Metrics + predictions are written under <figs_dir>/data/ so `python plots.py`
# can rebuild every figure from disk.
DATA_SUBDIR = "data"


def _slug(label: str) -> str:
    """Filesystem-safe version of a model label (e.g. 'full hla_seq' -> 'full_hla_seq')."""
    return "".join(c if c.isalnum() else "_" for c in label).strip("_").lower()


def save_run_data(results: dict[str, dict], figs_dir: str) -> None:
    """Persist each run's history + validation predictions to <figs_dir>/data/.

    `results` maps label -> {history, preds_hours, targets_hours, pearson}.
    Metrics go in one runs.json; the prediction arrays go in per-run .npz files.
    """
    data_dir = os.path.join(figs_dir, DATA_SUBDIR)
    os.makedirs(data_dir, exist_ok=True)

    # Clear stale arrays from a previous run so the dir matches runs.json exactly.
    for fname in os.listdir(data_dir):
        if fname.endswith(".npz"):
            os.remove(os.path.join(data_dir, fname))

    manifest = []
    for label, r in results.items():
        slug = _slug(label)
        # Arrays are saved separately because JSON can't hold them compactly.
        arrays = {
            "preds_hours": r["preds_hours"],
            "targets_hours": r["targets_hours"],
        }
        entry = {
            "label": label,
            "slug": slug,
            "pearson": r["pearson"],
            "history": r["history"],
        }
        # Include test predictions + metrics when the run has a test split.
        test = r.get("test")
        if test is not None:
            arrays["test_preds_hours"] = test["preds_hours"]
            arrays["test_targets_hours"] = test["targets_hours"]
            entry["test_metrics"] = {
                k: test[k]
                for k in ("loss", "mae_hours", "rmse_hours", "pearson", "spearman", "within_allele_spearman")
            }
        np.savez(os.path.join(data_dir, f"{slug}.npz"), **arrays)
        manifest.append(entry)
    with open(os.path.join(data_dir, "runs.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def load_run_data(figs_dir: str) -> dict[str, dict]:
    """Inverse of save_run_data: rebuild the results dict from <figs_dir>/data/."""
    data_dir = os.path.join(figs_dir, DATA_SUBDIR)
    runs_path = os.path.join(data_dir, "runs.json")
    if not os.path.exists(runs_path):
        raise FileNotFoundError(
            f"No saved run data at {runs_path}. Run compare.py first to "
            f"produce it, then rerun this to regenerate plots."
        )
    with open(runs_path) as f:
        manifest = json.load(f)

    results: dict[str, dict] = {}
    for entry in manifest:
        arrays = np.load(os.path.join(data_dir, f"{entry['slug']}.npz"))
        res = {
            "history": entry["history"],
            "preds_hours": arrays["preds_hours"],
            "targets_hours": arrays["targets_hours"],
            "pearson": entry["pearson"],
        }
        if "test_preds_hours" in arrays:
            res["test"] = {
                "preds_hours": arrays["test_preds_hours"],
                "targets_hours": arrays["test_targets_hours"],
                **entry.get("test_metrics", {}),
            }
        results[entry["label"]] = res
    return results


def make_all_plots(results: dict[str, dict], figs_dir: str) -> None:
    """Draw the right figures: single-model if one run, comparison if several."""
    figs_dir = f"{figs_dir}/compare"
    if len(results) == 1:
        (r,) = results.values()
        save_training_curves(r["history"], figs_dir)
        save_prediction_scatter(
            r["preds_hours"], r["targets_hours"], figs_dir, r["pearson"]
        )
    else:
        histories = {label: r["history"] for label, r in results.items()}
        save_comparison_curves(histories, figs_dir)
        save_comparison_scatter(results, figs_dir)


# --- Test-set evaluation suite (written to <figs_dir>/test/) ------------------


def _residuals_log(test: dict) -> tuple[np.ndarray, np.ndarray]:
    """Return (actual_hours, residual) where residual = predicted - actual in log space."""
    actual = test["targets_hours"]
    resid = np.log1p(test["preds_hours"]) - np.log1p(actual)
    return actual, resid


def _test_title(test: dict, label: str) -> str:
    return (
        f"{label} — TEST\n"
        f"Pearson {test['pearson']:.3f} | Spearman {test['spearman']:.3f} | "
        f"MAE {test['mae_hours']:.2f} h | RMSE {test['rmse_hours']:.2f} h"
    )


def _save_test_scatter(test: dict, path: str, label: str) -> None:
    """Predicted vs. actual half-life on the test set (log-log)."""
    targets, preds = test["targets_hours"], test["preds_hours"]
    plt.figure(figsize=(6, 6))
    plt.scatter(targets, preds, s=8, alpha=0.3)
    hi = float(max(targets.max(), preds.max()))
    plt.plot([0, hi], [0, hi], "k--", linewidth=1, label="perfect")
    plt.xlabel("actual thalf_hours")
    plt.ylabel("predicted thalf_hours")
    plt.xscale("log")
    plt.yscale("log")
    plt.title(_test_title(test, label))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_test_residuals(test: dict, path: str, label: str) -> None:
    """Residual (log-space pred - actual) vs. actual — shows bias across the range."""
    actual, resid = _residuals_log(test)
    plt.figure(figsize=(7, 5))
    plt.scatter(actual, resid, s=8, alpha=0.3)
    plt.axhline(0, color="k", linewidth=1)
    plt.xscale("log")
    plt.xlabel("actual thalf_hours")
    plt.ylabel("residual  log1p(pred) - log1p(actual)")
    plt.title(f"{label} — TEST residuals (>0 over-predicts, <0 under-predicts)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_test_residual_hist(test: dict, path: str, label: str) -> None:
    """Distribution of the log-space residuals (ideally centered on 0)."""
    _, resid = _residuals_log(test)
    plt.figure(figsize=(7, 5))
    plt.hist(resid, bins=40, alpha=0.8)
    plt.axvline(0, color="k", linewidth=1)
    plt.axvline(float(resid.mean()), color="C3", linestyle="--", linewidth=1,
                label=f"mean {resid.mean():.3f}")
    plt.xlabel("residual  log1p(pred) - log1p(actual)")
    plt.ylabel("count")
    plt.title(f"{label} — TEST residual distribution")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _save_test_metric_bars(tests: dict[str, dict], path: str) -> None:
    """Grouped bars comparing test metrics across models (correlations + errors)."""
    labels = list(tests)
    x = range(len(labels))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6 + 2 * len(labels), 5))

    # Three correlation bars: global Pearson/Spearman vs. within-allele ρ (the
    # honest ranking metric — near 0 for a model that only learns allele baselines).
    width = 0.27
    ax1.bar([i - width for i in x], [tests[l]["pearson"] for l in labels],
            width, label="Pearson (global)")
    ax1.bar(list(x), [tests[l]["spearman"] for l in labels],
            width, label="Spearman (global)")
    ax1.bar([i + width for i in x],
            [tests[l].get("within_allele_spearman", float("nan")) for l in labels],
            width, label="within-allele ρ")
    _add_reference("pearson", ax1)   # SOTA stability Pearson reference
    _add_reference("spearman", ax1)  # SOTA stability Spearman reference
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(labels, rotation=20, ha="right")
    ax1.set_ylabel("correlation (higher = better)")
    ax1.set_title("Test correlation (global vs. within-allele)")
    ax1.legend()
    ax1.grid(True, axis="y", alpha=0.3)

    ax2.bar([i - width / 2 for i in x], [tests[l]["mae_hours"] for l in labels],
            width, label="MAE (h)")
    ax2.bar([i + width / 2 for i in x], [tests[l]["rmse_hours"] for l in labels],
            width, label="RMSE (h)")
    ax2.set_xticks(list(x))
    ax2.set_xticklabels(labels, rotation=20, ha="right")
    ax2.set_ylabel("error in hours (lower = better)")
    ax2.set_title("Test error")
    ax2.legend()
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_test_plots(results: dict[str, dict], figs_dir: str) -> None:
    """Draw the test-set suite for every run that has test data, into <figs_dir>/test/.

    Per model: predicted-vs-actual scatter, residuals-vs-actual, residual
    histogram. With several models, also a grouped metric-comparison bar chart.
    Runs without a test split (e.g. old data) are silently skipped.
    """
    tests = {label: r["test"] for label, r in results.items() if r.get("test")}
    if not tests:
        return
    test_dir = os.path.join(figs_dir, "test")
    os.makedirs(test_dir, exist_ok=True)

    multi = len(tests) > 1
    for label, t in tests.items():
        pre = f"{_slug(label)}_" if multi else ""
        _save_test_scatter(t, os.path.join(test_dir, f"{pre}test_pred_vs_actual.png"), label)
        _save_test_residuals(t, os.path.join(test_dir, f"{pre}test_residuals.png"), label)
        _save_test_residual_hist(t, os.path.join(test_dir, f"{pre}test_residual_hist.png"), label)
    if multi:
        _save_test_metric_bars(tests, os.path.join(test_dir, "compare_test_metrics.png"))


def main() -> None:
    """Regenerate all figures from previously saved run data (no retraining)."""
    p = argparse.ArgumentParser(
        description="Regenerate figures from saved run data in <figs-dir>/data/"
    )
    p.add_argument("--figs-dir", default="figs", help="Folder holding data/ + figures")
    args = p.parse_args()

    results = load_run_data(args.figs_dir)
    make_all_plots(results, args.figs_dir)
    make_test_plots(results, args.figs_dir)
    print(
        f"Regenerated figures in {args.figs_dir}/ from "
        f"{len(results)} saved run(s): {', '.join(results)}"
    )


if __name__ == "__main__":
    main()
