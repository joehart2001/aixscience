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

    # Validation Pearson correlation.
    plt.figure(figsize=(7, 5))
    plt.plot(epochs, history["val_pearson"], color="C3", marker="o")
    plt.xlabel("epoch")
    plt.ylabel("Pearson r (log space)")
    plt.title("Validation correlation (out-of-domain proteins)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(figs_dir, "val_pearson.png"), dpi=150)
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
        ("val_pearson", "val Pearson r (log space)", "compare_val_pearson.png"),
    ]
    for key, ylabel, fname in panels:
        plt.figure(figsize=(7, 5))
        # Draw one line per model so they sit on identical axes.
        for label, history in histories.items():
            epochs = range(1, len(history[key]) + 1)
            plt.plot(epochs, history[key], marker="o", label=label)
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

    n = 1
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6), squeeze=False)
    ax = axes[0, 0]
    hi = 0
    lo = 1e5
    for (label, ev) in results.items():
        targets = ev["targets_hours"]
        preds = ev["preds_hours"]
        ax.scatter(targets, preds, s=8, alpha=0.3, label = f"{label} (Pearson r = {ev['pearson']:.3f})")
        lo = float(min(targets.min(), preds.min(), lo))
        hi = float(max(targets.max(), preds.max(), hi))
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfect")
    ax.set_xlabel("actual thalf_hours")
    ax.set_ylabel("predicted thalf_hours")
    plt.yscale("log")
    plt.xscale("log")
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
        np.savez(
            os.path.join(data_dir, f"{slug}.npz"),
            preds_hours=r["preds_hours"],
            targets_hours=r["targets_hours"],
        )
        manifest.append(
            {
                "label": label,
                "slug": slug,
                "pearson": r["pearson"],
                "history": r["history"],
            }
        )
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
        results[entry["label"]] = {
            "history": entry["history"],
            "preds_hours": arrays["preds_hours"],
            "targets_hours": arrays["targets_hours"],
            "pearson": entry["pearson"],
        }
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


def main() -> None:
    """Regenerate all figures from previously saved run data (no retraining)."""
    p = argparse.ArgumentParser(
        description="Regenerate figures from saved run data in <figs-dir>/data/"
    )
    p.add_argument("--figs-dir", default="figs", help="Folder holding data/ + figures")
    args = p.parse_args()

    results = load_run_data(args.figs_dir)
    make_all_plots(results, args.figs_dir)
    print(
        f"Regenerated figures in {args.figs_dir}/ from "
        f"{len(results)} saved run(s): {', '.join(results)}"
    )


if __name__ == "__main__":
    main()
