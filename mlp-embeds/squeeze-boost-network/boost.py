"""The 'boost' half: fit XGBoost on the SE-recalibrated features.

    python boost.py templates/split_C.yaml

Two stages, one config:

  1. Train the SE network exactly as `train.py` does (same engine, same
     best-by-val-loss checkpointing, same single final test evaluation).
  2. Freeze it, push every row through `model.features()` to get the
     recalibrated vector X~ = s * X, and fit a gradient-boosted tree ensemble
     on those features to predict the same target, log1p(thalf_hours).

Stage 2 is where the pairing earns its keep. Trees split on individual
features, so they benefit from an input where the informative channels have
already been amplified and the noisy ones damped — the gating is learned by
gradient descent, the thresholds on top of it by boosting.

Both stages are scored by `metrics_from_log` from train.py, so the MLP head and
the XGBoost head are measured by identical code, and both are written into one
run-data dir as separate labelled runs. That means the top-level
`figs/plots.py` can compare them (and the other networks) with no special
casing — point a template at this `figs_dir` with `run: <label> (SE+XGB)`.

Early stopping uses the validation split, which is the same split used to pick
the network checkpoint. The test split is touched exactly once, at the end.
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402
import torch  # noqa: E402
import xgboost as xgb  # noqa: E402

from config import build_results, load_config, print_summary, train_from_config
from data_seq import PeptideMHCDataset
from plots import make_all_plots, make_test_plots, save_run_data
from train import train_model
from train_common import metrics_from_log

# Defaults for the tree ensemble; override under `xgb:` in the YAML.
XGB_DEFAULTS = {
    "n_estimators": 2000,       # upper bound; early stopping picks the real count
    "learning_rate": 0.05,
    "max_depth": 6,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "early_stopping_rounds": 50,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train the SE network, then fit XGBoost on its features."
    )
    p.add_argument("config", help="Path to a YAML config file")
    return p.parse_args()


@torch.no_grad()
def extract_features(
    model, df, vocab, device, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    """Return (X~, target_log1p) for one dataframe, as numpy for XGBoost.

    Reuses `PeptideMHCDataset` so the encoding is bit-identical to training —
    re-encoding by hand here would be an easy way to introduce a silent skew
    between what the network saw and what the trees see.
    """
    model.eval()
    ds = PeptideMHCDataset(df, vocab)
    feats, targets = [], []
    for start in range(0, len(ds), batch_size):
        stop = min(start + batch_size, len(ds))
        idx = slice(start, stop)
        x = model.features(
            ds.peptide[idx].to(device),
            ds.hla[idx].to(device),
            ds.allele[idx].to(device),
        )
        feats.append(x.cpu().numpy())
        targets.append(ds.target[idx].numpy())
    return np.concatenate(feats), np.concatenate(targets)


def boost_curve(reg, X_val, y_val, alleles, max_points: int = 30) -> dict:
    """Validation metrics as a function of the number of boosting rounds.

    XGBoost has no epochs, but it does have a genuine learning curve: predicting
    with the first r trees gives the ensemble's state at round r. We walk that
    at a stride (capped at `max_points` evaluations, since each is a full
    predict over the validation split) rather than inventing a flat line.
    """
    n_rounds = reg.best_iteration + 1
    stride = max(1, n_rounds // max_points)
    rounds = list(range(stride, n_rounds + 1, stride))
    if rounds[-1] != n_rounds:
        rounds.append(n_rounds)

    curve = {
        "boost_rounds": rounds,
        "boost_val_loss": [],
        "boost_val_mae_hours": [],
        "boost_val_pearson": [],
    }
    for r in rounds:
        preds = reg.predict(X_val, iteration_range=(0, r))
        m = metrics_from_log(preds, y_val, alleles)
        # Same quantity as the network's val_loss: MSE in log1p space.
        curve["boost_val_loss"].append(float(np.mean((preds - y_val) ** 2)))
        curve["boost_val_mae_hours"].append(m["mae_hours"])
        curve["boost_val_pearson"].append(m["pearson"])
    return curve


def save_boost_curve(curve: dict, net_best_val_loss: float, figs_dir: str, label: str) -> None:
    """Plot val loss vs boosting rounds, with the network's best val loss as a line."""
    os.makedirs(os.path.join(figs_dir, "compare"), exist_ok=True)
    path = os.path.join(figs_dir, "compare", "boost_val_loss.png")

    plt.figure(figsize=(7, 5))
    plt.plot(curve["boost_rounds"], curve["boost_val_loss"], marker="o", label="SE+XGB")
    plt.axhline(
        net_best_val_loss,
        color="C1",
        linestyle="--",
        linewidth=1.2,
        label=f"SE+MLP best ({net_best_val_loss:.4f})",
    )
    plt.xlabel("boosting rounds")
    plt.ylabel("val MSE on log1p(thalf_hours)")
    plt.title(f"{label} — boosting curve (x is rounds, not epochs)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label = cfg["label"]
    print(f"=== Stage 1: SE network — {label} ({cfg['hla_col']}) ===")

    # --- Stage 1: the network, trained by the shared engine ------------------
    art: dict = {}
    history, best_eval, test_eval = train_from_config(
        cfg, lambda **kw: train_model(**kw, artifacts=art)
    )

    model, vocab, device = art["model"], art["vocab"], art["device"]
    if len(art["test_df"]) == 0:
        raise SystemExit("This split has no test rows; nothing to boost against.")

    # --- Stage 2: XGBoost on the recalibrated features -----------------------
    print(f"\n=== Stage 2: XGBoost on SE features — {label} ===")
    X_train, y_train = extract_features(model, art["train_df"], vocab, device)
    X_val, y_val = extract_features(model, art["val_df"], vocab, device)
    X_test, y_test = extract_features(model, art["test_df"], vocab, device)
    print(f"[boost] feature dim = {X_train.shape[1]}, train rows = {len(X_train)}")

    params = {**XGB_DEFAULTS, **(cfg["xgb"] or {})}
    reg = xgb.XGBRegressor(
        objective="reg:squarederror",
        random_state=cfg["seed"],
        n_jobs=0,  # 0 = use every core; this is the slow part on CPU
        **params,
    )
    # Early stopping on val — the same split that selected the network's
    # checkpoint, so the test split stays untouched until the very end.
    reg.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_rounds = reg.best_iteration + 1
    print(f"[boost] early stopping chose {best_rounds} of {params['n_estimators']} trees")

    # Score both heads with the same function the network's own eval uses.
    val_metrics = metrics_from_log(
        reg.predict(X_val), y_val, art["val_df"]["allele"].to_numpy()
    )
    test_metrics = metrics_from_log(
        reg.predict(X_test), y_test, art["test_df"]["allele"].to_numpy()
    )
    print(
        f"[boost] TEST | MAE {test_metrics['mae_hours']:.2f} h | "
        f"RMSE {test_metrics['rmse_hours']:.2f} h | "
        f"pearson {test_metrics['pearson']:.3f} | "
        f"spearman {test_metrics['spearman']:.3f} | "
        f"within_rho {test_metrics['within_allele_spearman']:.3f}"
    )

    # A real learning curve for the boosted model, measured on val at a stride
    # of boosting rounds. Its x-axis is *boosting rounds*, not epochs, so these
    # go under distinct keys: overlaying them on the network's epoch axis would
    # put two different units on one plot. The comparison that is actually
    # like-for-like is the test metrics, not the curves.
    # metrics_from_log reports no loss term; build_results and the plots expect
    # the key to exist, so supply the log-space MSE that the network reports.
    val_metrics["loss"] = float(np.mean((reg.predict(X_val) - y_val) ** 2))
    test_metrics["loss"] = float(np.mean((reg.predict(X_test) - y_test) ** 2))

    xgb_history = boost_curve(
        reg, X_val, y_val, art["val_df"]["allele"].to_numpy()
    )
    save_boost_curve(
        xgb_history,
        min(history["val_loss"]),
        cfg["figs_dir"],
        label,
    )

    # --- Persist both heads as two labelled runs in one figs dir -------------
    mlp_label, xgb_label = f"{label} (SE+MLP)", f"{label} (SE+XGB)"
    results = build_results(
        {mlp_label: history, xgb_label: xgb_history},
        {mlp_label: best_eval, xgb_label: val_metrics},
        {mlp_label: test_eval, xgb_label: test_metrics},
    )
    figs_dir = cfg["figs_dir"]
    save_run_data(results, figs_dir)
    # Only the network goes on the epoch-axis curve panels; the boosted model
    # gets its own rounds-axis figure above.
    make_all_plots({mlp_label: results[mlp_label]}, figs_dir)
    make_test_plots(results, figs_dir)
    print(f"\n[boost] saved figures + data to {figs_dir}/")

    # print_summary's val columns are per-epoch, so only the network has them;
    # both heads still appear in its test table.
    print_summary(
        {mlp_label: history}, {mlp_label: test_eval, xgb_label: test_metrics}
    )


if __name__ == "__main__":
    main()
