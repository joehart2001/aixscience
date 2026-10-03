"""Train one or more models from YAML config files and plot the results.

Each YAML in templates/ fully describes one model run (which HLA column, the
dataset, and all hyperparameters). See templates/hla_seq.yaml for the schema.

    # Train a single model (writes per-model figures to its figs_dir):
    python compare.py templates/pseudoseq.yaml

    # Train several and overlay them on shared comparison plots:
    python compare.py templates/hla_seq.yaml templates/pseudoseq.yaml

With two or more configs you get the comparison figures (compare_*.png); with a
single config you get the standard single-model figures.
"""

from __future__ import annotations

import argparse

from config import build_results, load_config, print_summary, train_from_config
from plots import make_all_plots, make_test_plots, save_run_data
from train import train_model


def main() -> None:
    p = argparse.ArgumentParser(description="Train/compare models from YAML")
    p.add_argument("configs", nargs="+", help="One or more YAML config paths")
    args = p.parse_args()

    configs = [load_config(path) for path in args.configs]

    histories: dict[str, dict] = {}
    best_evals: dict[str, dict] = {}
    test_evals: dict[str, dict] = {}
    for cfg in configs:
        label = cfg["label"]
        print(f"\n=== Training model: {label} (split {cfg['split']}) ===")
        history, best_eval, test_eval = train_from_config(cfg, train_model)
        histories[label] = history
        best_evals[label] = best_eval
        test_evals[label] = test_eval

    # Base dir "figs": run data -> figs/data/, comparison figures -> figs/compare/
    # (make_all_plots appends the "compare" subfolder). Regenerate later with
    # `python plots.py --figs-dir figs`.
    figs_dir = "figs"

    # Persist the underlying data so plots.py can regenerate figures later,
    # then draw them now (single-model or comparison, chosen automatically).
    results = build_results(histories, best_evals, test_evals)
    save_run_data(results, figs_dir)
    make_all_plots(results, figs_dir)
    make_test_plots(results, figs_dir)
    print(
        f"\n[compare] saved data to {figs_dir}/data/, val figures to "
        f"{figs_dir}/compare/, test figures to {figs_dir}/test/"
    )

    print_summary(histories, test_evals)


if __name__ == "__main__":
    main()
