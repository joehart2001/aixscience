"""Train one or more models from YAML config files and plot the results.

    # one split -> single-model figures in that config's figs_dir
    python compare.py templates/random.yaml

    # all four splits -> overlaid comparison figures in figs/
    python compare.py templates/*.yaml
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        print(f"\n=== Training model: {label} ({cfg.get('hla_col', cfg['split'])}) ===")
        history, best_eval, test_eval = train_from_config(cfg, train_model)
        histories[label] = history
        best_evals[label] = best_eval
        test_evals[label] = test_eval

    # Base dir "figs": run data -> figs/data/, comparison figures -> figs/compare/
    # (make_all_plots appends the "compare" subfolder), test suite -> figs/test/.
    figs_dir = "figs"
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
