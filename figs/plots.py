"""Inter-model comparison figures on the **test set**, across networks.

Each sibling network (`direct-network/`, `transformer-network/`,
`boltz-network/`, ...) persists its own runs to `<net>/figs/data/` — metrics in
`runs.json`, prediction arrays in per-run `.npz`. This script reads those files
and draws one comparison per *split*, so you can see how the architectures
differ on identical held-out rows. Nothing is retrained.

Usage (from aixscience/figs/):

    PY=../../.venv/bin/python
    $PY plots.py templates/split_C.yaml                 # one split
    $PY plots.py templates/*.yaml                       # every split + summary

One YAML per split lives in `templates/`; it names the runs to pull in. Output
goes to `compare/<out>/` plus, when several templates are given, an across-split
summary in `compare/`.

Loading and the metric-bar chart are reused from `direct-network/plots.py` (via
a sys.path insert, as c2-committee does with train.py) so a figure drawn here is
drawn by exactly the same code as the per-network ones.
"""

from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402
import yaml  # noqa: E402

# Reuse the per-network plotting helpers rather than reimplementing them.
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "direct-network"))
from plots import (  # noqa: E402
    _add_reference,
    _residuals_log,
    _save_test_metric_bars,
    _slug,
    load_run_data,
)

# Metrics shown in the across-split summary and the markdown table.
# (key, display name, "higher is better")
SUMMARY_METRICS = [
    ("pearson", "Pearson r", True),
    ("spearman", "Spearman ρ", True),
    ("within_allele_spearman", "within-allele ρ", True),
    ("mae_hours", "MAE (h)", False),
    ("rmse_hours", "RMSE (h)", False),
]


def load_spec(path: str) -> dict:
    """Read one comparison YAML and resolve its paths relative to the YAML."""
    with open(path) as f:
        spec = yaml.safe_load(f) or {}
    for key in ("label", "models"):
        if key not in spec:
            raise ValueError(f"{path}: missing required key '{key}'")
    # `out` defaults to a filesystem-safe version of the label.
    spec.setdefault("out", _slug(spec["label"]))
    spec.setdefault("caption", None)
    for m in spec["models"]:
        for key in ("label", "figs_dir", "run"):
            if key not in m:
                raise ValueError(f"{path}: model entry missing '{key}': {m}")
    # `figs_dir` is relative to the working directory, matching how the
    # per-network templates write `csv: ../Data/...` (i.e. run from figs/).
    return spec


def collect_tests(spec: dict) -> dict[str, dict]:
    """Return {model label -> that run's test dict} for one comparison spec.

    Raises if a named run has no test data, rather than silently dropping it —
    a missing model would quietly change what the comparison means.
    """
    tests: dict[str, dict] = {}
    for m in spec["models"]:
        results = load_run_data(m["figs_dir"])
        if m["run"] not in results:
            raise KeyError(
                f"Run {m['run']!r} not in {m['figs_dir']}/data/runs.json. "
                f"Available: {', '.join(results)}"
            )
        test = results[m["run"]].get("test")
        if test is None:
            raise KeyError(
                f"Run {m['run']!r} in {m['figs_dir']} has no saved test data; "
                f"retrain it so the test split is evaluated and persisted."
            )
        tests[m["label"]] = test
    return tests


def _check_same_rows(tests: dict[str, dict], spec_label: str) -> bool:
    """Warn unless every model was tested on identical rows.

    Comparing models across *different* test sets is the easiest way to draw a
    misleading bar chart (boltz-network, for instance, only covers HLA-A*02:01).
    We can't see row ids here, but identical targets in identical order is a
    strong check, so flag it loudly when it fails.
    """
    ref_label, ref = next(iter(tests.items()))
    for label, t in tests.items():
        same = len(t["targets_hours"]) == len(ref["targets_hours"]) and np.allclose(
            t["targets_hours"], ref["targets_hours"]
        )
        if not same:
            print(
                f"  WARNING [{spec_label}]: {label!r} was tested on different "
                f"rows than {ref_label!r} ({len(t['targets_hours'])} vs "
                f"{len(ref['targets_hours'])} rows) — metrics are NOT directly "
                f"comparable.",
                file=sys.stderr,
            )
            return False
    return True


def _footnote(fig, caption: str | None) -> None:
    """Draw an optional caveat under a figure (e.g. 'noisy: only 22 clusters')."""
    if caption:
        fig.text(0.5, 0.005, caption, ha="center", fontsize=9, style="italic")


# --- Per-split figures -------------------------------------------------------


def save_test_scatter_panels(
    tests: dict[str, dict], path: str, title: str, caption: str | None
) -> None:
    """Predicted-vs-actual on the test set, one panel per model, shared axes.

    Log-log, and every panel gets the *same* limits so the models are visually
    comparable rather than each being auto-scaled to flatter itself.
    """
    n = len(tests)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.8), squeeze=False)
    # Shared limits across panels, computed over every model's points.
    hi = max(
        float(max(t["targets_hours"].max(), t["preds_hours"].max()))
        for t in tests.values()
    )
    for ax, (label, t) in zip(axes.flatten(), tests.items()):
        targets, preds = t["targets_hours"], t["preds_hours"]
        ax.scatter(targets, preds, s=8, alpha=0.3)
        ax.plot([0, hi], [0, hi], "k--", linewidth=1, label="perfect")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("actual thalf_hours")
        ax.set_ylabel("predicted thalf_hours")
        ax.set_title(
            f"{label}\nPearson {t['pearson']:.3f} | "
            f"within-allele ρ {t.get('within_allele_spearman', float('nan')):.3f}"
        )
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    fig.tight_layout()
    _footnote(fig, caption)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_residual_overlay(
    tests: dict[str, dict], path: str, title: str, caption: str | None
) -> None:
    """Overlaid log-space residual distributions — who is biased, and which way.

    A model that has given up and predicts the dataset mean shows a narrow spike
    off-centre; a well-calibrated one is centred on 0.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    # One shared bin edge set so the histograms are genuinely comparable.
    all_resid = np.concatenate([_residuals_log(t)[1] for t in tests.values()])
    bins = np.linspace(float(all_resid.min()), float(all_resid.max()), 50)
    for label, t in tests.items():
        resid = _residuals_log(t)[1]
        ax.hist(resid, bins=bins, alpha=0.45, label=f"{label} (mean {resid.mean():+.3f})")
    ax.axvline(0, color="k", linewidth=1)
    ax.set_xlabel("residual  log1p(pred) - log1p(actual)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    _footnote(fig, caption)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_split_figures(spec: dict, out_root: str) -> dict[str, dict]:
    """Draw every per-split comparison figure; return the tests for the summary."""
    tests = collect_tests(spec)
    _check_same_rows(tests, spec["label"])

    out_dir = os.path.join(out_root, spec["out"])
    os.makedirs(out_dir, exist_ok=True)
    caption = spec["caption"]

    # Grouped metric bars — reused verbatim from direct-network/plots.py, so the
    # SOTA reference lines and metric choice match the per-network figures.
    _save_test_metric_bars(tests, os.path.join(out_dir, "compare_test_metrics.png"))
    save_test_scatter_panels(
        tests,
        os.path.join(out_dir, "compare_test_pred_vs_actual.png"),
        f"{spec['label']} — test predictions",
        caption,
    )
    save_residual_overlay(
        tests,
        os.path.join(out_dir, "compare_test_residual_hist.png"),
        f"{spec['label']} — test residual distribution",
        caption,
    )
    print(f"  {spec['label']}: {len(tests)} models -> {out_dir}/")
    return tests


# --- Across-split summary ----------------------------------------------------


def save_summary_bars(all_tests: dict[str, dict[str, dict]], path: str) -> None:
    """One panel per metric; x = split, one bar colour per model.

    This is the headline figure: it answers "which architecture wins, and does
    the answer change as the split gets harder?" in a single read.
    """
    splits = list(all_tests)
    # Union of model labels, in first-seen order, so a model missing from one
    # split (e.g. boltz has no C/C2) simply has no bar there.
    models: list[str] = []
    for tests in all_tests.values():
        for label in tests:
            if label not in models:
                models.append(label)

    metrics = [m for m in SUMMARY_METRICS if m[0] != "rmse_hours"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.5 * len(metrics), 5.2))
    width = 0.8 / max(len(models), 1)
    for ax, (key, name, higher_better) in zip(np.atleast_1d(axes), metrics):
        for i, model in enumerate(models):
            # NaN leaves a gap rather than a misleading zero-height bar.
            values = [
                all_tests[s].get(model, {}).get(key, float("nan")) for s in splits
            ]
            offset = (i - (len(models) - 1) / 2) * width
            ax.bar([x + offset for x in range(len(splits))], values, width, label=model)
        if key in ("pearson", "spearman"):
            _add_reference(key, ax)  # SOTA stability reference
        ax.axhline(0, color="k", linewidth=0.8)
        ax.set_xticks(range(len(splits)))
        ax.set_xticklabels(splits, rotation=20, ha="right")
        ax.set_ylabel(f"{name} ({'higher' if higher_better else 'lower'} = better)")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Test-set performance by architecture and split")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_summary_table(all_tests: dict[str, dict[str, dict]], path: str) -> None:
    """Write the same numbers as a markdown table, for pasting into notes."""
    lines = ["# Test-set metrics by split and model", ""]
    for split, tests in all_tests.items():
        lines += [
            f"## {split}",
            "",
            "| model | " + " | ".join(n for _, n, _ in SUMMARY_METRICS) + " |",
            "|---" * (len(SUMMARY_METRICS) + 1) + "|",
        ]
        for label, t in tests.items():
            cells = [f"{t.get(k, float('nan')):.3f}" for k, _, _ in SUMMARY_METRICS]
            lines.append(f"| {label} | " + " | ".join(cells) + " |")
        lines.append("")
    with open(path, "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        description=(
            "Inter-model test-set comparison figures, built from each network's "
            "saved figs/data/ (no retraining)."
        )
    )
    p.add_argument(
        "configs", nargs="+", help="Comparison YAML(s), e.g. templates/split_C.yaml"
    )
    p.add_argument("--out", default="compare", help="Output root folder")
    p.add_argument(
        "--summary-prefix",
        default="summary",
        help=(
            "Filename stem for the across-split summary. Give separate "
            "comparison families their own prefix, or the second run "
            "overwrites the first's summary."
        ),
    )
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    all_tests: dict[str, dict[str, dict]] = {}
    for path in args.configs:
        spec = load_spec(path)
        all_tests[spec["label"]] = make_split_figures(spec, args.out)

    # The summary only says something once there is more than one split.
    if len(all_tests) > 1:
        stem = os.path.join(args.out, f"{args.summary_prefix}_test_metrics")
        save_summary_bars(all_tests, f"{stem}.png")
        save_summary_table(all_tests, f"{stem}.md")
        print(f"  summary across {len(all_tests)} splits -> {args.out}/")
    print(f"Done. Figures in {args.out}/")


if __name__ == "__main__":
    main()
