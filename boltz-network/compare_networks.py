"""Compare direct-network vs boltz-network on the *same* rows and splits.

Both models are trained on the identical subset — one allele (HLA-A*02:01) and
only rows that have a Boltz embedding — using the same split columns (A and B).
The only difference is the input representation:

  - direct-network : peptide + HLA(pseudoseq) + allele as integer indices
  - boltz-network  : the precomputed Boltz2 embedding vector

Each model is trained by invoking its own train.py as a subprocess (the two
packages share module names like `data`/`model`, so they can't both be imported
in one process). We then load the saved run data from both and overlay them with
this directory's plots.py.

Examples:
    python compare_networks.py --out figs/compare-direct-boltz
    python compare_networks.py --subset pilot100 --out figs/compare-direct-boltz-pilot100
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile

import yaml

from plots import load_run_data, make_all_plots, make_test_plots, save_run_data

PY = "/home/wfarmilo/ai-for-science/.venv/bin/python"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # aixscience/
DIRECT_DIR = os.path.join(ROOT, "direct-network")
BOLTZ_DIR = os.path.join(ROOT, "boltz-network")

# Relative paths work from either network dir (both are siblings under aixscience/).
CSV = "../Data/rasmussen_et_al_dataset.csv"
SPLITS_CSV = "../Data/subsets/splits.csv"
ALLELE = "HLA-A*02:01"


def direct_config(split: str, emb_dir: str, figs_dir: str, epochs: int) -> dict:
    return {
        "label": f"direct-net {split}",
        "csv": CSV,
        "splits_csv": SPLITS_CSV,
        "split": split,
        "hla_col": "hla_pseudoseq",
        "allele": ALLELE,
        "embeddings_dir": emb_dir,  # restrict to the exact boltz rows
        "epochs": epochs,
        "figs_dir": figs_dir,
    }


def boltz_config(split: str, emb_dir: str, figs_dir: str, epochs: int) -> dict:
    return {
        "label": f"boltz-net {split}",
        "csv": CSV,
        "splits_csv": SPLITS_CSV,
        "split": split,
        "embeddings_dir": emb_dir,
        "feature_key": "features",
        "allele": ALLELE,
        "epochs": epochs,
        "figs_dir": figs_dir,
    }


def run_one(net_dir: str, cfg: dict, tmp: str, tag: str) -> dict:
    """Write a temp YAML, run that network's train.py, return its run-data entry."""
    figs_dir = os.path.join(tmp, tag)
    cfg = {**cfg, "figs_dir": figs_dir}
    cfg_path = os.path.join(tmp, f"{tag}.yaml")
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    print(f"\n>>> training {tag} ({os.path.basename(net_dir)})")
    proc = subprocess.run(
        [PY, "train.py", cfg_path],
        cwd=net_dir,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        print(proc.stdout[-2000:])
        print(proc.stderr[-2000:])
        raise RuntimeError(f"{tag} training failed (exit {proc.returncode})")
    # Echo the key lines (split sizes + TEST metrics).
    for line in proc.stdout.splitlines():
        if "rows =" in line or "TEST |" in line:
            print("   ", line)

    (entry,) = load_run_data(figs_dir).values()  # single-model run -> one entry
    return entry


def main() -> None:
    p = argparse.ArgumentParser(description="Compare direct-network vs boltz-network")
    p.add_argument("--subset", default="a0201", help="embeddings subdir (a0201 or pilot100)")
    p.add_argument("--out", default="figs/compare-direct-boltz", help="output figs dir")
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--splits", nargs="+", default=["A", "B"])
    args = p.parse_args()

    emb_dir = f"../Data/boltz2/boltz_embeddings/{args.subset}"

    merged: dict[str, dict] = {}
    with tempfile.TemporaryDirectory(prefix="netcmp_") as tmp:
        for split in args.splits:
            # Interleave so comparison legends read direct/boltz per split.
            merged[f"direct-net {split}"] = run_one(
                DIRECT_DIR, direct_config(split, emb_dir, "", args.epochs),
                tmp, f"direct_{split}",
            )
            merged[f"boltz-net {split}"] = run_one(
                BOLTZ_DIR, boltz_config(split, emb_dir, "", args.epochs),
                tmp, f"boltz_{split}",
            )

    # Persist merged data + draw overlaid val and test figures.
    save_run_data(merged, args.out)
    make_all_plots(merged, args.out)
    make_test_plots(merged, args.out)
    print(f"\n[compare_networks] saved to {args.out}/ (compare/, test/, data/)")

    # Summary table.
    print(f"\n{'model':16s} | test_Pearson | test_Spearman | within-allele_rho | test_MAE(h) | test_RMSE(h)")
    print("-" * 94)
    for label, r in merged.items():
        t = r.get("test")
        if t:
            print(
                f"{label:16s} | {t['pearson']:12.3f} | {t['spearman']:13.3f} | "
                f"{t.get('within_allele_spearman', float('nan')):17.3f} | "
                f"{t['mae_hours']:11.2f} | {t['rmse_hours']:12.2f}"
            )


if __name__ == "__main__":
    main()
