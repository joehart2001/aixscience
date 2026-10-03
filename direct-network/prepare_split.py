"""Add a flagged train/validation `split` column to a rasmussen-style CSV.

This is an **out-of-domain (OOD)** split: instead of scattering rows randomly, we
reserve a handful of entire proteins (distinct `hla_seq` values) for validation.
Every row belonging to a held-out protein goes to validation, and none of those
proteins appear in training. This tests whether the model generalizes to HLA
proteins it has never seen, rather than just memorizing per-protein quirks.

Example:
    python prepare_split.py \
        --in ../Data/rasmussen_et_al_dataset.csv \
        --out ../Data/rasmussen_et_al_dataset_split.csv \
        --n-holdout-proteins 5
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd


def main() -> None:
    p = argparse.ArgumentParser(description="Flag OOD train/validation splits")
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--out", dest="out_path", required=True)
    p.add_argument(
        "--n-holdout-proteins",
        type=int,
        default=5,
        help="Number of distinct hla_seq proteins reserved for validation.",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split-col", default="split")
    args = p.parse_args()

    df = pd.read_csv(args.in_path)

    # Pick which proteins to hold out, deterministically given the seed.
    proteins = np.sort(df["hla_seq"].unique())
    if args.n_holdout_proteins >= len(proteins):
        raise ValueError(
            f"Asked to hold out {args.n_holdout_proteins} proteins but only "
            f"{len(proteins)} exist."
        )
    rng = np.random.default_rng(args.seed)
    holdout = set(rng.choice(proteins, size=args.n_holdout_proteins, replace=False))

    is_val = df["hla_seq"].isin(holdout)
    df[args.split_col] = np.where(is_val, "validation", "train")

    df.to_csv(args.out_path, index=False)

    # Report which proteins were held out, by allele name (1:1 with hla_seq here).
    holdout_alleles = sorted(df.loc[is_val, "allele"].unique())
    n_val = int(is_val.sum())
    print(
        f"Wrote {args.out_path}: {len(df) - n_val} train / {n_val} validation "
        f"rows (column '{args.split_col}')."
    )
    print(
        f"Held out {args.n_holdout_proteins} proteins for OOD validation: "
        f"{', '.join(holdout_alleles)}"
    )


if __name__ == "__main__":
    main()
