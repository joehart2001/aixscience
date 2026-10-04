"""Shared data loading: allele/embedding filename helpers and the split loader.

Used by every network. `load_splits` joins the precomputed split assignments to
the sequence CSV by `row_id` and optionally restricts to one allele and/or to
rows that have a Boltz embedding on disk — so the sequence networks and the
boltz network can be pointed at exactly the same rows.
"""

from __future__ import annotations

import os
import re

import pandas as pd


def allele_slug(allele: str) -> str:
    """Filesystem form of an allele name: 'HLA-A*02:01' -> 'hla_a_02_01'."""
    return re.sub(r"[^a-z0-9]+", "_", allele.lower()).strip("_")


def embedding_filename(allele: str, peptide: str) -> str:
    """Map an allele-peptide complex to its Boltz embedding filename."""
    return f"{allele_slug(allele)}_{peptide.lower()}.npz"


def load_splits(
    sequences_csv: str,
    splits_csv: str,
    split: str,
    train_value: str = "train",
    val_value: str = "val",
    test_value: str = "test",
    allele: str | None = None,
    embeddings_dir: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) for one of the precomputed splits.

    The split assignments live in `splits_csv` (Data/subsets/splits.csv or
    embedded_splits.csv): one `split_<X>` column per strategy (A=random,
    B=by-peptide, C=by-allele, C2/C2e=by-allele-cluster) with values
    train/val/test. That file has no sequences, so we join it to `sequences_csv`
    (the rasmussen dataset) by `row_id` — the 0-based row index of the sequences
    file — to attach `hla_seq` and `hla_pseudoseq`. `test_df` may be empty if the
    split has no test rows.

    Optional subset filters: `allele` keeps only that allele; `embeddings_dir`
    keeps only rows whose Boltz embedding file exists there
    (`<allele_slug>_<peptide>.npz`, e.g. `hla_a_02_01_aefgpwqtv.npz`).
    """
    split_col = f"split_{split}"

    seq = pd.read_csv(sequences_csv).reset_index(names="row_id")
    sp = pd.read_csv(splits_csv)
    if split_col not in sp.columns:
        available = sorted(c[len("split_"):] for c in sp.columns if c.startswith("split_"))
        raise ValueError(
            f"Split '{split}' (column '{split_col}') not in {splits_csv}. "
            f"Available: {available}"
        )

    # Attach sequences by row_id, and verify the join lines up on allele/peptide.
    merged = sp.merge(
        seq[["row_id", "allele", "peptide", "hla_seq", "hla_pseudoseq"]],
        on="row_id",
        how="left",
        suffixes=("", "_seq"),
    )
    if merged["hla_seq"].isna().any():
        raise ValueError(
            f"Some row_ids in {splits_csv} have no match in {sequences_csv}."
        )
    mismatch = (merged["allele"] != merged["allele_seq"]) | (
        merged["peptide"] != merged["peptide_seq"]
    )
    if mismatch.any():
        raise ValueError(
            f"row_id mapping between {splits_csv} and {sequences_csv} is "
            f"inconsistent ({int(mismatch.sum())} rows differ on allele/peptide)."
        )
    merged = merged.drop(columns=["allele_seq", "peptide_seq"])

    # Optional subset filters (used so sequence and boltz nets see the same rows).
    if allele is not None:
        merged = merged[merged["allele"] == allele]
    if embeddings_dir is not None:
        has_emb = [
            os.path.exists(os.path.join(embeddings_dir, embedding_filename(a, p)))
            for a, p in zip(merged["allele"], merged["peptide"])
        ]
        merged = merged[has_emb]

    train_df = merged[merged[split_col] == train_value].reset_index(drop=True)
    val_df = merged[merged[split_col] == val_value].reset_index(drop=True)
    test_df = merged[merged[split_col] == test_value].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Column '{split_col}' did not yield both a '{train_value}' and a "
            f"'{val_value}' split."
        )
    return train_df, val_df, test_df
