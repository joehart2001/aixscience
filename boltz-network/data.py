"""Data loading for the Boltz-embedding half-life network.

Same task as direct-network (predict `thalf_hours`), but the allele-peptide
system is represented by a **precomputed Boltz2 embedding** instead of integer
amino-acid indices. Each allele-peptide complex has one `.npz` under an
embeddings directory (e.g. Data/boltz2/boltz_embeddings/a0201), named
`hla_a_02_01_<peptide>.npz`, holding several fixed-length feature vectors. We use
one of them (default the 1547-d `features` block concatenation) as the model input.

Because the embeddings currently cover a single allele (HLA-A*02:01), only the
random (A) and by-peptide (B) splits are meaningful; the allele-grouped splits
(C/C2) put everything in one fold.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def embedding_filename(peptide: str) -> str:
    """Map a peptide to its Boltz embedding filename (allele is A*02:01 here)."""
    return f"hla_a_02_01_{peptide.lower()}.npz"


def load_features(embeddings_dir: str, peptide: str, feature_key: str) -> np.ndarray:
    """Load one fixed-length feature vector for an allele-peptide complex."""
    path = os.path.join(embeddings_dir, embedding_filename(peptide))
    with np.load(path) as d:
        return d[feature_key].astype("float32")


@dataclass
class Scaler:
    """Per-feature standardization (z-score), learned from the training split.

    Boltz features live on very different scales, so we center/scale them before
    the MLP. This plays the role Vocab did in direct-network: built from training
    data only, then reused for val/test. `feature_key` records which embedding
    array was used so the same one is read everywhere.
    """

    mean: np.ndarray
    std: np.ndarray
    feature_key: str

    @property
    def dim(self) -> int:
        return int(len(self.mean))

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) / self.std

    @staticmethod
    def build(df: pd.DataFrame, embeddings_dir: str, feature_key: str) -> "Scaler":
        feats = np.stack(
            [load_features(embeddings_dir, p, feature_key) for p in df["peptide"]]
        )
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero on constant features
        return Scaler(mean, std, feature_key)


class BoltzDataset(Dataset):
    """Wraps a dataframe as standardized embedding tensors + log1p targets.

    As in direct-network, the regression target is `log1p(thalf_hours)` (undo
    with `expm1`), since half-lives span 0-257h and are heavily right-skewed.
    """

    def __init__(self, df: pd.DataFrame, embeddings_dir: str, scaler: Scaler):
        feats = [
            scaler.transform(load_features(embeddings_dir, p, scaler.feature_key))
            for p in df["peptide"]
        ]
        self.x = torch.tensor(np.stack(feats), dtype=torch.float32)
        self.target = torch.tensor(
            [math.log1p(t) for t in df["thalf_hours"]], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[i], "target": self.target[i]}


def load_splits(
    sequences_csv: str,
    splits_csv: str,
    split: str,
    embeddings_dir: str,
    allele: str = "HLA-A*02:01",
    train_value: str = "train",
    val_value: str = "val",
    test_value: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df, test_df) for one split, restricted to rows of
    `allele` that have a Boltz embedding on disk.

    Join and split selection mirror direct-network; the extra steps are (1) keep
    only the target allele and (2) drop rows whose embedding file is missing.
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

    merged = sp.merge(
        seq[["row_id", "allele", "peptide"]],
        on="row_id",
        how="left",
        suffixes=("", "_seq"),
    )
    mismatch = (merged["allele"] != merged["allele_seq"]) | (
        merged["peptide"] != merged["peptide_seq"]
    )
    if mismatch.any():
        raise ValueError(
            f"row_id mapping between {splits_csv} and {sequences_csv} is "
            f"inconsistent ({int(mismatch.sum())} rows differ)."
        )
    merged = merged.drop(columns=["allele_seq", "peptide_seq"])

    # Keep only the target allele with an embedding file present on disk.
    merged = merged[merged["allele"] == allele].copy()
    has_emb = merged["peptide"].map(
        lambda p: os.path.exists(os.path.join(embeddings_dir, embedding_filename(p)))
    )
    n_missing = int((~has_emb).sum())
    if n_missing:
        print(f"[data] dropping {n_missing} {allele} rows with no embedding in {embeddings_dir}")
    merged = merged[has_emb]

    train_df = merged[merged[split_col] == train_value].reset_index(drop=True)
    val_df = merged[merged[split_col] == val_value].reset_index(drop=True)
    test_df = merged[merged[split_col] == test_value].reset_index(drop=True)
    if len(train_df) == 0 or len(val_df) == 0:
        raise ValueError(
            f"Column '{split_col}' did not yield both a '{train_value}' and a "
            f"'{val_value}' split for allele {allele} with embeddings. "
            f"(Only random/A and by-peptide/B splits apply to a single allele.)"
        )
    return train_df, val_df, test_df
