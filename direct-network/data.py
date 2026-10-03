"""Data loading and encoding for the direct peptide-MHC half-life network.

The input is a CSV in the same form as `rasmussen_et_al_dataset.csv`:

    allele, peptide, thalf_hours, hla_seq, hla_pseudoseq

plus one extra column that flags the train/validation split (see `prepare_split.py`).
We only use three inputs to predict `thalf_hours`:

    peptide   -> short amino-acid string (9-mers in this dataset)
    hla_seq   -> the MHC amino-acid sequence (182 chars in this dataset)
    allele    -> a categorical name, e.g. "HLA-A*02:01"

Everything here is deliberately plain so the pipeline is easy to follow.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# The 20 standard amino acids. Index 0 is reserved for padding / unknown
# characters, so real amino acids start at index 1.
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
AA_TO_IDX = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
PAD_IDX = 0
AA_VOCAB_SIZE = len(AMINO_ACIDS) + 1  # +1 for the pad/unknown slot


def encode_sequence(seq: str, length: int) -> list[int]:
    """Turn an amino-acid string into a fixed-length list of integer indices.

    Unknown characters map to PAD_IDX. Sequences are truncated or right-padded
    so every example has the same length (required to batch them into a tensor).
    """
    # Map each character to its index, truncating anything past `length`.
    idxs = [AA_TO_IDX.get(aa, PAD_IDX) for aa in seq[:length]]
    # Right-pad with the pad index so every sequence ends up the same length.
    idxs += [PAD_IDX] * (length - len(idxs))
    return idxs


@dataclass
class Vocab:
    """Fixed lengths and the allele -> index lookup, learned from the data.

    We build this from the *training* split only, then reuse it for validation
    so the model never sees validation-specific vocabulary at setup time.
    """

    peptide_len: int
    hla_len: int
    allele_to_idx: dict[str, int]
    # Which CSV column provides the HLA sequence: "hla_seq" (full, 182 aa) or
    # "hla_pseudoseq" (34-aa contact-residue pseudosequence).
    hla_col: str = "hla_seq"

    @property
    def num_alleles(self) -> int:
        # +1 for an "unknown allele" slot at index 0.
        return len(self.allele_to_idx) + 1

    def allele_index(self, allele: str) -> int:
        return self.allele_to_idx.get(allele, 0)

    @staticmethod
    def build(df: pd.DataFrame, hla_col: str = "hla_seq") -> "Vocab":
        peptide_len = int(df["peptide"].str.len().max())
        # Length comes from whichever HLA column this model is using.
        hla_len = int(df[hla_col].str.len().max())
        unique_alleles = sorted(df["allele"].unique())
        allele_to_idx = {a: i + 1 for i, a in enumerate(unique_alleles)}
        return Vocab(peptide_len, hla_len, allele_to_idx, hla_col)


class PeptideMHCDataset(Dataset):
    """Wraps a dataframe of examples as encoded tensors.

    The regression target is `log1p(thalf_hours)`. Half-lives are heavily
    right-skewed (0 to ~257 hours), and log space makes the error distribution
    far more even, which is much easier for a simple network to fit. We undo the
    transform with `expm1` when we want predictions back in hours.
    """

    def __init__(self, df: pd.DataFrame, vocab: Vocab):
        self.vocab = vocab
        # Pre-encode every column once up front so batching is just indexing.
        self.peptide = torch.tensor(
            [encode_sequence(s, vocab.peptide_len) for s in df["peptide"]],
            dtype=torch.long,
        )
        # Encode whichever HLA column this model was configured to use.
        self.hla = torch.tensor(
            [encode_sequence(s, vocab.hla_len) for s in df[vocab.hla_col]],
            dtype=torch.long,
        )
        # Each allele name becomes a single integer index.
        self.allele = torch.tensor(
            [vocab.allele_index(a) for a in df["allele"]],
            dtype=torch.long,
        )
        # Target is log1p(hours); the model learns in this compressed space.
        self.target = torch.tensor(
            [math.log1p(t) for t in df["thalf_hours"]],
            dtype=torch.float32,
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {
            "peptide": self.peptide[i],
            "hla": self.hla[i],
            "allele": self.allele[i],
            "target": self.target[i],
        }


def load_splits(
    csv_path: str,
    split_col: str = "split",
    train_value: str = "train",
    val_value: str = "validation",
    seed: int = 0,
    n_holdout_proteins: int = 5,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the CSV and return (train_df, val_df).

    If the CSV already has a `split_col` flagging rows as train/validation, we
    honor it. Otherwise we fall back to an out-of-domain split that reserves a
    few entire proteins (distinct `hla_seq`) for validation, matching what
    prepare_split.py produces, so the pipeline still runs on the raw dataset.
    """
    df = pd.read_csv(csv_path)
    required = {"allele", "peptide", "thalf_hours", "hla_seq"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    if split_col in df.columns:
        train_df = df[df[split_col] == train_value].reset_index(drop=True)
        val_df = df[df[split_col] == val_value].reset_index(drop=True)
        if len(train_df) == 0 or len(val_df) == 0:
            raise ValueError(
                f"Column '{split_col}' present but did not yield both a "
                f"'{train_value}' and a '{val_value}' split."
            )
    else:
        print(
            f"[data] No '{split_col}' column found; falling back to an "
            f"out-of-domain split holding out {n_holdout_proteins} proteins "
            f"(seed={seed}). Use prepare_split.py to flag splits explicitly."
        )
        proteins = np.sort(df["hla_seq"].unique())
        rng = np.random.default_rng(seed)
        holdout = set(
            rng.choice(proteins, size=n_holdout_proteins, replace=False)
        )
        is_val = df["hla_seq"].isin(holdout)
        train_df = df[~is_val].reset_index(drop=True)
        val_df = df[is_val].reset_index(drop=True)

    return train_df, val_df
