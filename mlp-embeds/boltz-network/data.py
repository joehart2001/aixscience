"""Boltz-embedding data: feature loading, scaling, and the dataset.

Same task as the sequence networks (predict `thalf_hours`), but the allele-peptide
system is represented by a precomputed Boltz2 embedding. Each complex has one
`.npz` under an embeddings dir, named `<allele_slug>_<peptide>.npz`; we use one
fixed-length vector from it (default the 1547-d `features` block).

`allele_slug`, `embedding_filename`, and `load_splits` are shared — imported from
mlp-embeds/data_common.py.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from data_common import embedding_filename  # re-used for the per-complex filename


def load_features(
    embeddings_dir: str, allele: str, peptide: str, feature_key: str
) -> np.ndarray:
    """Load one fixed-length feature vector for an allele-peptide complex."""
    path = os.path.join(embeddings_dir, embedding_filename(allele, peptide))
    with np.load(path) as d:
        return d[feature_key].astype("float32")


@dataclass
class Scaler:
    """Per-feature standardization (z-score), learned from the training split.

    Boltz features live on very different scales, so we center/scale them before
    the MLP. This plays the role Vocab does for the sequence nets: built from
    training data only, then reused for val/test. `feature_key` records which
    embedding array was used so the same one is read everywhere.
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
            [
                load_features(embeddings_dir, a, p, feature_key)
                for a, p in zip(df["allele"], df["peptide"])
            ]
        )
        mean = feats.mean(axis=0)
        std = feats.std(axis=0)
        std[std == 0] = 1.0  # avoid divide-by-zero on constant features
        return Scaler(mean, std, feature_key)


class BoltzDataset(Dataset):
    """Wraps a dataframe as standardized embedding tensors + log1p targets.

    As in the sequence nets, the regression target is `log1p(thalf_hours)` (undo
    with `expm1`), since half-lives span 0-257h and are heavily right-skewed.
    """

    def __init__(self, df: pd.DataFrame, embeddings_dir: str, scaler: Scaler):
        feats = [
            scaler.transform(
                load_features(embeddings_dir, a, p, scaler.feature_key)
            )
            for a, p in zip(df["allele"], df["peptide"])
        ]
        self.x = torch.tensor(np.stack(feats), dtype=torch.float32)
        self.target = torch.tensor(
            [math.log1p(t) for t in df["thalf_hours"]], dtype=torch.float32
        )

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        return {"x": self.x[i], "target": self.target[i]}
