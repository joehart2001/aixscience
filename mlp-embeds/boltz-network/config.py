"""boltz-network config: DEFAULTS + train_from_config.

Shared helpers (load_config, build_results, print_summary) live in
mlp-embeds/config_common.py.
"""

from __future__ import annotations

from typing import Callable

from config_common import build_results, print_summary
from config_common import load_config as _load_config

# Defaults applied when a YAML omits a key, so configs can stay minimal.
# The `all75` embedding set now spans all 75 alleles / 22 clusters, so every
# split (A/B/C/C2) works on the embedded subset — same `splits.csv` as the
# sequence nets (the old embedded-cluster workaround split C2e is retired).
DEFAULTS = {
    "splits_csv": "../../Data/subsets/splits.csv",
    "split": "A",          # which split strategy: A, B, C, or C2
    "embeddings_dir": "../../Data/boltz2/boltz_embeddings/all75",
    "feature_key": "features",         # which Boltz embedding vector to use
    "allele": None,                    # None = every allele that has an embedding
    "train_value": "train",
    "val_value": "val",
    "epochs": 30,
    "batch_size": 128,
    "lr": 0.001,
    "weight_decay": 0.00001,
    "seed": 0,
    "device": "auto",      # CUDA if available (prod GPU), else CPU (local)
    "num_workers": 0,
    "figs_dir": "figs",
    "out": None,
}


def load_config(path: str) -> dict:
    return _load_config(path, DEFAULTS)


def train_from_config(cfg: dict, train_fn: Callable):
    """Call `train_fn` (train.train_model) with the parameters from `cfg`."""
    return train_fn(
        csv=cfg["csv"],
        splits_csv=cfg["splits_csv"],
        split=cfg["split"],
        embeddings_dir=cfg["embeddings_dir"],
        feature_key=cfg["feature_key"],
        allele=cfg["allele"],
        train_value=cfg["train_value"],
        val_value=cfg["val_value"],
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        seed=cfg["seed"],
        device=cfg["device"],
        num_workers=cfg["num_workers"],
        out=cfg["out"],
        tag=cfg["label"],
    )
