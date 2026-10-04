"""direct-network config: DEFAULTS + train_from_config.

Shared helpers (load_config, build_results, print_summary) live in
mlp-embeds/config_common.py.
"""

from __future__ import annotations

from typing import Callable

from config_common import build_results, print_summary
from config_common import load_config as _load_config

# Defaults applied when a YAML omits a key, so configs can stay minimal.
DEFAULTS = {
    "splits_csv": "../../Data/subsets/splits.csv",  # precomputed A/B/C/C2 splits
    "split": "A",          # which split strategy: A, B, C, or C2
    "hla_col": "hla_seq",
    "allele": None,          # optional: restrict to one allele (subset comparison)
    "embeddings_dir": None,  # optional: keep only rows with a Boltz embedding here
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
        hla_col=cfg["hla_col"],
        allele=cfg["allele"],
        embeddings_dir=cfg["embeddings_dir"],
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
