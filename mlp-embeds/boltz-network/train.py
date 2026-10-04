"""Train the Boltz-embedding half-life network from a YAML config.

Example:
    python train.py templates/random.yaml

Same shared training engine as the sequence networks; the inputs are precomputed
Boltz2 embedding vectors rather than integer amino-acid indices, so the model is
called as `model(batch["x"])`. To compare several splits, use compare.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from config import build_results, load_config, print_summary, train_from_config
from data import BoltzDataset, Scaler
from data_common import load_splits
from device_utils import resolve_device, seed_everything
from model import BoltzAffinityNet
from plots import make_all_plots, make_test_plots, save_run_data
from train_common import make_loaders, run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one model from a YAML config (see templates/)."
    )
    p.add_argument("config", help="Path to a YAML config file")
    return p.parse_args()


def _forward(model, batch):
    return model(batch["x"])


def train_model(
    csv: str,
    splits_csv: str,
    split: str,
    embeddings_dir: str,
    feature_key: str = "features",
    allele: str | None = None,
    train_value: str = "train",
    val_value: str = "val",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    device: str | None = None,
    num_workers: int = 0,
    test_value: str = "test",
    out: str | None = None,
    tag: str = "",
) -> tuple[dict[str, list[float]], dict, dict | None]:
    """Train one model and return (history, best_eval, test_eval).

    `split` selects the precomputed split; `embeddings_dir` holds the Boltz `.npz`
    files and `feature_key` picks which embedding vector to feed the model.
    `allele=None` keeps every allele that has an embedding.
    """
    device = resolve_device(device)
    seed_everything(seed, device)
    prefix = f"[{tag or split}]"
    print(f"{prefix} device = {device}")

    train_df, val_df, test_df = load_splits(
        csv, splits_csv=splits_csv, split=split,
        train_value=train_value, val_value=val_value, test_value=test_value,
        allele=allele, embeddings_dir=embeddings_dir,
    )
    print(
        f"{prefix} split={split} train rows = {len(train_df)}, "
        f"val rows = {len(val_df)}, test rows = {len(test_df)}"
    )

    # Feature scaler is fit on the training split only (like Vocab for seq nets).
    scaler = Scaler.build(train_df, embeddings_dir, feature_key)
    print(f"{prefix} feature_key={feature_key} input_dim={scaler.dim}")

    train_ds = BoltzDataset(train_df, embeddings_dir, scaler)
    val_ds = BoltzDataset(val_df, embeddings_dir, scaler)
    test_ds = BoltzDataset(test_df, embeddings_dir, scaler) if len(test_df) > 0 else None
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size, num_workers, device
    )

    model = BoltzAffinityNet(input_dim=scaler.dim).to(device)

    def checkpoint(m):
        return {
            "model_state": m.state_dict(),
            "scaler": scaler,
            "config": {"input_dim": scaler.dim, "feature_key": feature_key},
        }

    return run_training(
        model, train_loader, val_loader, test_loader,
        val_df["allele"].to_numpy(), test_df["allele"].to_numpy(),
        forward_fn=_forward,
        epochs=epochs, lr=lr, weight_decay=weight_decay, device=device,
        prefix=prefix, out=out, checkpoint=checkpoint,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label = cfg["label"]
    print(f"=== Training model: {label} (split {cfg['split']}) ===")

    history, best_eval, test_eval = train_from_config(cfg, train_model)

    results = build_results({label: history}, {label: best_eval}, {label: test_eval})
    figs_dir = cfg["figs_dir"]
    save_run_data(results, figs_dir)
    make_all_plots(results, figs_dir)
    make_test_plots(results, figs_dir)
    print(f"[train] saved figures + data to {figs_dir}/")

    print_summary({label: history}, {label: test_eval})


if __name__ == "__main__":
    main()
