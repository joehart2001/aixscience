"""Train the transformer peptide-MHC half-life network from a YAML config.

Example:
    python train.py templates/random.yaml

Same inputs and shared training engine as direct-network; only the encoder
(model.py) differs. To compare several splits on shared plots, use compare.py.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse

from config import build_results, load_config, print_summary, train_from_config
from data_common import load_splits
from data_seq import PeptideMHCDataset, Vocab
from device_utils import resolve_device, seed_everything
from model import DirectAffinityNet
from plots import make_all_plots, make_test_plots, save_run_data
from train_common import make_loaders, run_training


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train one model from a YAML config (see templates/)."
    )
    p.add_argument("config", help="Path to a YAML config file")
    return p.parse_args()


def train_model(
    csv: str,
    splits_csv: str,
    split: str,
    hla_col: str = "hla_seq",
    allele: str | None = None,
    embeddings_dir: str | None = None,
    train_value: str = "train",
    val_value: str = "val",
    epochs: int = 30,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    seed: int = 0,
    device: str | None = None,
    num_workers: int = 0,
    model_kwargs: dict | None = None,
    test_value: str = "test",
    out: str | None = None,
    tag: str = "",
) -> tuple[dict[str, list[float]], dict, dict | None]:
    """Train one model and return (history, best_eval, test_eval)."""
    device = resolve_device(device)
    seed_everything(seed, device)
    prefix = f"[{tag or hla_col}]"
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

    vocab = Vocab.build(train_df, hla_col=hla_col)
    print(
        f"{prefix} hla_col={vocab.hla_col} peptide_len={vocab.peptide_len} "
        f"hla_len={vocab.hla_len} num_alleles={vocab.num_alleles}"
    )

    train_ds = PeptideMHCDataset(train_df, vocab)
    val_ds = PeptideMHCDataset(val_df, vocab)
    test_ds = PeptideMHCDataset(test_df, vocab) if len(test_df) > 0 else None
    train_loader, val_loader, test_loader = make_loaders(
        train_ds, val_ds, test_ds, batch_size, num_workers, device
    )

    model = DirectAffinityNet(
        peptide_len=vocab.peptide_len,
        hla_len=vocab.hla_len,
        num_alleles=vocab.num_alleles,
        **(model_kwargs or {}),
    ).to(device)
    print(f"{prefix} transformer params = {sum(p.numel() for p in model.parameters()):,}")

    def checkpoint(m):
        return {
            "model_state": m.state_dict(),
            "vocab": vocab,
            "config": {
                "peptide_len": vocab.peptide_len,
                "hla_len": vocab.hla_len,
                "num_alleles": vocab.num_alleles,
            },
        }

    return run_training(
        model, train_loader, val_loader, test_loader,
        val_df["allele"].to_numpy(), test_df["allele"].to_numpy(),
        epochs=epochs, lr=lr, weight_decay=weight_decay, device=device,
        prefix=prefix, out=out, checkpoint=checkpoint,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    label = cfg["label"]
    print(f"=== Training model: {label} ({cfg.get('hla_col', cfg['split'])}) ===")

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
