# mlp-embeds

Four models that predict peptide–MHC class-I binding **half-life**
(`thalf_hours`) from the same data and splits, differing only in how they
represent the allele–peptide system. Shared machinery lives in flat modules here;
each `*-network/` holds only its unique encoder + config.

## Layout

```
mlp-embeds/
  plots.py          # all figures + run-data persistence (save_run_data/load_run_data)
  device_utils.py   # resolve_device / seed_everything / move_batch (CPU local, GPU prod)
  config_common.py  # load_config, build_results, print_summary
  data_common.py    # allele_slug, embedding_filename, load_splits (join + split select)
  data_seq.py       # AA encoding, Vocab, PeptideMHCDataset  (sequence nets)
  train_common.py   # evaluate, make_loaders, run_training (loop), metric helpers
  model_common.py   # mlp_head(in_dim, hidden=256, dropout=0.2)  (shared MLP head)

  direct-network/       model.py config.py train.py compare.py templates/ figs/
  transformer-network/  "   (encoder = self-attention)
  squeeze-boost-network/ " (+ boost.py; SE front-end, optional XGBoost stage)
  boltz-network/        + data.py (Scaler/BoltzDataset); frozen Boltz2 embeddings
  figs/                 cross-network comparison harness (see figs/README.md)
```

Each network keeps: `model.py` (its encoder, head via `mlp_head`), `config.py`
(`DEFAULTS` + `train_from_config`), a thin `train.py` (`train_model` builds the
model/dataset then calls `train_common.run_training`), and a 3-line `compare.py`.
Shared modules are imported from this directory via a 2-line `sys.path` bootstrap
at the top of each entry point.

## The four models

| Network | Allele–peptide representation |
|---|---|
| `direct-network` | peptide + HLA-pseudoseq + allele as integer indices → embed → flatten → MLP |
| `transformer-network` | the same indices → self-attention over the token stream |
| `squeeze-boost-network` | direct's front-end + a squeeze-excite block (optional XGBoost head) |
| `boltz-network` | a frozen, precomputed Boltz2 embedding vector → MLP |

## The four splits (sensible names → `split_<X>` column in `Data/subsets/`)

| Template | split | holds out |
|---|---|---|
| `random.yaml` | A | nothing (random 70/15/15) |
| `fix-peptide.yaml` | B | whole peptides |
| `fix-allele.yaml` | C | whole alleles |
| `fix-allele-group.yaml` | C2 (C2e for boltz) | whole allele clusters (unseen groove) |

Sequence nets use the full dataset (`Data/subsets/splits.csv`); boltz uses the
`all75` embeddings + `Data/subsets/embedded_splits.csv`. `Data/` at the repo root
is the source of truth and is never written to.

## Running

```bash
make direct          # or transformer / squeeze-boost / boltz
make all             # all four
make compare         # cross-network summary (figs/)
# equivalently, from a network dir:
cd direct-network && ../../../.venv/bin/python compare.py templates/*.yaml
```

Each `compare.py templates/*.yaml` trains the four splits and writes that
network's own `figs/`:

- `figs/compare/compare_val_{loss,mae}.png` — loss & MAE per split
- `figs/compare/compare_val_pearson.png` — global Pearson r per split
- `figs/compare/compare_val_within_rho.png` — within-allele ρ per split
- `figs/test/<split>_test_pred_vs_actual.png` — half-life parity on the test set
- `figs/test/compare_test_metrics.png` — grouped bars (Pearson / Spearman /
  within-allele ρ + MAE / RMSE), with SOTA reference lines
- `figs/data/` — `runs.json` + per-run `.npz`; rebuild any figure without
  retraining via `python plots.py --figs-dir figs`

## Compute

Device-agnostic (`device: auto` in a template): CPU locally, GPU in production.
Target is `log1p(thalf_hours)`; metrics are reported back in hours. See the repo
root `README.md` for the SOTA reference values drawn on the correlation figures.
