# direct-network

A small, readable PyTorch pipeline that **directly** predicts peptide-MHC
half-life (`thalf_hours`) from three inputs: the `peptide`, the `hla_seq`, and
the `allele`. No external structure prediction or pretrained embeddings — just
learned amino-acid/allele embeddings feeding a plain MLP.

## Files

| File | Purpose |
|------|---------|
| `data.py` | Amino-acid + allele encoding, the `PeptideMHCDataset`, and `load_splits` (honors a flagged `split` column, or falls back to a random split). |
| `model.py` | `DirectAffinityNet`: shared AA embedding → flatten → concat with allele embedding → MLP → 1 output. |
| `train.py` | Trains a **single** model from a YAML config: `train.py <config.yaml>`. Holds `train_model()` (the training engine) and `evaluate()`. |
| `compare.py` | Trains **one or more** models from YAML configs: `compare.py <config.yaml> [...]`; several overlay on shared comparison plots in `figs/compare/`. |
| `config.py` | Shared YAML loading (`load_config`, `DEFAULTS`) + helpers used by both `train.py` and `compare.py`. |
| `templates/` | One YAML per model variant (`hla_seq.yaml`, `pseudoseq.yaml`) holding all run parameters. |
| `plots.py` | Saves single-model and comparison figures; persists run data to `figs/data/`. Run `python plots.py` to **regenerate all figures from saved data** without retraining. |
| `device_utils.py` | Device selection helpers — picks GPU when available, else CPU. |
| `prepare_split.py` | Adds a flagged `train`/`validation` `split` column using an **out-of-domain** (held-out protein) split. |

## Input format

A CSV in the same form as `rasmussen_et_al_dataset.csv`:

```
allele, peptide, thalf_hours, hla_seq, hla_pseudoseq
```

plus one extra column (default name `split`) flagging each row as `train` or
`validation`. Use `prepare_split.py` to generate it.

### Out-of-domain split

Rather than scattering rows randomly, `prepare_split.py` reserves ~5 entire
proteins (distinct `hla_seq` values) for validation: every row for a held-out
protein goes to validation, and those proteins never appear in training. This
tests generalization to **unseen HLA proteins** — a much harder and more honest
signal than a random split. Held-out alleles are absent from the training
vocabulary, so at validation time they fall back to the model's "unknown allele"
slot and predictions must rely on the `hla_seq` and `peptide` inputs.

## Quick start

From this directory (uses the repo's `.venv`):

```bash
PY=../../.venv/bin/python

# 1. Flag an out-of-domain split holding out 5 proteins for validation.
$PY prepare_split.py \
    --in ../Data/rasmussen_et_al_dataset.csv \
    --out ../Data/rasmussen_et_al_dataset_split.csv \
    --n-holdout-proteins 5

# 2. Train a single model from a YAML config in templates/.
$PY train.py templates/pseudoseq.yaml

# 3. Or train several and overlay them on comparison plots.
$PY compare.py templates/hla_seq.yaml templates/pseudoseq.yaml
```

Every parameter (HLA column, dataset path, epochs, learning rate, seed, device,
output locations, …) lives in its own `templates/*.yaml` — edit those rather than
passing flags. `train.py` takes exactly one config; `compare.py` takes one or more.

If the CSV has **no** split column, the pipeline logs a warning and falls back to
the same out-of-domain held-out-protein split so it still runs.

## Two models

| Model | HLA input | Length |
|-------|-----------|--------|
| full `hla_seq` | entire MHC sequence | 182 aa |
| `hla_pseudoseq` | contact residues lining the binding groove | 34 aa |

Both are identical networks trained on the same out-of-domain split; only the HLA
column differs. On held-out proteins the **pseudosequence model generalizes
noticeably better** — the 34 contact residues carry most of the binding signal
with far less noise than the full sequence (a 15-epoch run gave best val loss
~0.68 vs ~0.89 and Pearson ~0.60 vs ~0.43).

## Figures

After a single training run, `figs/` contains per-model plots:

- `loss_curve.png` — train vs. validation loss per epoch.
- `val_mae.png` — validation MAE (hours) per epoch.
- `val_pearson.png` — validation Pearson correlation per epoch.
- `pred_vs_actual.png` — predicted vs. actual half-life on the held-out proteins.

After `compare.py`, `figs/` also contains overlaid comparison plots:

- `compare_val_loss.png`, `compare_val_mae.png`, `compare_val_pearson.png` —
  both models on the same axes per epoch.
- `compare_pred_vs_actual.png` — side-by-side predicted-vs-actual scatter.

### Regenerating figures without retraining

Every run also writes its metrics + validation predictions to `figs/data/`
(`runs.json` + per-run `.npz`). To restyle or rebuild the plots from that saved
data — no training required:

```bash
$PY plots.py --figs-dir figs
```

## Compute (CPU local / GPU production)

The code is device-agnostic: it uses the GPU when one is available and falls back
to CPU otherwise (`device_utils.resolve_device`). Set `device:` in a YAML (or
`--device` on `train.py`) to `auto` (default), `cpu`, or `cuda`; raise
`num_workers` on a production GPU box for faster data loading. Locally it runs on
CPU; the same code runs unchanged on a GPU in production.

## Design notes

- **Target is `log1p(thalf_hours)`.** Half-lives span 0–~257 hours and are
  heavily right-skewed; log space makes the regression much easier. Predictions
  are converted back to hours with `expm1` for the reported MAE.
- **Vocabulary is built from the training split only** (sequence lengths and the
  allele lookup). Unseen alleles at inference map to an "unknown" slot.
- **Fixed lengths**: in this dataset every peptide is a 9-mer and every
  `hla_seq` is 182 chars, so sequences are simply flattened after embedding.
  Shorter/longer sequences are padded/truncated automatically.
