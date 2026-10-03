# boltz-network

Same task and **same model complexity** as `direct-network` (predict peptide–MHC
half-life `thalf_hours`), but the allele-peptide system is represented by a
**precomputed Boltz2 embedding** instead of integer amino-acid indices. Most code
is copied from `direct-network`; only the input representation changed.

## What's different from direct-network

| | direct-network | boltz-network |
|---|---|---|
| Input | peptide + HLA sequence + allele as integer indices | one Boltz2 embedding vector per allele-peptide complex |
| `data.py` | amino-acid encoding + `Vocab` | `.npz` loading + `Scaler` (z-score from train) |
| `model.py` | embeddings → flatten → MLP | embedding vector → MLP (same 2×256 ReLU + dropout) |
| Splits used | A, B, C, C2 | **A and B only** (embeddings cover one allele) |

`train.py`, `compare.py`, `config.py`, `plots.py`, `device_utils.py` are the same
as direct-network (apart from the data/model wiring), so the **outputs are the
identical figure suite**: training curves, val predicted-vs-actual, and the full
test suite (scatter, residuals, residual histogram, metric-comparison bars).

## Embeddings

Per-complex `.npz` files under `Data/boltz2/boltz_embeddings/`:
- `a0201/` — all HLA-A*02:01 complexes (945 peptides).
- `pilot100/` — first 100 CSV rows (quick smoke tests).

Each file has several fixed-length vectors; the default `feature_key: features`
is the 1547-d concatenation of 9 blocks (peptide/contact single-rep mean+max,
cross pair-rep summaries, token counts, confidence). Set `feature_key` in the
YAML to use a different one (e.g. `peptide_s_mean`).

Only HLA-A*02:01 has embeddings, so allele-grouped splits (C/C2) collapse to a
single fold; only random (A) and by-peptide (B) splits are meaningful here.

## Quick start

```bash
PY=../../.venv/bin/python

# Train a single model on one split (split chosen inside the YAML).
$PY train.py templates/split_A.yaml

# Compare the two applicable splits on shared plots.
$PY compare.py templates/split_A.yaml templates/split_B.yaml

# Rebuild all figures (val + test) from saved run data, no retraining.
$PY plots.py --figs-dir figs
```

For a fast check, copy a template, set `embeddings_dir` to `.../pilot100` and
lower `epochs`.

## Figures

- `figs/compare/` — training curves + val predicted-vs-actual (overlaid for a
  comparison run).
- `figs/test/` — per-split test scatter / residuals / residual histogram, plus
  `compare_test_metrics.png` (Pearson/Spearman and MAE/RMSE bars).
- `figs/data/` — `runs.json` + per-run `.npz` so `plots.py` can regenerate
  everything without retraining.
