# boltz-network

Same task and **same model complexity** as `direct-network` (predict peptide–MHC
half-life `thalf_hours`), but the allele-peptide system is represented by a
**precomputed Boltz2 embedding** instead of integer amino-acid indices. Only
`data.py` (`Scaler`/`BoltzDataset`), `model.py`, and `config.py` are unique; the
training loop, metrics, and plots are shared from `mlp-embeds/`. See
[`../README.md`](../README.md) for the shared layout, splits, and how to run.

## What's different from direct-network

| | direct-network | boltz-network |
|---|---|---|
| Input | peptide + HLA sequence + allele as integer indices | one Boltz2 embedding vector per allele-peptide complex |
| `data.py` | amino-acid encoding + `Vocab` | `.npz` loading + `Scaler` (z-score from train) |
| `model.py` | embeddings → flatten → MLP | embedding vector → MLP (same 2×256 ReLU + dropout) |
| Splits used | A, B, C, C2 on the `all75` rows | A, B, C, C2 on the `all75` embeddings (75 alleles) |

`train.py`, `compare.py`, `config.py`, `plots.py`, `device_utils.py` are the same
as direct-network (apart from the data/model wiring), so the **outputs are the
identical figure suite**: training curves, val predicted-vs-actual, and the full
test suite (scatter, residuals, residual histogram, metric-comparison bars).

## Embeddings

Per-complex `.npz` files under `Data/boltz2/boltz_embeddings/`:
- `all75/` — the embedded rows across all 75 alleles (~14.5k; the canonical templates use this).
- `a0201/` — HLA-A*02:01 only (945 peptides; used by the `figs/` matched 3-way).
- `pilot100/` — first 100 rows, for quick smoke tests.

Each file has several fixed-length vectors; the default `feature_key: features`
is the 1547-d concatenation of 9 blocks (peptide/contact single-rep mean+max,
cross pair-rep summaries, token counts, confidence). Set `feature_key` in the
YAML to use a different one (e.g. `peptide_s_mean`).

Because `all75` spans all 75 alleles / 22 clusters, all four splits from
`Data/subsets/splits.csv` apply. A single-allele set like
`a0201` supports only random (A) / by-peptide (B).

## Run

```bash
# all four splits -> figs/all75_split_*/   (or: make make_model boltz-network  from mlp-embeds/)
for t in templates/*.yaml; do ../../../.venv/bin/python train.py $t; done
```

See [`../README.md`](../README.md) for the figures produced and the shared
layout. For a fast check, copy a template, point `embeddings_dir` at `pilot100`,
and lower `epochs`. `compare_networks.py` runs the separate matched-A*02:01
three-way comparison (see `../figs/README.md`).
