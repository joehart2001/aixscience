# figs/ — inter-model comparisons

Cross-network comparison on the **test set**. Each network
(`direct-network/`, `transformer-network/`, `boltz-network/`) already persists
its runs to `<net>/figs/data/` — metrics in `runs.json`, prediction arrays in
per-run `.npz`. This folder reloads those files and replots them on shared
axes, so **nothing is retrained**; rerun it any time a network's run data
changes.

```
figs/
├── plots.py                   # the comparison script
├── templates/
│   ├── split_*.yaml           # direct vs transformer, full dataset, splits A/B/C/C2
│   └── a0201_split_*.yaml     # direct vs transformer vs boltz2, matched rows, A/B
├── matched-a0201/             # run data for the three-way comparison (see below)
└── compare/                   # output
    ├── <out>/                 # per-comparison figures
    └── summary_test_*.png|md
```

## Two comparisons

| Templates | Models | Rows | Splits |
|---|---|---|---|
| `split_*.yaml` | direct, transformer, SE+MLP, SE+XGB | full dataset (~28k) | A, B, C, C2 |
| `a0201_split_*.yaml` | direct, transformer, **boltz2 (frozen)** | HLA-A\*02:01 with embeddings (~945 peptides) | A, B |

They are separate on purpose: boltz2 embeddings exist for one allele only, so a
three-way comparison is possible only on that subset, and that subset cannot
support the allele-grouped splits (C/C2 collapse to a single fold). Run them
separately — mixing them in one invocation would put incomparable bars in the
same summary.

```bash
# Full dataset, direct vs transformer.
$PY plots.py templates/split_*.yaml --summary-prefix summary_full

# Three-way: first train all three on identical rows, then plot.
cd ../boltz-network && $PY compare_networks.py --out ../figs/matched-a0201
cd ../figs && $PY plots.py templates/a0201_split_*.yaml --summary-prefix summary_a0201
```

`--summary-prefix` matters: the across-split summary is written to the root of
`--out`, so two comparison families sharing a prefix would overwrite each
other's summary (the per-split subfolders never collide).

`compare_networks.py` trains each network by invoking its own `train.py` as a
subprocess (the three packages share module names like `data`/`model`, so they
cannot be imported into one process) and writes **one shared run-data dir** —
which is why the three `models:` entries in those templates share a `figs_dir`
and differ only by `run`.

## Usage

```bash
PY=../../.venv/bin/python   # run from aixscience/figs/

$PY plots.py templates/split_C.yaml      # one split
$PY plots.py templates/*.yaml            # every split + the across-split summary
$PY plots.py templates/*.yaml --out /tmp/cmp   # write elsewhere
```

Paths inside a template are relative to **the directory you run from**, matching
how the per-network templates write `csv: ../Data/...`.

## Template schema

```yaml
label: split C (grouped by allele)   # title + summary x-axis label
out: split_C                         # subfolder under compare/ (default: slug of label)
caption: >-                          # optional caveat drawn under each figure
  Only 22 clusters, so ~3 land in test.
models:
  - label: direct (MLP)              # legend/bar name
    figs_dir: ../direct-network/figs # a network's figs dir (must contain data/runs.json)
    run: split C (allele)            # the run's label inside that runs.json
```

A missing `run` raises with the list of available labels rather than silently
dropping a model, since a vanished bar would quietly change what the figure
means.

## Figures

Per split, in `compare/split_<X>/`:

| File | Shows |
|---|---|
| `compare_test_metrics.png` | grouped bars: Pearson / Spearman / within-allele ρ, and MAE / RMSE |
| `compare_test_pred_vs_actual.png` | predicted-vs-actual scatter, one panel per model, **shared log-log limits** so no panel is auto-scaled to flatter itself |
| `compare_test_residual_hist.png` | overlaid log-space residual distributions on shared bins — who is biased and which way |

Across splits, in `compare/`:

- `summary_test_metrics.png` — the headline: one panel per metric, x = split, one
  bar colour per model. Answers "which architecture wins, and does that change
  as the split gets harder?"
- `summary_test_metrics.md` — the same numbers as a markdown table.

The metric bars and the SOTA reference lines are imported from
`direct-network/plots.py` (via a `sys.path` insert, as `c2-committee` does with
`train.py`), so these figures are drawn by the same code as the per-network ones.

## Comparing like with like

`plots.py` checks that every model in a comparison has identical test targets in
identical order and prints a loud **WARNING** to stderr when they differ. It
still draws the figure — sometimes the mismatch is the point — but the bars are
not comparable.

This is why **`boltz-network` is absent from the default templates**: its runs
cover HLA-A\*02:01 only, so it is scored on a different, much smaller set of
rows. To bring it in honestly, first retrain direct/transformer with matching
`allele` + `embeddings_dir` filters (both are config keys for exactly this
purpose), then add a third `models:` entry.

## Current results

### Full dataset — direct vs transformer (`summary_full_test_metrics.md`)

| split | direct (MLP) r | transformer r | direct within-ρ | transformer within-ρ |
|---|---|---|---|---|
| A (random) | **0.796** | 0.693 | **0.598** | 0.448 |
| B (peptide) | **0.757** | 0.685 | **0.550** | 0.449 |
| C (allele) | **0.640** | 0.443 | **0.364** | 0.151 |
| C2 (cluster) | −0.009 | −0.138 | 0.065 | 0.077 |

The MLP wins everywhere. Two caveats before reading that as an architecture
verdict: the transformer is the *smaller* model (242K vs ~850K params), and on
A/B its train and val loss were still falling at the epoch budget — it is
under-trained, not outclassed. On C it genuinely overfits (train loss falls
while val loss climbs).

Both models collapse to chance on C2, and the residual histogram shows why the
MAE column is misleading there: both have a mean log-residual near **−0.57**,
i.e. they systematically *under*-predict an unseen groove rather than ranking it
badly. Low MAE on C2 reflects a conservative constant-ish prediction, not skill.

### Matched A\*02:01 rows — three-way (`summary_a0201_test_metrics.md`)

| split | direct (MLP) | transformer | boltz2 (frozen) |
|---|---|---|---|
| A (random) | **0.691** | 0.163 | 0.658 |
| B (peptide) | 0.550 | 0.403 | **0.601** |

Pearson r; ~945 peptides, so ~670 training rows. With one allele, within-allele
ρ equals global Spearman.

**Frozen Boltz2 features win where it counts.** On B — unseen peptides, the
harder and more meaningful of the two — boltz2 beats the from-scratch MLP on
every metric (r 0.601 vs 0.550, MAE 4.82 vs 5.17 h). On the random split it ties
it. That is the expected shape for a pretrained representation: its advantage
shows up under distribution shift, not on memorisable rows.

**The transformer collapses at this scale**, far below its own full-dataset
numbers (r 0.693 on A). 670 training rows is nowhere near enough to learn
attention from scratch, and A being *worse* than B is itself a sign of an
unconverged model rather than a meaningful ordering. Treat these two bars as a
floor, not a verdict on the architecture.
