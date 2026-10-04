# squeeze-boost-network

Naive embeddings (exactly as in `direct-network`) → **1-D Squeeze-and-Excitation
channel recalibration** → two heads: the usual MLP, and **XGBoost fitted on the
recalibrated features**. The gating is learned by gradient descent; the splits
on top of it by boosting.

Everything except `model.py` and `boost.py` is copied from `direct-network`;
`data.py`, `plots.py`, `compare.py` and `device_utils.py` are unchanged.

## The squeeze-and-excitation block

The feature vector reaching the head is the direct-network one: the flattened
peptide embedding, the flattened HLA-pseudosequence embedding, and the allele
embedding, concatenated — 704 channels at the default widths.

| Step | What happens |
|---|---|
| **Squeeze** | The vector is already 1-D, so flattening has done the pooling a 2-D SE block would do. To still give the gate a cheap summary, we compute **mean and variance per field** (peptide / HLA / allele) and append those 6 numbers to its input. Set `squeeze_stats: false` for the plain form, where the gate reads X alone. |
| **Excitation** | Bottleneck gate `s = σ(W₂ · ReLU(W₁ · X))`, with `W₁: (C/r, C)` and `W₂: (C, C/r)` at reduction ratio **r = 8**. |
| **Scale** | `X̃ = s ⊙ X`, elementwise. |

Because `s` is computed per example, the same channel can be suppressed for one
HLA molecule and amplified for another — damping noisy pocket interactions and
emphasising the anchor positions that matter for *that* groove. A plain MLP must
commit to one fixed weight per channel instead.

`SqueezeExcite1d.gate_values()` returns `s` on its own, for inspecting which
channels actually get damped.

## The boost stage

`boost.py` runs both stages from one config:

1. Train the SE network with the shared engine — same best-by-val-loss
   checkpointing, same single final test evaluation.
2. Freeze it, push every row through `model.features()` to get `X̃`, and fit an
   XGBoost regressor on those 704 features against the same target,
   `log1p(thalf_hours)`.

Trees split on individual features, so they gain from an input whose informative
channels have already been amplified. Early stopping uses the **validation**
split — the same one that picked the network checkpoint — so the test split is
touched exactly once, at the end.

Both heads are scored by `metrics_from_log` imported from `train.py`, i.e. the
identical code that scores every other network in this repo.

## Quick start

```bash
PY=../../.venv/bin/python

$PY boost.py templates/split_C.yaml     # SE network + XGBoost, both scored
$PY train.py templates/split_C.yaml     # SE network alone (no boost stage)
$PY compare.py templates/split_*.yaml   # SE network across all four splits
$PY plots.py --figs-dir figs/split_C    # rebuild figures from saved data
```

`boost.py` writes **two labelled runs** into one figs dir — `<label> (SE+MLP)`
and `<label> (SE+XGB)` — so the top-level `figs/plots.py` can compare them
against the other networks with no special casing. Point a template entry at
this `figs_dir` with the matching `run:`.

## Config

Direct-network's schema plus two blocks:

```yaml
model:                   # passed to the model; omit any key for its default
  reduction: 8           # SE bottleneck ratio r
  squeeze_stats: true    # append per-field mean+variance to the gate input
  hidden_dim: 256        # head width (matches direct-network's MLP)
  dropout: 0.2

xgb:                     # boost.py only; train.py ignores this
  n_estimators: 2000     # upper bound; early stopping picks the real count
  learning_rate: 0.05
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8
  min_child_weight: 5
  early_stopping_rounds: 50
```

The MLP head is left at direct-network's exact shape on purpose, so a
difference in results is attributable to the SE block rather than to extra head
capacity.

## Note on the shared engine

`train_model()` here gained two parameters over direct-network's: `model_kwargs`
(the `model:` block) and `artifacts`, an optional out-parameter dict that gets
filled with the trained model, vocab and dataframes. `boost.py` needs those to
extract features. It is an out-parameter rather than an extra return value
specifically so the `(history, best_eval, test_eval)` contract stays identical
to the other networks', keeping `config.py` and `compare.py` in sync with them.

## Results (test set, full dataset)

Pearson r / within-allele rho, against the other networks:

| split | direct (MLP) | transformer | **SE+MLP** | SE+XGB |
|---|---|---|---|---|
| A (random) | 0.796 / 0.598 | 0.693 / 0.448 | **0.818 / 0.643** | 0.811 / 0.643 |
| B (peptide) | 0.757 / 0.550 | 0.685 / 0.449 | 0.767 / 0.579 | **0.774 / 0.589** |
| C (allele) | 0.640 / 0.364 | 0.443 / 0.151 | **0.700 / 0.426** | 0.602 / 0.284 |
| C2 (cluster) | -0.009 / 0.065 | -0.138 / 0.077 | -0.162 / 0.031 | -0.233 / -0.127 |

**The SE block earns its place**: SE+MLP beats direct-network on A, B and C with
an identical head, and the gap is widest on C (r 0.700 vs 0.640, within-allele
rho 0.426 vs 0.364, MAE 2.98 vs 3.47 h) — the realistic unseen-allele case. That
is what per-example channel gating is supposed to buy.

**The boost stage is not clearly worth it.** It helps slightly on B, costs a
little on A, and costs a lot on C (0.602 vs 0.700). The C case is instructive:
XGBoost reaches a *better* validation loss than the network (1.256 vs 1.341) but
a worse test score, and early stopping fired at 38 rounds against 1893 on A.
With only ~60 training alleles, the trees fit the validation fold that is
selecting them. See `figs/split_C/compare/boost_val_loss.png`.

**C2 remains at chance for every architecture**, and SE+XGB is the worst of
them. Nothing here touches the unseen-groove problem.

## A note on the boosting curve

XGBoost has no epochs, so its per-round validation curve is recorded under
`boost_rounds` / `boost_val_*` and drawn in its own figure,
`figs/<split>/compare/boost_val_loss.png`, with the network's best val loss as a
dashed reference. It is deliberately kept off the epoch-axis panels: boosting
rounds and epochs are different units, and overlaying them would imply a
comparison that is not being made. The like-for-like comparison between the two
heads is the test suite, where both appear.

## Dependency

Adds **xgboost** (3.4.1, installed into the shared `.venv`; numpy stays at
1.26.4). It is *not* recorded in `pyproject.toml`/`uv.lock` yet, because
`uv.lock` is currently committed with unresolved merge-conflict markers and
`uv add` refuses to parse it — resolve that first, then run
`uv add "xgboost>=2.0"`.
