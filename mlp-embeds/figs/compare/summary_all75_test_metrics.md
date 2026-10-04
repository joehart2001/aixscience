# Test-set metrics by split and model

## A (random)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.757 | 0.736 | 0.520 | 4.366 | 11.391 |
| transformer | 0.655 | 0.653 | 0.378 | 4.851 | 11.932 |
| SE+MLP | 0.773 | 0.745 | 0.564 | 4.252 | 11.019 |
| SE+XGB | 0.773 | 0.739 | 0.552 | 4.198 | 11.014 |
| boltz2 (frozen) | 0.792 | 0.766 | 0.576 | 4.196 | 11.297 |

## B (peptide)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.715 | 0.718 | 0.433 | 4.982 | 13.051 |
| transformer | 0.643 | 0.667 | 0.381 | 5.214 | 12.803 |
| SE+MLP | 0.723 | 0.724 | 0.457 | 4.783 | 12.200 |
| SE+XGB | 0.720 | 0.720 | 0.468 | 4.633 | 11.768 |
| boltz2 (frozen) | 0.764 | 0.761 | 0.566 | 4.525 | 11.337 |

## C (allele)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.584 | 0.501 | 0.387 | 4.359 | 8.355 |
| transformer | 0.379 | 0.365 | 0.143 | 4.841 | 9.126 |
| SE+MLP | 0.663 | 0.607 | 0.434 | 4.269 | 8.296 |
| SE+XGB | 0.666 | 0.654 | 0.353 | 4.203 | 7.645 |
| boltz2 (frozen) | 0.792 | 0.773 | 0.578 | 3.252 | 6.479 |

## C2e (cluster)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | -0.034 | 0.024 | 0.017 | 3.215 | 6.680 |
| transformer | 0.158 | 0.216 | 0.027 | 3.192 | 6.471 |
| SE+MLP | 0.116 | 0.144 | 0.026 | 3.198 | 6.312 |
| SE+XGB | 0.014 | 0.103 | 0.059 | 3.279 | 6.380 |
| boltz2 (frozen) | 0.511 | 0.545 | 0.279 | 2.912 | 5.777 |
