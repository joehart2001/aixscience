# Test-set metrics by split and model

## A*02:01 split A (random split)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.691 | 0.687 | 0.687 | 5.836 | 9.214 |
| transformer | 0.163 | 0.149 | 0.149 | 7.793 | 11.888 |
| boltz2 (frozen) | 0.658 | 0.685 | 0.685 | 6.054 | 10.324 |

## A*02:01 split B (grouped by peptide)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.550 | 0.520 | 0.520 | 5.173 | 7.986 |
| transformer | 0.403 | 0.408 | 0.408 | 6.112 | 9.548 |
| boltz2 (frozen) | 0.601 | 0.583 | 0.583 | 4.821 | 7.511 |
