# Test-set metrics by split and model

## split A (random split)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.796 | 0.762 | 0.598 | 3.417 | 9.010 |
| transformer | 0.693 | 0.680 | 0.448 | 4.003 | 10.137 |

## split B (grouped by peptide)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.757 | 0.740 | 0.550 | 3.699 | 9.326 |
| transformer | 0.685 | 0.689 | 0.449 | 4.184 | 11.065 |

## split C (grouped by allele)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.640 | 0.602 | 0.364 | 3.467 | 6.638 |
| transformer | 0.443 | 0.405 | 0.151 | 4.223 | 8.042 |

## split C2 (grouped by allele cluster)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | -0.009 | 0.017 | 0.065 | 3.323 | 6.256 |
| transformer | -0.138 | -0.165 | 0.077 | 3.280 | 6.255 |
