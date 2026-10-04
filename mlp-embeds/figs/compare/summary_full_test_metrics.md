# Test-set metrics by split and model

## split A (random split)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.796 | 0.762 | 0.598 | 3.417 | 9.010 |
| transformer | 0.693 | 0.680 | 0.448 | 4.003 | 10.137 |
| SE+MLP | 0.818 | 0.788 | 0.643 | 3.198 | 8.511 |
| SE+XGB | 0.811 | 0.785 | 0.643 | 3.272 | 8.898 |

## split B (grouped by peptide)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.757 | 0.740 | 0.550 | 3.699 | 9.326 |
| transformer | 0.685 | 0.689 | 0.449 | 4.184 | 11.065 |
| SE+MLP | 0.767 | 0.756 | 0.579 | 3.870 | 14.691 |
| SE+XGB | 0.774 | 0.766 | 0.589 | 3.656 | 9.986 |

## split C (grouped by allele)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.640 | 0.602 | 0.364 | 3.467 | 6.638 |
| transformer | 0.443 | 0.405 | 0.151 | 4.223 | 8.042 |
| SE+MLP | 0.700 | 0.629 | 0.426 | 2.980 | 5.559 |
| SE+XGB | 0.602 | 0.559 | 0.284 | 3.835 | 7.255 |

## split C2 (grouped by allele cluster)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | -0.009 | 0.017 | 0.065 | 3.323 | 6.256 |
| transformer | -0.138 | -0.165 | 0.077 | 3.280 | 6.255 |
| SE+MLP | -0.162 | -0.157 | 0.031 | 3.427 | 6.276 |
| SE+XGB | -0.233 | -0.264 | -0.127 | 3.305 | 5.954 |
