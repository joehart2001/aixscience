# Test-set metrics by split and model

## A (random)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.728 | 0.701 | 0.490 | 3.700 | 9.993 |
| transformer | 0.634 | 0.636 | 0.340 | 4.104 | 10.582 |
| SE+MLP | 0.749 | 0.718 | 0.521 | 3.489 | 9.342 |
| SE+XGB | 0.741 | 0.719 | 0.547 | 3.535 | 9.701 |
| boltz2 (frozen) | 0.794 | 0.776 | 0.595 | 3.374 | 9.545 |

## B (peptide)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.696 | 0.689 | 0.461 | 4.303 | 12.490 |
| transformer | 0.596 | 0.621 | 0.305 | 4.474 | 10.672 |
| SE+MLP | 0.712 | 0.714 | 0.461 | 4.563 | 17.082 |
| SE+XGB | 0.729 | 0.728 | 0.528 | 3.808 | 10.207 |
| boltz2 (frozen) | 0.761 | 0.758 | 0.580 | 3.699 | 9.951 |

## C (allele)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | 0.570 | 0.543 | 0.256 | 3.770 | 6.688 |
| transformer | 0.375 | 0.304 | 0.123 | 4.167 | 8.054 |
| SE+MLP | 0.592 | 0.562 | 0.206 | 3.605 | 6.271 |
| SE+XGB | 0.487 | 0.481 | 0.139 | 4.156 | 7.940 |
| boltz2 (frozen) | 0.714 | 0.703 | 0.548 | 3.164 | 6.368 |

## C2 (cluster)

| model | Pearson r | Spearman ρ | within-allele ρ | MAE (h) | RMSE (h) |
|---|---|---|---|---|---|
| direct (MLP) | -0.065 | -0.054 | 0.015 | 3.468 | 6.835 |
| transformer | -0.098 | -0.127 | 0.041 | 3.397 | 6.911 |
| SE+MLP | -0.108 | -0.184 | 0.069 | 3.506 | 6.891 |
| SE+XGB | -0.075 | -0.177 | -0.047 | 3.303 | 6.594 |
| boltz2 (frozen) | 0.425 | 0.485 | 0.440 | 3.030 | 6.548 |
