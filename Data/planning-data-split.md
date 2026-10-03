Suggested Splits of Data:

In all cases have a ~70/15/15 split between train/val/test.

A: Random
Seed is set to 3102026
B: Group by Peptide - each unique peptide only appears in one of the splits
Seed is set to 1032026
C: Group by each allele - each unique allele only appears in one of the splits
Seed is set to 2026103
C2: Group by allele cluster - alleles sharing >=30 of their 34 contact positions
are treated as one unit, so near-identical grooves cannot be split across train
and test. 75 alleles -> 22 clusters. Each cluster only appears in one of the splits.
Seed is set to 20260310

C tests a new allele where a close relative was in training (the realistic
clinical case). C2 tests a genuinely unseen groove. The gap between them
measures how much performance is near-neighbour lookup rather than
generalisation.

Note: C2 has only 22 groups, so val and test hold ~3 clusters each and those
estimates are noisy. Report with the spread, not as a point value.

All splits stratified on zero-fraction and log half-life.

Save in the csv format