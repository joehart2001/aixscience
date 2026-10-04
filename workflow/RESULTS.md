# Do protein foundation models improve peptide–HLA stability prediction?

London AI x Science Hackathon, Serova Protein Engineering Track, 3–4 October 2026.

Data: Rasmussen et al. 2016, *J Immunol* 197(4):1517–1524, doi:10.4049/jimmunol.1600582 —
28,166 measured peptide–HLA class I complex half-lives, 75 alleles, all 9-mers.

**Answer: yes, but only on the question that matters.** Where the allele has been
seen before, a sequence one-hot is sufficient and Boltz-2 adds almost nothing.
Where the allele is new, the sequence baseline collapses to nothing or worse,
and Boltz-2 is the only representation that survives.

---

## 1. The result depends entirely on the split

| split | what is held out | sequence (trees) | Boltz pooled (trees) |
| --- | --- | --- | --- |
| A | nothing (random) | +0.747 | +0.761 |
| B | peptides | +0.766 | +0.755 |
| **C2** | **whole pseudosequence clusters** | **−0.198** | **+0.604** |

Spearman rank correlation on log10 half-life, held-out test fold.

On split_C2 no allele in the test fold appears in training, and neither does its
binding groove. The sequence baseline does not merely fail there: it goes
**negative**, ranking unseen alleles backwards. Training correlation is +0.769
against test −0.198, which is memorisation of allele identity rather than
learning of binding chemistry. The stronger learner is the more harmful one —
the perceptron, which underfits, manages +0.097.

## 2. Every combination containing Boltz wins; none without it does

All 31 combinations of five feature blocks, 14,997 complexes, split_C2,
3 seeds each, both gradient-boosted trees and a perceptron
(`split_C2_mace/permutation_table.txt`).

| | n | test Spearman (trees) | median |
| --- | --- | --- | --- |
| contains the Boltz block | 16 | +0.514 to +0.619 | +0.557 |
| does not | 15 | −0.226 to +0.029 | −0.099 |

Perfect separation, no overlap. Best is `seq+boltz` at **+0.619**; Boltz alone
reaches +0.603, so sequence contributes a real but small +0.016 once Boltz
anchors the prediction.

**Every MACE block costs performance**: boltz +0.603 → +boltz+node +0.571 →
+node+edge +0.556 → all five blocks +0.548. The 28,807-dimensional model is
worse than the 1,547-dimensional one.

## 3. Boltz discriminates peptides inside grooves it has never seen

Within one cluster the groove is nearly fixed, so allele identity cannot
contribute and the correlation is pure peptide discrimination
(`split_C2_mace/per_cluster_gbm_boltz.csv`).

| fold | cluster | n | alleles | ρ |
| --- | --- | --- | --- | --- |
| test | 19 | 654 | 3 | +0.638 |
| test | 16 | 273 | 1 | +0.484 |
| test | 3 | 1378 | 5 | +0.445 |
| test | 5 | 187 | 1 | +0.442 |
| val | 14 | 249 | 1 | +0.542 |
| val | 18 | 250 | 1 | +0.456 |
| val | 7 | 1488 | 10 | +0.430 |
| val | 20 | 256 | 1 | +0.376 |
| val | 10 | 255 | 1 | +0.159 |

Eight of nine measurable held-out clusters fall between +0.38 and +0.64. The
overall test figure of +0.614 exceeds most individual clusters, so part of it is
between-cluster ranking; the honest statement is **+0.614 overall, +0.38–0.64
within unseen grooves**.

## 4. Why: the useful representation is the one that does *not* encode identity

UMAP over 14,997 complexes, with three diagnostics per block (`umap_energy/`).
*Extrapolation* is the median embedding distance from a test complex to its
nearest training complex, divided by the typical train–train spacing.

| block | dims | allele purity | cluster purity | extrapolation |
| --- | --- | --- | --- | --- |
| pseudosequence one-hot | 680 | 98.8% | 100.0% | **5517×** |
| MACE site energies | PCA 50 | 75.2% | 99.4% | **61×** |
| Boltz pooled | PCA 50 | **50.3%** | 91.8% | **26×** |
| MACE edge features | PCA 50 | 30.3% | 70.8% | 2× |
| MACE node descriptors | PCA 50 | 9.6% | 37.4% | 1× |
| peptide one-hot | 180 | 3.7% | 24.2% | 2× |

*Purity = fraction of a complex's 10 nearest neighbours sharing its allele or
cluster. All MACE blocks and Boltz are reduced by PCA to 50 components on the
training fold before embedding, so these rows are directly comparable; the two
one-hots are narrow enough to embed raw. Unreduced, the energies give 84.8% and
423× and Boltz 47.6% and 18× — the same ordering, larger numbers.*

**There are two distinct failure modes, not one.**

*Unreachable representations.* The pseudosequence one-hot and the MACE site
energies describe the groove so completely that each allele forms its own
island. A held-out allele then lands where the model has never had a gradient,
and it extrapolates into empty space — which is how a score becomes *negative*
rather than merely zero. Half-life is also uniform within each island: these
features capture *which groove*, not *which peptide fits it*.

*In-distribution but uninformative.* The MACE node and edge blocks are the
opposite case. They have low allele purity and sit essentially on top of the
training distribution (1–2×), yet still score +0.02 and −0.07 on held-out
alleles. Nothing is unreachable; the features simply do not carry the target
across grooves. Diluting Boltz with them costs performance monotonically.

Boltz sits between the two and fails in neither way: it **mixes alleles**
(a complex's neighbours are a different allele half the time), it stays close
to the training distribution, and half-life varies systematically within its
regions rather than being constant inside islands.

31.6% of the variance in log half-life lies between alleles and **68.4% within
them**, so a representation that only separates alleles cannot address most of
the problem even in principle.

**The sequence baseline's failure is localised precisely.** Peptide sequence
space is well covered under both splits (extrapolation 2–4×, allele purity
3.7%), so held-out peptides are not unfamiliar. The collapse comes entirely from
the pseudosequence half of the input, at 5517× extrapolation — a lookup table
with no entry for the test alleles.

## 5. What did not work

Well-supported negatives, all on identical rows with identical training recipes.

- **MACE-OFF24 descriptors** (node, site energies, cross-chain edge features) on
  14,997 complexes: +0.02, −0.17, −0.07 on held-out alleles. The edge features
  were the strongest hypothesis — a cross-chain contact cannot be a disguised
  sequence feature — and they still do not transfer.
- **Cross-attention** over Boltz trunk tokens with a learned distance bias:
  +0.459 on split_C2, below the plain perceptron's +0.510. Attention alone
  (without sequence and pooled features in the head) reaches only +0.244.
- **Low-rank bilinear** peptide × groove interaction: level with or behind a
  perceptron everywhere; rank 128 worse than rank 64.
- **Charge descriptors**, sequence- and structure-derived (Coulomb proxy, salt
  bridges, per-position charge environment): add +0.017 to sequence alone, and
  **+0.0007** once Boltz is present.
- **MACE rigid-translation Hessian** on 129 complexes, with an escape-coordinate
  projection: +0.261 ± 0.101 alone, and `hess+boltz` is identical to `boltz` to
  three decimals.

The pattern is consistent: explicit physical descriptors carry real information
that Boltz has already absorbed.

## 6. Method notes

- **Boltz-2 predictions**: 14,507 complexes computed on Modal L40S GPUs in
  5h49m with zero failures, plus 491 reused — 14,998 total spanning all 75
  alleles and all 22 clusters, stratified on half-life.
  Validated against a reference run: RMSD 0.03–0.10 Å, ipTM/pLDDT to 4 dp,
  embedding correlation > 0.9995.
- **Controls that changed the reading.** An allele-mean predictor using *no
  peptide information* scores **+0.536** on split_B, so roughly 70% of every
  split_B number is allele identity. Gradient-boosted trees reach +0.755 from
  sequence alone on split_B, against the perceptron's +0.718 — part of what
  earlier runs attributed to Boltz was the perceptron underfitting a one-hot.
- **Target**: log10 half-life, zeros (20.2% of rows, below the assay's detection
  limit) floored at half the smallest positive value. A censored likelihood was
  not tried and is the most obvious unexplored improvement.
- **Standardisation** is fitted on the training fold only, throughout.
- All training runs are stored per seed with per-epoch loss, the fitted model,
  full hyperparameters and dataset record, plus loss and parity figures.

## 7. Limitations

- One structure per complex, one seed, no conformational ensemble. Half-life is
  a kinetic quantity and a single static pose cannot express a barrier.
- MACE-OFF24 is trained on gas-phase organic chemistry: no dielectric screening,
  no desolvation penalty. An implicit-solvent comparison was scoped but not run.
- The Hessian result rests on 129 complexes of a single allele, so it tests only
  the regime where sequence is already sufficient.
- split_C2 has 5 test clusters; cluster 21 holds 7 complexes and is unmeasurable.
- NetMHCstabpan is not a fair comparator — it was trained on this entire dataset.

## Where things are

```
workflow/split_C2_mace/      the 31-combination sweep, permutation_table.{txt,csv}
workflow/split_B_mace/       the same on held-out peptides (21/31 trees)
workflow/split_C2_full/      architectures: perceptron, trees, cross-attention
workflow/split_B_arch/       architectures incl. bilinear, on held-out peptides
workflow/umap_energy/        seven UMAPs with purity and extrapolation diagnostics
workflow/hessian/            escape coordinate and Hessian invariants
workflow/mace-readout.md     design decisions for the MACE feature extraction
claude-experiments/          lab books and earlier single-allele experiments
/share/ijp30/hackathons/aixscience2026/boltz_inputs_and_predictions/
                             structures, embeddings, inputs, alignments, provenance
```
