# Boltz-2 peptide-HLA half-life head: implementation plan

## Objective

Build the smallest defensible model that predicts measured peptide-HLA complex
half-life from frozen Boltz-2 representations, then test whether the existing
MACE-MH-1/`omol` interaction score adds useful information.

The first version should answer one question:

> Does a supervised head on frozen Boltz-2 interface features rank unseen
> peptides better than a sequence-only baseline and the raw MACE score?

This is a representation-learning experiment, not a full Boltz-2 fine-tune.
Boltz weights remain frozen throughout the initial implementation.

## Scope and decisions

- Start with HLA-A\*02:01, which has 1,023 labelled examples in the supplied
  dataset.
- Represent each complex as HLA heavy chain A, beta2-microglobulin B, and
  peptide C, matching the existing pilot.
- Use one deterministic Boltz-2 prediction per complex initially.
- Extract compact interface features during inference. Do not save the full
  pair representation for every complex.
- Train a small hurdle head because approximately 20% of measurements are
  reported as exactly zero:
  - a classifier for zero/undetected versus positive half-life;
  - a regressor for `log1p(half_life_hours)` on positive observations.
- Keep MACE out of the first head. Add its interaction energy later as one
  scalar feature so its incremental value can be measured cleanly.
- Use peptide-disjoint splits to prevent the same peptide sequence appearing
  in both training and evaluation data.

## Non-goals for the first implementation

- Fine-tuning the Boltz trunk or diffusion model.
- Reproducing Boltz's full training infrastructure.
- Claiming calibrated dissociation kinetics from three-dimensional energy
  alone.
- Processing all 28,166 complexes before the A02:01 experiment passes its
  validation gates.
- Treating reported zero values as literal molecular half-lives of zero. They
  are handled as a separate assay-floor/undetected class.

## Proposed architecture

```text
HLA + beta2m + peptide
          |
          v
   frozen Boltz-2 trunk
          |
          +-- single-token representation s
          +-- pair representation z
          +-- predicted coordinates and confidence
          |
          v
 peptide-HLA interface pooling
          |
          v
 compact feature vector (+ optional MACE scalar)
          |
          v
 shared MLP
      /           \
 P(positive)    log1p(half-life) for positives
```

### Interface masks

- `peptide_mask`: tokens belonging to chain C.
- `receptor_mask`: valid protein tokens not belonging to chain C. This is
  important because Boltz's current small-molecule affinity logic identifies
  all protein tokens as receptor tokens; that would overlap the peptide when
  the binder is itself a protein chain.
- `cross_mask`: peptide-to-receptor and receptor-to-peptide token pairs.
- Optionally restrict the receptor side to predicted contacts within 10 A.

### Pooled features

Start with a small, fixed set rather than learning a new pairformer:

1. Mean and max pooling of peptide single-token features.
2. Mean pooling of receptor single-token features near the peptide.
3. Mean, max, and distance-weighted mean pooling of cross-interface pair
   features.
4. Boltz confidence values such as ipTM and interface confidence.
5. Simple metadata: peptide length and allele identity when the experiment is
   extended beyond one allele.

The extractor must immediately reduce token and pair tensors to a one-dimensional
feature vector on the accelerator. Only that compact vector is moved to CPU and
written to disk. This avoids storing an `O(number_of_tokens^2)` tensor per
complex.

### Prediction head and loss

Use a two-layer MLP with LayerNorm, GELU, and modest dropout. It produces:

- `positive_logit`, trained with binary cross-entropy on `half_life > 0`;
- `log_half_life`, trained with Smooth L1 loss only where `half_life > 0`.

The combined training loss is:

```text
loss = BCE(positive_logit, is_positive)
     + lambda_reg * SmoothL1(log_half_life, log1p(half_life))[is_positive]
```

Tune `lambda_reg` only on the validation set. For reporting, retain both head
outputs and calculate a continuous score such as
`sigmoid(positive_logit) * expm1(log_half_life)`. Rank-based metrics should also
be reported directly from the regression score to ensure the choice of
combination is not hiding poor ordering.

## Dataset and evaluation protocol

### Data checks

Before model work, validate and record:

- required columns, units, missing values, and duplicate rows;
- half-life distribution overall and per allele;
- exact-zero count and plausible assay ceiling/floor effects;
- duplicate peptide-HLA pairs;
- peptides appearing with multiple alleles;
- HLA sequence and pseudosequence duplicates.

### Splits

Create the split once, save it as CSV, and reuse it for every ablation.

For the A02:01 prototype:

- 70% train, 15% validation, 15% test;
- group by exact peptide sequence;
- stratify groups approximately by zero/positive status and positive target
  quantile;
- seed all split generation and training.

For the later multi-allele experiment:

- primary split: peptide-disjoint across all alleles;
- harder secondary split: hold out clusters of similar HLA pseudosequences;
- never select hyperparameters using either test split.

### Metrics

Primary metric:

- Spearman rank correlation on measured half-life.

Secondary metrics:

- Spearman on positive-only measurements;
- RMSE and MAE on `log1p(half_life)` for positives;
- AUROC and average precision for zero versus positive;
- macro-average per-allele Spearman in the multi-allele experiment;
- bootstrap 95% confidence intervals on test-set differences between models.

## Required comparisons

Use the identical saved split for every model:

1. Constant/median predictor as a sanity check.
2. Sequence-only baseline using peptide amino-acid composition and position
   encoding, or frozen protein-language-model peptide embeddings if already
   available.
3. Existing raw MACE interaction-energy score.
4. Frozen Boltz interface head.
5. Frozen Boltz interface head plus MACE interaction energy.

The combined model is only useful if it improves held-out metrics over the
Boltz-only model, not merely training performance.

## Implementation phases

### Phase 0: lock the experiment contract

Deliverables:

- a versioned A02:01 train/validation/test split;
- a dataset summary JSON or Markdown file;
- a single metrics schema shared by all baselines;
- fixed random seeds and configuration files.

Acceptance checks:

- no peptide occurs in more than one split;
- target distributions are comparable across splits;
- the split can be regenerated byte-for-byte from its configuration.

Estimated new code: 100-150 lines.

### Phase 1: implement and validate interface pooling

Add a narrow adapter around the cloned Boltz-2 model. Reuse its pretrained
checkpoint loading and featurisation, but expose the frozen single and pair
representations immediately before the existing affinity machinery.

Validate pooling on the three structures from the existing pilot before
running a larger batch:

- peptide and receptor masks are disjoint;
- the peptide mask selects exactly the chain C residues;
- pooled feature dimensionality is constant;
- features contain no NaN or infinite values;
- repeated extraction with the same seed produces the same vector;
- saved vectors are small enough to cache for the full dataset.

Estimated new code: 100-180 lines plus focused tests.

### Phase 2: cache A02:01 features on `fast10`

Run feature extraction for the 1,023 A02:01 complexes using the existing remote
GPU environment at `/home/jh2536/hackathon/aixscience`.

Implementation requirements:

- resumable processing with one output record per complex;
- skip already completed, validated records;
- record Boltz version, checkpoint identity, input hash, seed, and inference
  settings in every output manifest;
- write failures separately without aborting the batch;
- never write full pair tensors to the cache;
- record wall time and peak GPU memory for capacity planning.

Start with a 16-complex smoke batch, inspect masks and outputs, then process 100
examples. Only launch all 1,023 after those checks pass.

Estimated new code: 80-140 lines, mostly orchestration and manifests.

### Phase 3: train the half-life head

Train only the small prediction head from cached feature vectors. This makes
model development quick and keeps Boltz inference separate from supervised
training.

Training requirements:

- fit normalisation statistics on the training split only;
- early-stop on validation Spearman with a secondary check on log-RMSE;
- save the best checkpoint, configuration, feature schema, and metrics;
- run at least five seeds and report mean plus spread;
- confirm the model can overfit a tiny 16-example subset as a plumbing test.

Estimated new code: 120-180 lines.

### Phase 4: add baselines and MACE late fusion

Train the sequence baseline on the same split. Use the existing MACE scoring
pipeline to generate interaction energies for the A02:01 structures, join them
by peptide-HLA sample identifier, normalise using training data only, and append
the energy as one scalar to the Boltz feature vector.

Do not regenerate MACE scores merely to match row ordering; joins must use a
stable sample identifier and reject missing or duplicate matches.

Acceptance decision:

- continue with MACE only if Boltz + MACE improves validation performance
  consistently across seeds and improves or preserves held-out test
  performance;
- otherwise retain MACE as a reported negative result and use the simpler
  Boltz-only model.

Estimated new code: 80-140 lines.

### Phase 5: scale or fine-tune selectively

Proceed only if the frozen-head experiment beats the sequence baseline.

In order of increasing cost and risk:

1. Expand from A02:01 to all alleles and add allele representations.
2. Replace fixed pooling with a small trainable cross-interface attention or
   pairformer block.
3. Unfreeze only the new interface module or the final Boltz representation
   block with a very small learning rate.
4. Consider full-trunk fine-tuning only if the dataset and compute budget justify
   it.

Each step must be evaluated as an ablation against the frozen model.

## Proposed project layout

Keep experimental code in this repository and make only the smallest necessary
adapter change to the cloned `boltz` repository.

```text
aixscience_repo/
  configs/
    half_life_a02.yaml
  data/
    splits/
      a02_seed42.csv
  src/boltz_mace/
    dataset.py
    split.py
    boltz_features.py
    interface_pooling.py
    half_life_head.py
    train_half_life.py
    evaluate_half_life.py
  scripts/
    extract_boltz_features.py
    run_fast10_extraction.sh
  tests/
    test_split.py
    test_interface_pooling.py
    test_half_life_head.py
  results/
    half_life_head/
```

If the required Boltz tensors cannot be accessed cleanly without changing the
clone, add one explicitly documented inference hook there rather than copying
Boltz model code into this repository.

## Reproducibility and artefacts

Every experiment should preserve:

- dataset checksum and split file;
- Git revisions of this repository and the Boltz clone;
- Boltz and MACE checkpoint identifiers;
- complete inference and training configuration;
- random seeds;
- compact cached features and feature-schema version;
- per-example predictions, aggregate metrics, and failure manifest;
- hardware and runtime summary.

Do not commit checkpoints, full Boltz outputs, or large feature caches. Commit
the split definitions, configs, lightweight result tables, and summaries.

## Tests

Minimum automated coverage:

- split leakage and deterministic split generation;
- peptide/receptor mask disjointness;
- pooling invariance to padded tokens;
- correct handling of an empty contact set;
- fixed output dimensions across complexes;
- loss masking when a batch contains no positive examples;
- finite forward and backward passes;
- train-only feature normalisation;
- stable joining of Boltz and MACE features;
- a small end-to-end test using synthetic tensors without loading Boltz weights.

## Risks and mitigations

| Risk | Mitigation |
|---|---|
| Boltz structures are confident but insensitive to half-life | Compare representations with sequence baselines and inspect variance before scaling. |
| Full pair embeddings consume excessive storage | Pool on GPU during inference and save one compact vector per complex. |
| Peptide leaks into the receptor mask | Define receptor as valid protein tokens excluding chain C and test disjointness. |
| Exact zeros distort regression | Use a hurdle head and separately report classification and positive-only regression. |
| Repeated peptides leak across alleles | Group all rows by peptide sequence before splitting. |
| MACE energy is not a kinetic observable | Treat it as one optional feature and require held-out incremental value. |
| A02:01 conclusions do not generalise | Treat A02:01 as a prototype, then run peptide- and HLA-cluster-held-out evaluation. |
| Feature extraction fails partway through | Use atomic per-example outputs, a manifest, validation, and resumable execution. |

## Effort estimate

For the A02:01 frozen-head prototype:

- 400-700 lines of implementation and tests;
- approximately one focused day for a first working version;
- additional GPU time for Boltz feature extraction;
- another half to one day for baselines, seed sweeps, and result analysis.

A native Boltz training integration with distributed training, data modules, and
checkpoint compatibility would likely require 800-1,500 lines and several days.
It is not required to answer the initial scientific question.

## Definition of done

The minimal experiment is complete when:

1. The saved test set has never been used for model or threshold selection.
2. All A02:01 examples have compact Boltz features or a documented failure.
3. Constant, sequence-only, MACE-only, Boltz-only, and Boltz-plus-MACE results
   are evaluated on the identical split.
4. Metrics are reported across at least five training seeds with bootstrap
   confidence intervals on the held-out predictions.
5. The result is reproducible from committed configs, split files, and scripts.
6. A written conclusion states whether Boltz adds signal beyond sequence and
   whether MACE adds signal beyond Boltz.
