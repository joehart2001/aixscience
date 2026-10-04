I want to take one of these MACE models, freeze the descriptors, and then add a readout head on top of the descriptors which predicts the binding affinity.
Ideas of how to do this:
* It will probably have to be a global head. The simplest thing would be to take the features on the atoms along the backbone (N,O,C,CA) of the peptide and residue, have some order them go in (or account for permutation) as use these to predict the stability. 
* One could take the backbone atoms or some pooled atomic descriptor (depending on residue)
* We include the HLA and the peptide

1. Get the atomic descriptors
2. Pool by residue
3. 
---

## Representation decided (3 Oct 2026)

### The dataset is uniform at residue level, not at atom level

All 28,166 rows: peptide 9 residues, `hla_seq` 182, `hla_pseudoseq` 34. No gaps,
no insertions, standard 20 amino acids only. So a fixed-length token sequence is
available for free.

Atom counts are *not* fixed: a peptide has 61-92 heavy atoms depending on
sequence (GLY 4 per residue, TRP 14). Any fixed-shape tensor therefore has to be
indexed by residue, with some rule for collapsing the variable atom set inside.

### Chosen layout: five slots per residue

    0  N          exact, single atom
    1  CA         exact, single atom
    2  C          exact, single atom
    3  O          exact, except at the C-terminus (see OXT)
    4  sidechain  mean over the residue's remaining heavy atoms

Averaging happens *only* in slot 4, where the atom count genuinely varies (1 for
ALA, up to 10 for TRP). The four backbone slots hold single atoms, so a TRP
backbone is directly comparable to a GLY backbone with nothing lost.

Rejected alternatives: whole-residue mean `(L, 1024)` averages over 4-14 atoms
and discards within-residue detail; backbone-only `(L, 4, 1024)` throws the side
chain away entirely; canonical atom slots with a mask `(L, 15, 1024)` is lossless
but 3x the width, which is unaffordable at the sample sizes reachable here.

### Decision: OXT folded into slot 3

The C-terminal residue carries two carboxylate oxygens, `O` and `OXT`. They are
resonance-equivalent, so slot 3 at the last residue is their **mean**.

The alternatives were worse. Leaving `OXT` in the side-chain slot is chemically
wrong - it is backbone carboxyl, not side chain, and it would contaminate slot 4
at exactly one position. Giving `OXT` its own sixth slot would leave that slot
empty at 8 of 9 peptide positions (181 of 182 HLA positions), wasting a sixth of
the tensor to encode one atom.

Consequence to remember: slot 3 is exact everywhere **except** the final residue,
where it is an average of two degenerate atoms. If a model ever shows an anomaly
localised to the last position's slot 3, this is the first thing to check.

### Decision: glycine side chains are zeroed and masked

Glycine has no side-chain heavy atom, so slot 4 does not exist for it. Those
slots are set to **zero** and flagged `False` in an accompanying boolean mask.
In the 100-complex peptide set this is 39 of 900 residue slots (4.3%).

A zero vector is an acceptable stand-in, but it is a *fabricated* value, not a
measurement - it sits at the origin of descriptor space, which is not a neutral
location. The mask is shipped alongside `X` so a head can attend over it or
learn its own absent-residue embedding instead of consuming the zeros as data.
A head that ignores the mask is training on fabricated vectors; that is a
modelling choice, and it should be a deliberate one.

### Hydrogens excluded

PDBFixer places hydrogens from residue templates, so their descriptors encode
our pH-7 protonation assumptions rather than anything Boltz predicted. Heavy
atoms only, throughout.

### Dimensionality warning

Peptide alone is 9 x 5 x 1024 = 46,080 features. With the HLA that becomes
191 x 5 x 1024 = 977,920. Against 100 complexes this is hopeless, and it is not
speculation: going from 1,024 to 9,216 pooled features at n=100 already *reduced*
out-of-fold Spearman from +0.34 to +0.25 (`positional_comparison.csv`). This
tensor is the shape to grow into once there are thousands of structures, not
something to fit today.

### Control that must accompany any result

A residue's mean descriptor is dominated by residue identity - a tryptophan
environment vector looks tryptophan-ish wherever it sits. So a head on these
features can learn a sequence predictor with extra steps and appear to be using
structure. Every descriptor result needs a sequence-only baseline (one-hot
9 x 20, same split, comparable head capacity) reported next to it. The brief asks
precisely whether foundation models beat sequence-based supervised baselines, so
this baseline is the result, not a footnote.

---

## Model switched to MACE-OFF24 medium (3 Oct 2026)

`~/.cache/mace/MACE-OFF24_medium.model`, fetched from
`github.com/ACEsuit/mace-off/raw/main/mace_off24/`. The installed mace 0.3.14
only has OFF23 URLs built in, so OFF24 must be passed as an explicit file path.

Why it replaces the OMOL / MH-1 models used earlier:

| | OMOL extra-large | MH-1 | **OFF24 medium** |
|---|---|---|---|
| whole 3,047-atom complex | out of memory | out of memory | **0.31 s, 10.6 GB** |
| invariant features/atom | 1,024 | 1,024 | **256** |
| cutoff | 6 A, 3 layers | 6 A, 2 layers | 6 A, 2 layers |

Both earlier models had to have the complex split or truncated to fit. OFF24
ingests it whole, so peptide *and* HLA descriptors come from one forward pass on
the intact complex, which is what makes an HLA-side tensor practical at all. The
4x narrower feature vector also cuts directly into the dimensionality problem.

**Caveat, and it is not a small one.** MACE-OFF is trained on neutral organic
molecules and takes no total-charge input. ARG+, LYS+, ASP-, GLU- and both
termini are out of distribution. The earlier OMOL runs did accept a formal
charge. So descriptors near charged groups are less trustworthy here, and the
net-charge control becomes more important rather than less.

### Shapes produced (`claude-experiments/003-boltz-a0201-100/tensors_off24.npz`)

    X_pep   (100,   9, 5, 256)   mask_pep   4,461 / 4,500 slots present
    X_hla   (100, 182, 5, 256)   mask_hla  89,500 / 91,000 slots present

98 MB, 28 s for all 100 complexes. OXT folded exactly 200 times (2 per complex,
one terminus per chain) and peptide glycine masking came out at 39, matching the
independent count from the OMOL-based tensor - the slot logic is model-agnostic.

### First look at signal (5-fold out-of-fold, partial least squares, scaling inside folds)

| representation | dims | rho | charge-residual |
|---|---|---|---|
| peptide: pooled backbone | 256 | +0.238 | +0.146 |
| peptide: pooled all slots | 256 | +0.255 | +0.194 |
| **peptide: per-slot, pooled residues** | **1,280** | **+0.336** | **+0.244** (p=0.014) |
| peptide: full backbone slots | 9,216 | +0.171 | +0.071 |
| peptide: full 5-slot tensor | 11,520 | +0.167 | +0.090 |
| HLA: pooled all slots | 256 | -0.143 | -0.121 |

Reference: OMOL/MH-1 pooled backbone gave +0.339 / +0.226 on 1,024 features.
Net-charge baseline alone is +0.229.

Three things follow.

**OFF24 matches OMOL at a quarter of the width.** Per-slot pooling reaches
+0.336 against OMOL's +0.339, with a slightly *better* charge-residual
(+0.244 vs +0.226). The expensive 1,024-feature model bought nothing.

**Keeping the slots but pooling the positions is the sweet spot.** Distinguishing
N/CA/C/O/sidechain while averaging over the 9 residues roughly doubles rho
against pooling everything (+0.336 vs +0.255). Keeping positions separate as
well collapses it to +0.167 - the same dimensionality wall seen before, now
confirmed on a second model and a second feature width. This is no longer a
quirk of one descriptor set.

**The HLA side carries no signal on its own, as designed.** rho = -0.143,
interval spanning zero. All 100 complexes share one allele sequence, so the only
HLA variation is what the bound peptide induces - and at n=100 that is not
recoverable. The HLA tensor only becomes testable across multiple alleles.

---

## Readout head result (3 Oct 2026) — negative

Full detail in `claude-experiments/LABBOOK.md`, section 005 and its addendum.

Scaled to the complete 1,023-complex HLA-A*02:01 set using a teammate's Boltz
runs (with alignments, complete molecule). Frozen MACE-OFF24 features, 5-slot
per-residue pooling, then a readout head.

Split: project `split_B`, peptide-grouped, 722 / 144 / 157.
Head: GELU, dropout 0.3, Adam 1e-3 / weight decay 1e-4, batch 32, early stopping
patience 40 on validation, 5 seeds.

                                        train      val     test
    seq_onehot      180 in,    27,329     +0.972   +0.703   +0.621 +/- 0.013
    struct_shared  mask-aware, 196,833    +0.992   +0.632   +0.577 +/- 0.024
    struct_flat  11,520 in, 1,478,849     +0.985   +0.604   +0.516 +/- 0.024
    combined     concatenated, ~1.5M      +0.984   +0.618   +0.509 +/- 0.010
    two_stage    sequence then structure  +0.981   +0.712   +0.626 +/- 0.015

Every model memorises the training set (train rho 0.97-0.99) even at the
early-stopping point. Validation ranks the models identically to test, so the
verdict is not an unlucky test draw.

A one-hot of the peptide sequence beats every frozen-feature head. Adding
structure on top of sequence is worth +0.005, below the seed spread.

The dimensionality warning written above was correct and is now resolved: at
n=100 the full tensor scored +0.167, at n=1,023 it scores +0.558. Pooling was
never better, it was a symptom of too few samples. But breaking that wall did
not change the verdict against the sequence baseline.

The control demanded in the section above has now been run, and it came out
against the foundation-model story.
