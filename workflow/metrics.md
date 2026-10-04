# Boltz-2 output metrics

What Boltz-2 writes for each predicted complex, what each field means, and what
we measured when we tested whether any of them predicts peptide-HLA stability.

## Files produced per complex

Boltz writes five files per input into
`<out_dir>/boltz_results_<input_dir>/predictions/<name>/`:

| file | size | contents |
| --- | --- | --- |
| `<name>_model_0.pdb` | ~127 KB | the predicted 3D structure |
| `confidence_<name>_model_0.json` | ~650 B | scalar confidence summary (below) |
| `pae_<name>_model_0.npz` | ~129 KB | residue x residue predicted aligned error |
| `pde_<name>_model_0.npz` | ~115 KB | residue x residue predicted distance error |
| `plddt_<name>_model_0.npz` | ~800 B | per-residue pLDDT |

About 470 KB per complex, so ~13 GB if ever run over the full 28,166-row
dataset. `--output_format mmcif` or discarding the PAE/PDE matrices cuts this
sharply; those two matrices are ~97% of the bulk.

`model_0` is the first (and by default only) diffusion sample. `--diffusion_samples N`
produces `model_0 .. model_{N-1}`, ranked by confidence.

## The confidence JSON, field by field

All values are 0-1. The `i` prefix throughout means **interface** — restricted to
residue pairs spanning two different chains, rather than the whole complex.

### Headline

**`confidence_score`** — Boltz's own weighted blend of the metrics below, used to
rank diffusion samples against each other. Convenient as a single number, but it
mixes interface and intra-chain quality, so it is the least interpretable field
here. Prefer a specific metric when testing a hypothesis.

### Positioning of chains relative to each other

**`iptm`** — *interface predicted TM-score*. Confidence that the chains are
positioned correctly **with respect to one another**. This is the key metric for
a complex: a model can fold both chains perfectly and still dock them wrongly, in
which case pLDDT stays high while ipTM collapses. For peptide-HLA this answers
"is the peptide in the groove, the right way round, in the right register?"

**`ptm`** — *predicted TM-score*, the whole-complex version. Dominated by the
larger chain, so for a 182-residue HLA plus a 9-mer it mostly reports on the HLA
fold and says little about the peptide.

**`pair_chains_iptm`** — ipTM broken out per chain pair, as a nested dict keyed by
chain index (`"0"` = chain A = HLA, `"1"` = chain B = peptide). The off-diagonal
entry `["0"]["1"]` is the A-B interface specifically; diagonal entries are
within-chain. With two chains this is the most targeted confidence number
available, and in our data it had ~6x more spread than `iptm`.

**`chains_ptm`** — pTM per individual chain. `["1"]` is the peptide alone.

**`ligand_iptm` / `protein_iptm`** — ipTM restricted to ligand or protein
interfaces. `ligand_iptm` is **0.0** for peptide-HLA because the peptide is
declared as a `protein` chain, not a ligand. This is also why Boltz-2's affinity
module cannot be used here: it requires the binder chain to be a `ligand`
(SMILES/CCD), and raises `"Affinity is currently only supported for ligands"`
otherwise.

### Local geometric confidence

**`complex_plddt`** — *predicted Local Distance Difference Test*, averaged over
the complex. Per-residue confidence in **local** geometry: "do I know the
environment around this residue?" High pLDDT with low ipTM means well-folded
chains, wrongly docked.

**`complex_iplddt`** — the same, restricted to interface residues. More relevant
than `complex_plddt` for binding questions, since it ignores the well-determined
scaffold interior.

Per-residue values live in `plddt_*.npz`; the PDB B-factor column carries the
same numbers (scaled 0-100), so standard structure viewers colour by pLDDT.

### Expected error

**`complex_pde`** — *predicted distance error*, mean over the complex. Unlike the
others, **lower is better** — it is an error estimate, not a confidence.

**`complex_ipde`** — predicted distance error at the interface. The one to watch
for binding, and the field with the largest dynamic range in our test.

## Measured result: these do NOT predict stability

Experiment `claude-experiments/003-boltz-a0201-100`: 100 `HLA-A*02:01` complexes
drawn from the `split_B` train fold, stratified across half-life deciles,
spanning **0.00 to 48.1 h (a 481x range)**. Single-sequence mode (no MSA), since
with a fixed allele the HLA MSA is identical for every complex and cannot
discriminate between peptides. 6 min 21 s total, ~3.8 s per complex.

Spearman correlation against measured `thalf_hours`, 2000-sample bootstrap CI:

| metric | observed range | sd | Spearman | 95% CI | p |
| --- | --- | --- | --- | --- | --- |
| `confidence_score` | 0.9710-0.9889 | 0.0030 | +0.015 | [-0.18, +0.24] | 0.88 |
| `iptm` | 0.9737-0.9920 | 0.0047 | +0.016 | [-0.19, +0.23] | 0.88 |
| `ptm` | 0.9742-0.9892 | 0.0040 | -0.127 | [-0.32, +0.07] | 0.21 |
| `complex_plddt` | 0.9688-0.9886 | 0.0034 | +0.005 | [-0.18, +0.21] | 0.96 |
| `complex_iplddt` | 0.9547-0.9892 | 0.0043 | +0.093 | [-0.10, +0.29] | 0.36 |
| `complex_pde` | 0.2561-0.2788 | 0.0057 | +0.046 | [-0.17, +0.25] | 0.65 |
| `complex_ipde` | 0.2714-0.4196 | 0.0359 | -0.127 | [-0.33, +0.08] | 0.21 |
| `chain_ptm_peptide` | 0.9563-0.9916 | 0.0064 | +0.106 | [-0.10, +0.31] | 0.29 |
| `pair_iptm_AB` | 0.8633-0.9795 | 0.0168 | +0.125 | [-0.09, +0.31] | 0.22 |

**Every confidence interval contains zero. No metric reaches p < 0.2.**

The binned view is equally flat — median `complex_ipde` by half-life decile moves
between 0.311 and 0.355 with no monotonic trend, while the median half-life
across those same deciles rises from 0.10 h to 27.0 h.

The deeper problem is **dynamic range**. `iptm` spans 0.0183 across a 481-fold
change in half-life, and 8 of the 9 metrics have sd < 0.02. There is almost
nothing to correlate with. Boltz-2 is uniformly, nearly maximally confident about
every 9-mer in a class I groove.

### Why this is unsurprising in hindsight

Structure predictors are trained to reproduce experimentally solved structures,
and a crystallised peptide-HLA complex is by construction one where the peptide
*did* bind. The model has no training signal for "this peptide binds, but only
briefly". Confidence reports *how sure the model is about the coordinates*, not
*how long the complex survives* — and for a conserved groove with a 9-mer, it is
sure about the coordinates either way.

Stability is also dominated by the dissociation rate, which is a kinetic
property. A single static structure contains no kinetics.

### What this rules out, and what it does not

Ruled out: using Boltz-2 confidence scalars directly as stability features. Do not
spend compute predicting structures just to read these numbers.

Not ruled out, in rough order of promise:

1. **Physical interaction energy** from the predicted structures — the
   `boltz-mace-stability` route. The structures themselves may be fine even
   though the confidence scores are uninformative; the 20 contact positions and
   P2/P9 anchor burial were recovered correctly in experiment 001.
2. **Geometric descriptors** from the PDB — buried surface area, anchor-pocket
   distances, contact counts. Physical measurements rather than model
   self-reports.
3. **The PAE submatrix** restricted to peptide-HLA residue pairs. The scalars
   average this signal with the well-determined scaffold; the raw matrix does
   not. `pair_iptm_AB` is the closest scalar proxy and had the best spread of the
   lot, which weakly supports looking here.
4. **Protein language model embeddings**, which is a different hypothesis
   entirely and untouched by this result.

### Caveats

- One allele only. `HLA-A*02:01` is promiscuous (2 zeros in 100 vs 20% globally),
  so this tests graded discrimination among binders, not binder vs non-binder.
- Single-sequence mode. An MSA would sharpen the HLA scaffold but is constant
  across these 100 complexes, so it could not create peptide-dependent signal.
- n=100. Enough to exclude a strong correlation; a true |rho| < 0.2 would not be
  reliably detected.
- Beta-2-microglobulin omitted. Real class I complexes are heterotrimers.

Reproduce: `claude-experiments/003-boltz-a0201-100/{prepare,analyse}.py`

---

# Result 2: interaction energies

Same 100 complexes, step 4-5 of `workflow.md`. Structures from Boltz-2,
hydrogens from PDBFixer at pH 7, energies from MACE.

## Setup and the choices that made it run

`E_int = E(A.P) - E(A) - E(P)`, all three at the fixed complex geometry, so this
is an instantaneous interaction energy with no relaxation.

Three decisions were forced by the 24 GB card, and each is worth recording:

**Energy only, no forces.** Skipping the autograd graph raised the feasible
system size from ~1,250 to ~2,250 atoms in float64. We never need forces - no
optimisation, no dynamics.

**float32.** The full 3,047-atom complex needs ~30 GB in float64 and OOMs; in
float32 it fits in 14.9 GB and runs in 0.57 s. The cost was measured rather than
assumed, by computing `E_int` both ways on a truncated system where both fit:

| | |
| --- | --- |
| max \|float32 - float64\| | 0.009 eV |
| `E_int` spread across complexes | 2.554 eV |
| error as fraction of signal | **0.36%** |

**No truncation.** Since the full system fits, there is no cluster
approximation. This matters: the receptive field of MH-1 is 12 A (6 A x 2
layers) and exactness under truncation needs 2 x r_eff = 24 A, which retains
99.6% of atoms anyway. Truncation was never going to help - the peptide sits at
the centre of a protein only 55 A across, so an 18 A sphere around it already
engulfs 94% of the structure.

**Models.** MH-1 (`omol` head) ran all 100 complexes in 107 s. **MACE-OMOL
failed on all 100** - it has 3 interaction layers to MH-1's 2 and OOMs even at
float32 energy-only. OMOL needs a larger card.

Charges come from PDBFixer protonation: the HLA is -1 in every complex (same
allele), peptides span -3 to +3.

## Measured result: E_int does NOT predict stability

`E_int` ranged -6.74 to +3.06 eV, median -4.25.

| relationship | Spearman | 95% CI | p |
| --- | --- | --- | --- |
| `E_int` vs half-life | +0.121 | [-0.08, +0.32] | 0.232 |
| `E_int` vs peptide charge | **+0.330** | | **8.1e-04** |
| peptide charge vs half-life | **+0.229** | [+0.01, +0.42] | **0.022** |
| charge-residual of `E_int` vs half-life | +0.024 | [-0.18, +0.22] | 0.816 |

Read together these say something sharper than "no correlation".

1. `E_int` does not predict half-life - its CI contains zero.
2. `E_int` is strongly driven by **net peptide charge**, which is exactly the
   predicted failure of a vacuum interaction energy: a -1 HLA against peptides
   from -3 to +3 produces Coulomb terms far larger than the binding-chemistry
   differences of interest. For reference, changing the peptide charge by one
   unit at fixed geometry moves its energy by ~5.9 eV.
3. Net charge *does* weakly but significantly predict half-life. More positively
   charged 9-mers sit longer in the HLA-A*02:01 groove.
4. Removing the charge dependence from `E_int` leaves **nothing** (rho = +0.024).

**So the entire predictive content of `E_int` is net charge, and it recovers
that signal worse than simply counting it.** Summing Arg/Lys minus Asp/Glu from
the sequence gives rho = 0.229; the full pipeline of structure prediction,
protonation and quantum-ML energies gives rho = 0.121. The expensive route is
outperformed by a one-line operation on the peptide string.

### Why

**No desolvation.** A charged peptide pays a large desolvation penalty on
binding which a vacuum calculation ignores entirely. This is what MM-GBSA
exists to correct, and `openmm` is already a dependency.

**Half-life is kinetic.** `E_int` is a well depth; the off-rate depends on the
dissociation barrier. Even an exact binding energy is a proxy for the wrong
quantity.

### Caveats

- One allele. The charge effect may be specific to the A*02:01 groove rather
  than general - worth checking across alleles before relying on it.
- rho = 0.229 at n=100 barely clears zero, and is marginal across several tested
  correlations.
- Charge may proxy amino-acid composition generally, not electrostatics as such.
- The 46 neutral peptides are a confound-free subset, at the cost of power.

Reproduce: `claude-experiments/003-boltz-a0201-100/{energies,analyse_energies}.py`
Figure: `fig_energy_vs_halflife.png`

---

# Descriptors

MACE invariant descriptors for the **peptide atoms in the bound complex** -
computed on the full 3,047-atom system, then sliced to the peptide. Each atom's
descriptor therefore encodes its environment including the surrounding groove;
the same peptide in isolation would give different values.

Only invariant (l=0) components are kept. Raw `node_feats` contain equivariant
parts that rotate with the molecule, so they are not valid ML features - two
identical complexes in different orientations would give different numbers.

MH-1: 2 layers x 512 invariant features = **1024 per atom**. 100 complexes in
55 s.

| output | contents |
| --- | --- |
| `descriptors/<name>.npz` | per-atom, (n_peptide_atoms x 1024), with symbols, sequence, half-life. 64 MB total |
| `descriptors_pooled.npz` | mean- and sum-pooled, (100 x 1024), with half-life and peptide charge. 791 KB |

**1024 features for 100 samples.** Any model fitted here will overfit trivially.
Use strong regularisation, fit only on the training split, and treat in-sample
performance as meaningless. Mean-pooling is length-invariant; sum-pooling is not,
and peptide atom counts vary 120-182, so sum-pooled features partly encode size.

Reproduce: `claude-experiments/003-boltz-a0201-100/descriptors.py`
