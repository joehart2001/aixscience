# Brief: MACE-OFF24 descriptors for the 75-allele peptide–HLA stability set

For a fresh Claude Code session on **phy-tcm-pc79**. You are not resuming an
earlier conversation; everything you need is here or on disk. Another session is
running on pc75 and owns the Boltz prediction job, the sync, and all the
Boltz-based training. **Do not touch anything it owns** (list at the bottom).

Start with:

    cd /share/ijp30/hackathons/aixscience
    tmux new -s mace

`/share` and `/home` are both network mounts shared with pc75, so no data needs
to move between machines.

---

## 1. The scientific question

London AI x Science Hackathon, Serova Protein Engineering Track. The question
is whether open protein foundation models improve prediction of peptide–HLA
class I complex **stability** (half-life, `thalf_hours`) over sequence-based
supervised baselines. The organisers explicitly want an honest answer —
**a well-supported negative result counts as a result.** Do not chase a number.

Target is half-life, not binding affinity: it is dominated by the off-rate.
Heavily skewed, so work in log10 and treat Spearman as the primary metric.
About 18–20% of rows are recorded as exactly 0 h; floor them at half the
smallest positive value and be aware they form a visible band in any parity plot.

Dataset: `Data/rasmussen_et_al_dataset.csv`, 28,166 rows, from Rasmussen et al.
2016, J Immunol 197(4):1517–1524, doi:10.4049/jimmunol.1600582 — **cite this in
any submission.** All peptides are 9-mers; `allele`, `hla_seq` and
`hla_pseudoseq` are functionally dependent on each other.

## 2. Your job

Produce MACE-OFF24 descriptors for the Boltz-2 structures of the 75-allele
subset, pool them per residue, and test them against the sequence baseline
under the same recipe the Boltz representations were tested with.

This is a real gap in the comparison, not a nice-to-have. The only MACE result
so far is single-allele (experiment 005), where it lost to sequence. Across 75
alleles the receptor genuinely varies, so the representation has something new
to say — and this time **you must keep the HLA side**, which 005 deliberately
dropped. See §5.

## 3. What already exists — read these first

Read in this order; they contain decisions you should not re-litigate:

- `workflow/mace-readout.md` — the design record. OXT folded into the O slot,
  glycine side chains zeroed and masked, hydrogens excluded, why MACE-OFF24,
  the dimensionality warning, and the requirement that a sequence baseline is
  run **before** any representation is believed.
- `claude-experiments/005-a0201-1023/prep.py` — PDBFixer protonation at pH 7,
  12 Å truncation around the peptide, labelled extended xyz. **Reuse this.**
- `claude-experiments/005-a0201-1023/features.py` — MACE-OFF24 pass and slicing.
- `claude-experiments/005-a0201-1023/slots.py` — the 5-slot pooling. Reuse
  unchanged.
- `workflow/split_A_early_sync/` — the Boltz comparison you must match:
  `train_runs.py` (architecture and recipe), `data_split_a.py` (loader,
  baseline construction), `parity.py`, `summarise.py`, and
  `runs/summary_table.txt` for the numbers to beat.

### The truncation is exact, not an approximation

MACE-OFF24 medium has `r_max` 6 Å and 2 layers, so an atom's feature vector
depends on exactly the atoms within 2 × 6 = 12 Å. Keeping the 12 Å ball around
the peptide leaves peptide features bit-exact. This was verified, not assumed:
12 Å reproduced features to max|diff| 1.2e-6 (float32 round-off) where 8 Å was
off by 1.4e-2. **If you change the model, recheck this** — a different `r_max`
or layer count changes the required radius.

Note the structures are the biologically complete complex: chain A the full HLA
heavy chain (276 residues, alpha1+alpha2+alpha3), chain B beta-2 microglobulin
(99), chain C the 9-mer peptide. ~6,120 atoms protonated, ~2,050 after
truncation. Beta-2 microglobulin and alpha3 contribute almost nothing directly
— they are outside the receptive field — but they matter indirectly by changing
the groove conformation Boltz predicted.

## 4. Inputs, and the one thing you must wait for

Structures land here as the pc75 job and sync progress:

    /share/ijp30/hackathons/aixscience2026/boltz_inputs_and_predictions/
        predictions/all75/<sample_id>/     <- what you need
        embeddings/all75/<sample_id>.npz   <- not yours; Boltz features

Plus 491 reused HLA-A*02:01 complexes under the teammate's tree at
`/share/jh2536/hackathons/aixscience2026/boltz_inputs_and_predictions/`.

The manifest of what was selected, with splits, is
`claude-experiments/006-modal-boltz/subset_manifest.csv`
(`sample_id, allele, peptide, thalf_hours, cluster, split_C2, split_B, status`).

**Structures arrive gradually.** Target is 14,998 complexes; a few thousand will
be present when you start. Write every stage to resume from what exists and skip
what is already done, and re-run as more arrive. Do not block waiting.

## 5. Design decisions specific to this run

**Keep the HLA side.** In 005 the HLA was dropped because all 1,023 complexes
shared one allele, so it was a constant and carried no signal (ρ = −0.143,
interval spanning zero). That reasoning does not transfer. With 75 alleles the
groove varies, and the receptor is where the allele-specific signal lives — the
pseudosequence alone reached ρ = +0.529 on split_A. Pool the HLA residues that
fall inside the 12 Å ball, and treat peptide-only and peptide+HLA as separate
representations so the contribution is measurable rather than assumed.

**The baseline is peptide one-hot PLUS pseudosequence one-hot (860 inputs).**
Not peptide alone. Across 75 alleles a peptide-only baseline is a straw man:
peptide alone gives +0.447, pseudosequence alone +0.529, the two together
+0.747. That jump is interaction — peptide-groove fit — and it is the bar.

**Watch the dimensionality.** The 5-slot tensor is 9 × 5 × 256 = 11,520 inputs
for the peptide alone, and adding HLA residues multiplies that. In 005 the full
tensor went from +0.167 at n=100 to +0.558 at n=1,023 — it was a sample-size
symptom, not a pooling failure. With ~15,000 complexes you have more room, but a
representation that is wider than it is informative will still lose. Test a
pooled form alongside the full tensor.

**Splits.** `split_C2` (held-out pseudosequence clusters) is the headline — it
is the honest test. `split_A` is the random split and **leaks**, because each
allele recurs with an identical receptor sequence; use it only as a ceiling and
label it as such everywhere. `split_B` holds out peptides. All three are in
`Data/subsets/splits.csv`, keyed on (allele, peptide).

## 6. Match the existing recipe exactly

Only the input may differ, or the comparison means nothing:

    Linear(n_in, 128) -> GELU -> Dropout(0.3)
    Linear(128,   32) -> GELU -> Dropout(0.3)
    Linear(32,     1)

Adam, lr 1e-3, weight decay 1e-4, batch 32, MSE on standardised log10 half-life,
max 300 epochs, early stopping on validation loss with patience 40, best-val
weights restored, 5 seeds. Feature standardisation from the **train fold only**
— earlier work in this project leaked by scaling before splitting, so put
scaling inside the fold. Compare with a paired bootstrap of the test Spearman
against the sequence baseline, 2,000 resamples of complexes.

Output layout, as requested by the user — one folder per training run:

    runs/<representation>/seed<k>/
        train.log  epoch_log.csv  model.pt  run_details.json
        fig_loss.png  fig_parity.png  predictions.npz
    runs/fig_parity_all.png      grid, representations down, folds across
    runs/summary_table.txt       train/val/test Spearman per representation

Report train, val **and** test Spearman — never test alone. Note that averaging
seed predictions then correlating (an ensemble) scores higher than averaging the
per-seed correlations; if you report both, label which is which.

## 7. Environment

Use the repo venv: `/share/ijp30/hackathons/aixscience/.venv/bin/python`
(mace 0.3.16, ase 3.29.0, torch 2.14.1+cu130, pdbfixer, sklearn 1.6.1).
Model weights: `~/.cache/mace/MACE-OFF24_medium.model` — 256 features, `r_max`
6 Å, 2 layers, giving 2 × 256 invariant features per atom.
Check what is on pc79's GPU before launching anything large.

## 8. Rules

- **Never delete files.** No `rm`, under any circumstances. Move things to a
  backup folder instead.
- Work in `claude-experiments/` with a subdirectory per experiment and keep a
  lab book. Next free number is **008**. (`claude-experiments/` is gitignored.)
- Ask the user if unsure rather than guessing.
- Avoid acronyms the user has not used first.
- Report honestly: if a result is negative, say so plainly. If you retract an
  earlier claim, state it in one line and move on. Several findings in this
  project were already withdrawn after proper cross-validation — a number that
  looks good in-sample usually isn't.

### Owned by the pc75 session — do not touch

- `claude-experiments/006-modal-boltz/` — the Modal app, the running prediction
  job, `sync.py`, `subset_manifest.csv`. Read it; do not run or edit it.
- The Modal volumes and the `phla-boltz` app. Do not launch Modal jobs.
- `embeddings/all75/` — read-only for you.
- `workflow/split_A_early_sync/` — read it to copy the recipe; write nothing.
- The git branches. `ijp_dev` is active; do not commit or push without asking.

Write only under `claude-experiments/008-*/` and, if the user asks for it, a new
folder under `workflow/`.
