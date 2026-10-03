# Boltz-2 x MACE peptide-HLA stability pilot

This repository tests whether a frozen-coordinate atomistic interaction score can
rank the measured stability of peptide-HLA complexes.

The pilot uses:

- **Boltz-2** to predict peptide-HLA-A\*02:01-beta2m complex structures.
- **MACE-MH-1**, specifically the case-sensitive `omol` head, to calculate an
  interaction-energy proxy.
- The NetMHCstabpan dataset supplied with the London AI x Science protein
  engineering challenge.

The score is:

```text
E_interaction = E(complex) - E(receptor) - E(peptide)
```

All three single-point energies use identical frozen coordinates. Lower values
are ranked as stronger interactions. This is an exploratory structural score,
not a calculation of the kinetic dissociation barrier measured by half-life.

## Setup

Python 3.11 or 3.12 is recommended.

```bash
uv sync
```

MACE-MH-1 is distributed under the Academic Software License. Loading
`model="mh-1"` downloads it into the MACE cache if it is not already present.

## Run the minimal pilot

Create five inputs spanning the HLA-A\*02:01 half-life distribution:

```bash
uv run python -m boltz_mace.prepare_inputs \
  --dataset ../rasmussen_et_al_dataset.csv \
  --n-samples 5
```

Run Boltz-2. Single-sequence mode is the reproducible, no-network default:

```bash
uv run boltz predict inputs/pilot/boltz \
  --out_dir outputs/boltz \
  --cache outputs/cache/boltz \
  --accelerator gpu \
  --model boltz2 \
  --diffusion_samples 1 \
  --recycling_steps 3 \
  --sampling_steps 50 \
  --seed 42 \
  --use_potentials
```

For higher-quality HLA and beta2m representations, recreate the inputs with
`--use-msa-server` and add `--use_msa_server` to the Boltz command. The 9-mer
peptide remains in single-sequence mode.

Score the structures:

```bash
XDG_CACHE_HOME=outputs/cache uv run python -m boltz_mace.score \
  --manifest inputs/pilot/manifest.csv \
  --predictions-dir outputs/boltz \
  --model mh-1 \
  --head omol \
  --device cpu \
  --dtype float64 \
  --hydrogen-seed 0
```

The scorer repairs missing terminal atoms, adds hydrogens at pH 7.4, keeps
complete receptor residues within 10 A of the peptide, writes
`results/pilot_scores.csv`, and reports Spearman's rank correlation. Boltz
confidence metrics are retained to identify unreliable complexes.

Float64 is intentional: the interaction score subtracts large molecular total
energies, and float32 loses meaningful precision through cancellation.

## Important limitations

- Boltz-2's affinity head is currently intended for small-molecule ligands, so
  it is not used for the peptide chain.
- MACE-MH-1/OMOL is a molecular foundation potential, not a protein force field.
- Static interaction energy is not dissociation kinetics and omits solvent,
  entropy, and conformational rearrangement.
- The initial sample is a pipeline and signal check. Any substantive conclusion
  needs many more peptides, multiple Boltz samples, and held-out evaluation.
