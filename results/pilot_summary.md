# Three-peptide pilot result

## Configuration

- Allele: HLA-A*02:01
- Peptides: observed minimum, median, and maximum half-life examples
- Structure model: Boltz-2, one sample, 3 recycles, 50 diffusion steps,
  physical steering, seed 42, single-sequence mode
- Structure: extracellular HLA heavy chain, beta2-microglobulin, and peptide
- Preparation: PDBFixer terminal repair and pH 7.4 hydrogenation on OpenMM's
  deterministic Reference platform, hydrogen seed 0
- Atomistic model: MACE-MH-1, `omol` head, float64
- Score: frozen-coordinate `E(complex) - E(receptor) - E(peptide)` on complete
  receptor residues within 10 A of the peptide

## Result

Lower MACE interaction energy is ranked as stronger.

| Predicted rank | Peptide | Measured half-life (h) | MACE interaction energy (eV) | Boltz ipTM |
|---:|---|---:|---:|---:|
| 1 | HLSTAFARV | 3.9 | -9.465737 | 0.981995 |
| 2 | ALVSEVTEV | 48.1 | -9.126578 | 0.987832 |
| 3 | ALAAGIVGV | 0.0 | -7.309104 | 0.980767 |

Spearman rho is 0.50 (`p = 0.667`, `n = 3`) when more negative interaction
energy is treated as more stable. The zero-half-life peptide is correctly ranked
weakest, but the ordering of the two nonzero examples is reversed. This validates
the pipeline only; three deliberately extreme examples cannot establish
predictive performance.
