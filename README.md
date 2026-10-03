# aixscience

Peptide–MHC class I binding **half-life (stability)** prediction (Serova Protein
Engineering Track Challenge). See `direct-network/` (sequence-based models) and
`boltz-network/` (Boltz2-embedding models).

## References

### Dataset
- Rasmussen M, Fenoy E, Harndahl M, Kristensen AB, Nielsen IK, Nielsen M, Buus S.
  **Pan-Specific Prediction of Peptide–MHC Class I Complex Stability, a Correlate
  of T Cell Immunogenicity.** *The Journal of Immunology.* 2016;197(4):1517–1524.
  doi:[10.4049/jimmunol.1600582](https://doi.org/10.4049/jimmunol.1600582) ·
  [PubMed](https://pubmed.ncbi.nlm.nih.gov/27402703/)
  — source of the `rasmussen_et_al_dataset.csv` stability measurements and the
  NetMHCstabpan method (pan-specific PCC = 0.676, global rescaling).

### State-of-the-art reference values (plotted as dashed lines on correlation figures)
- **Peptide:MHC Binding Stability Prediction Using Protein Language Models.**
  *bioRxiv* 2026.
  doi:[10.64898/2026.06.28.735023](https://doi.org/10.64898/2026.06.28.735023) ·
  [bioRxiv](https://www.biorxiv.org/content/10.64898/2026.06.28.735023v1)
  — best model (MINT Transfer) on the NetMHCstabpan test set under a
  leakage-controlled 80%-identity peptide-cluster split: **Pearson r = 0.76,
  Spearman ρ = 0.79** (the `REFERENCES` values in `*/plots.py`). NetMHCstabpan
  scores ρ = 0.88 on that set but it is leakage-inflated.

### Other methods cited for context (binding *affinity*, not stability)
- Reynisson B, Alvarez B, Paul S, Peters B, Nielsen M. **NetMHCpan-4.1 / NetMHCIIpan-4.0.**
  *Nucleic Acids Research.* 2020;48(W1):W449–W454.
  doi:[10.1093/nar/gkaa379](https://doi.org/10.1093/nar/gkaa379) — affinity Pearson r ≈ 0.78–0.82.
- O'Donnell TJ, Rubinsteyn A, Laserson U. **MHCflurry 2.0.** *Cell Systems.*
  2020;11(1):42–48. doi:[10.1016/j.cels.2020.06.010](https://doi.org/10.1016/j.cels.2020.06.010)
  — affinity Pearson r ≈ 0.76–0.80.
- Chu Y, et al. **TransPHLA (a transformer-based method for pMHC binding, Zhao et al.).**
  *Nature Machine Intelligence.* 2021. — epitope classification AUC ≈ 0.93–0.96.

> Note: affinity (KD/IC50) is a different, less kinetically noisy target than
> stability; those numbers are context only and are **not** used as reference
> lines. The reference lines are the stability-specific values above.
