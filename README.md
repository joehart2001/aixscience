# aixscience

Peptide–MHC class I binding **half-life (stability)** prediction (Serova Protein
Engineering Track Challenge).

<p align="center">
  <a href="https://joehart2001.github.io/aixscience/" target="_blank" rel="noopener">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="demo_app_html/docs/protein.gif">
      <img src="demo_app_html/docs/protein-light.gif" alt="HLA-A*02:01 with the peptide ALLENIHRV bound in the groove, rotating" width="460">
    </picture>
  </a>
</p>

<p align="center">
  <a href="https://joehart2001.github.io/aixscience/" target="_blank" rel="noopener"><strong>▶&nbsp; Watch the interactive demo</strong></a><br>
  <sub>Six scenes, 66 s. How four representations of a peptide–HLA pair are built,
  and where each one breaks. Scrub it, or jump to any scene.</sub>
</p>

<p align="center">
  <em>Above: HLA-A*02:01 with <code>ALLENIHRV</code> in the groove — the real Boltz-2
  prediction, t½ 44.7 h, complex pLDDT 0.988.</em>
</p>

**Frozen Boltz-2 complex embeddings beat every from-scratch sequence model, and
the margin grows with how hard the split is.** On the cluster split, where no
peptide group leaks between train and test, they are the only representation
left above chance.

| split | held out | direct (MLP) | transformer | SE+MLP | SE+XGB | **boltz2 (frozen)** |
|---|---|---|---|---|---|---|
| A | nothing (shuffled) | 0.757 | 0.655 | 0.773 | 0.773 | **0.792** |
| B | peptides | 0.715 | 0.643 | 0.723 | 0.720 | **0.764** |
| C | alleles | 0.584 | 0.379 | 0.663 | 0.666 | **0.792** |
| C2e | clusters | −0.034 | 0.158 | 0.116 | 0.014 | **0.511** |

<sub>Pearson r on held-out test, 9,031 embedded rows across 54 alleles. Full
table in `mlp-embeds/figs/compare/summary_all75_test_metrics.md`. C2e holds only
2 test clusters, so read that row as a spread rather than a point value.</sub>

- **`mlp-embeds/`** — the models: four `*-network/` frameworks (direct,
  transformer, squeeze-boost, boltz) sharing flat modules, plus a cross-network
  comparison harness in `mlp-embeds/figs/`. See `mlp-embeds/README.md`.
- **`Data/`** — dataset, precomputed splits, and Boltz2 embeddings (source of
  truth; models read from it, never write to it).
- **`demo_app_html/`** — the demo above, plus the structure pipeline explainer
  and the renderer behind the rotating complex. Each builds to one
  dependency-free HTML file; published to [Pages](https://joehart2001.github.io/aixscience/) by
  `.github/workflows/pages.yml`, which gates the deploy on the build checks.
  See `demo_app_html/README.md`.

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
