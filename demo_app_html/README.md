# demo_app_html

Two self-contained animated explainers for the peptide–HLA half-life work. Each
builds to a single HTML file with no runtime dependencies, so it can be opened
from disk, dropped into a slide, or published behind a link.

| Page | What it shows |
|---|---|
| `dist/levels.html` | The representation ladder: direct MLP baseline → SE gate → transformer → frozen Boltz-2, and where each one breaks across the four held-out splits. |
| `dist/pipeline.html` | The structure pipeline: sequence dataset → Boltz-2 complex → frozen MACE features → shallow head. |

```bash
python build.py     # src/ -> dist/
node verify.js      # the checks below; exits non-zero on failure
open dist/levels.html
```

## Where the numbers come from

Every value in `levels.html` comes from
`../mlp-embeds/figs/compare/summary_all75_test_metrics.md`. `verify.js` reads
that file **in place** — no copy — re-parses it and compares all 100 cells
against the page, so the two cannot drift apart silently. Regenerate the
metrics, re-run `node verify.js`, and it will tell you which cells moved.

Five models over four splits, ordered by decreasing leakage:

| split | held out | direct | transformer | SE+MLP | SE+XGB | **boltz2** |
|---|---|---|---|---|---|---|
| A | nothing (shuffled) | 0.757 | 0.655 | 0.773 | 0.773 | **0.792** |
| B | peptides | 0.715 | 0.643 | 0.723 | 0.720 | **0.764** |
| C | alleles | 0.584 | 0.379 | 0.663 | 0.666 | **0.792** |
| C2e | clusters | −0.034 | 0.158 | 0.116 | 0.014 | **0.511** |

Frozen Boltz-2 takes every metric on every split except RMSE on A, where
`SE+XGB` wins 11.014 to 11.297.

**Two caveats the pages carry on their face.** C2e holds only 2 test clusters,
so that row is a spread rather than a point value. And there is no Level 0 run:
`direct-network` defaults to `hla_col: hla_seq`, but every committed template
overrides it to `hla_pseudoseq`, so the full 182-residue path exists in code and
was never trained.

## The structure

`src/data/allenihrv.json` is one real Boltz-2 prediction —
`hla_a_02_01_allenihrv`, measured t½ 44.7 h, complex pLDDT 0.988 — extracted
from the ModelCIF on `fast10`:

```
/share/ijp30/hackathons/aixscience2026/boltz_inputs_and_predictions/
  predictions/all75/hla_a_02_01_allenihrv/hla_a_02_01_allenihrv_model_0.cif
```

To swap in a different complex:

```bash
python tools/extract_structure.py /path/to/<id>_model_0.cif > src/data/allenihrv.json
python tools/pack_structure.py    src/data/allenihrv.json  > src/data/boltz_allenihrv.js
python build.py && node verify.js
```

`extract_structure.py` keeps the peptide's heavy atoms and the first 180 Cα of
the heavy chain, then rewrites every coordinate into a frame derived from the
structure itself — x along the peptide N→C, y from the peptide centroid toward
the platform centroid, z their cross product. That is why the groove stays
horizontal at every rotation without a hand-tuned camera.

Note that Boltz writes heavy atoms only. Hydrogens enter during preparation,
somewhere between the CIF and MACE, which is worth pinning down since it decides
whether the atom-to-residue mapping is built from the CIF or from the protonated
structure.

## How the cartoon is drawn

No DSSP and no viewer library. `src/lib/ss.js` scores each 5-residue window of
the Cα trace against ideal helix geometry (i→i+3 ≈ 5.3 Å, i→i+4 ≈ 6.2 Å) and
ideal extended geometry (9.9 Å, 12.4 Å), takes whichever fits better, then drops
runs too short to be real. On this chain it recovers the two groove helices at
59–87 and 139–180 plus nine strands, which `verify.js` asserts.

Rendering is a painter's algorithm. Each ribbon segment becomes one quad and
each peptide atom and bond one primitive; all ~1,550 sort far-to-near every
frame, so the sheet weaves correctly behind the helices as the complex turns.
Ribbon thickness is two darker strokes along each quad's long edges, and depth
drives lighting through the perspective factor. Peptide bonds are found at a
1.95 Å cutoff, giving 74 bonds over 74 atoms with 8 inter-residue links and one
ring at His P7.

## What verify.js checks

1. All 100 metric values match the source markdown.
2. Scene count, rail entries and chips agree, and each chip seeks into its own scene.
3. Every reveal finishes at least 1 s before its scene is cut. This guards a real
   bug that shipped once: content was still appearing when the cut came, so the
   final element of each scene flashed up and vanished.
4. The cartoon finds 2 helices and ≥7 strands, and the peptide is one connected
   chain with exactly 8 peptide bonds.

## Layout

```
build.py                      src/ -> dist/
verify.js                     correctness checks
src/
  levels.head.html            markup, tokens, prose, metrics table
  levels.body.js              scenes, cartoon renderer, transport
  pipeline.html               self-contained, copied through
  data/
    allenihrv.json            extracted structure
    boltz_allenihrv.js        same, packed for inlining
  lib/ss.js                   Cα-only secondary structure (page + verify)
tools/
  extract_structure.py        ModelCIF -> JSON
  pack_structure.py           JSON -> inline constant
dist/                         generated, safe to delete
```

Metrics are not vendored here; `verify.js` reads
`../mlp-embeds/figs/compare/summary_all75_test_metrics.md` directly.

`levels.html` is assembled rather than written directly because the structure
constant has to be inlined inside the IIFE before `ATOMS` and `TRACE` are
derived from it. Edit `src/`, never `dist/`.

## Editing notes

Colours are CSS custom properties on `:root`, redefined for dark mode in two
guarded blocks. The canvas reads them through `pal()` on resize and on theme
change, so the animation follows the page theme; never hardcode a colour in the
drawing code.

Scene timing is one normalised clock `t ∈ [0,1]`, with each sub-animation a
`sub(t, start, end)` window. To give a scene more room, widen its slice in `CUT`
and re-run `verify.js` — the dwell check will catch it if a reveal now runs past
the cut.
