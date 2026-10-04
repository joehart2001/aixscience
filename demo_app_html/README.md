# demo_app_html

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/protein.gif">
    <img src="docs/protein-light.gif" alt="HLA-A*02:01 with the peptide ALLENIHRV bound in the groove, rotating" width="460">
  </picture>
</p>
<p align="center">
  <em>HLA-A*02:01 with <code>ALLENIHRV</code> in the groove &mdash; the real Boltz-2 prediction,
  t&frac12; 44.7&nbsp;h, complex pLDDT 0.988. Rendered by <code>tools/render_protein.py</code>
  from the deposited coordinates.</em>
</p>

Two self-contained animated explainers for the peptide–HLA half-life work. Each
builds to a single HTML file with no runtime dependencies, so it can be opened
from disk, dropped into a slide, or published behind a link.

| Page | What it shows |
|---|---|
| `dist/pipeline.html` | A plain-language walkthrough: measurements → Boltz-2 complex → frozen MACE descriptors → trainable regression head. |
| `dist/levels.html` | A like-for-like comparison of the direct MLP, SE variants, transformer, and frozen Boltz-2 representation across four held-out splits. |

**Live: [https://joehart2001.github.io/aixscience/](https://joehart2001.github.io/aixscience/)** — published from `main` by
`.github/workflows/pages.yml`, which runs the build and every check below and
refuses to deploy if any of them fail.

```bash
python build.py                      # src/ -> dist/
node verify.js                       # correctness checks
node tools/check_text.js             # label collisions, desktop
node tools/check_text.js --width 400 # and phone
open dist/local/index.html           # or: cd dist/local && python3 -m http.server
```

`dist/` is generated and gitignored. `dist/local/` is what Pages serves, and the
only copies with their own `<head>`: the publishable fragments in `dist/` omit
it because the artifact platform supplies one, which means opening them
directly shows mojibake for every degree sign and Greek letter.

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
so the size of that gap should be treated cautiously. And SE is presented as a
variant of the direct baseline because it reweights the same 704 input channels
rather than introducing a different raw representation.

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

The structure coordinates are literal. The moving messages, feature bars, and
compressed neural-network nodes in `pipeline.html` are explanatory schematics;
they communicate data flow rather than exact activations or layer widths.

## The README animation

```bash
python tools/render_protein.py                       # docs/protein.gif
python tools/render_protein.py --theme light         # docs/protein-light.gif
python tools/render_protein.py --frames 120 --fps 12 # slower and smoother
```

Defaults are 460x288, 90 frames at 9 fps: one revolution every 10 s, 4 degrees
per frame. Those two numbers trade off against each other — dropping the frame
rate alone slows the spin but makes each step more visible, so a slower
rotation needs *more* frames to stay smooth, and the GIF grows with them. The
pair of themes is about 4.5 MB; if that becomes a nuisance, drop the light
variant and serve the dark one to both themes.

A faithful port of the canvas renderer — same secondary structure assignment,
same painter's algorithm — drawn at 3× and downsampled, because PIL does not
antialias polygons. Frames go to ffmpeg's `palettegen`/`paletteuse`, which
produces a much better palette than quantising each frame independently.
Needs Pillow, numpy and ffmpeg. Both themes are committed so the `<picture>`
element above can follow the reader's GitHub theme.

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
  render_protein.py           the rotating GIF above
docs/                         rendered GIFs, committed
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
