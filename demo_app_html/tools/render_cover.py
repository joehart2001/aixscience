"""Render the repository cover image.

1280x640, the size GitHub uses for a social preview. Reuses the cartoon
renderer from render_protein.py so the complex on the cover is the same real
Boltz-2 prediction the demo turns, and reads the headline numbers out of the
metrics table rather than restating them.

    python tools/render_cover.py            # docs/cover.png
    python tools/render_cover.py --theme light
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import sys

from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from render_protein import (  # noqa: E402
    BOND_CUTOFF, THEMES, assign_ss, dist, fit, frame, runs_of,
)

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRUCTURE = ROOT / "src" / "data" / "allenihrv.json"
METRICS = ROOT.parent / "mlp-embeds" / "figs" / "compare" / "summary_test_metrics.md"

W, H = 1280, 640
SANS = "/System/Library/Fonts/HelveticaNeue.ttc"
MONO = "/System/Library/Fonts/Menlo.ttc"
SPLITS = [("A", "random"), ("B", "peptide"), ("C", "allele"), ("C2", "cluster")]
SEQ_MODELS = ["direct (MLP)", "transformer", "SE+MLP", "SE+XGB"]


def load_metrics() -> dict:
    rows, split = {}, None
    for line in METRICS.read_text().split("\n"):
        head = re.match(r"^##\s+(\S+)", line)
        if head:
            split = head.group(1)
            continue
        cells = [c.strip() for c in line.split("|") if c.strip()]
        if len(cells) == 6 and cells[0] != "model" and not set(cells[1]) <= set("-"):
            rows[(cells[0], split)] = float(cells[1])
    return rows


def font(path: str, size: int, index: int = 0) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size, index=index)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    theme = THEMES[args.theme]
    ink = (233, 239, 245) if args.theme == "dark" else (22, 32, 43)
    mute = (138, 153, 168) if args.theme == "dark" else (93, 107, 122)
    line = (39, 52, 66) if args.theme == "dark" else (210, 217, 224)
    accent = tuple(int(theme["chain"].lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    warn = (214, 106, 106) if args.theme == "dark" else (163, 58, 58)

    img = Image.new("RGB", (W, H), theme["bg"])
    d = ImageDraw.Draw(img)
    m = load_metrics()

    # --- the complex, rendered by the same code the GIF uses, bled off the right
    data = json.loads(STRUCTURE.read_text())
    trace, atoms = data["trace"], [a["p"] for a in data["peptide"]]
    bonds = [(i, j) for i in range(len(atoms)) for j in range(i + 1, len(atoms))
             if dist(atoms[i], atoms[j]) < BOND_CUTOFF]
    pw, ph = 700, 560
    zoom, oy = fit(trace, atoms, 1, pw, ph, margin=0.97)
    cfg = dict(theme=theme, trace=trace, atoms=atoms, bonds=bonds,
               runs=runs_of(assign_ss(trace)), zoom=zoom, oy=oy, w=pw, h=ph)
    mol = frame(2.42, cfg)
    img.paste(mol, (W - pw - 10, (H - ph) // 2))

    # fade the molecule into the background on its left edge
    grad = Image.new("L", (200, H), 0)
    gd = ImageDraw.Draw(grad)
    for x in range(200):
        gd.line([(x, 0), (x, H)], fill=int(255 * (1 - x / 200) ** 1.3))
    img.paste(Image.new("RGB", (200, H), theme["bg"]), (W - pw - 10, 0), grad)

    # --- title block
    x0, y = 64, 58
    d.text((x0, y), "SEROVA PROTEIN ENGINEERING TRACK", font=font(MONO, 13), fill=accent)
    y += 26
    d.text((x0, y), "Boltz-τ", font=font(SANS, 92, 1), fill=ink)
    y += 118
    for part, col in ((" peptide–HLA binding stability from a", mute),):
        d.text((x0, y), "Predicting" + part, font=font(SANS, 25), fill=col)
    d.text((x0, y + 34), "frozen structural foundation model.", font=font(SANS, 25), fill=mute)

    y += 92
    d.line([(x0, y), (x0 + 108, y)], fill=accent, width=3)

    # --- the headline: the split where the sequence models stop working
    y += 38
    d.text((x0, y), "On held-out allele clusters", font=font(SANS, 21, 1), fill=ink)
    y += 42

    best_seq = {s: max(m[(mod, s)] for mod in SEQ_MODELS) for s, _ in SPLITS}
    pairs = [("Boltz-τ readout", m[("boltz2 (frozen)", "C2")], accent),
             ("best sequence model", best_seq["C2"], warn)]
    for label, val, col in pairs:
        d.text((x0, y + 2), f"{val:+.3f}".replace("-", "−"),
               font=font(MONO, 46, 1), fill=col)
        d.text((x0 + 212, y + 22), label, font=font(SANS, 20), fill=mute)
        y += 66

    # --- the four splits, as a strip
    y += 4
    d.text((x0, y), "PEARSON r BY HELD-OUT SPLIT", font=font(MONO, 12), fill=mute)
    y += 20
    cw = 112
    for i, (key, name) in enumerate(SPLITS):
        cx = x0 + i * cw
        bv = m[("boltz2 (frozen)", key)]
        d.text((cx, y), f"{key} · {name}", font=font(MONO, 11), fill=mute)
        # track from -0.2 to 1.0 with a tick at zero
        tw, ty, th = cw - 22, y + 20, 7
        zx = cx + tw * (0.2 / 1.2)
        d.rectangle([cx, ty, cx + tw, ty + th], fill=line)
        vx = cx + tw * ((bv + 0.2) / 1.2)
        d.rectangle([min(zx, vx), ty, max(zx, vx), ty + th],
                    fill=accent if bv > 0 else warn)
        d.rectangle([zx - 1, ty - 2, zx, ty + th + 2], fill=mute)
        d.text((cx, ty + 15), f"{bv:.3f}".replace("-", "−"),
               font=font(MONO, 17, 1), fill=ink)

    d.text((x0, H - 40), "joehart2001.github.io/boltz-tau", font=font(MONO, 15), fill=accent)

    out = pathlib.Path(args.out) if args.out else \
        ROOT / "docs" / f"cover{'' if args.theme == 'dark' else '-light'}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print(f"{out.relative_to(ROOT)}  {W}x{H}  {out.stat().st_size / 1024:,.0f} KB",
          file=sys.stderr)


if __name__ == "__main__":
    main()
