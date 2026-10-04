"""Render the rotating peptide-HLA cartoon to an animated GIF for the README.

A faithful port of the canvas renderer in src/levels.body.js: the same Ca-only
secondary structure assignment, the same painter's algorithm over ribbon quads,
peptide sticks and atoms. Frames are drawn at 3x and downsampled, because PIL
does not antialias polygons, then handed to ffmpeg's palettegen/paletteuse,
which makes a far better GIF palette than a per-frame quantiser.

    python tools/render_protein.py                   # docs/protein.gif
    python tools/render_protein.py --theme light
    python tools/render_protein.py --size 320 --frames 36
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import shutil
import subprocess
import sys
import tempfile

from PIL import Image, ImageDraw

ROOT = pathlib.Path(__file__).resolve().parent.parent
STRUCTURE = ROOT / "src" / "data" / "allenihrv.json"

THEMES = {
    "dark":  dict(bg=(15, 21, 28),    chain="#9b8ff0", peptide="#e4a062"),
    "light": dict(bg=(238, 241, 244), chain="#6558c4", peptide="#c9752f"),
}

SS_HELIX = {2: 5.5, 3: 5.3, 4: 6.2}
SS_STRAND = {2: 6.7, 3: 9.9, 4: 12.4}
SS_TOL = {2: 1.0, 3: 1.1, 4: 1.3}
MIN_RUN = {"H": 6, "E": 3}

BOND_CUTOFF = 1.95     # Angstrom
HELIX_W, STRAND_W, COIL_W = 1.15, 0.85, 0.28   # half-widths, Angstrom
BALL_R, STICK_W = 0.26, 0.30
SMOOTH = {"H": 4, "E": 2, "C": 3}              # helices smooth toward their axis
ARROW_HEAD = 0.68      # fraction of a strand that is body, rest is the head
SS = 3                 # supersample factor


# ---------------------------------------------------------------- geometry

def dist(a, b):
    return math.dist(a, b)


def assign_ss(ca: list[list[float]]) -> str:
    """Score each window against ideal helix and ideal strand, take the better."""
    n = len(ca)

    def fit(i, target):
        total = 0.0
        for k in (2, 3, 4):
            if i + k >= n:
                return 1e9
            total += ((dist(ca[i], ca[i + k]) - target[k]) / SS_TOL[k]) ** 2
        return total / 3

    ss = ["C"] * n
    for i in range(n - 4):
        h, e = fit(i, SS_HELIX), fit(i, SS_STRAND)
        if min(h, e) > 1:
            continue
        kind = "H" if h < e else "E"
        for k in range(i, i + 5):
            if ss[k] == "C":
                ss[k] = kind

    for _ in range(2):                       # drop runs too short to be real
        start = 0
        for i in range(1, n + 1):
            if i == n or ss[i] != ss[start]:
                kind = ss[start]
                if kind != "C" and i - start < MIN_RUN[kind]:
                    for k in range(start, i):
                        ss[k] = "C"
                start = i
    return "".join(ss)


def runs_of(ss: str) -> list[tuple[str, int, int]]:
    out, start = [], 0
    for i in range(1, len(ss) + 1):
        if i == len(ss) or ss[i] != ss[start]:
            out.append((ss[start], max(0, start - 1), min(len(ss) - 1, i)))
            start = i
    return out


def chaikin(pts, iters):
    for _ in range(iters):
        q = [pts[0]]
        for a, b in zip(pts, pts[1:]):
            q.append([a[i] * .75 + b[i] * .25 for i in range(3)])
            q.append([a[i] * .25 + b[i] * .75 for i in range(3)])
        q.append(pts[-1])
        pts = q
    return pts


def project(p, ang, cx, cy, zoom, oy=0.0):
    c, s = math.cos(ang), math.sin(ang)
    x = p[0] * c + p[2] * s
    z = -p[0] * s + p[2] * c
    k = 52.0 / (52.0 + z)                     # coordinates here are Angstrom
    return cx + x * k * zoom, cy + (p[1] - oy) * k * zoom, k, z


def fit(trace, atoms, frames, w, h, margin=0.94):
    """Zoom and vertical offset so the complex fills the frame at every angle."""
    pts = trace + atoms
    oy = (min(p[1] for p in pts) + max(p[1] for p in pts)) / 2
    x0 = y0 = 1e9
    x1 = y1 = -1e9
    for i in range(frames):
        for q in pts:
            px, py, _, _ = project(q, 2 * math.pi * i / frames, 0, 0, 1, oy)
            x0, x1 = min(x0, px), max(x1, px)
            y0, y1 = min(y0, py), max(y1, py)
    pad = HELIX_W * 2.2                       # ribbons stick out past the centreline
    return min(w * margin / (x1 - x0 + pad), h * margin / (y1 - y0 + pad)), oy


def shade(hex_colour: str, t: float) -> tuple[int, int, int]:
    h = hex_colour.lstrip("#")
    rgb = [int(h[i:i + 2], 16) for i in (0, 2, 4)]
    target = 255 if t > 0 else 0
    a = abs(t)
    return tuple(round(c + (target - c) * a) for c in rgb)


# ----------------------------------------------------------------- drawing

def frame(ang, cfg) -> Image.Image:
    w, h = cfg["w"] * SS, cfg["h"] * SS
    img = Image.new("RGB", (w, h), cfg["theme"]["bg"])
    d = ImageDraw.Draw(img)
    cx, cy = w / 2, h / 2
    zoom = cfg["zoom"] * SS
    oy = cfg["oy"]
    chain, pep = cfg["theme"]["chain"], cfg["theme"]["peptide"]
    prims = []

    for kind, a, b in cfg["runs"]:
        raw = cfg["trace"][a:b + 1]
        if len(raw) < 2:
            continue
        sm = chaikin(raw, SMOOTH[kind])
        sp = [project(q, ang, cx, cy, zoom, oy) for q in sm]
        n = len(sp)

        def half_width(u):
            if kind == "H":
                return HELIX_W
            if kind == "E":
                return STRAND_W if u < ARROW_HEAD else \
                    STRAND_W * 2.8 * (1 - (u - ARROW_HEAD) / (1 - ARROW_HEAD))
            return COIL_W

        left, right = [], []
        for i in range(n):
            pa, pc = sp[max(0, i - 1)], sp[min(n - 1, i + 1)]
            dx, dy = pc[0] - pa[0], pc[1] - pa[1]
            m = math.hypot(dx, dy) or 1
            w = half_width(i / (n - 1)) * sp[i][2] * zoom
            left.append((sp[i][0] - dy / m * w, sp[i][1] + dx / m * w))
            right.append((sp[i][0] + dy / m * w, sp[i][1] - dx / m * w))
        for i in range(n - 1):
            prims.append(dict(z=(sp[i][3] + sp[i + 1][3]) / 2, t=kind,
                              quad=[left[i], left[i + 1], right[i + 1], right[i]],
                              k=(sp[i][2] + sp[i + 1][2]) / 2))

    ap = [project(a, ang, cx, cy, zoom, oy) for a in cfg["atoms"]]
    for s in ap:
        prims.append(dict(z=s[3], t="P", xy=(s[0], s[1]), r=BALL_R * s[2] * zoom, k=s[2]))
    for i, j in cfg["bonds"]:
        a, b = ap[i], ap[j]
        prims.append(dict(z=(a[3] + b[3]) / 2, t="B", xy=(a[0], a[1]), xy2=(b[0], b[1]),
                          w=STICK_W * (a[2] + b[2]) / 2 * zoom, k=(a[2] + b[2]) / 2))

    prims.sort(key=lambda o: -o["z"])

    for o in prims:
        lit = max(-.40, min(.40, (o["k"] - .86) * 2.7))
        if o["t"] == "B":
            d.line([o["xy"], o["xy2"]], fill=shade(pep, -.45),
                   width=max(2, round(o["w"] + 1.5 * SS)), joint="curve")
            d.line([o["xy"], o["xy2"]], fill=shade(pep, lit * .9),
                   width=max(1, round(o["w"])), joint="curve")
        elif o["t"] == "P":
            r = max(1.0, o["r"])
            d.ellipse([o["xy"][0] - r, o["xy"][1] - r, o["xy"][0] + r, o["xy"][1] + r],
                      fill=shade(pep, lit * .9), outline=shade(pep, -.45), width=max(1, SS // 2))
        else:
            base = shade(chain, -.12) if o["t"] == "C" else shade(chain, 0)
            face = tuple(round(c + (255 if lit > 0 else 0 - c) * 0) for c in base)
            face = shade("#%02x%02x%02x" % base, lit)
            d.polygon(o["quad"], fill=face)
            edge = shade(chain, -.46)
            w = max(1, round((.8 if o["t"] == "C" else 1.05) * SS))
            d.line([o["quad"][0], o["quad"][1]], fill=edge, width=w)
            d.line([o["quad"][2], o["quad"][3]], fill=edge, width=w)

    return img.resize((cfg["w"], cfg["h"]), Image.LANCZOS)


# -------------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=560)
    ap.add_argument("--height", type=int, default=340)
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--fps", type=int, default=16)
    ap.add_argument("--theme", choices=list(THEMES), default="dark")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = json.loads(STRUCTURE.read_text())
    trace = data["trace"]
    ss = assign_ss(trace)
    runs = runs_of(ss)
    atoms = [a["p"] for a in data["peptide"]]
    bonds = [(i, j)
             for i in range(len(atoms)) for j in range(i + 1, len(atoms))
             if dist(atoms[i], atoms[j]) < BOND_CUTOFF]

    zoom, oy = fit(trace, atoms, args.frames, args.width, args.height)
    cfg = dict(theme=THEMES[args.theme], trace=trace, atoms=atoms, bonds=bonds,
               runs=runs, zoom=zoom, oy=oy, w=args.width, h=args.height)

    print(f"{data['sequence']}  {len(atoms)} atoms, {len(bonds)} bonds, "
          f"{sum(1 for r in runs if r[0]=='H')} helices, "
          f"{sum(1 for r in runs if r[0]=='E')} strands, "
          f"zoom {zoom:.2f} px/A", file=sys.stderr)

    out = pathlib.Path(args.out) if args.out else \
        ROOT / "docs" / f"protein{'' if args.theme=='dark' else '-light'}.gif"
    out.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        for i in range(args.frames):
            frame(2 * math.pi * i / args.frames, cfg).save(tmp / f"f{i:04d}.png")
            print(f"\r  frame {i+1}/{args.frames}", end="", file=sys.stderr)
        print(file=sys.stderr)

        if not shutil.which("ffmpeg"):
            sys.exit("ffmpeg not found")
        pal = tmp / "palette.png"
        run = lambda c: subprocess.run(c, check=True, capture_output=True)
        run(["ffmpeg", "-y", "-i", str(tmp / "f%04d.png"),
             "-vf", "palettegen=max_colors=144:stats_mode=diff", str(pal)])
        run(["ffmpeg", "-y", "-framerate", str(args.fps), "-i", str(tmp / "f%04d.png"),
             "-i", str(pal), "-lavfi", "paletteuse=dither=bayer:bayer_scale=4",
             "-loop", "0", str(out)])

    kb = out.stat().st_size / 1024
    print(f"{out.relative_to(ROOT)}  {args.width}x{args.height}  {args.frames} frames  "
          f"{args.frames/args.fps:.1f}s loop  {kb:,.0f} KB", file=sys.stderr)


if __name__ == "__main__":
    main()
