"""Assemble dist/*.html from src/.

levels.html is head + inlined structure data + body, in that order, because
the body refers to ATOMS and TRACE at module scope. pipeline.html is already
self-contained and is copied through.

    python build.py
"""

from __future__ import annotations

import pathlib
import shutil

ROOT = pathlib.Path(__file__).parent
SRC, DIST = ROOT / "src", ROOT / "dist"

GEOMETRY = """
const SC=10;
const ATOMS=BOLTZ.pep.map(([ri,el,x,y,z])=>({ri,el,p:{x:x*SC,y:y*SC,z:z*SC},r:el===1?2.4:el===0?2.2:2.1}));
const TRACE=BOLTZ.tr.map(([x,y,z])=>({x:x*SC,y:y*SC,z:z*SC}));
"""


def build_levels() -> None:
    head = (SRC / "levels.head.html").read_text()
    body = (SRC / "levels.body.js").read_text()
    data = (SRC / "data" / "boltz_allenihrv.js").read_text()

    # the structure constant has to land inside the IIFE, before first use
    marker = "(function(){\n"
    assert body.startswith(marker), "levels.body.js must open with an IIFE"
    body = body.replace(marker, marker + data + GEOMETRY, 1)

    out = DIST / "levels.html"
    out.write_text(head + "\n<script>\n" + body + "</script>\n")
    print(f"  dist/levels.html    {out.stat().st_size:>7,} bytes")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    build_levels()
    shutil.copy(SRC / "pipeline.html", DIST / "pipeline.html")
    print(f"  dist/pipeline.html  {(DIST / 'pipeline.html').stat().st_size:>7,} bytes")


if __name__ == "__main__":
    main()
