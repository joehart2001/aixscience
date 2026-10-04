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


# Two shapes of the same pages.
#
# dist/        publishable fragments. The artifact platform wraps the page in
#              its own <!doctype>/<head> at publish time, so these must NOT
#              carry one. pipeline rides along as a sibling file; sibling files
#              are served raw, so that one does need a skeleton, and its link
#              back to levels has to point at the artifact root.
# dist/local/  complete documents for opening from disk or a plain file server,
#              which supply no charset of their own. Without the meta the
#              browser reads UTF-8 as Latin-1 and every degree sign, middle dot
#              and Greek letter turns to mojibake.
STANDALONE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<style>:root{color-scheme:light}body{margin:0}img{max-width:100%}[hidden]{display:none!important}</style>
</head><body>
__PAGE__
</body></html>
"""


def build_pipeline() -> None:
    page = (SRC / "pipeline.html").read_text()
    standalone = STANDALONE.replace("__PAGE__", page.replace('href="levels.html"', 'href="./"'))
    (DIST / "pipeline.standalone.html").write_text(standalone)
    print(f"  dist/pipeline.standalone.html  "
          f"{(DIST / 'pipeline.standalone.html').stat().st_size:>7,} bytes  (publishes beside levels)")


def build_local() -> None:
    """Browsable copies, and exactly what GitHub Pages serves.

    levels becomes index.html so Pages has a root document, which means
    pipeline's link back has to point at the directory rather than a filename.
    """
    local = DIST / "local"
    local.mkdir(exist_ok=True)
    pages = {
        "index.html": (DIST / "levels.html").read_text()
                      .replace('href="levels.html"', 'href="./"'),
        "pipeline.html": (SRC / "pipeline.html").read_text()
                         .replace('href="levels.html"', 'href="./"'),
    }
    for name, body in pages.items():
        out = local / name
        out.write_text(STANDALONE.replace("__PAGE__", body))
        print(f"  dist/local/{name:<14} {out.stat().st_size:>7,} bytes")


def main() -> None:
    DIST.mkdir(exist_ok=True)
    build_levels()
    build_pipeline()
    build_local()


if __name__ == "__main__":
    main()
