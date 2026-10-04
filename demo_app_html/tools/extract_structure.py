"""Pull a Boltz-2 prediction out of a ModelCIF into the compact JSON the demos use.

Reads one `*_model_0.cif`, keeps the peptide chain's heavy atoms and the
heavy chain's first 180 Calphas, and rewrites every coordinate into a frame
built from the structure itself:

  x  along the peptide N -> C
  y  from the peptide centroid toward the platform centroid (the sheet floor)
  z  their cross product

so the groove lies horizontally at any rotation with no hand-tuned camera.

    python tools/extract_structure.py <path to .cif> > src/data/allenihrv.json
"""

from __future__ import annotations

import json
import math
import sys

AA3TO1 = dict(
    ALA="A", ARG="R", ASN="N", ASP="D", CYS="C", GLN="Q", GLU="E", GLY="G",
    HIS="H", ILE="I", LEU="L", LYS="K", MET="M", PHE="F", PRO="P", SER="S",
    THR="T", TRP="W", TYR="Y", VAL="V",
)
PLATFORM_RESIDUES = 180  # the alpha1/alpha2 domain


def read_atoms(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            if line.startswith(("ATOM", "HETATM")):
                p = line.split()
                rows.append(
                    dict(el=p[2], atom=p[3], comp=p[5], seq=int(p[7]), asym=p[9],
                         x=float(p[10]), y=float(p[11]), z=float(p[12]), b=float(p[17]))
                )
    return rows


def vec(r):
    return [r["x"], r["y"], r["z"]]


def sub(a, b):
    return [a[i] - b[i] for i in range(3)]


def dot(a, b):
    return sum(a[i] * b[i] for i in range(3))


def cross(a, b):
    return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]


def unit(a):
    m = math.sqrt(dot(a, a))
    return [q / m for q in a]


def centroid(pts):
    n = len(pts)
    return [sum(p[i] for p in pts) / n for i in range(3)]


def main(path: str) -> None:
    rows = read_atoms(path)
    chains: dict[str, list[dict]] = {}
    for r in rows:
        chains.setdefault(r["asym"], []).append(r)
    sizes = {k: max(x["seq"] for x in v) for k, v in chains.items()}

    # the peptide is the only 9-residue chain; the heavy chain is the longest
    peptide = chains[next(k for k, n in sizes.items() if n == 9)]
    platform = [r for r in chains[max(sizes, key=sizes.get)]
                if r["atom"] == "CA" and r["seq"] <= PLATFORM_RESIDUES]

    pep_ca = [vec(r) for r in peptide if r["atom"] == "CA"]
    origin = centroid(pep_ca)

    ex = unit(sub(pep_ca[-1], pep_ca[0]))
    down = sub(centroid([vec(r) for r in platform]), origin)
    ey = unit(sub(down, [ex[i] * dot(down, ex) for i in range(3)]))
    ez = cross(ex, ey)

    def transform(r):
        d = sub(vec(r), origin)
        return [round(dot(d, ex), 2), round(dot(d, ey), 2), round(dot(d, ez), 2)]

    seqs = sorted({r["seq"] for r in peptide})
    sequence = "".join(
        AA3TO1[next(r for r in peptide if r["seq"] == s)["comp"]] for s in seqs
    )

    out = dict(
        sequence=sequence,
        peptide=[dict(e=r["el"], r=seqs.index(r["seq"]), p=transform(r)) for r in peptide],
        trace=[transform(r) for r in platform],
        plddt_peptide=round(sum(r["b"] for r in peptide) / len(peptide), 1),
    )
    print(f"{sequence}: {len(out['peptide'])} heavy atoms, "
          f"{len(out['trace'])} Calpha, pLDDT {out['plddt_peptide']}", file=sys.stderr)
    print(json.dumps(out, separators=(",", ":")))


if __name__ == "__main__":
    main(sys.argv[1])
