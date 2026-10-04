"""Pack the extracted JSON into the inline constant the pages embed.

    python tools/pack_structure.py src/data/allenihrv.json > src/data/boltz_allenihrv.js
"""

from __future__ import annotations

import json
import sys

ELEMENTS = {"N": 0, "C": 1, "O": 2}

d = json.load(open(sys.argv[1]))
blob = {
    "seq": d["sequence"],
    "pep": [[a["r"], ELEMENTS[a["e"]]] + a["p"] for a in d["peptide"]],
    "tr": [[round(v, 1) for v in p] for p in d["trace"]],
}
print("const BOLTZ=" + json.dumps(blob, separators=(",", ":")) + ";")
