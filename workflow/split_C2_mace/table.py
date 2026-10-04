"""Turn the permutation sweep into one table, read from the run records.

Reads runs/<family>_<blocks>/seed*/run_details.json rather than parsing the
log, so the numbers carry a per-seed spread and nothing depends on log
formatting. Both families are shown side by side: the sweep ran gradient
boosted trees and the perceptron over the same 31 block combinations, the same
frozen 14,997 complexes and the same split_C2 folds.

Writes permutation_table.csv and permutation_table.txt.
"""

import glob
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
BLOCKS = ["seq", "boltz", "node", "energy", "edge"]


def collect():
    out = {}
    for d in sorted(glob.glob(os.path.join(RUNS, "*", "seed*"))):
        j = os.path.join(d, "run_details.json")
        if not os.path.exists(j):
            continue
        det = json.load(open(j))
        name = det["representation"]
        fam, tag = name.split("_", 1)
        out.setdefault((tag, fam), {"dim": det["input_dim"], "rho": [],
                                    "rounds": []})
        out[(tag, fam)]["rho"].append(det["spearman"])
        out[(tag, fam)]["rounds"].append(det["epochs_run"])
    return out


data = collect()
tags = sorted({t for t, _ in data},
              key=lambda t: (len(t.split("+")), t))
rows = []
for t in tags:
    r = {"blocks": t, "n_blocks": len(t.split("+"))}
    for fam in ("gbm", "mlp"):
        k = (t, fam)
        if k not in data:
            continue
        g = lambda f: np.array([s[f] for s in data[k]["rho"]])
        r["dim"] = data[k]["dim"]
        r[f"{fam}_train"] = g("train").mean()
        r[f"{fam}_val"] = g("val").mean()
        r[f"{fam}_test"] = g("test").mean()
        r[f"{fam}_test_sd"] = g("test").std()
        r[f"{fam}_seeds"] = len(data[k]["rho"])
    rows.append(r)

df = pd.DataFrame(rows)
df["has_boltz"] = df.blocks.str.contains("boltz")
df = df.sort_values("gbm_test", ascending=False)
df.to_csv(os.path.join(HERE, "permutation_table.csv"), index=False)

L = []
L.append("# split_C2 permutation sweep: all combinations of five feature blocks")
L.append("#")
L.append("# seq    peptide one-hot + NetMHCpan pseudosequence one-hot      (860)")
L.append("# boltz  Boltz-2 pooled trunk blocks                            (1547)")
L.append("# node   MACE descriptors on the peptide, 9 residues x 5 slots (11520)")
L.append("# energy MACE per-residue site energies, peptide and groove      (430)")
L.append("# edge   MACE peptide<->groove cross-chain edge features       (14450)")
L.append("#")
L.append("# 14,997 complexes, 75 alleles, 22 clusters. split_C2 holds out whole")
L.append("# pseudosequence clusters, so no allele in the test fold is seen in")
L.append("# training. Spearman, mean over seeds. Both families use the same rows.")
L.append("#")
L.append(f"{'blocks':<28} {'dim':>6}  {'GBM train':>9} {'val':>7} {'test':>15}"
         f"   {'MLP train':>9} {'val':>7} {'test':>15}")
for r in df.itertuples():
    def cell(fam):
        t = getattr(r, f"{fam}_test", np.nan)
        if not np.isfinite(t):
            return f"{'-':>9} {'-':>7} {'-':>15}"
        return (f"{getattr(r, f'{fam}_train'):>+9.3f} "
                f"{getattr(r, f'{fam}_val'):>+7.3f} "
                f"{t:>+8.3f} +/-{getattr(r, f'{fam}_test_sd'):5.3f}")
    L.append(f"{r.blocks:<28} {int(r.dim):>6}  {cell('gbm')}   {cell('mlp')}")

L.append("")
L.append("# summary: does the combination contain the Boltz block?")
for has, g in df.groupby("has_boltz"):
    L.append(f"#   {'with' if has else 'without':<8} boltz  n={len(g):>2}  "
             f"GBM test {g.gbm_test.min():+.3f} to {g.gbm_test.max():+.3f}"
             f"   median {g.gbm_test.median():+.3f}")
txt = "\n".join(L)
open(os.path.join(HERE, "permutation_table.txt"), "w").write(txt + "\n")
print(txt)
