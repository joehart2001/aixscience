"""All 31 permutations of the five feature blocks, for both model families.

Blocks: seq, boltz, node, energy, edge (see mace_blocks.py).
Models:  MLP   -- the perceptron used throughout, inputs standardised on train
         GBM   -- LightGBM, no standardisation needed

Split is split_B: peptides held out, alleles shared. The companion to the
split_C2 sweep. Sequence is strong here (trees +0.755) because the groove has
been seen, so this says which blocks add anything when sequence already works --
the opposite regime to split_C2, where sequence collapses.

Three seeds per combination rather than five: 31 x 2 x 3 = 186 runs. The best
combinations can be re-run at five seeds afterwards.

Run folders follow the usual layout, named "<family>_<blocks joined by +>".
"""

import itertools
import json
import os
import sys
import time

import numpy as np

import trees
from mace_blocks import have, load_blocks
from train_mlp import MLP, MLP_TEXT
from trainer import FROZEN, RUNS, SPLIT, train
from data_split import load

SEEDS = [0, 1, 2]
BLOCKS = ["seq", "boltz", "node", "energy", "edge"]
ONLY = sys.argv[1:] or None          # e.g. "python run_perms.py gbm" / "mlp"

D = load(SPLIT)
ids = D["df"].sample_id.to_numpy()
keep = np.array([have(s) for s in ids])
print(f"{len(ids):,} rows with Boltz; {keep.sum():,} also have MACE features")

# freeze to the intersection
if os.path.exists(FROZEN):
    frozen = [l.strip() for l in open(FROZEN) if l.strip()]
    keep = np.isin(ids, frozen)
else:
    open(FROZEN, "w").write("\n".join(ids[keep]) + "\n")
    print(f"froze {keep.sum():,} complexes")

import trainer
D = trainer._subset(D, keep)
ids, idx, tr = D["df"].sample_id.to_numpy(), D["idx"], D["tr"]
for k, v in idx.items():
    s = D["df"].iloc[v]
    print(f"  {k:<6} {len(v):>6,}  clusters {s.cluster.nunique():>3}  "
          f"alleles {s.allele.nunique():>3}")

print("reading MACE features ...", flush=True)
M = load_blocks(list(ids))
std, F = D["std"], D["F"]
RAW = {"seq": D["oh"], "boltz": F["pooled"],
       "node": M["node"], "energy": M["energy"], "edge": M["edge"]}
STD = {k: std(v) for k, v in RAW.items()}
print("block dimensions: " + "  ".join(f"{k} {v.shape[1]:,}"
                                       for k, v in RAW.items()), flush=True)

combos = [c for n in range(1, len(BLOCKS) + 1)
          for c in itertools.combinations(BLOCKS, n)]
print(f"{len(combos)} combinations x {len(SEEDS)} seeds x "
      f"{'1' if ONLY else '2'} families\n", flush=True)

def done(name):
    """True when every seed of this run already has a finished record."""
    return all(os.path.exists(os.path.join(RUNS, name, f"seed{s}",
                                           "run_details.json")) for s in SEEDS)


t0 = time.time()
for c in combos:
    tag = "+".join(c)
    if ONLY is None or "mlp" in ONLY:
        X = np.hstack([STD[b] for b in c]).astype(np.float32)
        name = f"mlp_{tag}"
        if done(name):
            print(f"{name:<40} already done, skipping", flush=True)
            continue
        rs = [train(name, f"perceptron on {tag}", (lambda X=X: MLP(X.shape[1])),
                    (X,), D, s, MLP_TEXT, {"blocks": list(c)}) for s in SEEDS]
        r = lambda k: np.mean([x["rho"][k] for x in rs])
        print(f"{name:<40} dim {X.shape[1]:>6,}  train {r('train'):+.3f}  "
              f"val {r('val'):+.3f}  test {r('test'):+.3f}  "
              f"[{time.time()-t0:5.0f}s]", flush=True)
        del X
    if ONLY is None or "gbm" in ONLY:
        X = np.ascontiguousarray(np.hstack([RAW[b] for b in c]),
                                 dtype=np.float32)
        name = f"gbm_{tag}"
        if done(name):
            print(f"{name:<40} already done, skipping", flush=True)
            continue
        rs = [trees.run(name, f"boosted trees on {tag}", X, D, s)
              for s in SEEDS]
        r = lambda k: np.mean([x["rho"][k] for x in rs])
        print(f"{name:<40} dim {X.shape[1]:>6,}  train {r('train'):+.3f}  "
              f"val {r('val'):+.3f}  test {r('test'):+.3f}  "
              f"[{time.time()-t0:5.0f}s]", flush=True)
        del X
print(f"\ndone in {(time.time()-t0)/60:.1f} min")
