"""Paired bootstrap across every architecture in runs/, on the frozen rows.

Two baselines are reported, because which one you choose changes the headline:
    mlp_seq  the perceptron on sequence one-hot -- what earlier work used
    gbm_seq  gradient-boosted trees on the same inputs -- the strong baseline

Predictions are the 5-seed ensemble (seeds averaged, then correlated), which is
what the parity grid plots. 2,000 resamples of complexes, paired.
"""

import glob
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
NBOOT = 2000

pred, meta = {}, {}
for d in sorted(glob.glob(os.path.join(RUNS, "*", "seed*"))):
    j = os.path.join(d, "run_details.json")
    f = os.path.join(d, "predictions.npz")
    if not (os.path.exists(j) and os.path.exists(f)):
        continue
    det = json.load(open(j))
    n = det["representation"]
    z = np.load(f)
    pred.setdefault(n, []).append(z["pred_test"])
    meta.setdefault(n, {"dim": det["input_dim"], "params": det["n_parameters"],
                        "family": det.get("model_family", "neural network"),
                        "rho": []})
    meta[n]["rho"].append(det["spearman"])
    y = z["true_test"]

ens = {n: np.mean(v, 0) for n, v in pred.items()}
rng = np.random.default_rng(0)
boot = rng.integers(0, len(y), size=(NBOOT, len(y)))
rho_b = {n: np.array([spearmanr(p[b], y[b]).statistic for b in boot])
         for n, p in ens.items()}

rows = []
for n in ens:
    m = meta[n]
    g = lambda k: np.mean([r[k] for r in m["rho"]])
    row = dict(rep=n, family=m["family"], dim=m["dim"], params=m["params"],
               rho_train=g("train"), rho_val=g("val"), rho_test=g("test"),
               rho_test_sd=np.std([r["test"] for r in m["rho"]]),
               rho_test_ens=float(spearmanr(ens[n], y).statistic))
    for base in ("mlp_seq", "gbm_seq"):
        if base in rho_b:
            d = rho_b[n] - rho_b[base]
            lo, hi = np.percentile(d, [2.5, 97.5])
            row[f"d_{base}"] = d.mean()
            row[f"ci_{base}"] = f"[{lo:+.3f}, {hi:+.3f}]"
            row[f"beats_{base}"] = "YES" if lo > 0 else ("WORSE" if hi < 0 else "ns")
    rows.append(row)

res = pd.DataFrame(rows).sort_values("rho_test", ascending=False)
res.to_csv(os.path.join(RUNS, "_summary", "architecture_comparison.csv"), index=False)

w = max(len(n) for n in ens)
print(f"{'representation':<{w}} {'family':<24} {'dim':>5} {'params':>9} "
      f"{'train':>7} {'val':>7} {'test':>7}  {'vs mlp_seq':>20} "
      f"{'vs gbm_seq':>20}")
for r in res.itertuples():
    print(f"{r.rep:<{w}} {r.family:<24} {r.dim:>5} {r.params:>9,} "
          f"{r.rho_train:>+7.3f} {r.rho_val:>+7.3f} {r.rho_test:>+7.3f}  "
          f"{r.d_mlp_seq:>+6.3f} {r.ci_mlp_seq:>17} "
          f"{r.d_gbm_seq:>+6.3f} {r.ci_gbm_seq:>17}")
