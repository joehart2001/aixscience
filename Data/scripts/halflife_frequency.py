"""Frequency table of every distinct half-life value in the dataset.

Writes halflife_frequency.csv (all 944 values, ordered by share) and prints
the head. Values in the source CSV are rounded to 1-2 decimals, which is the
only reason they repeat at all.
"""

import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)                 # Data/
CSV = os.path.join(DATA, "rasmussen_et_al_dataset.csv")
PLOTS = os.path.join(DATA, "plots")
WEIGHTS = os.path.join(DATA, "weights")
SUBSETS = os.path.join(DATA, "subsets")
df = pd.read_csv(CSV)
t = df["thalf_hours"]
n = len(t)

vc = t.value_counts()
tab = pd.DataFrame({
    "thalf_hours": vc.index,
    "count": vc.values,
    "pct": 100.0 * vc.values / n,
})
tab = tab.sort_values(["pct", "thalf_hours"], ascending=[False, True]).reset_index(drop=True)
tab.insert(0, "rank", tab.index + 1)
tab["cum_pct"] = tab["pct"].cumsum()

# Normalised log-frequency: log_f_i = log(f_i) / sum_j log(f_j), f = count / n.
# Every f_i < 1 so every log(f_i) < 0 and the denominator is negative too, which
# makes log_f_i positive and sum to exactly 1 over the 944 distinct values.
# Note this REVERSES the frequency ordering: a rare value has a more negative
# log and therefore a LARGER log_f_i than a common one.
f = tab["count"] / n
logf = np.log(f)
tab["f"] = f
tab["log_f_i"] = logf / logf.sum()

tab["pct"] = tab["pct"].round(4)
tab["cum_pct"] = tab["cum_pct"].round(4)
tab["f"] = tab["f"].round(8)
tab["log_f_i"] = tab["log_f_i"].round(8)

out = os.path.join(WEIGHTS, "halflife_frequency.csv")
tab.to_csv(out, index=False)

print(f"{n:,} rows | {t.nunique()} distinct half-life values\n")
print(f"{'rank':>4}  {'t_half (h)':>10}  {'count':>6}  {'% of data':>9}  {'cum %':>7}  {'log_f_i':>9}")
print("-" * 58)
for r in tab.head(30).itertuples():
    print(f"{r.rank:>4}  {r.thalf_hours:>10.2f}  {r.count:>6,}  {r.pct:>8.2f}%  "
          f"{r.cum_pct:>6.2f}%  {r.log_f_i:>9.6f}")

print("-" * 58)
print(f"sum of log_f_i over all {len(tab)} values : {tab['log_f_i'].sum():.10f}")
print(f"log_f_i range                  : {tab['log_f_i'].min():.6f} (most common) "
      f"-> {tab['log_f_i'].max():.6f} (rarest)")
print(f"values occurring exactly once : {(tab['count'] == 1).sum()}")
print(f"top 10 values cover           : {tab.head(10).pct.sum():.1f}% of the dataset")
print(f"top 30 values cover           : {tab.head(30).pct.sum():.1f}% of the dataset")
print(f"\nfull table ({len(tab)} rows) written to {out}")
