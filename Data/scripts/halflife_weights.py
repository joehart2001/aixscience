"""Sample weights that downweight abundant half-life values.

Family: w_i ∝ (1 / f_bin(i)) ** alpha, normalised so the mean weight over rows
is exactly 1 (so the loss scale is unchanged) and clipped to bound the tail.

  alpha = 0.0   no reweighting (uniform)
  alpha = 0.5   square-root inverse frequency  <- recommended default
  alpha = 1.0   full inverse frequency (aggressive)

Binning rather than exact value: thalf_hours is a continuous measurement
rounded to 1-2 decimals, so exact-value classes are an artefact of rounding
(0.1 and 0.11 would be separate classes, and the 376 singleton values would
each draw the maximum weight despite being single noisy measurements).
Zeros get their own bin because they are censored non-binders, not a rounded
measurement.

Writes halflife_weights.csv: one row per dataset row, with the bin, the bin
frequency, and the weight at each alpha.
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
N_BINS = 20            # log-spaced bins over the non-zero values
CLIP = (0.1, 10.0)     # bound the weight range after normalisation

# w ∝ (1/f) ** alpha, so alpha < 0 gives w ∝ f ** |alpha| (upweights abundance).
SCHEMES = {
    "w_sqrt_f":      -0.5,   # w ∝ sqrt(f)      - upweights abundant (opposite effect)
    "w_uniform":      0.0,   # no reweighting
    "w_inv_sqrt_f":   0.5,   # w ∝ 1/sqrt(f)    - RECOMMENDED
    "w_inv_f":        1.0,   # w ∝ 1/f          - flattens bins entirely
}
ALPHAS = list(SCHEMES.values())

df = pd.read_csv(CSV)
t = df["thalf_hours"].to_numpy()
n = len(t)

# ---- bin: zeros in bin 0, non-zeros in log-spaced bins 1..N_BINS ----------
nz = t > 0
edges = np.logspace(np.log10(t[nz].min()), np.log10(t[nz].max()), N_BINS + 1)
edges[-1] *= 1.000001                       # make the top edge inclusive
bin_idx = np.zeros(n, dtype=int)
bin_idx[nz] = np.digitize(t[nz], edges)     # 1..N_BINS

# Merge undersized bins into their neighbour. thalf_hours is rounded to 0.1,
# so the lowest log bins are nearly empty (one bin held a single row) and
# would otherwise draw the maximum weight off one noisy measurement.
MIN_COUNT = 100
while True:
    c = np.bincount(bin_idx, minlength=N_BINS + 1)
    live = [b for b in range(1, N_BINS + 1) if c[b] > 0]
    small = [b for b in live if c[b] < MIN_COUNT]
    if not small:
        break
    b = small[0]
    nbrs = [x for x in live if x != b]
    tgt = min(nbrs, key=lambda x: (abs(x - b), c[x]))   # nearest, then smallest
    bin_idx[bin_idx == b] = tgt
    edges[min(b, tgt)] = edges[min(b, tgt)]             # edges only label output

counts = np.bincount(bin_idx, minlength=N_BINS + 1)
f_bin = counts / n                          # fraction of the dataset per bin

# ---- weights -------------------------------------------------------------
out = pd.DataFrame({
    "thalf_hours": t,
    "bin": bin_idx,
    "bin_count": counts[bin_idx],
    "bin_frac": f_bin[bin_idx],
})

for name, a in SCHEMES.items():
    with np.errstate(divide="ignore"):
        w_bin = np.where(counts > 0, (1.0 / np.maximum(f_bin, 1e-12)) ** a, 0.0)
    w = w_bin[bin_idx]
    w = w / w.mean()                        # mean weight == 1 over rows
    w = np.clip(w, *CLIP)
    w = w / w.mean()                        # renormalise after clipping
    out[name] = w

# Class-level sampled share each scheme implies, for the summary table only.
sampled_share = {
    name: (lambda s: s / s.sum())(np.array(
        [out.loc[out.bin == b, name].sum() for b in range(N_BINS + 1)]))
    for name in SCHEMES
}

out.to_csv(os.path.join(WEIGHTS, "halflife_weights.csv"), index=False)

# ---- report --------------------------------------------------------------
# Label each surviving bin by the actual data range it covers (bins may have
# been merged, so the original edges no longer describe them).
lbl = {0: "0 (censored)"}
for b in range(1, N_BINS + 1):
    m = bin_idx == b
    if m.any():
        lbl[b] = f"{t[m].min():.2f}-{t[m].max():.2f}"
print(f"{n:,} rows | zeros kept as their own bin | {N_BINS} log-spaced bins above 0\n")
hdr = f"{'bin':>3}  {'t_half range (h)':>18}  {'count':>6}  {'% data':>7}"
hdr += "".join(f"  {name:>13}" for name in SCHEMES)
print(hdr); print("-" * len(hdr))

rows = []
for b in range(N_BINS + 1):
    if counts[b] == 0:
        continue
    row = f"{b:>3}  {lbl[b]:>18}  {counts[b]:>6,}  {100*f_bin[b]:>6.2f}%"
    rec = {"bin": b, "t_half_range": lbl[b], "count": counts[b],
           "pct_of_data": round(100 * f_bin[b], 4)}
    for name in SCHEMES:
        v = out.loc[out.bin == b, name].iloc[0]
        row += f"  {v:>13.3f}"
        rec[name] = round(v, 6)
        rec[f"{name}_sampled_share_pct"] = round(100 * sampled_share[name][b], 4)
    rows.append(rec)
    print(row)
print("-" * len(hdr))

summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(WEIGHTS, "halflife_weight_schemes.csv"), index=False)

for name, a in SCHEMES.items():
    w = out[name]
    print(f"{name:>13} (alpha={a:>4}): mean {w.mean():.4f}  "
          f"min {w.min():.3f}  max {w.max():.3f}  "
          f"max/min {w.max()/w.min():>5.1f}x  "
          f"zeros hold {100*w[out.bin == 0].sum()/w.sum():>5.1f}% of total weight "
          f"(unweighted {100*f_bin[0]:.1f}%)")
print("\nwrote halflife_weights.csv (per row) and "
      "halflife_weight_schemes.csv (per bin)")
