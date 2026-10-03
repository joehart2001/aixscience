"""Build the train/val/test splits described in planning-data-split.md.

Four splits, all 70/15/15, each with its own fixed seed:

  A   random rows
  B   grouped by peptide        - no peptide crosses splits
  C   grouped by allele         - no allele crosses splits
  C2  grouped by allele cluster - alleles sharing >= IDENTITY_MIN of their 34
                                  contact positions are one unit, so
                                  near-identical grooves cannot be separated

All four are stratified on the two things that would otherwise skew a fold:
whether the measurement is a censored zero, and the magnitude of the half-life.

Outputs
  splits.csv          one row per dataset row, in the original order, with a
                      train/val/test label for each of the four splits
  split_summary.csv   per split x fold: rows, share, zero fraction, median
  pseudoseq_clusters.csv is NOT rewritten here; see plot_halflife_by_cluster.py
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

SEEDS = {"A": 3102026, "B": 1032026, "C": 2026103, "C2": 20260310}
TARGET = {"train": 0.70, "val": 0.15, "test": 0.15}
IDENTITY_MIN = 30          # of 34 contact positions, for the C2 clustering
N_STRATA = 4

df = pd.read_csv(CSV)
df["row_id"] = np.arange(len(df))
N = len(df)


# --------------------------------------------------------------- clustering
def allele_clusters(frame, identity_min):
    """Single-linkage clusters of alleles on pseudosequence identity."""
    al = frame.drop_duplicates("allele")[["allele", "hla_pseudoseq"]].reset_index(drop=True)
    P = np.frombuffer("".join(al.hla_pseudoseq).encode(), dtype="S1").reshape(len(al), 34)
    sim = (P[:, None, :] == P[None, :, :]).sum(-1)

    parent = list(range(len(al)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(len(al)):
        for j in range(i + 1, len(al)):
            if sim[i, j] >= identity_min:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)

    al["root"] = [find(i) for i in range(len(al))]
    codes = {r: k for k, r in enumerate(sorted(al.root.unique()))}
    al["cluster"] = al.root.map(codes)
    return dict(zip(al.allele, al.cluster))


df["cluster"] = df.allele.map(allele_clusters(df, IDENTITY_MIN))
df["is_zero"] = df.thalf_hours == 0


# ------------------------------------------------------------- stratum keys
def quantile_bin(values, n_bins):
    """Rank-based binning that tolerates ties and tiny inputs."""
    r = pd.Series(values).rank(method="first")
    n = min(n_bins, max(1, len(set(values))))
    return pd.qcut(r, n, labels=False, duplicates="drop").to_numpy()


def assign_rows(strata, seed):
    """Stratified split of individual rows (used for split A)."""
    rng = np.random.default_rng(seed)
    out = np.empty(len(strata), dtype=object)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        rng.shuffle(idx)
        n_tr = int(round(TARGET["train"] * len(idx)))
        n_va = int(round(TARGET["val"] * len(idx)))
        out[idx[:n_tr]] = "train"
        out[idx[n_tr:n_tr + n_va]] = "val"
        out[idx[n_tr + n_va:]] = "test"
    return out


N_RESTARTS = 4000


def _score(g, chosen, global_zero):
    """Lower is better: row shares off target, plus zero-fraction imbalance."""
    cost = 0.0
    for f, target in TARGET.items():
        members = [k for k, v in chosen.items() if v == f]
        n = g.n[members].sum()
        if n == 0:
            return np.inf
        share = n / g.n.sum()
        zero = (g.zero[members] * g.n[members]).sum() / n
        cost += 12.0 * abs(share - target) + abs(zero - global_zero)
    return cost


def assign_groups(frame, key, seed):
    """Stratified, size-aware assignment of whole groups to folds.

    Greedy largest-first within strata gets the row shares roughly right, but
    with very uneven group sizes (C2 clusters span 7 to 3,701 rows) a single
    greedy pass can leave a fold both undersized and skewed. So we run many
    seeded greedy passes over shuffled stratum/group orders and keep the
    assignment that best matches the target shares and the global zero
    fraction. Deterministic for a given seed.
    """
    rng = np.random.default_rng(seed)
    g = frame.groupby(key).agg(
        n=("thalf_hours", "size"),
        zero=("is_zero", "mean"),
        med=("thalf_hours", "median"),
    )
    n_strata = max(1, min(N_STRATA, len(g) // 3))
    g["stratum"] = (quantile_bin(g.zero.to_numpy(), n_strata) * n_strata
                    + quantile_bin(g.med.to_numpy(), n_strata))
    global_zero = (g.zero * g.n).sum() / g.n.sum()

    # Many groups make the search unnecessary - one greedy pass is already exact.
    restarts = 1 if len(g) > 200 else N_RESTARTS

    best, best_cost = None, np.inf
    for _ in range(restarts):
        got = {f: 0 for f in TARGET}
        chosen = {}
        for s in rng.permutation(g.stratum.unique()):
            sub = g[g.stratum == s]
            order = sub.assign(j=rng.random(len(sub))).sort_values(
                ["n", "j"], ascending=[False, True]).index
            for grp in order:
                deficit = {f: TARGET[f] * (sum(got.values()) + g.n[grp]) - got[f]
                           for f in TARGET}
                fold = max(deficit, key=deficit.get)
                chosen[grp] = fold
                got[fold] += g.n[grp]
        cost = _score(g, chosen, global_zero)
        if cost < best_cost:
            best, best_cost = dict(chosen), cost
    return frame[key].map(best).to_numpy()


# ------------------------------------------------------------------- build
row_stratum = (df.is_zero.astype(int).to_numpy() * (N_STRATA + 1)
               + quantile_bin(df.thalf_hours.to_numpy(), N_STRATA))

df["split_A"] = assign_rows(row_stratum, SEEDS["A"])
df["split_B"] = assign_groups(df, "peptide", SEEDS["B"])
df["split_C"] = assign_groups(df, "allele", SEEDS["C"])
df["split_C2"] = assign_groups(df, "cluster", SEEDS["C2"])

SPLITS = ["A", "B", "C", "C2"]
GROUP_KEY = {"A": None, "B": "peptide", "C": "allele", "C2": "cluster"}


# ------------------------------------------------------------ verification
print("leakage checks")
ok = True
for s in SPLITS:
    key = GROUP_KEY[s]
    if key is None:
        print(f"  {s:>2}: random rows, no grouping constraint")
        continue
    crossing = df.groupby(key)[f"split_{s}"].nunique()
    bad = int((crossing > 1).sum())
    ok &= bad == 0
    print(f"  {s:>2}: {crossing.size:>5} {key:<8} groups, {bad} crossing a split "
          f"{'OK' if bad == 0 else 'FAIL'}")

# C2 must also keep whole alleles together, and must separate the clusters.
assert df.groupby("allele").split_C2.nunique().max() == 1, "C2 split an allele"
# Clusters that are near-identical must not appear in two folds (by construction).
print(f"  C2: alleles never split across folds OK")
assert ok, "group leakage detected"

rows = []
print("\nfold composition")
for s in SPLITS:
    col = f"split_{s}"
    print(f"\n  split {s}"
          + (f"  (grouped by {GROUP_KEY[s]})" if GROUP_KEY[s] else "  (random rows)"))
    print(f"    {'fold':<6} {'rows':>7} {'share':>7} {'zero%':>7} {'median h':>9} "
          f"{'groups':>7} {'alleles':>8}")
    for fold in ["train", "val", "test"]:
        m = df[col] == fold
        ngrp = df.loc[m, GROUP_KEY[s]].nunique() if GROUP_KEY[s] else m.sum()
        print(f"    {fold:<6} {m.sum():>7,} {100*m.mean():>6.1f}% "
              f"{100*df.loc[m, 'is_zero'].mean():>6.1f}% "
              f"{df.loc[m, 'thalf_hours'].median():>9.2f} {ngrp:>7,} "
              f"{df.loc[m, 'allele'].nunique():>8}")
        rows.append({"split": s, "fold": fold, "rows": int(m.sum()),
                     "share_pct": round(100 * m.mean(), 2),
                     "zero_pct": round(100 * df.loc[m, "is_zero"].mean(), 2),
                     "median_thalf": round(df.loc[m, "thalf_hours"].median(), 3),
                     "n_groups": int(ngrp),
                     "n_alleles": int(df.loc[m, "allele"].nunique()),
                     "n_peptides": int(df.loc[m, "peptide"].nunique())})

# Which alleles / clusters are held out - needed when reporting C and C2.
print("\nheld-out alleles (C) and clusters (C2)")
for s in ["C", "C2"]:
    key = GROUP_KEY[s]
    for fold in ["val", "test"]:
        held = sorted(df.loc[df[f"split_{s}"] == fold, key].unique())
        shown = ", ".join(str(h).replace("HLA-", "") for h in held[:10])
        print(f"  {s:>2} {fold:<5}: {len(held)} {key}s -> {shown}"
              + (" ..." if len(held) > 10 else ""))

cols = ["row_id", "allele", "peptide", "thalf_hours", "cluster"] + \
       [f"split_{s}" for s in SPLITS]
df[cols].to_csv(os.path.join(SUBSETS, "splits.csv"), index=False)
pd.DataFrame(rows).to_csv(os.path.join(SUBSETS, "split_summary.csv"), index=False)
print("\nwrote splits.csv and split_summary.csv")
