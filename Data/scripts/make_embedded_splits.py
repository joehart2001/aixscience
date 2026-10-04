"""Add a C2-style cluster split that works on the Boltz-embedded subset.

Why this exists: `all75` has embeddings for 9,031 of the 28,166 rows (54 of 75
alleles). The clusters that `splits.csv` holds out for C2 are almost entirely
alleles with no embedding, so on that subset C2 comes out as 8,945 train / 86
val / **0 test** — unusable. Splits A, B and C survive the restriction fine
(their folds are spread across alleles), but C2 does not.

So we keep the C2 *idea* — hold out whole pseudosequence clusters, so a
near-identical groove can never span folds — and re-run the same stratified
assignment over only the embedded rows. The result is `split_C2e`.

`split_C2e` is NOT the same partition as `split_C2`; it cannot be, since the
cluster pool differs. Compare C2e results against the other all75 runs, not
against the full-dataset C2 numbers.

Output: Data/subsets/embedded_splits.csv — every row of splits.csv (so row_id
joins still work), carrying split_A/B/C verbatim plus split_C2e. Rows without
an embedding are labelled "excluded" in split_C2e, which load_splits drops
automatically since it only keeps train/val/test.

    python make_embedded_splits.py [--embeddings-dir ../boltz2/boltz_embeddings/all75]
"""

import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.dirname(HERE)
SUBSETS = os.path.join(DATA, "subsets")
MAKE_SPLITS = os.path.join(HERE, "make_splits.py")

# Its own seed, so re-running never silently reshuffles the other splits.
SEED_C2E = 20261004


def load_make_splits_helpers() -> dict:
    """Reuse make_splits.py's assignment logic without triggering its writes.

    That module does everything at import time, including overwriting
    splits.csv, so importing it is destructive. Everything we need
    (`assign_groups` and the constants it closes over) is defined above its
    build section, so we execute only that prefix.
    """
    src = open(MAKE_SPLITS).read()
    marker = "# ------------------------------------------------------------------- build"
    if marker not in src:
        raise RuntimeError(
            f"{MAKE_SPLITS} no longer has the expected build-section marker; "
            f"update this script rather than duplicating its logic."
        )
    # The prefix resolves its own paths from __file__, which exec() does not
    # provide; seed it so those point at make_splits.py's real location.
    ns: dict = {"__file__": MAKE_SPLITS, "__name__": "make_splits_prefix"}
    exec(compile(src[: src.index(marker)], MAKE_SPLITS, "exec"), ns)
    return ns


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--embeddings-dir",
        default=os.path.join(DATA, "boltz2", "boltz_embeddings", "all75"),
    )
    p.add_argument("--out", default=os.path.join(SUBSETS, "embedded_splits.csv"))
    args = p.parse_args()

    ns = load_make_splits_helpers()
    assign_groups = ns["assign_groups"]
    embedding_filename = lambda a, p_: (  # noqa: E731 - mirrors data.py's rule
        f"{__import__('re').sub(r'[^a-z0-9]+', '_', a.lower()).strip('_')}_{p_.lower()}.npz"
    )

    splits = pd.read_csv(os.path.join(SUBSETS, "splits.csv"))
    files = set(os.listdir(args.embeddings_dir))
    has_emb = np.array(
        [
            embedding_filename(a, pep) in files
            for a, pep in zip(splits["allele"], splits["peptide"])
        ]
    )
    print(
        f"{has_emb.sum():,} of {len(splits):,} rows have an embedding "
        f"({splits.loc[has_emb, 'allele'].nunique()} of "
        f"{splits['allele'].nunique()} alleles, "
        f"{splits.loc[has_emb, 'cluster'].nunique()} of "
        f"{splits['cluster'].nunique()} clusters)"
    )

    sub = splits[has_emb].copy()
    # assign_groups needs the columns its stratification reads.
    sub["is_zero"] = sub["thalf_hours"] == 0
    sub["split_C2e"] = assign_groups(sub, "cluster", SEED_C2E)

    splits["split_C2e"] = "excluded"
    splits.loc[sub.index, "split_C2e"] = sub["split_C2e"].to_numpy()

    # Same guarantees C2 gives: whole clusters, and whole alleles, per fold.
    assert sub.groupby("cluster")["split_C2e"].nunique().max() == 1, "cluster split"
    assert sub.groupby("allele")["split_C2e"].nunique().max() == 1, "allele split"
    counts = sub["split_C2e"].value_counts()
    for fold in ("train", "val", "test"):
        if counts.get(fold, 0) == 0:
            raise SystemExit(f"split_C2e produced no '{fold}' rows; adjust SEED_C2E")

    print("\n  fold      rows   share   zero%   clusters  alleles")
    for fold in ("train", "val", "test"):
        m = sub["split_C2e"] == fold
        print(
            f"  {fold:<6} {int(m.sum()):>7,} {m.mean():>7.1%} "
            f"{sub.loc[m, 'is_zero'].mean():>7.1%} "
            f"{sub.loc[m, 'cluster'].nunique():>10} "
            f"{sub.loc[m, 'allele'].nunique():>8}"
        )

    splits.to_csv(args.out, index=False)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
