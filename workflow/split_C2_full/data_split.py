"""Loader for the early-sync split_B run over whatever Boltz embeddings exist.

split_B holds out PEPTIDES: a 9-mer in the test fold appears nowhere in train,
under any allele. That removes the near-duplicate leakage that makes split_A an
optimistic ceiling, so these numbers are a real generalisation estimate for the
question "a new peptide against an allele we have seen". It does NOT test a new
allele -- every receptor in test is also in train. That is split_C2's job, and
those rows are still being predicted.

Baseline is peptide one-hot PLUS NetMHCpan pseudosequence one-hot: across 75
alleles the receptor varies, so a peptide-only baseline would be a straw man.
"""

import os

import numpy as np
import pandas as pd

MAN = "/share/ijp30/hackathons/aixscience/claude-experiments/006-modal-boltz/subset_manifest.csv"
CSV = "/share/ijp30/hackathons/aixscience/Data/rasmussen_et_al_dataset.csv"
SPLITS = "/share/ijp30/hackathons/aixscience/Data/subsets/splits.csv"
EMB_NEW = "/share/ijp30/hackathons/aixscience2026/boltz_inputs_and_predictions/embeddings/all75"
EMB_OLD = "/share/jh2536/hackathons/aixscience2026/boltz_inputs_and_predictions/embeddings/a0201"
AA = "ACDEFGHIKLMNPQRSTVWY"


def onehot(seqs, n):
    out = np.zeros((len(seqs), n * 20), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:n]):
            if c in AA:
                out[i, j * 20 + AA.index(c)] = 1
    return out


def load(split_col="split_C2"):
    man = pd.read_csv(MAN)
    src = pd.read_csv(CSV)
    sp = pd.read_csv(SPLITS)

    key = list(zip(sp.allele, sp.peptide))
    split_of = dict(zip(key, sp[split_col]))
    pseudo = dict(zip(zip(src.allele, src.peptide), src.hla_pseudoseq))

    rows, F = [], {k: [] for k in ("pooled", "tok", "pmean", "recep", "conf")}
    n_miss = n_bad = 0
    for r in man.itertuples():
        for d in (EMB_NEW, EMB_OLD):
            f = os.path.join(d, f"{r.sample_id}.npz")
            if os.path.exists(f):
                break
        else:
            n_miss += 1
            continue
        try:
            z = np.load(f, allow_pickle=True)
            m = z["peptide_token_mask"]
            assert m.sum() == 9, f"{r.sample_id}: {m.sum()} peptide tokens"
            F["tok"].append(z["s"][m].ravel())
            F["pooled"].append(z["features"])
            F["pmean"].append(z["peptide_s_mean"])
            F["recep"].append(z["contact_receptor_s_mean"])
            F["conf"].append(z["confidence"])
        except Exception:                 # mid-write or truncated
            n_bad += 1
            continue
        rows.append(r)

    df = pd.DataFrame(rows).reset_index(drop=True)
    F = {k: np.asarray(v, dtype=np.float32) for k, v in F.items()}
    df[split_col] = [split_of[(a, p)] for a, p in zip(df.allele, df.peptide)]

    y = df.thalf_hours.to_numpy(dtype=np.float64)
    ly = np.log10(np.where(y > 0, y, y[y > 0].min() / 2)).astype(np.float32)

    ps = [pseudo[(a, p)] for a, p in zip(df.allele, df.peptide)]
    oh_pep, oh_hla = onehot(df.peptide.tolist(), 9), onehot(ps, 34)

    where = df[split_col].to_numpy()
    idx = {k: np.flatnonzero(where == k) for k in ("train", "val", "test")}
    tr = idx["train"]
    ym, ys = ly[tr].mean(), ly[tr].std()

    def std(A):
        return ((A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)).astype(np.float32)

    import charge_struct as _cs
    zs = np.load(_cs.OUT, allow_pickle=True)
    cmap = dict(zip(zs["names"], zs["X"]))
    miss = [i for i, n in enumerate(df.sample_id) if n not in cmap]
    if miss:
        keep = np.array([n in cmap for n in df.sample_id])
        print(f"  dropping {len(miss)} rows without structure charges")
        df = df[keep].reset_index(drop=True)
        F = {k: v[keep] for k, v in F.items()}
        oh_pep, oh_hla = oh_pep[keep], oh_hla[keep]
        y, ly, ps = y[keep], ly[keep], [q for q, m in zip(ps, keep) if m]
        where = df[split_col].to_numpy()
        idx = {k: np.flatnonzero(where == k) for k in ("train", "val", "test")}
        tr = idx["train"]
        ym, ys = ly[tr].mean(), ly[tr].std()

        def std(A):
            return ((A - A[tr].mean(0)) / (A[tr].std(0) + 1e-6)).astype(np.float32)

    F["charge_struct"] = np.asarray([cmap[n] for n in df.sample_id],
                                    dtype=np.float32)
    import charges as _ch
    F["charge_seq"] = _ch.sequence_charge_block(df.peptide.tolist(), ps)

    return dict(df=df, F=F, oh_pep=oh_pep, oh_hla=oh_hla, pseudoseq=ps,
                oh=np.hstack([oh_pep, oh_hla]), std=std, y=y, ly=ly,
                yz=((ly - ym) / ys).astype(np.float32), ym=ym, ys=ys,
                idx=idx, tr=tr, n_missing=n_miss, n_bad=n_bad)


if __name__ == "__main__":
    D = load()
    df, idx = D["df"], D["idx"]
    print(f"loaded {len(df):,}   not yet predicted {D['n_missing']:,}   "
          f"unreadable {D['n_bad']:,}")
    print(f"alleles {df.allele.nunique()}  clusters {df.cluster.nunique()}")
    for k, v in idx.items():
        s = df.iloc[v]
        print(f"  {k:<6} {len(v):>6,}  alleles {s.allele.nunique():>3}  "
              f"zero-halflife {(s.thalf_hours==0).mean()*100:4.1f}%  "
              f"median t1/2 {s.thalf_hours.median():.2f} h")
    for k, v in D["F"].items():
        print(f"  {k:<8} {v.shape}")
