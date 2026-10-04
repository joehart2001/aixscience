"""Extract peptide and contacting-receptor tokens for the cross-attention model.

The trunk single representation s is (384, 384): one vector per token, ordered
chain A (HLA heavy, tokens 0-275), chain B (beta-2 microglobulin, 276-374),
chain C (peptide, 375-383). Verified against peptide_token_mask, which is True
exactly on 375-383.

The pooled feature vector averages those 384-dim token vectors away. Cross
attention wants them kept, so for each complex this saves:

    P     (9, 384)    the peptide token vectors, in sequence order
    R     (K, 384)    the K receptor tokens nearest the peptide
    Dm    (9, K)      alpha-carbon distance from each peptide token to each
                      kept receptor token, used as an attention bias

Token centres are alpha-carbons read from the predicted structure, in the same
chain order as the tokens, so index i of the coordinate array is token i.

K = 48 covers the groove comfortably: a 12 A shell around the peptide holds
roughly 35-45 receptor residues in these complexes. Tokens are ordered by
distance, nearest first, so a smaller K can be taken later by slicing.
"""

import os

import numpy as np

from trainer import FROZEN, get_data

EMB = ("/share/ijp30/hackathons/aixscience2026/"
       "boltz_inputs_and_predictions/embeddings/all75")
EMB_OLD = ("/share/jh2536/hackathons/aixscience2026/"
           "boltz_inputs_and_predictions/embeddings/a0201")
PRED = ("/share/ijp30/hackathons/aixscience2026/"
        "boltz_inputs_and_predictions/predictions/all75")
PRED_OLD = ("/share/jh2536/hackathons/aixscience2026/"
            "boltz_inputs_and_predictions/predictions/a0201")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tokens.npz")
K = 48
CHAIN_ORDER = ["A", "B", "C"]


def ca_coords(cif):
    """Alpha-carbon coordinates in token order: chain A, then B, then C."""
    per = {c: [] for c in CHAIN_ORDER}
    with open(cif) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            f = line.split()
            if f[3] != "CA":
                continue
            ch = f[15]
            if ch in per:
                per[ch].append((float(f[10]), float(f[11]), float(f[12])))
    return np.asarray([p for c in CHAIN_ORDER for p in per[c]], dtype=np.float32)


def find(base_new, base_old, sid, suffix=""):
    for b in (base_new, base_old):
        p = os.path.join(b, sid + suffix) if suffix else os.path.join(b, sid)
        if os.path.exists(p):
            return p
    return None


def one(sid):
    e = find(EMB, EMB_OLD, sid, ".npz")
    d = find(PRED, PRED_OLD, sid)
    if e is None or d is None:
        return None
    cif = os.path.join(d, f"{sid}_model_0.cif")
    if not os.path.exists(cif):
        return None
    z = np.load(e, allow_pickle=True)
    s, m = z["s"], z["peptide_token_mask"]
    xyz = ca_coords(cif)
    if len(xyz) != len(m):
        return None                      # token/residue mismatch; skip loudly
    pep = np.flatnonzero(m)
    rec = np.flatnonzero(~m)
    dist = np.linalg.norm(xyz[pep][:, None, :] - xyz[rec][None, :, :], axis=-1)
    near = np.argsort(dist.min(0))[:K]
    return (s[pep].astype(np.float32),
            s[rec[near]].astype(np.float32),
            dist[:, near].astype(np.float32))


if __name__ == "__main__":
    D = get_data()
    ids = D["df"].sample_id.to_numpy()
    P = np.zeros((len(ids), 9, 384), dtype=np.float32)
    R = np.zeros((len(ids), K, 384), dtype=np.float32)
    Dm = np.zeros((len(ids), 9, K), dtype=np.float32)
    bad = []
    for i, sid in enumerate(ids):
        r = one(sid)
        if r is None:
            bad.append(sid)
            continue
        P[i], R[i], Dm[i] = r
        if (i + 1) % 1000 == 0:
            print(f"  {i+1:,}/{len(ids):,}", flush=True)
    if bad:
        raise SystemExit(f"{len(bad)} complexes could not be prepared, "
                         f"e.g. {bad[:3]} -- the frozen set requires all of them")
    np.savez(OUT, ids=np.asarray(ids, dtype="U64"), P=P, R=R, D=Dm, K=K)
    print(f"wrote {OUT}  P {P.shape}  R {R.shape}  D {Dm.shape}")
    print(f"  contact distance: median {np.median(Dm.min(1)):.1f} A, "
          f"furthest kept token {Dm.min(1).max():.1f} A")
