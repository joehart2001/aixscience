"""Structure-derived charge descriptors from the predicted complexes.

The sequence block in charges.py knows which residues are charged but nothing
about where they are. These use the Boltz coordinates, so they can express
whether a charged peptide residue actually faces an oppositely charged groove
residue -- which is what an off-rate should care about -- rather than only that
both exist somewhere.

Per complex, computed on the charged side-chain centres (the charge-bearing
atoms of Asp/Glu/Lys/Arg/His, plus the backbone termini of the peptide):

    coulomb_sum          sum_ij q_i q_j / d_ij over peptide x receptor pairs,
                         a screened-free, dielectric-free Coulomb proxy
    coulomb_attract      the same restricted to q_i q_j < 0 (favourable)
    coulomb_repel        the same restricted to q_i q_j > 0
    n_salt_bridges       opposite-charge pairs within 4 A
    n_close_repulsive    like-charge pairs within 4 A
    receptor_q_8/12 A    net receptor charge within 8 and 12 A of any peptide atom
    pos_q_env (9)        per peptide position, sum of receptor q / d within 12 A
    pep_q_sidechain      net peptide charge from the same charged-centre model
    min_d_saltbridge     closest opposite-charge contact distance (20 if none)

20 values. All are invariant to rotation and translation.

Note the reused HLA-A*02:01 complexes appear in both prediction trees; rows are
keyed on the complex name, so the duplicate is collapsed rather than counted twice.

Resumable: writes one row per complex to charge_struct.npz and skips what is
already there, so it can be re-run as more structures arrive.
"""

import glob
import os
import sys
from multiprocessing import Pool

import numpy as np
from scipy.spatial import cKDTree

PRED = ("/share/ijp30/hackathons/aixscience2026/"
        "boltz_inputs_and_predictions/predictions/all75")
PRED_OLD = ("/share/jh2536/hackathons/aixscience2026/"
            "boltz_inputs_and_predictions/predictions/a0201")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charge_struct.npz")
PEPTIDE_CHAIN = "C"

# charge-bearing side-chain atoms and the charge carried there
CHARGED = {
    "ASP": (["OD1", "OD2"], -1.0),
    "GLU": (["OE1", "OE2"], -1.0),
    "LYS": (["NZ"], +1.0),
    "ARG": (["NH1", "NH2", "NE"], +1.0),
    "HIS": (["ND1", "NE2"], +0.1),
}
NAMES = (["coulomb_sum", "coulomb_attract", "coulomb_repel", "n_salt_bridges",
          "n_close_repulsive", "receptor_q_8A", "receptor_q_12A"]
         + [f"pos_q_env_{i+1}" for i in range(9)]
         + ["pep_q_sidechain", "min_d_saltbridge", "n_pep_charged",
            "n_recep_charged_12A"])


def read_cif(path):
    """-> chain, resseq, resname, atomname, xyz  for every ATOM record."""
    ch, rs, rn, an, xyz = [], [], [], [], []
    with open(path) as fh:
        for line in fh:
            if not line.startswith("ATOM"):
                continue
            f = line.split()
            # columns fixed by the Boltz writer, verified against the header
            an.append(f[3]); rn.append(f[5]); rs.append(int(f[7]))
            ch.append(f[15]); xyz.append((float(f[10]), float(f[11]), float(f[12])))
    return (np.array(ch), np.array(rs), np.array(rn), np.array(an),
            np.asarray(xyz, dtype=np.float64))


def charge_centres(ch, rs, rn, an, xyz, mask):
    """One point per charged group: mean of its charge-bearing atoms."""
    pts, qs, res = [], [], []
    sel = np.flatnonzero(mask)
    if not len(sel):
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    key = np.array([f"{c}_{r}" for c, r in zip(ch[sel], rs[sel])])
    for k in np.unique(key):
        i = sel[key == k]
        name = rn[i][0]
        if name not in CHARGED:
            continue
        atoms, q = CHARGED[name]
        j = i[np.isin(an[i], atoms)]
        if not len(j):
            continue
        pts.append(xyz[j].mean(0)); qs.append(q); res.append(rs[i][0])
    if not pts:
        return np.zeros((0, 3)), np.zeros(0), np.zeros(0, dtype=int)
    return np.asarray(pts), np.asarray(qs), np.asarray(res)


def one(path):
    try:
        ch, rs, rn, an, xyz = read_cif(path)
        pep = ch == PEPTIDE_CHAIN
        rec = ~pep
        if pep.sum() == 0 or rec.sum() == 0:
            return None
        P, qP, resP = charge_centres(ch, rs, rn, an, xyz, pep)
        R, qR, _ = charge_centres(ch, rs, rn, an, xyz, rec)

        # peptide termini, from the first and last peptide residue backbone
        pres = np.unique(rs[pep])
        for r, q, nm in ((pres.min(), +1.0, "N"), (pres.max(), -1.0, "OXT")):
            j = np.flatnonzero(pep & (rs == r) & (an == nm))
            if not len(j):
                j = np.flatnonzero(pep & (rs == r) & (an == ("N" if q > 0 else "C")))
            if len(j):
                P = np.vstack([P, xyz[j[0]]]) if len(P) else xyz[j[:1]]
                qP = np.append(qP, q); resP = np.append(resP, r)

        if len(P) == 0 or len(R) == 0:
            d = np.zeros(len(NAMES), dtype=np.float32)
            d[NAMES.index("min_d_saltbridge")] = 20.0
            return os.path.basename(os.path.dirname(path)), d

        D = np.linalg.norm(P[:, None, :] - R[None, :, :], axis=-1)
        D = np.maximum(D, 1.0)                      # avoid a blow-up at contact
        QQ = qP[:, None] * qR[None, :]
        cou = QQ / D
        attract, repel = cou[QQ < 0].sum(), cou[QQ > 0].sum()
        close = D <= 4.0
        n_salt = int((close & (QQ < 0)).sum())
        n_rep = int((close & (QQ > 0)).sum())
        md = float(D[QQ < 0].min()) if (QQ < 0).any() else 20.0

        pep_xyz = xyz[pep]
        tree = cKDTree(R)

        def near(radius):
            """Indices of receptor charge centres within `radius` of any peptide atom."""
            hits = tree.query_ball_point(pep_xyz, radius)
            flat = [i for h in hits for i in h]
            return np.unique(flat).astype(int) if flat else np.zeros(0, dtype=int)

        u8, u12 = near(8.0), near(12.0)
        q8 = qR[u8].sum() if len(u8) else 0.0
        q12 = qR[u12].sum() if len(u12) else 0.0

        # per peptide position: q / d over receptor charges within 12 A
        env = np.zeros(9, dtype=np.float64)
        for k, r in enumerate(pres[:9]):
            a = xyz[pep & (rs == r)]
            if not len(a):
                continue
            dd = np.linalg.norm(a[:, None, :] - R[None, :, :], axis=-1).min(0)
            m = dd <= 12.0
            if m.any():
                env[k] = (qR[m] / np.maximum(dd[m], 1.0)).sum()

        v = np.array([cou.sum(), attract, repel, n_salt, n_rep, q8, q12,
                      *env, qP.sum(), md, len(qP), len(u12)], dtype=np.float32)
        return os.path.basename(os.path.dirname(path)), v
    except Exception as e:                                   # keep going
        return ("ERR:" + os.path.basename(os.path.dirname(path)), str(e))


if __name__ == "__main__":
    have = {}
    if os.path.exists(OUT):
        z = np.load(OUT, allow_pickle=True)
        have = dict(zip(z["names"], z["X"]))
        print(f"{len(have):,} already computed")

    paths = sorted(glob.glob(f"{PRED}/*/*_model_0.cif")) + \
        sorted(glob.glob(f"{PRED_OLD}/*/*_model_0.cif"))
    todo = [p for p in paths
            if os.path.basename(os.path.dirname(p)) not in have]
    print(f"{len(paths):,} structures on disk, {len(todo):,} to do", flush=True)

    errs = 0
    with Pool(int(os.environ.get("NPROC", "16"))) as pool:
        for n, r in enumerate(pool.imap_unordered(one, todo, chunksize=16), 1):
            if r is None:
                continue
            k, v = r
            if k.startswith("ERR:"):
                errs += 1
                if errs <= 3:
                    print("  " + k + "  " + str(v), flush=True)
                continue
            have[k] = v
            if n % 1000 == 0:
                print(f"  {n:,}/{len(todo):,}", flush=True)

    names = np.array(sorted(have))
    X = np.asarray([have[k] for k in names], dtype=np.float32)
    np.savez_compressed(OUT, names=names, X=X, feature_names=np.array(NAMES))
    print(f"wrote {OUT}: {X.shape}, {errs} errors")
