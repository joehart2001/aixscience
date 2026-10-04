"""The escape coordinate, and invariants of the 3x3 rigid-translation Hessian.

H is the curvature of the energy against rigid displacement of the whole
peptide, so its eigenvalues are stiffnesses and its eigenvectors are directions.
The eigenvalues are already rotation invariant. The eigenvectors are NOT -- they
rotate with Boltz's arbitrary output orientation -- so they can only enter a
model contracted with another vector that rotates the same way.

The escape direction supplies that vector, and it is pure geometry, available
without the Hessian:

    e = normalise( centre of mass of the peptide
                   - centroid of the 34 pseudosequence (groove) residues )

Both rotate together with the complex, so every quantity below is invariant.

    H_ee          e^T H e, the stiffness against pulling the peptide straight
                  out of the groove. The single most direct statement about the
                  off-rate that three modes can make.
    H_ee_inv      1 / H_ee, the compliance; large means easily pulled out
    cos_soft      |v_min . e|, how closely the softest direction aligns with
                  escape. 1 means the easiest motion IS leaving
    cos_stiff     |v_max . e|
    lam_soft/mid/stiff   sorted eigenvalues
    anisotropy    lam_stiff / lam_soft
    trace, det, log_det   invariants of H
    F_e           net force projected on e (these structures are not stationary,
                  so this is a real quantity, not noise)
    F_mag         net force magnitude
    depth         |COM_peptide - groove centroid|
    burial        receptor heavy atoms within 5 A of any peptide heavy atom
    n_negative    eigenvalues below zero (a non-stationary structure can have them)

Geometry comes from the same truncated structures the Hessian was computed on,
so atom ordering and coordinates match exactly.
"""

import glob
import os

import numpy as np
from ase.io import read

XYZ = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
       "008-mace-all75/xyz_trunc_c16_hmin")
HESS = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
        "008-mace-all75/hess_c16_hmin")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "escape.npz")

NAMES = ["H_ee", "H_ee_inv", "cos_soft", "cos_stiff", "lam_soft", "lam_mid",
         "lam_stiff", "anisotropy", "trace", "det", "log_det", "F_e", "F_mag",
         "depth", "burial", "n_negative"]


def escape_direction(atoms):
    """Groove centroid -> peptide centre of mass, normalised."""
    part = atoms.arrays["part"]
    pep = part == "peptide"
    grv = part == "hla_pseudo"
    m = atoms.get_masses()[pep]
    com = (atoms.positions[pep] * m[:, None]).sum(0) / m.sum()
    cen = atoms.positions[grv].mean(0)
    v = com - cen
    return v / np.linalg.norm(v), com, cen


def one(sid):
    a = read(os.path.join(XYZ, f"{sid}.xyz"))
    z = np.load(os.path.join(HESS, f"{sid}.npz"), allow_pickle=True)
    H = np.asarray(z["H"], float)
    H = 0.5 * (H + H.T)                       # symmetrise; already symmetric
    lam, V = np.linalg.eigh(H)                # ascending
    e, com, cen = escape_direction(a)

    H_ee = float(e @ H @ e)
    part, Z = a.arrays["part"], a.numbers
    pep_h = (part == "peptide") & (Z > 1)
    rec_h = (part != "peptide") & (Z > 1)
    d = np.linalg.norm(a.positions[rec_h][:, None, :]
                       - a.positions[pep_h][None, :, :], axis=-1)
    burial = float((d.min(1) <= 5.0).sum())
    F = np.asarray(z["net_force"], float)

    return np.array([
        H_ee,
        1.0 / H_ee if abs(H_ee) > 1e-6 else 0.0,
        abs(float(V[:, 0] @ e)),
        abs(float(V[:, 2] @ e)),
        lam[0], lam[1], lam[2],
        lam[2] / lam[0] if lam[0] > 1e-6 else 0.0,
        float(np.trace(H)), float(np.linalg.det(H)),
        float(np.log(abs(np.linalg.det(H)) + 1e-12)),
        float(F @ e), float(np.linalg.norm(F)),
        float(np.linalg.norm(com - cen)), burial,
        float((lam < 0).sum()),
    ], dtype=np.float32)


if __name__ == "__main__":
    sids = sorted(os.path.basename(f)[:-4]
                  for f in glob.glob(f"{HESS}/*.npz"))
    print(f"{len(sids)} Hessians available")
    rows, keep = [], []
    for i, s in enumerate(sids):
        try:
            rows.append(one(s))
            keep.append(s)
        except Exception as exc:
            print(f"  {s}: {exc}")
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(sids)}", flush=True)
    X = np.asarray(rows, dtype=np.float32)
    np.savez(OUT, names=np.asarray(keep, dtype="U64"), X=X,
             feature_names=np.asarray(NAMES, dtype="U20"))
    print(f"wrote {OUT}  {X.shape}")
    for i, n in enumerate(NAMES):
        print(f"  {n:<12} mean {X[:, i].mean():+10.3f}  sd {X[:, i].std():9.3f}"
              f"  min {X[:, i].min():+9.3f}  max {X[:, i].max():+9.3f}")
