"""Assemble the MACE feature blocks produced by the pc79 session.

Five families, kept separate so permutations of them are meaningful:

  seq      peptide one-hot + NetMHCpan pseudosequence one-hot          (860)
  boltz    the Boltz pooled blocks                                    (1547)
  node     MACE descriptors on the peptide residues, 5 slots x 256   (11520)
           N, CA, C, O, sidechain; glycine side chains masked to zero
  energy   per-residue MACE site energies, peptide and groove          (430)
           sum and mean per slot, E0-subtracted
  edge     peptide<->groove CROSS-CHAIN edges from MACE's own 6 A graph
           counts, closest approach, hydrogen-bond counts, the radial
           basis, element composition, and the partner node descriptor,
           for both the peptide and the groove side               (14450)

The edge block is the one that matters for the brief's question. A node
descriptor is dominated by residue identity, so a head on it can learn a
sequence predictor by another route -- the most likely reading of the earlier
single-allele negative. A cross-chain edge exists only where the two chains
actually touch, so it cannot be a disguised sequence feature. That it should
carry the signal is also what the Boltz pair representation already suggested:
the three cross_z blocks hold 65% of the tree gain on held-out alleles.

Masked slots are zero, matching how the masks are applied in experiment 005.
"""

import os

import numpy as np

FEAT = ("/share/ijp30/hackathons/aixscience/claude-experiments/"
        "008-mace-all75/feat_c16_hmin")

EDGE_KEYS = ["Epep_n", "Epep_dmin", "Epep_hb", "Epep_rad", "Epep_elem",
             "Epep_partner", "Ehla_n", "Ehla_dmin", "Ehla_hb", "Ehla_rad",
             "Ehla_elem", "Ehla_partner"]
ENERGY_KEYS = ["Epep_sum", "Epep_mean", "Ehla_sum", "Ehla_mean"]


def have(sid):
    return os.path.exists(os.path.join(FEAT, f"{sid}.npz"))


def read(sid):
    """-> dict of flat float32 vectors: node, energy, edge."""
    z = np.load(os.path.join(FEAT, f"{sid}.npz"), allow_pickle=True)
    node = (z["X_pep"] * z["mask_pep"][..., None]).astype(np.float32).ravel()
    energy = np.concatenate([np.asarray(z[k], np.float32).ravel()
                             for k in ENERGY_KEYS])
    edge = np.concatenate([np.asarray(z[k], np.float32).ravel()
                           for k in EDGE_KEYS]
                          + [np.asarray([z["n_cross_edges"], z["n_hbonds"]],
                                        np.float32)])
    return {"node": node, "energy": energy, "edge": edge}


def load_blocks(ids, verbose=True):
    first = read(ids[0])
    out = {k: np.zeros((len(ids), len(v)), np.float32)
           for k, v in first.items()}
    for k, v in first.items():
        out[k][0] = v
    for i, sid in enumerate(ids[1:], 1):
        for k, v in read(sid).items():
            out[k][i] = v
        if verbose and (i + 1) % 2000 == 0:
            print(f"  {i+1:,}/{len(ids):,}", flush=True)
    return out


if __name__ == "__main__":
    import glob
    ids = [os.path.basename(f)[:-4] for f in sorted(glob.glob(f"{FEAT}/*.npz"))]
    print(f"{len(ids):,} MACE feature files")
    b = read(ids[0])
    for k, v in b.items():
        print(f"  {k:<8} {v.shape[0]:>6,}")
