"""Charge descriptors for the peptide and for the structure it sits in.

Two families, deliberately separated so their contributions are distinguishable.

SEQUENCE-DERIVED (no structure needed, available for every row)
    peptide:  net charge, counts of positive / negative / histidine residues,
              per-position charge (9 values), and the charge of the two anchor
              positions singled out (P2 and P9, the classic HLA anchors)
    groove:   the same net/count summary over the 34-residue NetMHCpan
              pseudosequence, plus its per-position charge (34 values)
    pair:     peptide net charge x groove net charge, and their sum -- a crude
              complementarity term, since an off-rate should care about whether
              opposite charges face each other rather than about either alone

STRUCTURE-DERIVED (needs the predicted coordinates; see charge_struct.py)

Charges are formal, at pH 7: Asp/Glu -1, Lys/Arg +1, His +0.1 (roughly its
fractional protonation near pH 7), termini +1 / -1 where included.

A WARNING FROM EARLIER IN THIS PROJECT. Peptide net charge looked predictive on
a 100-complex stratified sample (rho = +0.229) and then came out at -0.071
across all 1,023 of the same allele. That claim was withdrawn. Treat anything
here as unproven until it survives a held-out split with the usual bootstrap.
"""

import numpy as np

Q = {"D": -1.0, "E": -1.0, "K": +1.0, "R": +1.0, "H": +0.1}
NTERM, CTERM = +1.0, -1.0


def res_charges(seq):
    return np.array([Q.get(c, 0.0) for c in seq], dtype=np.float32)


def summary(seq, termini=False):
    """net charge, n_positive, n_negative, n_histidine, absolute charge."""
    q = res_charges(seq)
    net = q.sum() + (NTERM + CTERM if termini else 0.0)
    return np.array([net, (q > 0.5).sum(), (q < -0.5).sum(),
                     seq.count("H"), np.abs(q).sum()], dtype=np.float32)


def peptide_charge(seqs):
    """(N, 9 + 5 + 2) : per-position charge, summary, and the two anchor charges."""
    out = []
    for s in seqs:
        q = res_charges(s)
        out.append(np.concatenate([q, summary(s, termini=True),
                                   [q[1], q[8]]]))      # P2 and P9 anchors
    return np.asarray(out, dtype=np.float32)


def groove_charge(pseudoseqs):
    """(N, 34 + 5) : per-position pseudosequence charge and its summary."""
    out = []
    for s in pseudoseqs:
        out.append(np.concatenate([res_charges(s), summary(s)]))
    return np.asarray(out, dtype=np.float32)


def pair_terms(seqs, pseudoseqs):
    """(N, 4) : product, sum, and absolute difference of the two net charges."""
    out = []
    for p, g in zip(seqs, pseudoseqs):
        a = res_charges(p).sum() + NTERM + CTERM
        b = res_charges(g).sum()
        out.append([a, b, a * b, abs(a - b)])
    return np.asarray(out, dtype=np.float32)


def sequence_charge_block(peptides, pseudoseqs):
    """Everything above concatenated: (N, 16 + 39 + 4) = (N, 59)."""
    return np.hstack([peptide_charge(peptides),
                      groove_charge(pseudoseqs),
                      pair_terms(peptides, pseudoseqs)]).astype(np.float32)
