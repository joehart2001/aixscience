from __future__ import annotations

from pathlib import Path

import pandas as pd

from boltz_mace.prepare_inputs import (
    load_alpha3,
    make_boltz_input,
    select_quantile_spanning_rows,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_quantile_selection_spans_observed_range() -> None:
    frame = pd.DataFrame(
        {
            "peptide": [f"P{i}" for i in range(9)],
            "thalf_hours": list(range(9)),
        }
    )
    selected = select_quantile_spanning_rows(frame, 3)
    assert selected["thalf_hours"].tolist() == [0, 4, 8]


def test_boltz_input_has_three_protein_chains() -> None:
    payload = make_boltz_input("ACDE", "FGHI", "AAAAAAAAA", use_msa_server=False)
    proteins = [entry["protein"] for entry in payload["sequences"]]
    assert [protein["id"] for protein in proteins] == ["A", "B", "C"]
    assert all(protein["msa"] == "empty" for protein in proteins)
    assert proteins[0]["sequence"] == "ACDEFGHI"
    assert proteins[-1]["sequence"] == "AAAAAAAAA"


def test_alpha3_table_covers_dataset() -> None:
    alpha3 = load_alpha3()
    dataset = pd.read_csv(REPO_ROOT / "Data" / "rasmussen_et_al_dataset.csv")
    assert set(dataset["allele"]) == set(alpha3)
    assert {len(sequence) for sequence in alpha3.values()} == {94}
    # HLA-A*02:01 heavy chain residues 183-276 in PDB 1AKJ.
    assert alpha3["HLA-A*02:01"] == (
        "DAPKTHMTHHAVSDHEATLRCWALSFYPAEITLTWQRDGEDQTQDTELVETRPAGDGTFQKWAAVVV"
        "PSGQEQRYTCHVQHEGLPKPLTLRWEP"
    )
    assert alpha3["HLA-B*07:02"] != alpha3["HLA-A*02:01"]
