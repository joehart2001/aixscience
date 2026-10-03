from __future__ import annotations

import pandas as pd

from boltz_mace.prepare_inputs import make_boltz_input, select_quantile_spanning_rows


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
    payload = make_boltz_input("ACDE", "AAAAAAAAA", use_msa_server=False)
    proteins = [entry["protein"] for entry in payload["sequences"]]
    assert [protein["id"] for protein in proteins] == ["A", "B", "C"]
    assert all(protein["msa"] == "empty" for protein in proteins)
    assert proteins[-1]["sequence"] == "AAAAAAAAA"

