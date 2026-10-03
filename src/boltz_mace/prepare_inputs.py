"""Create a small, distribution-spanning Boltz-2 peptide-HLA pilot."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# The challenge CSV supplies mature residues 1-182 (alpha1/alpha2). Each allele's
# alpha3 domain comes from IPD-IMGT/HLA via build_alpha3.py, and
# beta-2-microglobulin is included as a separate chain.
ALPHA3_TABLE = Path(__file__).resolve().parents[2] / "Data" / "hla_alpha3.csv"
BETA2_MICROGLOBULIN = (
    "IQRTPKIQVYSRHPAENGKSNFLNCYVSGFHPSDIEVDLLKNGERIEKVEHSDLSFSKDWSFYLLYY"
    "TEFTPTEKDEYACRVNHVTLSQPKIVKWDRDM"
)


def safe_slug(text: str) -> str:
    """Return a filesystem-safe, stable identifier."""
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_").lower()


def select_quantile_spanning_rows(
    frame: pd.DataFrame, n_samples: int
) -> pd.DataFrame:
    """Select observed rows nearest evenly spaced target quantiles."""
    if n_samples < 2:
        raise ValueError("n_samples must be at least 2")
    if len(frame) < n_samples:
        raise ValueError("n_samples cannot exceed the number of available rows")

    ordered = frame.sort_values(["thalf_hours", "peptide"]).reset_index(drop=True)
    indices = np.linspace(0, len(ordered) - 1, n_samples).round().astype(int)
    return ordered.iloc[indices].reset_index(drop=True)


def load_alpha3(path: Path = ALPHA3_TABLE) -> dict[str, str]:
    """Map dataset allele names to their HLA alpha3-domain sequences."""
    table = pd.read_csv(path)
    return dict(zip(table["allele"], table["alpha3"], strict=True))


def make_boltz_input(
    hla_sequence: str, alpha3: str, peptide: str, use_msa_server: bool
) -> dict:
    """Build a Boltz-2 input dictionary for one pHLA complex."""
    protein_msa = {} if use_msa_server else {"msa": "empty"}
    return {
        "version": 1,
        "sequences": [
            {
                "protein": {
                    "id": "A",
                    "sequence": hla_sequence + alpha3,
                    **protein_msa,
                }
            },
            {
                "protein": {
                    "id": "B",
                    "sequence": BETA2_MICROGLOBULIN,
                    **protein_msa,
                }
            },
            {
                "protein": {
                    "id": "C",
                    "sequence": peptide,
                    # A 9-mer MSA is not informative and adds an avoidable call.
                    "msa": "empty",
                }
            },
        ],
    }


def prepare_inputs(
    dataset: Path,
    output_dir: Path,
    allele: str,
    n_samples: int,
    use_msa_server: bool,
) -> Path:
    """Write Boltz YAML inputs and return the manifest path."""
    frame = pd.read_csv(dataset)
    required = {"allele", "peptide", "thalf_hours", "hla_seq"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")

    allele_frame = frame.loc[frame["allele"] == allele].copy()
    if allele_frame.empty:
        raise ValueError(f"No rows found for allele {allele!r}")
    alpha3 = load_alpha3().get(allele)
    if alpha3 is None:
        raise ValueError(f"No alpha3 domain for {allele!r} in {ALPHA3_TABLE}")
    if allele_frame["hla_seq"].nunique() != 1:
        raise ValueError(f"Expected one HLA sequence for {allele!r}")

    selected = select_quantile_spanning_rows(allele_frame, n_samples)
    output_dir.mkdir(parents=True, exist_ok=True)
    boltz_dir = output_dir / "boltz"
    boltz_dir.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, str | float]] = []

    for _, row in selected.iterrows():
        peptide = str(row["peptide"])
        sample_id = f"{safe_slug(allele)}_{peptide.lower()}"
        yaml_path = boltz_dir / f"{sample_id}.yaml"
        payload = make_boltz_input(
            str(row["hla_seq"]), alpha3, peptide, use_msa_server=use_msa_server
        )
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
        manifest_rows.append(
            {
                "sample_id": sample_id,
                "allele": allele,
                "peptide": peptide,
                "thalf_hours": float(row["thalf_hours"]),
                "boltz_input": str(yaml_path),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    return manifest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("../rasmussen_et_al_dataset.csv"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("inputs/pilot"))
    parser.add_argument("--allele", default="HLA-A*02:01")
    parser.add_argument("--n-samples", type=int, default=5)
    parser.add_argument(
        "--use-msa-server",
        action="store_true",
        help="Let `boltz predict --use_msa_server` obtain HLA and beta2m MSAs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = prepare_inputs(
        dataset=args.dataset,
        output_dir=args.output_dir,
        allele=args.allele,
        n_samples=args.n_samples,
        use_msa_server=args.use_msa_server,
    )
    print(f"Wrote {args.n_samples} Boltz inputs and {manifest}")


if __name__ == "__main__":
    main()
