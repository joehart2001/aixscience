"""Predict one peptide-HLA complex with Boltz-2 straight from the challenge CSV."""

import subprocess
from pathlib import Path

import pandas as pd
import torch
import yaml

from boltz_mace.prepare_inputs import load_alpha3, make_boltz_input, safe_slug

REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent / "out"

row = pd.read_csv(REPO_ROOT / "Data" / "rasmussen_et_al_dataset.csv").iloc[0]
sample_id = f"{safe_slug(row['allele'])}_{row['peptide'].lower()}"
yaml_path = OUT_DIR / f"{sample_id}.yaml"
yaml_path.parent.mkdir(parents=True, exist_ok=True)

# use_msa_server=False writes `msa: empty`, so this runs with no network calls.
payload = make_boltz_input(
    row["hla_seq"],
    load_alpha3()[row["allele"]],
    row["peptide"],
    use_msa_server=False,
)
yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False))
print(f"{sample_id}  thalf={row['thalf_hours']} h  ->  {yaml_path}")

subprocess.run(
    [
        "boltz",
        "predict",
        str(yaml_path),
        "--out_dir",
        str(OUT_DIR),
        "--accelerator",
        "gpu" if torch.cuda.is_available() else "cpu",
        "--override",
    ],
    check=True,
)
