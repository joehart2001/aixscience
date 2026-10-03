"""Score Boltz peptide-HLA structures with MACE-MH-1's OMOL head."""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from ase import Atoms
from mace.calculators import mace_mp
from openmm import Platform, unit
from pdbfixer import PDBFixer
from scipy.stats import spearmanr


@dataclass(frozen=True)
class PreparedComplex:
    atoms: Atoms
    chain_ids: np.ndarray
    residue_keys: np.ndarray


def locate_prediction(predictions_dir: Path, sample_id: str) -> Path:
    """Locate Boltz's highest-confidence structure for a sample."""
    candidates = [
        predictions_dir / "predictions" / sample_id / f"{sample_id}_model_0.cif",
        predictions_dir / sample_id / f"{sample_id}_model_0.cif",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    nested = list(
        predictions_dir.glob(
            f"boltz_results_*/predictions/{sample_id}/{sample_id}_model_0.cif"
        )
    )
    if len(nested) == 1:
        return nested[0]
    if len(nested) > 1:
        raise RuntimeError(f"Found multiple predictions for {sample_id}: {nested}")
    raise FileNotFoundError(
        f"Could not find {sample_id}_model_0.cif under {predictions_dir}"
    )


def add_hydrogens(cif_path: Path, ph: float, seed: int) -> PreparedComplex:
    """Repair Boltz protein termini and add standard hydrogens with PDBFixer."""
    random.seed(seed)
    # OpenMM's multithreaded CPU minimizer can produce slightly different proton
    # coordinates between runs. The Reference platform is slower but deterministic
    # for this one-time hydrogen-only minimization.
    platform = Platform.getPlatformByName("Reference")
    fixer = PDBFixer(filename=str(cif_path), platform=platform)
    fixer.findMissingResidues()
    # Boltz supplies the requested sequence. Do not fabricate any missing loops,
    # but do repair missing heavy atoms such as terminal OXT atoms.
    fixer.missingResidues = {}
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)

    symbols: list[str] = []
    chain_ids: list[str] = []
    residue_keys: list[str] = []
    for atom in fixer.topology.atoms():
        if atom.element is None:
            raise ValueError(f"Atom {atom} has no chemical element")
        symbols.append(atom.element.symbol)
        chain_ids.append(atom.residue.chain.id)
        residue_keys.append(f"{atom.residue.chain.id}:{atom.residue.id}")

    positions = np.asarray(
        fixer.positions.value_in_unit(unit.angstrom), dtype=np.float64
    )
    return PreparedComplex(
        atoms=Atoms(symbols=symbols, positions=positions, pbc=False),
        chain_ids=np.asarray(chain_ids),
        residue_keys=np.asarray(residue_keys),
    )


def crop_interface(
    prepared: PreparedComplex, peptide_chain: str, radius: float
) -> tuple[Atoms, np.ndarray]:
    """Keep the peptide and whole receptor residues within radius Angstrom."""
    peptide_mask = prepared.chain_ids == peptide_chain
    if not peptide_mask.any():
        raise ValueError(
            f"Peptide chain {peptide_chain!r} is absent; found "
            f"{sorted(set(prepared.chain_ids))}"
        )
    receptor_mask = ~peptide_mask
    receptor_positions = prepared.atoms.positions[receptor_mask]
    peptide_positions = prepared.atoms.positions[peptide_mask]

    near_receptor = np.zeros(receptor_positions.shape[0], dtype=bool)
    radius_sq = radius**2
    for peptide_position in peptide_positions:
        displacement = receptor_positions - peptide_position
        near_receptor |= np.einsum("ij,ij->i", displacement, displacement) <= radius_sq

    near_residues = set(prepared.residue_keys[receptor_mask][near_receptor])
    keep_mask = peptide_mask | np.isin(prepared.residue_keys, list(near_residues))
    return prepared.atoms[keep_mask], prepared.chain_ids[keep_mask]


def mace_interaction_energy(
    complex_atoms: Atoms,
    chain_ids: np.ndarray,
    peptide_chain: str,
    calculator,
) -> tuple[float, float, float, float]:
    """Return complex, receptor, peptide, and interaction energies in eV."""
    peptide_mask = chain_ids == peptide_chain
    receptor_mask = ~peptide_mask
    if not peptide_mask.any() or not receptor_mask.any():
        raise ValueError("Both peptide and receptor atoms are required")

    systems = [
        complex_atoms.copy(),
        complex_atoms[receptor_mask],
        complex_atoms[peptide_mask],
    ]
    energies: list[float] = []
    for atoms in systems:
        atoms.calc = calculator
        energies.append(float(atoms.get_potential_energy()))
    complex_energy, receptor_energy, peptide_energy = energies
    interaction_energy = complex_energy - receptor_energy - peptide_energy
    return complex_energy, receptor_energy, peptide_energy, interaction_energy


def load_confidence(cif_path: Path) -> dict:
    """Load the confidence JSON next to a Boltz structure when present."""
    sample_id = cif_path.parent.name
    path = cif_path.parent / f"confidence_{sample_id}_model_0.json"
    return json.loads(path.read_text()) if path.exists() else {}


def score_manifest(
    manifest: Path,
    predictions_dir: Path,
    output_csv: Path,
    model: str,
    head: str,
    device: str,
    dtype: str,
    peptide_chain: str,
    crop_radius: float,
    ph: float,
    hydrogen_seed: int,
    limit: int | None = None,
) -> pd.DataFrame:
    """Score every completed structure in a pilot manifest."""
    frame = pd.read_csv(manifest)
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        frame = frame.head(limit)
    model_path = Path(model)
    model_spec = str(model_path.resolve()) if model_path.exists() else model
    calculator = mace_mp(
        model=model_spec,
        head=head,
        device=device,
        default_dtype=dtype,
        dispersion=False,
    )

    results: list[dict] = []
    for row in frame.to_dict(orient="records"):
        cif_path = locate_prediction(predictions_dir, row["sample_id"])
        prepared = add_hydrogens(cif_path, ph=ph, seed=hydrogen_seed)
        cropped, chain_ids = crop_interface(
            prepared, peptide_chain=peptide_chain, radius=crop_radius
        )
        e_complex, e_receptor, e_peptide, e_interaction = mace_interaction_energy(
            cropped,
            chain_ids,
            peptide_chain=peptide_chain,
            calculator=calculator,
        )
        confidence = load_confidence(cif_path)
        results.append(
            {
                **row,
                "structure_path": str(cif_path),
                "n_atoms_full": len(prepared.atoms),
                "n_atoms_scored": len(cropped),
                "mace_complex_eV": e_complex,
                "mace_receptor_eV": e_receptor,
                "mace_peptide_eV": e_peptide,
                "mace_interaction_eV": e_interaction,
                "boltz_confidence": confidence.get("confidence_score"),
                "boltz_iptm": confidence.get("iptm"),
                "boltz_complex_iplddt": confidence.get("complex_iplddt"),
                "hydrogen_seed": hydrogen_seed,
            }
        )

    result = pd.DataFrame(results)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_csv, index=False, quoting=csv.QUOTE_MINIMAL)
    if len(result) >= 3:
        # Lower interaction energy means stronger predicted interaction.
        rho, pvalue = spearmanr(
            -result["mace_interaction_eV"], result["thalf_hours"]
        )
        print(f"Spearman rho={rho:.3f}, p={pvalue:.3g}, n={len(result)}")
    columns = ["peptide", "thalf_hours", "mace_interaction_eV", "boltz_iptm"]
    print(result.sort_values("mace_interaction_eV")[columns].to_string(index=False))
    print(f"Wrote {output_csv}")
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("inputs/pilot/manifest.csv")
    )
    parser.add_argument("--predictions-dir", type=Path, default=Path("outputs/boltz"))
    parser.add_argument("--output", type=Path, default=Path("results/pilot_scores.csv"))
    parser.add_argument("--model", default="mh-1")
    parser.add_argument("--head", default="omol")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float64")
    parser.add_argument("--peptide-chain", default="C")
    parser.add_argument("--crop-radius", type=float, default=10.0)
    parser.add_argument("--ph", type=float, default=7.4)
    parser.add_argument("--hydrogen-seed", type=int, default=0)
    parser.add_argument("--limit", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    score_manifest(
        manifest=args.manifest,
        predictions_dir=args.predictions_dir,
        output_csv=args.output,
        model=args.model,
        head=args.head,
        device=args.device,
        dtype=args.dtype,
        peptide_chain=args.peptide_chain,
        crop_radius=args.crop_radius,
        ph=args.ph,
        hydrogen_seed=args.hydrogen_seed,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
