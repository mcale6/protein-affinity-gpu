import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .utils import residue_constants


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):  # noqa: D401
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def build_sasa_records(
    complex_sasa: Any,
    relative_sasa: Any,
    target,
    binder,
    chain_labels: tuple[str, str],
) -> np.ndarray:
    """Convert atom- and residue-level SASA values into a structured array."""
    atom_types = np.array(residue_constants.atom_types)
    restype_lookup = np.array(
        [residue_constants.restype_1to3[restype] for restype in residue_constants.restypes],
        dtype="U3",
    )
    atoms_per_residue = residue_constants.atom_type_num

    target_residue_count = len(target.aatype)
    binder_residue_count = len(binder.aatype)
    total_residues = target_residue_count + binder_residue_count

    combined_mask = np.asarray(np.concatenate([target.atom_mask, binder.atom_mask]).ravel(), dtype=bool)
    residue_names = np.concatenate(
        [restype_lookup[np.asarray(target.aatype)], restype_lookup[np.asarray(binder.aatype)]]
    )
    residue_indices = np.concatenate([target.residue_index, binder.residue_index]).astype(int)
    residue_chains = np.concatenate(
        [
            np.full(target_residue_count, chain_labels[0], dtype="U2"),
            np.full(binder_residue_count, chain_labels[1], dtype="U2"),
        ]
    )

    atom_names = np.tile(atom_types, total_residues)
    chain_ids_atom = np.repeat(residue_chains, atoms_per_residue)
    residue_names_atom = np.repeat(residue_names, atoms_per_residue)
    residue_indices_atom = np.repeat(residue_indices, atoms_per_residue)
    relative_sasa_atom = np.repeat(np.asarray(relative_sasa), atoms_per_residue)

    dtype = [
        ("chain", "U2"),
        ("resname", "U3"),
        ("resindex", "i4"),
        ("atomname", "U4"),
        ("atom_sasa", "f4"),
        ("relative_sasa", "f4"),
    ]
    filtered_rows = list(
        zip(
            chain_ids_atom[combined_mask],
            residue_names_atom[combined_mask],
            residue_indices_atom[combined_mask],
            atom_names[combined_mask],
            np.asarray(complex_sasa)[combined_mask],
            relative_sasa_atom[combined_mask],
        )
    )
    return np.array(filtered_rows, dtype=dtype)


@dataclass
class ContactAnalysis:
    """Results from analyzing interface contacts."""

    values: list

    def __post_init__(self):
        self.values = [float(value) for value in self.values]
        if len(self.values) != 6:
            raise ValueError("Contact values must be a list of length 6")

    def to_dict(self) -> dict[str, float]:
        total_contacts = sum(self.values)
        charged_contacts = self.values[1] + self.values[3] + self.values[5]
        polar_contacts = self.values[2] + self.values[4] + self.values[5]
        aliphatic_contacts = self.values[0] + self.values[3] + self.values[4]
        return {
            "AA": self.values[0],
            "CC": self.values[1],
            "PP": self.values[2],
            "AC": self.values[3],
            "AP": self.values[4],
            "CP": self.values[5],
            "IC": total_contacts,
            "chargedC": charged_contacts,
            "polarC": polar_contacts,
            "aliphaticC": aliphatic_contacts,
        }


@dataclass
class ProdigyResults:
    contact_types: ContactAnalysis
    binding_affinity: np.float32
    dissociation_constant: np.float32
    nis_aliphatic: np.float32
    nis_charged: np.float32
    nis_polar: np.float32
    structure_id: str = "_"
    sasa_data: np.ndarray | None = None

    def to_dict(self) -> dict[str, Any]:
        sasa_list = []
        if self.sasa_data is not None:
            for row in self.sasa_data:
                sasa_list.append(
                    {
                        "chain": row["chain"],
                        "resname": row["resname"],
                        "resindex": int(row["resindex"]),
                        "atomname": row["atomname"],
                        "atom_sasa": float(row["atom_sasa"]),
                        "relative_sasa": float(row["relative_sasa"]),
                    }
                )

        return {
            "structure_id": self.structure_id,
            "ba_val": float(self.binding_affinity),
            "kd": float(self.dissociation_constant),
            "contacts": self.contact_types.to_dict(),
            "nis": {
                "aliphatic": float(self.nis_aliphatic),
                "charged": float(self.nis_charged),
                "polar": float(self.nis_polar),
            },
            "sasa_data": sasa_list,
        }

    def save_results(self, output_dir: str | Path) -> Path:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{self.structure_id}_results.json"
        output_path.write_text(json.dumps(self.to_dict(), indent=2, cls=NumpyEncoder))
        return output_path

    def __str__(self) -> str:
        contact_types_dict = self.contact_types.to_dict()
        return (
            "------------------------\n"
            "PRODIGY Analysis Results\n"
            "------------------------\n"
            f"Binding Energy (DG): {self.binding_affinity:.2f} kcal/mol\n"
            f"Dissociation Constant (Kd): {self.dissociation_constant:.2e} M\n"
            "------------------------\n"
            "\nContact Analysis:\n"
            f"  Charged-Charged: {contact_types_dict['CC']:.1f}\n"
            f"  Polar-Polar: {contact_types_dict['PP']:.1f}\n"
            f"  Aliphatic-Aliphatic: {contact_types_dict['AA']:.1f}\n"
            f"  Aliphatic-Charged: {contact_types_dict['AC']:.1f}\n"
            f"  Aliphatic-Polar: {contact_types_dict['AP']:.1f}\n"
            f"  Charged-Polar: {contact_types_dict['CP']:.1f}\n"
            "------------------------\n"
            "\nNon-Interacting Surface:\n"
            f"  Aliphatic: {self.nis_aliphatic:.1f}%\n"
            f"  Charged: {self.nis_charged:.1f}%\n"
            f"  Polar: {self.nis_polar:.1f}%\n"
            "------------------------\n"
        )
