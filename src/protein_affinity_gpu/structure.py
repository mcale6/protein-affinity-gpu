# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Protein structure model plus load/sanitize helpers."""

import dataclasses
import io
import logging
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Tuple

import numpy as np
from Bio import PDB
from Bio.PDB import MMCIFParser, PDBParser, is_aa
from Bio.PDB.Structure import Structure

from .utils import residue_constants

LOGGER = logging.getLogger(__name__)

FeatureDict = Mapping[str, np.ndarray]
ModelOutput = Mapping[str, Any]

PDB_CHAIN_IDS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
PDB_MAX_CHAINS = len(PDB_CHAIN_IDS)


@dataclasses.dataclass(frozen=True)
class Protein:
    """Protein structure representation."""

    atom_positions: np.ndarray
    aatype: np.ndarray
    atom_mask: np.ndarray
    residue_index: np.ndarray
    chain_index: np.ndarray
    b_factors: np.ndarray

    def __post_init__(self):
        if len(np.unique(self.chain_index)) > PDB_MAX_CHAINS:
            raise ValueError(
                f"Cannot build an instance with more than {PDB_MAX_CHAINS} chains "
                "because these cannot be written to PDB format."
            )


def from_bio_structure(structure: Structure, chain_id: Optional[str] = None) -> Protein:
    """Build a canonical Protein object from a Biopython structure."""
    models = list(structure.get_models())
    if len(models) != 1:
        raise ValueError(f"Only single model PDBs/mmCIFs are supported. Found {len(models)} models.")

    model = models[0]
    atom_positions = []
    aatype = []
    atom_mask = []
    residue_index = []
    chain_ids = []
    b_factors = []

    for chain in model:
        if chain_id is not None and chain.id != chain_id:
            continue

        for residue in chain:
            if residue.id[2] != " ":
                raise ValueError(
                    f"PDB/mmCIF contains an insertion code at chain {chain.id} and "
                    f"residue index {residue.id[1]}. These are not supported."
                )

            residue_code = residue_constants.restype_3to1.get(residue.resname, "X")
            restype_idx = residue_constants.restype_order.get(residue_code, residue_constants.restype_num)
            positions = np.zeros((residue_constants.atom_type_num, 3))
            mask = np.zeros((residue_constants.atom_type_num,))
            residue_b_factors = np.zeros((residue_constants.atom_type_num,))

            for atom in residue:
                if atom.name not in residue_constants.atom_types:
                    continue
                atom_order = residue_constants.atom_order[atom.name]
                positions[atom_order] = atom.coord
                mask[atom_order] = 1.0
                residue_b_factors[atom_order] = atom.bfactor

            if np.sum(mask) < 0.5:
                continue

            aatype.append(restype_idx)
            atom_positions.append(positions)
            atom_mask.append(mask)
            residue_index.append(residue.id[1])
            chain_ids.append(chain.id)
            b_factors.append(residue_b_factors)

    unique_chain_ids = np.unique(chain_ids)
    chain_id_mapping = {chain_name: index for index, chain_name in enumerate(unique_chain_ids)}
    chain_index = np.array([chain_id_mapping[chain_name] for chain_name in chain_ids])

    return Protein(
        atom_positions=np.array(atom_positions),
        atom_mask=np.array(atom_mask),
        aatype=np.array(aatype),
        residue_index=np.array(residue_index),
        chain_index=chain_index,
        b_factors=np.array(b_factors),
    )


def from_pdb_string(pdb_str: str, chain_id: Optional[str] = None) -> Protein:
    """Construct a Protein object from a PDB string."""
    with io.StringIO(pdb_str) as pdb_handle:
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure(id="none", file=pdb_handle)
        return from_bio_structure(structure, chain_id)


def from_mmcif_string(mmcif_str: str, chain_id: Optional[str] = None) -> Protein:
    """Construct a Protein object from an mmCIF string."""
    with io.StringIO(mmcif_str) as mmcif_handle:
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure(structure_id="none", filename=mmcif_handle)
        return from_bio_structure(structure, chain_id)


def parse_structure_file(structure_path: str | Path) -> Structure:
    """Parse a structure file from disk."""
    structure_path = Path(structure_path)
    suffix = structure_path.suffix.lower()
    if suffix not in {".pdb", ".ent", ".cif", ".mmcif"}:
        raise ValueError(f"Unsupported structure format: {structure_path.suffix}")

    parser = PDB.PDBParser(QUIET=True) if suffix in {".pdb", ".ent"} else PDB.MMCIFParser(QUIET=True)
    return parser.get_structure(structure_path.stem, str(structure_path))


def sanitize_structure(structure: Structure, selected_chains: Iterable[str] | None = None) -> Structure:
    """Remove unsupported content and normalize a structure in-place."""
    selected_chains = list(selected_chains or [])
    models = list(structure.get_models())
    if not models:
        raise ValueError("Structure does not contain any models.")

    keep_model = models[0]
    for model in models[1:]:
        structure.detach_child(model.id)
        LOGGER.info("Removed extra model %s", model.id)

    current_chains = {chain.id for chain in keep_model}
    if selected_chains:
        missing = sorted(set(selected_chains) - current_chains)
        if missing:
            raise ValueError(f"Selected chain(s) not present in structure: {', '.join(missing)}")

        for chain in list(keep_model):
            if chain.id not in selected_chains:
                keep_model.detach_child(chain.id)
                LOGGER.info("Removed unselected chain %s", chain.id)

    for atom in list(structure.get_atoms()):
        if not atom.is_disordered():
            continue
        residue = atom.parent
        selected_atom = atom.selected_child
        selected_atom.altloc = " "
        selected_atom.disordered_flag = 0
        residue.detach_child(atom.id)
        residue.add(selected_atom)

    for chain in list(keep_model):
        for residue in list(chain):
            if residue.get_id()[2] != " ":
                chain.detach_child(residue.id)
                LOGGER.info("Removed residue with insertion code %s", residue.id)
                continue

            if residue.id[0][0] in {"W", "H"} or not is_aa(residue, standard=True):
                chain.detach_child(residue.id)
                LOGGER.info("Removed unsupported residue %s", residue.id)
                continue

            for atom in list(residue):
                if atom.element == "H" or atom.get_name().startswith("H"):
                    residue.detach_child(atom.id)

            if not list(residue.get_atoms()):
                chain.detach_child(residue.id)

        if not list(chain.get_residues()):
            keep_model.detach_child(chain.id)

    return structure


def load_structure(structure_path: str | Path, chain_id: str | None = None, sanitize: bool = True) -> Protein:
    """Load a single-chain structure into the canonical Protein representation."""
    structure = parse_structure_file(structure_path)
    if sanitize:
        sanitize_structure(structure, [chain_id] if chain_id else None)
    return from_bio_structure(structure, chain_id=chain_id)


def load_complex(
    structure_path: str | Path,
    selection: str = "A,B",
    sanitize: bool = True,
) -> tuple[Protein, Protein]:
    """Load a two-chain complex into canonical Protein representations."""
    target_chain, binder_chain = [chain.strip() for chain in selection.split(",")]
    if not target_chain or not binder_chain:
        raise ValueError("Selection must contain exactly two chain identifiers, e.g. 'A,B'.")

    structure = parse_structure_file(structure_path)
    if sanitize:
        sanitize_structure(structure, [target_chain, binder_chain])

    target = from_bio_structure(structure, chain_id=target_chain)
    binder = from_bio_structure(structure, chain_id=binder_chain)
    return target, binder


def _chain_end(atom_index, end_resname, chain_name, residue_index) -> str:
    return f"{'TER':<6}{atom_index:>5}      {end_resname:>3} {chain_name:>1}{residue_index:>4}"


def to_pdb(prot: Protein) -> str:
    """Convert a Protein object back to PDB format."""
    restypes = residue_constants.restypes + ["X"]
    atom_types = residue_constants.atom_types
    residue_name = lambda index: residue_constants.restype_1to3.get(restypes[index], "UNK")

    atom_mask = prot.atom_mask
    aatype = prot.aatype
    atom_positions = prot.atom_positions
    residue_index = prot.residue_index.astype(np.int32)
    chain_index = prot.chain_index.astype(np.int32)
    b_factors = prot.b_factors

    if np.any(aatype > residue_constants.restype_num):
        raise ValueError("Invalid aatypes.")

    chain_ids = {}
    for chain_number in np.unique(chain_index):
        if chain_number >= PDB_MAX_CHAINS:
            raise ValueError(f"The PDB format supports at most {PDB_MAX_CHAINS} chains.")
        chain_ids[chain_number] = PDB_CHAIN_IDS[chain_number]

    pdb_lines = ["MODEL     1"]
    atom_index = 1
    last_chain_index = chain_index[0]

    for residue_idx in range(aatype.shape[0]):
        if last_chain_index != chain_index[residue_idx]:
            pdb_lines.append(
                _chain_end(
                    atom_index,
                    residue_name(aatype[residue_idx - 1]),
                    chain_ids[chain_index[residue_idx - 1]],
                    residue_index[residue_idx - 1],
                )
            )
            last_chain_index = chain_index[residue_idx]
            atom_index += 1

        residue_name_3 = residue_name(aatype[residue_idx])
        for atom_name, position, mask, b_factor in zip(
            atom_types,
            atom_positions[residue_idx],
            atom_mask[residue_idx],
            b_factors[residue_idx],
        ):
            if mask < 0.5:
                continue

            name = atom_name if len(atom_name) == 4 else f" {atom_name}"
            element = atom_name[0]
            atom_line = (
                f"{'ATOM':<6}{atom_index:>5} {name:<4}{'':>1}"
                f"{residue_name_3:>3} {chain_ids[chain_index[residue_idx]]:>1}"
                f"{residue_index[residue_idx]:>4}{'':>1}   "
                f"{position[0]:>8.3f}{position[1]:>8.3f}{position[2]:>8.3f}"
                f"{1.00:>6.2f}{b_factor:>6.2f}          "
                f"{element:>2}{'':>2}"
            )
            pdb_lines.append(atom_line)
            atom_index += 1

    pdb_lines.append(
        _chain_end(atom_index, residue_name(aatype[-1]), chain_ids[chain_index[-1]], residue_index[-1])
    )
    pdb_lines.extend(["ENDMDL", "END"])
    return "\n".join(line.ljust(80) for line in pdb_lines) + "\n"


def ideal_atom_mask(prot: Protein) -> np.ndarray:
    """Compute the ideal heavy-atom mask for a Protein."""
    return residue_constants.STANDARD_ATOM_MASK[prot.aatype]


def from_prediction(
    features: FeatureDict,
    result: ModelOutput,
    b_factors: Optional[np.ndarray] = None,
    remove_leading_feature_dimension: bool = True,
) -> Protein:
    """Assemble a Protein object from model outputs."""
    fold_output = result["structure_module"]

    def maybe_remove_leading_dim(array: np.ndarray) -> np.ndarray:
        return array[0] if remove_leading_feature_dimension else array

    if "asym_id" in features:
        chain_index = maybe_remove_leading_dim(features["asym_id"])
    else:
        chain_index = np.zeros_like(maybe_remove_leading_dim(features["aatype"]))

    if b_factors is None:
        b_factors = np.zeros_like(fold_output["final_atom_mask"])

    return Protein(
        aatype=maybe_remove_leading_dim(features["aatype"]),
        atom_positions=fold_output["final_atom_positions"],
        atom_mask=fold_output["final_atom_mask"],
        residue_index=maybe_remove_leading_dim(features["residue_index"]) + 1,
        chain_index=chain_index,
        b_factors=b_factors,
    )
