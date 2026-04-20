from typing import Dict, NamedTuple
from pathlib import Path
from collections import defaultdict
import numpy as np

from ..resources import data_path

class AtomInfo(NamedTuple):
    """Store atom information."""
    radius: float
    is_polar: bool

class ResidueLibrary:
    """Handles atom radii."""
    def __init__(self, library_input: Path = None):
        if library_input is None:
            with data_path("vdw.radii") as default_path:
                library_text = default_path.read_text()
        else:
            library_text = Path(library_input).read_text()
        self.residue_atoms = defaultdict(dict)
        self._parse_library(library_text)
        self.radii_matrix = self._build_radii_matrix()
        self.radii_matrix_atom14 = self._build_radii_matrix_atom14()

    def _parse_library(self, text: str):
        current_residue = None
        for line in text.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('RESIDUE'):
                parts = line.split()
                current_residue = parts[2]
            elif line.startswith('ATOM'):
                if current_residue:
                    atom_name = line[5:9].strip()
                    parts = line[9:].strip().split()
                    radius = float(parts[0])
                    is_polar = bool(int(parts[1]))
                    self.residue_atoms[current_residue][atom_name] = AtomInfo(radius, is_polar)

    def get_radius(self, residue: str, atom: str, element: str = None) -> float:
        atom_info = self.residue_atoms.get(residue, {}).get(atom)
        if atom_info:
            return atom_info.radius
        # Use default radius for element if not found
        element_radii = {
            'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'S': 1.80,
            'P': 1.80, 'FE': 1.47, 'ZN': 1.39, 'MG': 1.73
        }
        return element_radii.get(element.upper(), 1.80)

    def is_polar(self, residue: str, atom: str) -> bool:
        atom_info = self.residue_atoms.get(residue, {}).get(atom)
        return bool(atom_info and atom_info.is_polar)
    
    def _build_radii_matrix(self) -> np.ndarray:
        """Build matrix of atom radii for all residue types from residue_constants.
        Array of shape [n_residue_types, n_atoms] containing radii values
        """
        from . import residue_constants

        radii_by_aa: Dict[str, list[float]] = {}
        for aa in residue_constants.restypes:
            res_name = residue_constants.restype_1to3[aa]
            radii_by_aa[aa] = [
                self.get_radius(res_name, atom_name, atom_name[0])
                for atom_name in residue_constants.atom_types
            ]
        return np.array([radii_by_aa[aa] for aa in residue_constants.restypes])

    def _build_radii_matrix_atom14(self) -> np.ndarray:
        """Same as ``radii_matrix`` but packed into the atom14 layout.

        Empty atom14 slots (padding past the residue's real atom count) get
        radius 0 so the SASA probe radius collapses to the probe alone — then
        the atom14 mask zeros them out entirely.
        """
        from . import residue_constants

        radii = np.zeros((residue_constants.restype_num, 14), dtype=np.float32)
        for restype_idx, aa in enumerate(residue_constants.restypes):
            res_name = residue_constants.restype_1to3[aa]
            atom14_names = residue_constants.restype_name_to_atom14_names[res_name]
            for slot, atom_name in enumerate(atom14_names):
                if not atom_name:
                    continue
                radii[restype_idx, slot] = self.get_radius(
                    res_name, atom_name, atom_name[0]
                )
        return radii

default_library = ResidueLibrary()
