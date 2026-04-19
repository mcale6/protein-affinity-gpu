from protein_affinity_gpu.utils import residue_constants
from protein_affinity_gpu.utils.residue_library import ResidueLibrary


def test_residue_library_builds_radii_matrix():
    library = ResidueLibrary()
    assert library.radii_matrix.shape == (
        len(residue_constants.restypes),
        residue_constants.atom_type_num,
    )
    assert library.get_radius("ALA", "CA", "C") > 0
