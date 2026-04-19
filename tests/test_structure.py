from pathlib import Path

from protein_affinity_gpu.structure import load_complex


def test_load_complex_sanitizes_hydrogens_and_waters(tmp_path: Path):
    pdb_text = """\
ATOM      1  N   ALA A   1      11.104  13.207  12.678  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.150  12.500  1.00 20.00           C
ATOM      3  C   ALA A   1      13.055  11.714  12.210  1.00 20.00           C
ATOM      4  O   ALA A   1      12.351  10.751  12.480  1.00 20.00           O
ATOM      5  H   ALA A   1      10.500  13.800  12.900  1.00 20.00           H
HETATM    6  O   HOH A 201      10.000  10.000  10.000  1.00 20.00           O
ATOM      7  N   GLY B   1      14.300  11.530  11.650  1.00 20.00           N
ATOM      8  CA  GLY B   1      14.910  10.187  11.360  1.00 20.00           C
ATOM      9  C   GLY B   1      16.409  10.264  11.678  1.00 20.00           C
ATOM     10  O   GLY B   1      17.032  11.321  11.532  1.00 20.00           O
TER
END
"""
    structure_path = tmp_path / "mini_complex.pdb"
    structure_path.write_text(pdb_text)

    target, binder = load_complex(structure_path, selection="A,B")
    assert target.atom_mask.sum() == 4.0
    assert binder.atom_mask.sum() == 4.0
    assert len(target.aatype) == 1
    assert len(binder.aatype) == 1
