from pathlib import Path

from Bio.PDB import PDBParser

from protein_affinity_gpu.cpu import _select_structure_chains


def test_select_structure_chains_detaches_unselected_chains(tmp_path: Path):
    pdb_text = """\
ATOM      1  N   ALA A   1      11.104  13.207  12.678  1.00 20.00           N
ATOM      2  CA  ALA A   1      12.560  13.150  12.500  1.00 20.00           C
ATOM      3  N   GLY B   1      14.300  11.530  11.650  1.00 20.00           N
ATOM      4  CA  GLY B   1      14.910  10.187  11.360  1.00 20.00           C
ATOM      5  N   SER C   1      16.300  10.530  10.650  1.00 20.00           N
ATOM      6  CA  SER C   1      16.910   9.187  10.360  1.00 20.00           C
TER
END
"""
    structure_path = tmp_path / "three_chain.pdb"
    structure_path.write_text(pdb_text)

    structure = PDBParser(QUIET=True).get_structure("mini", str(structure_path))
    model = structure[0]

    selected = _select_structure_chains(model, ["A", "C"])

    assert [chain.id for chain in model] == ["A", "B", "C"]
    assert [chain.id for chain in selected] == ["A", "C"]
