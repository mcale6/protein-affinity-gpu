from protein_affinity_gpu.utils.resources import read_text_resource


def test_can_read_packaged_resource():
    contents = read_text_resource("vdw.radii")
    assert "RESIDUE" in contents
