import json
from pathlib import Path

import numpy as np

from protein_affinity_gpu.results import ContactAnalysis, ProdigyResults


def test_results_serialize_and_save(tmp_path: Path):
    sasa_dtype = [
        ("chain", "U2"),
        ("resname", "U3"),
        ("resindex", "i4"),
        ("atomname", "U4"),
        ("atom_sasa", "f4"),
        ("relative_sasa", "f4"),
    ]
    sasa_data = np.array([("A", "ALA", 1, "CA", 12.5, 0.8)], dtype=sasa_dtype)
    results = ProdigyResults(
        structure_id="demo",
        contact_types=ContactAnalysis([1, 2, 3, 4, 5, 6]),
        binding_affinity=np.float32(-12.3),
        dissociation_constant=np.float32(1.2e-9),
        nis_aliphatic=np.float32(11.1),
        nis_charged=np.float32(22.2),
        nis_polar=np.float32(33.3),
        sasa_data=sasa_data,
    )

    output_path = results.save_results(tmp_path)
    payload = json.loads(output_path.read_text())
    assert payload["structure_id"] == "demo"
    assert payload["contacts"]["IC"] == 21.0
    assert payload["sasa_data"][0]["atomname"] == "CA"
