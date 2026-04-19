import math
from pathlib import Path

import pytest


def test_tinygrad_prediction_finite_and_close_to_cpu():
    pytest.importorskip("tinygrad")
    pytest.importorskip("prodigy_prot")
    pytest.importorskip("freesasa")

    from protein_affinity_gpu import predict_binding_affinity, predict_binding_affinity_tinygrad

    fixture = Path("benchmarks/fixtures/1A2K.pdb")
    tg_result = predict_binding_affinity_tinygrad(fixture, selection="A,B")

    dg = float(tg_result.binding_affinity)
    kd = float(tg_result.dissociation_constant)
    assert math.isfinite(dg)
    assert math.isfinite(kd)
    assert kd > 0

    cpu_result = predict_binding_affinity(fixture, selection="A,B")
    assert abs(float(cpu_result.binding_affinity) - dg) < 0.75
    assert abs(
        cpu_result.contact_types.to_dict()["IC"] - tg_result.contact_types.to_dict()["IC"]
    ) < 10.0
