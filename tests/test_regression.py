from pathlib import Path

import pytest


def test_cpu_and_jax_predictions_stay_reasonably_close():
    pytest.importorskip("jax")
    pytest.importorskip("prodigy_prot")
    pytest.importorskip("freesasa")

    from protein_affinity_gpu import predict_binding_affinity, predict_binding_affinity_jax

    fixture = Path("benchmarks/fixtures/1A2K.pdb")
    cpu_result = predict_binding_affinity(fixture, selection="A,B")
    jax_result = predict_binding_affinity_jax(fixture, selection="A,B")

    assert abs(float(cpu_result.binding_affinity) - float(jax_result.binding_affinity)) < 0.75
    assert abs(
        cpu_result.contact_types.to_dict()["IC"] - jax_result.contact_types.to_dict()["IC"]
    ) < 10.0
