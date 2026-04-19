import protein_affinity_gpu
from benchmarks import benchmark as benchmark_script
from benchmarks import compare as compare_script
from protein_affinity_gpu.cli import predict as predict_cli


def test_top_level_import_smoke():
    assert protein_affinity_gpu.__version__ == "1.6.9"
    assert callable(protein_affinity_gpu.predict_binding_affinity)
    assert callable(protein_affinity_gpu.predict_binding_affinity_jax)
    assert callable(protein_affinity_gpu.predict_binding_affinity_tinygrad)
    assert callable(protein_affinity_gpu.load_complex)


def test_cli_modules_import():
    assert callable(predict_cli.main)
    assert callable(benchmark_script.main)
    assert callable(compare_script.main)
