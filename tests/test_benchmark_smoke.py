from pathlib import Path

from benchmarks import benchmark as benchmark_cli


class DummyResult:
    def __init__(self, structure_id: str):
        self.structure_id = structure_id

    def to_dict(self):
        return {"structure_id": self.structure_id, "ba_val": -1.0}


def test_benchmark_harness_runs_and_skips_cuda(monkeypatch, tmp_path: Path):
    fixture = Path("benchmarks/fixtures/1A2K.pdb")
    monkeypatch.setattr(
        benchmark_cli,
        "predict_binding_affinity",
        lambda struct_path, **kwargs: DummyResult(Path(struct_path).stem),
    )
    monkeypatch.setattr(benchmark_cli, "cuda_available", lambda: False)

    output_path, report = benchmark_cli.run_benchmark(
        input_path=fixture,
        output_dir=tmp_path,
        repeats=2,
        targets=("cpu", "cuda"),
    )

    assert output_path.exists()
    assert any(result["target"] == "cpu" and result["status"] == "ok" for result in report["results"])
    assert any(result["target"] == "cuda" and result["status"] == "skipped" for result in report["results"])
