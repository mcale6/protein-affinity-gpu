from pathlib import Path

from benchmarks import sasa_benchmark


def test_run_sasa_benchmark_writes_summary_artifacts(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("pdb_id\tchain1\tchain2\n1ABC\tA\tB\n")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    (structures_dir / "1ABC.pdb").write_text("HEADER\n")

    def fake_run_benchmark(**kwargs):
        output_dir = kwargs["output_dir"]
        report = {
            "results": [
                {
                    "structure_id": "1ABC",
                    "target": "cpu",
                    "status": "ok",
                    "n_atoms": 42,
                    "cold_time_seconds": 1.25,
                    "warm_mean_seconds": 0.012,
                }
            ]
        }
        report_path = output_dir / "benchmark_results.json"
        report_path.write_text("{}")
        return report_path, report

    monkeypatch.setattr(sasa_benchmark, "run_experimental_benchmark", fake_run_benchmark)

    summary = sasa_benchmark.run_sasa_benchmark(
        manifest_path=manifest,
        structures_dir=structures_dir,
        output_dir=tmp_path / "output",
        repeats=2,
        targets=("cpu",),
        limit=1,
    )

    assert summary["ok"] == 1
    assert summary["error"] == 0
    assert summary["download"]["existing"] == 1
    assert Path(summary["artifacts"]["summary_json"]).exists()
    assert Path(summary["artifacts"]["rows_csv"]).exists()
    assert Path(summary["artifacts"]["warm_ms_wide_csv"]).exists()
    assert Path(summary["artifacts"]["plot_png"]).exists()
    assert (tmp_path / "output" / "manifest_subset.tsv").exists()


def test_parse_targets_accepts_csv_and_sequences():
    assert sasa_benchmark.parse_targets("cpu,jax") == ("cpu", "jax")
    assert sasa_benchmark.parse_targets(("cpu", "tinygrad")) == ("cpu", "tinygrad")
