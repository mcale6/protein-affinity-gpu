import json
from pathlib import Path

from benchmarks import compare as compare_cli


class DummyResult:
    def __init__(self, structure_id: str, offset: float):
        self.structure_id = structure_id
        self.offset = offset

    def to_dict(self):
        return {
            "structure_id": self.structure_id,
            "ba_val": -10.0 + self.offset,
            "kd": 1.0e-9 + self.offset,
            "contacts": {
                "AA": 10.0 + self.offset,
                "CC": 1.0 + self.offset,
                "PP": 2.0 + self.offset,
                "AC": 3.0 + self.offset,
                "AP": 4.0 + self.offset,
                "CP": 5.0 + self.offset,
                "IC": 25.0 + self.offset,
                "chargedC": 9.0 + self.offset,
                "polarC": 11.0 + self.offset,
                "aliphaticC": 17.0 + self.offset,
            },
            "nis": {
                "aliphatic": 11.0 + self.offset,
                "charged": 22.0 + self.offset,
                "polar": 33.0 + self.offset,
            },
            "sasa_data": [],
        }


def test_run_comparison_writes_summary_and_rows(monkeypatch, tmp_path: Path):
    manifest = tmp_path / "dataset.tsv"
    manifest.write_text(
        "\n".join(
            [
                "pdb_id\tchain1\tchain2\tdifficulty\tn_virtual_xl\tbest_lrmsd_no_xl\tbest_lrmsd_7xl_10best\tbest_lrmsd_7xl\tref_table",
                "1MQ8\tA\tB\tM\t23\t55.1\t4.3\t4.3\tKahraman-2013-T3",
                "1JK9\tA\tB\tD\t31\t40.3\t6.8\t7.2\tKahraman-2013-T3",
            ]
        )
        + "\n"
    )

    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    for pdb_id in ("1MQ8", "1JK9"):
        (structures_dir / f"{pdb_id}.pdb").write_text("HEADER dummy\nEND\n")

    def cpu_predictor(struct_path, **kwargs):
        return DummyResult(Path(struct_path).stem, offset=0.0)

    def jax_predictor(struct_path, **kwargs):
        return DummyResult(Path(struct_path).stem, offset=0.5)

    monkeypatch.setattr(compare_cli, "predict_binding_affinity", cpu_predictor)
    monkeypatch.setattr(compare_cli, "_load_jax_predictor", lambda: jax_predictor)
    monkeypatch.setattr(compare_cli, "get_jax_backend_name", lambda: "gpu")
    monkeypatch.setattr(compare_cli, "tinygrad_available", lambda: False)
    monkeypatch.setattr(compare_cli, "get_tinygrad_backend_name", lambda: "unavailable")

    rows_path, summary_path, summary = compare_cli.run_comparison(
        manifest_path=manifest,
        structures_dir=structures_dir,
        output_dir=tmp_path / "output",
        repeats=2,
        make_plots=False,
    )

    assert rows_path.exists()
    assert summary_path.exists()
    assert summary["completed_structures"] == 2
    assert summary["failed_structures"] == 0
    assert summary["jax_backend"] == "gpu"
    assert summary["metrics"]["ba_val"]["count"] == 2
    assert summary["metrics"]["contacts_aa"]["count"] == 2
    assert summary["timing"]["count"] == 2

    saved = json.loads(summary_path.read_text())
    assert saved["completed_structures"] == 2
    assert "ba_val" in saved["metrics"]
    assert "contacts_aa" in saved["metrics"]
