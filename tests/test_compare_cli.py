import csv
import json
from pathlib import Path

from benchmarks import compare as compare_cli
from benchmarks.compare import BackendSpec


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
            "sasa_data": [{"atom_sasa": 1.0 + self.offset}],
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

    def tinygrad_batch_predictor(struct_path, **kwargs):
        return DummyResult(Path(struct_path).stem, offset=0.2)

    def tinygrad_single_predictor(struct_path, **kwargs):
        return DummyResult(Path(struct_path).stem, offset=0.3)

    fake_backends = {
        "cpu": BackendSpec("cpu", lambda: cpu_predictor, "CPU"),
        "tinygrad-batch": BackendSpec("tinygrad-batch", lambda: tinygrad_batch_predictor, "TG-B"),
        "tinygrad-single": BackendSpec("tinygrad-single", lambda: tinygrad_single_predictor, "TG-S"),
    }
    monkeypatch.setattr(compare_cli, "BACKENDS", fake_backends)
    monkeypatch.setattr(compare_cli, "count_atom14_atoms", lambda path, selection: 1234)

    rows_path, summary_path, summary = compare_cli.run_comparison(
        manifest_path=manifest,
        structures_dir=structures_dir,
        output_dir=tmp_path / "output",
        backends=["cpu", "tinygrad-batch", "tinygrad-single"],
        repeats=2,
        make_plot=False,
    )

    assert rows_path.exists()
    assert summary_path.exists()
    assert summary["repeats"] == 2
    assert summary["backends"] == ["cpu", "tinygrad-batch", "tinygrad-single"]
    for name in ("cpu", "tinygrad-batch", "tinygrad-single"):
        entry = summary["per_backend"][name]
        assert entry["completed"] == 2
        assert entry["failed"] == 0

    saved = json.loads(summary_path.read_text())
    assert saved["per_backend"]["cpu"]["completed"] == 2

    with rows_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    first = rows[0]
    assert first["cpu_status"] == "ok"
    assert first["tinygrad-batch_status"] == "ok"
    assert first["tinygrad-single_status"] == "ok"
    assert float(first["cpu_ba_val"]) == -10.0
    assert float(first["tinygrad-batch_contacts_ic"]) == 25.2
    assert int(first["n_atoms_atom14"]) == 1234
