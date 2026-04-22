"""Smoke tests for the local benchmark harness.

Mocks every backend loader so the test runs without JAX/tinygrad or any GPU,
then verifies the unified CSV + summary JSON are produced with the expected
row/column shape.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from benchmarks import benchmark as benchmark_cli
from benchmarks.sasa import sasa_benchmark


class DummyResult:
    def __init__(self, structure_id: str, offset: float = 0.0):
        self.structure_id = structure_id
        self.offset = offset

    def to_dict(self):
        return {
            "structure_id": self.structure_id,
            "ba_val": -10.0 + self.offset,
            "kd": 1e-9 + self.offset,
            "contacts": {
                "AA": 10.0, "CC": 1.0, "PP": 2.0, "AC": 3.0, "AP": 4.0, "CP": 5.0,
                "IC": 25.0, "chargedC": 9.0, "polarC": 11.0, "aliphaticC": 17.0,
            },
            "nis": {"aliphatic": 11.0, "charged": 22.0, "polar": 33.0},
            "sasa_data": [{"atom_sasa": 1.0 + self.offset}],
        }


@pytest.fixture
def manifest_and_structures(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("pdb_id\tchain1\tchain2\n1ABC\tA\tB\n1XYZ\tA\tB\n")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    for pdb_id in ("1ABC", "1XYZ"):
        (structures_dir / f"{pdb_id}.pdb").write_text("HEADER dummy\nEND\n")
    return manifest, structures_dir


def test_run_benchmark_writes_results_and_summary(
    monkeypatch, tmp_path: Path, manifest_and_structures
):
    manifest, structures_dir = manifest_and_structures

    def fake_loader():
        return lambda struct_path, **_: DummyResult(Path(struct_path).stem)

    fake_backends = {
        name: sasa_benchmark.BackendSpec(name, fake_loader, display)
        for name, display in (
            ("cpu", "CPU"),
            ("tinygrad-single", "TG-S"),
            ("tinygrad-batch", "TG-B"),
        )
    }
    monkeypatch.setattr(sasa_benchmark, "BACKENDS", fake_backends)
    monkeypatch.setattr(sasa_benchmark, "count_atom14_atoms", lambda p, s: 1234)

    summary = sasa_benchmark.run_benchmark(
        manifest_path=manifest,
        structures_dir=structures_dir,
        output_dir=tmp_path / "output",
        backends=("cpu", "tinygrad-single", "tinygrad-batch"),
        repeats=2,
        device="test-cpu",
    )

    rows_path = Path(summary["artifacts"]["rows_csv"])
    summary_path = Path(summary["artifacts"]["summary_json"])
    assert rows_path.exists()
    assert summary_path.exists()
    assert summary["device"] == "test-cpu"

    with rows_path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    first = rows[0]
    assert int(first["n_atoms_atom14"]) == 1234
    for name in ("cpu", "tinygrad-single", "tinygrad-batch"):
        assert first[f"{name}_status"] == "ok"
        assert first[f"{name}_ba_val"]

    saved = json.loads(summary_path.read_text())
    for name in ("cpu", "tinygrad-single", "tinygrad-batch"):
        assert saved["per_backend"][name]["completed"] == 2
        assert saved["per_backend"][name]["failed"] == 0


def test_local_cli_defaults_to_local_targets():
    parser = benchmark_cli.build_parser()
    args = parser.parse_args(["--manifest", "/dev/null"])
    assert tuple(args.targets) == sasa_benchmark.LOCAL_DEFAULT_TARGETS


def test_unknown_backend_rejected(tmp_path: Path, manifest_and_structures):
    manifest, structures_dir = manifest_and_structures
    with pytest.raises(ValueError, match="Unknown backend"):
        sasa_benchmark.run_benchmark(
            manifest_path=manifest,
            structures_dir=structures_dir,
            output_dir=tmp_path / "output",
            backends=("no-such-backend",),
        )
