"""Tests for the shared benchmark helpers in benchmarks/sasa/sasa_benchmark.py."""
from __future__ import annotations

from pathlib import Path

from benchmarks.sasa import sasa_benchmark


def test_backends_registry_has_six_entries():
    assert set(sasa_benchmark.BACKENDS) == {
        "cpu",
        "jax-single", "jax-batch", "jax-scan",
        "tinygrad-single", "tinygrad-batch",
    }


def test_default_target_sets_are_disjoint_by_backend():
    assert "cpu" in sasa_benchmark.LOCAL_DEFAULT_TARGETS
    assert "cpu" not in sasa_benchmark.GPU_DEFAULT_TARGETS
    assert set(sasa_benchmark.LOCAL_DEFAULT_TARGETS).issubset(sasa_benchmark.BACKENDS)
    assert set(sasa_benchmark.GPU_DEFAULT_TARGETS).issubset(sasa_benchmark.BACKENDS)


def test_load_and_write_manifest_roundtrip(tmp_path: Path):
    original = tmp_path / "manifest.tsv"
    original.write_text(
        "pdb_id\tchain1\tchain2\n1ABC\tA\tB\n1XYZ\tA\tC\n"
    )
    rows = sasa_benchmark.load_manifest_rows(original)
    assert rows == [
        {"pdb_id": "1ABC", "chain1": "A", "chain2": "B"},
        {"pdb_id": "1XYZ", "chain1": "A", "chain2": "C"},
    ]
    copy_path = tmp_path / "copy.tsv"
    sasa_benchmark.write_manifest_rows(rows[:1], copy_path)
    assert copy_path.exists()
    reread = sasa_benchmark.load_manifest_rows(copy_path)
    assert reread == rows[:1]


def test_materialize_manifest_limits_rows(tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text(
        "pdb_id\tchain1\tchain2\n1ABC\tA\tB\n1XYZ\tA\tC\n2DEF\tA\tB\n"
    )
    subset = sasa_benchmark.materialize_manifest(
        manifest, tmp_path / "out", limit=2
    )
    assert subset != manifest
    rows = sasa_benchmark.load_manifest_rows(subset)
    assert [r["pdb_id"] for r in rows] == ["1ABC", "1XYZ"]


def test_materialize_manifest_returns_original_when_no_limit(tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("pdb_id\tchain1\tchain2\n1ABC\tA\tB\n")
    assert sasa_benchmark.materialize_manifest(
        manifest, tmp_path / "out", limit=None
    ) == manifest
    assert sasa_benchmark.materialize_manifest(
        manifest, tmp_path / "out", limit=0
    ) == manifest


def test_resolve_structure_path_finds_pdb_or_cif(tmp_path: Path):
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    (structures_dir / "1abc.pdb").write_text("HEADER\n")
    # File lookup is case-insensitive on macOS; just verify we resolve to
    # the right suffix/stem regardless of which casing the filesystem keeps.
    resolved = sasa_benchmark.resolve_structure_path(structures_dir, "1ABC")
    assert resolved.stem.upper() == "1ABC"
    assert resolved.suffix == ".pdb"
    assert resolved.exists()


def test_ensure_manifest_structures_counts_existing(tmp_path: Path):
    manifest = tmp_path / "manifest.tsv"
    manifest.write_text("pdb_id\tchain1\tchain2\n1ABC\tA\tB\n")
    structures_dir = tmp_path / "structures"
    structures_dir.mkdir()
    (structures_dir / "1ABC.pdb").write_text("HEADER\n")
    summary = sasa_benchmark.ensure_manifest_structures(manifest, structures_dir)
    assert summary == {"requested": 1, "downloaded": 0, "existing": 1}


def test_clear_tinygrad_caches_is_safe_without_backends():
    # Should not raise even if the optional caches can't be imported.
    sasa_benchmark.clear_tinygrad_caches()


def test_snapshot_memory_returns_dict():
    snap = sasa_benchmark.snapshot_memory()
    assert isinstance(snap, dict)
    # rss_mb is always set on darwin/linux via resource.getrusage.
    assert "rss_mb" in snap


def test_extract_scalar_metrics_flattens_expected_fields():
    result = {
        "ba_val": -10.0, "kd": 1e-9,
        "contacts": {
            "AA": 10.0, "CC": 1.0, "PP": 2.0, "AC": 3.0, "AP": 4.0, "CP": 5.0,
            "IC": 25.0, "chargedC": 9.0, "polarC": 11.0, "aliphaticC": 17.0,
        },
        "nis": {"aliphatic": 11.0, "charged": 22.0, "polar": 33.0},
        "sasa_data": [{"atom_sasa": 1.0}, {"atom_sasa": 2.5}],
    }
    metrics = sasa_benchmark.extract_scalar_metrics(result)
    assert metrics["ba_val"] == -10.0
    assert metrics["nis_charged"] == 22.0
    assert metrics["contacts_ic"] == 25.0
    assert metrics["sasa_sum"] == 3.5
