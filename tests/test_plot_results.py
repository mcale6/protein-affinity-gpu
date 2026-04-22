"""Tests for benchmarks/plot_results.py — merge multiple results.csv files on
``pdb_id`` and render the comparison figure from the merged rows.
"""
from __future__ import annotations

import csv
from pathlib import Path

from benchmarks import plot_results


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def test_load_rows_merges_on_pdb_id(tmp_path: Path):
    local_csv = tmp_path / "local.csv"
    gpu_csv = tmp_path / "gpu.csv"

    local_cols = [
        "pdb_id", "chain1", "chain2", "n_atoms_atom14", "device",
        "cpu_status", "cpu_warm_mean_seconds", "cpu_ba_val", "cpu_sasa_sum",
    ]
    gpu_cols = [
        "pdb_id", "chain1", "chain2", "n_atoms_atom14", "device",
        "jax-batch_status", "jax-batch_warm_mean_seconds",
        "jax-batch_ba_val", "jax-batch_sasa_sum",
    ]
    _write_csv(local_csv, local_cols, [
        {"pdb_id": "1MQ8", "chain1": "A", "chain2": "B",
         "n_atoms_atom14": 1234, "device": "cpu",
         "cpu_status": "ok", "cpu_warm_mean_seconds": 0.5,
         "cpu_ba_val": -10.0, "cpu_sasa_sum": 100.0},
        {"pdb_id": "1JK9", "chain1": "A", "chain2": "B",
         "n_atoms_atom14": 2000, "device": "cpu",
         "cpu_status": "ok", "cpu_warm_mean_seconds": 0.7,
         "cpu_ba_val": -11.0, "cpu_sasa_sum": 110.0},
    ])
    _write_csv(gpu_csv, gpu_cols, [
        {"pdb_id": "1MQ8", "chain1": "A", "chain2": "B",
         "n_atoms_atom14": 1234, "device": "gpu",
         "jax-batch_status": "ok", "jax-batch_warm_mean_seconds": 0.05,
         "jax-batch_ba_val": -10.1, "jax-batch_sasa_sum": 100.3},
        {"pdb_id": "1JK9", "chain1": "A", "chain2": "B",
         "n_atoms_atom14": 2000, "device": "gpu",
         "jax-batch_status": "ok", "jax-batch_warm_mean_seconds": 0.08,
         "jax-batch_ba_val": -11.1, "jax-batch_sasa_sum": 110.4},
    ])

    rows, backends = plot_results.load_rows_from_csvs([local_csv, gpu_csv])
    assert len(rows) == 2
    assert set(backends) == {"cpu", "jax-batch"}
    by_pdb = {r["pdb_id"]: r for r in rows}
    assert by_pdb["1MQ8"]["cpu_ba_val"] == -10.0
    assert by_pdb["1MQ8"]["jax-batch_ba_val"] == -10.1
    assert by_pdb["1JK9"]["cpu_sasa_sum"] == 110.0
    assert by_pdb["1JK9"]["jax-batch_sasa_sum"] == 110.4


def test_plot_figure_writes_png(tmp_path: Path):
    csv_path = tmp_path / "results.csv"
    columns = [
        "pdb_id", "chain1", "chain2", "n_atoms_atom14", "device",
        "cpu_status", "cpu_warm_mean_seconds",
        "cpu_ba_val", "cpu_sasa_sum",
        "cpu_contacts_ic", "cpu_contacts_charged", "cpu_contacts_polar",
        "cpu_contacts_aliphatic",
        "cpu_nis_aliphatic", "cpu_nis_charged", "cpu_nis_polar",
        "jax-batch_status", "jax-batch_warm_mean_seconds",
        "jax-batch_ba_val", "jax-batch_sasa_sum",
        "jax-batch_contacts_ic", "jax-batch_contacts_charged",
        "jax-batch_contacts_polar", "jax-batch_contacts_aliphatic",
        "jax-batch_nis_aliphatic", "jax-batch_nis_charged",
        "jax-batch_nis_polar",
    ]
    rows = [
        {"pdb_id": f"1ABC{i}", "chain1": "A", "chain2": "B",
         "n_atoms_atom14": 1000 + i * 200, "device": "cpu",
         "cpu_status": "ok", "cpu_warm_mean_seconds": 0.5 + i * 0.1,
         "cpu_ba_val": -10.0 - i, "cpu_sasa_sum": 100.0 + i,
         "cpu_contacts_ic": 25.0, "cpu_contacts_charged": 9.0,
         "cpu_contacts_polar": 11.0, "cpu_contacts_aliphatic": 17.0,
         "cpu_nis_aliphatic": 11.0, "cpu_nis_charged": 22.0,
         "cpu_nis_polar": 33.0,
         "jax-batch_status": "ok", "jax-batch_warm_mean_seconds": 0.05 + i * 0.01,
         "jax-batch_ba_val": -10.0 - i + 0.1, "jax-batch_sasa_sum": 100.0 + i + 0.3,
         "jax-batch_contacts_ic": 25.0, "jax-batch_contacts_charged": 9.0,
         "jax-batch_contacts_polar": 11.0, "jax-batch_contacts_aliphatic": 17.0,
         "jax-batch_nis_aliphatic": 11.0, "jax-batch_nis_charged": 22.0,
         "jax-batch_nis_polar": 33.0}
        for i in range(3)
    ]
    _write_csv(csv_path, columns, rows)

    fig_path = tmp_path / "figure.png"
    loaded, backends = plot_results.load_rows_from_csvs([csv_path])
    plot_results.plot_figure(loaded, backends, fig_path)
    assert fig_path.exists()
    assert fig_path.stat().st_size > 0


def test_cli_rejects_missing_csv(tmp_path: Path, capsys):
    missing = tmp_path / "does_not_exist.csv"
    try:
        plot_results.main([str(missing), "--output-dir", str(tmp_path)])
    except SystemExit as exc:
        assert exc.code != 0
    else:
        raise AssertionError("CLI should have exited on missing CSV")
