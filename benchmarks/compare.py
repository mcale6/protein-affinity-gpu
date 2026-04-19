#!/usr/bin/env python3

import argparse
import csv
import json
import logging
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from benchmarks.benchmark import cuda_available, tinygrad_available  # noqa: E402
    from protein_affinity_gpu.cli.predict import (  # noqa: E402
        _load_jax_predictor,
        _load_tinygrad_predictor,
    )
    from protein_affinity_gpu.cpu import predict_binding_affinity  # noqa: E402
    from protein_affinity_gpu.resources import format_duration  # noqa: E402
    from protein_affinity_gpu.results import NumpyEncoder  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guidance
    raise SystemExit(
        f"Missing Python dependency '{exc.name}'. "
        "Use the repo virtualenv or install dependencies with "
        "\".venv/bin/python -m pip install -e '.[compare]'\"."
    ) from exc

LOGGER = logging.getLogger(__name__)
SUPPORTED_STRUCTURE_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")
CONTACT_FIELD_MAP = {
    "AA": "contacts_aa",
    "CC": "contacts_cc",
    "PP": "contacts_pp",
    "AC": "contacts_ac",
    "AP": "contacts_ap",
    "CP": "contacts_cp",
    "IC": "contacts_ic",
    "chargedC": "contacts_charged",
    "polarC": "contacts_polar",
    "aliphaticC": "contacts_aliphatic",
}
OVERVIEW_METRICS = [
    {"name": "ba_val", "title": "Binding affinity (DG)"},
    {"name": "contacts_ic", "title": "Interface contacts"},
    {"name": "nis_aliphatic", "title": "NIS aliphatic"},
    {"name": "nis_charged", "title": "NIS charged"},
    {"name": "nis_polar", "title": "NIS polar"},
]
RAW_CONTACT_METRICS = [
    {"name": "contacts_aa", "title": "AA contacts"},
    {"name": "contacts_cc", "title": "CC contacts"},
    {"name": "contacts_pp", "title": "PP contacts"},
    {"name": "contacts_ac", "title": "AC contacts"},
    {"name": "contacts_ap", "title": "AP contacts"},
    {"name": "contacts_cp", "title": "CP contacts"},
]
DERIVED_CONTACT_METRICS = [
    {"name": "contacts_charged", "title": "Charged contacts"},
    {"name": "contacts_polar", "title": "Polar contacts"},
    {"name": "contacts_aliphatic", "title": "Aliphatic contacts"},
]
SUMMARY_METRICS = OVERVIEW_METRICS + RAW_CONTACT_METRICS + DERIVED_CONTACT_METRICS
BACKEND_VALUE_NAMES = [
    "ba_val",
    "kd",
    "nis_aliphatic",
    "nis_charged",
    "nis_polar",
    *CONTACT_FIELD_MAP.values(),
]


def get_jax_backend_name() -> str:
    try:
        import jax
    except ImportError:
        return "unavailable"
    return str(jax.default_backend()).lower()


def get_tinygrad_backend_name() -> str:
    try:
        from tinygrad import Device
    except Exception:
        return "unavailable"
    return str(Device.DEFAULT).lower()


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    manifest_path = Path(manifest_path)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [
            {key: (value.strip() if isinstance(value, str) else value) for key, value in row.items()}
            for row in reader
            if row.get("pdb_id")
        ]
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return rows


def resolve_structure_path(structures_dir: Path, pdb_id: str) -> Path:
    normalized = pdb_id.upper()
    for suffix in SUPPORTED_STRUCTURE_SUFFIXES:
        exact = structures_dir / f"{normalized}{suffix}"
        if exact.exists():
            return exact

        lower = structures_dir / f"{normalized.lower()}{suffix}"
        if lower.exists():
            return lower

    raise FileNotFoundError(f"Structure not found for {pdb_id} in {structures_dir}")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return float(value)


def _backend_metric_key(prefix: str, metric_name: str) -> str:
    return f"{prefix}_{metric_name}"


def _benchmark_predictor(
    predictor,
    structure_path: Path,
    repeats: int,
    selection: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    sphere_points: int,
) -> dict[str, Any]:
    timings = []
    last_result = None

    for _ in range(repeats):
        start_time = time.perf_counter()
        last_result = predictor(
            struct_path=structure_path,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
            save_results=False,
            quiet=True,
        )
        timings.append(time.perf_counter() - start_time)

    cold_time = timings[0]
    warm_times = timings[1:]
    warm_mean = mean(warm_times) if warm_times else cold_time
    return {
        "status": "ok",
        "cold_time_seconds": cold_time,
        "cold_time_formatted": format_duration(cold_time),
        "warm_times_seconds": warm_times,
        "warm_mean_seconds": warm_mean,
        "warm_mean_formatted": format_duration(warm_mean),
        "result": last_result.to_dict() if last_result is not None else None,
    }


def _run_backend(
    predictor,
    structure_path: Path,
    repeats: int,
    selection: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    sphere_points: int,
) -> dict[str, Any]:
    try:
        return _benchmark_predictor(
            predictor=predictor,
            structure_path=structure_path,
            repeats=repeats,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
        )
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        return {"status": "error", "error": str(exc)}


def _paired_success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if row["cpu_status"] == "ok" and row["jax_status"] == "ok"]


def _tinygrad_success_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        row
        for row in rows
        if row["cpu_status"] == "ok" and row.get("tinygrad_status") == "ok"
    ]


def _compute_metric_stats(rows: list[dict[str, Any]], cpu_key: str, jax_key: str) -> dict[str, Any]:
    cpu_values = np.asarray([row[cpu_key] for row in rows], dtype=float)
    jax_values = np.asarray([row[jax_key] for row in rows], dtype=float)
    diff = jax_values - cpu_values

    stats: dict[str, Any] = {
        "count": int(cpu_values.size),
        "cpu_mean": float(np.mean(cpu_values)),
        "jax_mean": float(np.mean(jax_values)),
        "mae": float(np.mean(np.abs(diff))),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "mean_signed_error": float(np.mean(diff)),
        "max_abs_error": float(np.max(np.abs(diff))),
    }
    if cpu_values.size > 1 and np.std(cpu_values) > 0 and np.std(jax_values) > 0:
        stats["pearson_r"] = float(np.corrcoef(cpu_values, jax_values)[0, 1])
    else:
        stats["pearson_r"] = None
    return stats


def _compute_metric_block(rows: list[dict[str, Any]], other_prefix: str) -> dict[str, Any]:
    return {
        metric["name"]: _compute_metric_stats(
            rows,
            _backend_metric_key("cpu", metric["name"]),
            _backend_metric_key(other_prefix, metric["name"]),
        )
        for metric in SUMMARY_METRICS
    }


def _compute_timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_cold = [row["cpu_cold_time_seconds"] for row in rows]
    cpu_warm = [row["cpu_warm_mean_seconds"] for row in rows]
    jax_cold = [row["jax_cold_time_seconds"] for row in rows]
    jax_warm = [row["jax_warm_mean_seconds"] for row in rows]
    speedups = [row["warm_speedup"] for row in rows if row["warm_speedup"] is not None]

    return {
        "count": len(rows),
        "cpu_cold_mean_seconds": float(mean(cpu_cold)),
        "cpu_warm_mean_seconds": float(mean(cpu_warm)),
        "jax_cold_mean_seconds": float(mean(jax_cold)),
        "jax_warm_mean_seconds": float(mean(jax_warm)),
        "mean_warm_speedup": float(mean(speedups)) if speedups else None,
        "median_warm_speedup": float(median(speedups)) if speedups else None,
        "max_warm_speedup": float(max(speedups)) if speedups else None,
        "min_warm_speedup": float(min(speedups)) if speedups else None,
    }


def build_summary(
    rows: list[dict[str, Any]],
    failures: list[dict[str, Any]],
    repeats: int,
    jax_backend: str,
    tinygrad_backend: str,
    manifest_path: Path,
    structures_dir: Path,
) -> dict[str, Any]:
    successful = _paired_success_rows(rows)
    tinygrad_successful = _tinygrad_success_rows(rows)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "structures_dir": str(structures_dir),
        "repeats": repeats,
        "jax_backend": jax_backend,
        "tinygrad_backend": tinygrad_backend,
        "completed_structures": len(successful),
        "completed_tinygrad_structures": len(tinygrad_successful),
        "failed_structures": len(failures),
        "metrics": {},
        "tinygrad_metrics": {},
        "timing": {},
        "tinygrad_timing": {},
        "failures": failures,
    }
    if successful:
        summary["metrics"] = _compute_metric_block(successful, "jax")
        summary["timing"] = _compute_timing_summary(successful)

    if tinygrad_successful:
        summary["tinygrad_metrics"] = _compute_metric_block(tinygrad_successful, "tinygrad")
        summary["tinygrad_timing"] = _compute_tinygrad_timing_summary(tinygrad_successful)
    return summary


def _compute_tinygrad_timing_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cpu_warm = [row["cpu_warm_mean_seconds"] for row in rows]
    tg_cold = [row["tinygrad_cold_time_seconds"] for row in rows]
    tg_warm = [row["tinygrad_warm_mean_seconds"] for row in rows]
    speedups = [row["tinygrad_warm_speedup"] for row in rows if row.get("tinygrad_warm_speedup") is not None]

    return {
        "count": len(rows),
        "cpu_warm_mean_seconds": float(mean(cpu_warm)),
        "tinygrad_cold_mean_seconds": float(mean(tg_cold)),
        "tinygrad_warm_mean_seconds": float(mean(tg_warm)),
        "mean_warm_speedup": float(mean(speedups)) if speedups else None,
        "median_warm_speedup": float(median(speedups)) if speedups else None,
        "max_warm_speedup": float(max(speedups)) if speedups else None,
        "min_warm_speedup": float(min(speedups)) if speedups else None,
    }


def _format_stat(value: float | None, precision: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{precision}f}"


def _import_matplotlib_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("Plot generation requires the optional 'matplotlib' dependency.") from exc
    return plt


def _add_identity_line(axis, x_values: np.ndarray, y_values: np.ndarray) -> None:
    minimum = min(np.min(x_values), np.min(y_values))
    maximum = max(np.max(x_values), np.max(y_values))
    padding = max((maximum - minimum) * 0.05, 1e-6)
    axis.plot(
        [minimum - padding, maximum + padding],
        [minimum - padding, maximum + padding],
        linestyle="--",
        linewidth=1.0,
        color="#6c757d",
        zorder=1,
    )


def _plot_metric_grid(
    rows: list[dict[str, Any]],
    metrics: list[dict[str, str]],
    summary_metrics: dict[str, Any],
    output_path: Path,
    title: str,
    other_prefix: str,
    other_label: str,
) -> None:
    plt = _import_matplotlib_pyplot()
    columns = 3 if len(metrics) > 4 else 2
    rows_count = int(math.ceil(len(metrics) / columns))
    figure, axes = plt.subplots(
        rows_count,
        columns,
        figsize=(columns * 5.2, rows_count * 4.4),
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes).ravel()

    for axis, metric in zip(axes, metrics):
        metric_name = metric["name"]
        cpu_key = _backend_metric_key("cpu", metric_name)
        other_key = _backend_metric_key(other_prefix, metric_name)
        cpu_values = np.asarray([row[cpu_key] for row in rows], dtype=float)
        other_values = np.asarray([row[other_key] for row in rows], dtype=float)
        axis.scatter(cpu_values, other_values, s=48, color="#1f77b4", alpha=0.9, zorder=2)
        _add_identity_line(axis, cpu_values, other_values)

        for row, cpu_value, other_value in zip(rows, cpu_values, other_values):
            axis.annotate(
                row["pdb_id"],
                (cpu_value, other_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        metric_stats = summary_metrics.get(metric_name, {})
        axis.text(
            0.03,
            0.97,
            "\n".join(
                [
                    f"n={metric_stats.get('count', 0)}",
                    f"r={_format_stat(metric_stats.get('pearson_r'))}",
                    f"MAE={_format_stat(metric_stats.get('mae'))}",
                    f"RMSE={_format_stat(metric_stats.get('rmse'))}",
                ]
            ),
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d7de"},
        )
        axis.set_title(metric["title"])
        axis.set_xlabel("CPU")
        axis.set_ylabel(other_label)
        axis.grid(alpha=0.25)

    for axis in axes[len(metrics) :]:
        axis.axis("off")

    figure.suptitle(title, fontsize=14)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _plot_timing(
    rows: list[dict[str, Any]],
    timing_summary: dict[str, Any],
    output_path: Path,
    title: str,
    other_prefix: str,
    other_label: str,
    speedup_key: str,
) -> None:
    plt = _import_matplotlib_pyplot()
    cpu_times = np.asarray([row["cpu_warm_mean_seconds"] for row in rows], dtype=float)
    other_times = np.asarray([row[f"{other_prefix}_warm_mean_seconds"] for row in rows], dtype=float)
    sorted_rows = sorted(rows, key=lambda row: row.get(speedup_key) or 0.0, reverse=True)

    figure, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    axes[0].scatter(cpu_times, other_times, s=48, color="#2ca02c", alpha=0.9, zorder=2)
    _add_identity_line(axes[0], cpu_times, other_times)
    for row, cpu_time, other_time in zip(rows, cpu_times, other_times):
        axes[0].annotate(
            row["pdb_id"],
            (cpu_time, other_time),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=7,
        )
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("CPU warm mean (s)")
    axes[0].set_ylabel(f"{other_label} warm mean (s)")
    axes[0].set_title("Warm timing comparison")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].text(
        0.03,
        0.97,
        "\n".join(
            [
                f"n={timing_summary.get('count', 0)}",
                f"mean speedup={_format_stat(timing_summary.get('mean_warm_speedup'))}x",
                f"median speedup={_format_stat(timing_summary.get('median_warm_speedup'))}x",
            ]
        ),
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d0d7de"},
    )

    labels = [row["pdb_id"] for row in sorted_rows]
    speedups = [row.get(speedup_key) or 0.0 for row in sorted_rows]
    axes[1].bar(labels, speedups, color="#ff7f0e")
    axes[1].axhline(1.0, linestyle="--", linewidth=1.0, color="#6c757d")
    axes[1].set_ylabel(f"CPU / {other_label} warm speedup")
    axes[1].set_title("Per-structure speedup")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].grid(axis="y", alpha=0.25)

    figure.suptitle(title, fontsize=14)
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def _write_rows_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return

    metric_fieldnames = [
        _backend_metric_key(prefix, metric_name)
        for prefix in ("cpu", "jax", "tinygrad")
        for metric_name in BACKEND_VALUE_NAMES
    ]
    fieldnames = [
        "pdb_id",
        "chain1",
        "chain2",
        "difficulty",
        "n_virtual_xl",
        "best_lrmsd_no_xl",
        "best_lrmsd_7xl_10best",
        "best_lrmsd_7xl",
        "ref_table",
        "structure_path",
        "cpu_status",
        "jax_status",
        "tinygrad_status",
        "cpu_error",
        "jax_error",
        "tinygrad_error",
        "ba_abs_diff",
        "tinygrad_ba_abs_diff",
        "contacts_ic_abs_diff",
        "tinygrad_contacts_ic_abs_diff",
        "cpu_cold_time_seconds",
        "cpu_warm_mean_seconds",
        "jax_cold_time_seconds",
        "jax_warm_mean_seconds",
        "tinygrad_cold_time_seconds",
        "tinygrad_warm_mean_seconds",
        "warm_speedup",
        "tinygrad_warm_speedup",
        *metric_fieldnames,
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: row.get(name) for name in fieldnames})


def _extract_backend_metrics(prefix: str, payload: dict[str, Any]) -> dict[str, Any]:
    if payload["status"] != "ok":
        return {_backend_metric_key(prefix, metric_name): None for metric_name in BACKEND_VALUE_NAMES}

    result = payload["result"]
    metrics = {
        _backend_metric_key(prefix, "ba_val"): float(result["ba_val"]),
        _backend_metric_key(prefix, "kd"): float(result["kd"]),
        _backend_metric_key(prefix, "nis_aliphatic"): float(result["nis"]["aliphatic"]),
        _backend_metric_key(prefix, "nis_charged"): float(result["nis"]["charged"]),
        _backend_metric_key(prefix, "nis_polar"): float(result["nis"]["polar"]),
    }
    metrics.update(
        {
            _backend_metric_key(prefix, field_name): float(result["contacts"][contact_key])
            for contact_key, field_name in CONTACT_FIELD_MAP.items()
        }
    )
    return metrics


def run_comparison(
    manifest_path: Path,
    structures_dir: Path,
    output_dir: Path,
    repeats: int = 3,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    require_gpu: bool = False,
    make_plots: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    if repeats < 1:
        raise ValueError("repeats must be at least 1")

    manifest_path = Path(manifest_path)
    structures_dir = Path(structures_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if require_gpu and not cuda_available():
        raise RuntimeError("No GPU-backed JAX device detected. Re-run without --require-gpu to allow fallback.")

    try:
        jax_predictor = _load_jax_predictor()
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError("JAX comparison requires the optional 'jax' and 'jaxlib' dependencies.") from exc

    tinygrad_predictor = _load_tinygrad_predictor() if tinygrad_available() else None

    rows = []
    failures = []
    for manifest_row in load_manifest(manifest_path):
        structure_path = resolve_structure_path(structures_dir, manifest_row["pdb_id"])
        selection = f"{manifest_row['chain1']},{manifest_row['chain2']}"
        cpu_payload = _run_backend(
            predictor=predict_binding_affinity,
            structure_path=structure_path,
            repeats=repeats,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
        )
        jax_payload = _run_backend(
            predictor=jax_predictor,
            structure_path=structure_path,
            repeats=repeats,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
        )
        if tinygrad_predictor is not None:
            tinygrad_payload = _run_backend(
                predictor=tinygrad_predictor,
                structure_path=structure_path,
                repeats=repeats,
                selection=selection,
                temperature=temperature,
                distance_cutoff=distance_cutoff,
                acc_threshold=acc_threshold,
                sphere_points=sphere_points,
            )
        else:
            tinygrad_payload = {"status": "skipped", "error": "tinygrad not installed"}

        row = dict(manifest_row)
        row.update(
            {
                "structure_path": str(structure_path),
                "cpu_status": cpu_payload["status"],
                "jax_status": jax_payload["status"],
                "tinygrad_status": tinygrad_payload["status"],
                "cpu_error": cpu_payload.get("error"),
                "jax_error": jax_payload.get("error"),
                "tinygrad_error": tinygrad_payload.get("error"),
                "cpu_cold_time_seconds": _safe_float(cpu_payload.get("cold_time_seconds")),
                "cpu_warm_mean_seconds": _safe_float(cpu_payload.get("warm_mean_seconds")),
                "jax_cold_time_seconds": _safe_float(jax_payload.get("cold_time_seconds")),
                "jax_warm_mean_seconds": _safe_float(jax_payload.get("warm_mean_seconds")),
                "tinygrad_cold_time_seconds": _safe_float(tinygrad_payload.get("cold_time_seconds")),
                "tinygrad_warm_mean_seconds": _safe_float(tinygrad_payload.get("warm_mean_seconds")),
            }
        )

        row.update(_extract_backend_metrics("cpu", cpu_payload))
        row.update(_extract_backend_metrics("jax", jax_payload))
        row.update(_extract_backend_metrics("tinygrad", tinygrad_payload))

        if cpu_payload["status"] == "ok" and jax_payload["status"] == "ok":
            row["ba_abs_diff"] = abs(row["jax_ba_val"] - row["cpu_ba_val"])
            row["contacts_ic_abs_diff"] = abs(row["jax_contacts_ic"] - row["cpu_contacts_ic"])
            row["warm_speedup"] = (
                row["cpu_warm_mean_seconds"] / row["jax_warm_mean_seconds"]
                if row["jax_warm_mean_seconds"]
                else None
            )
        else:
            row["ba_abs_diff"] = None
            row["contacts_ic_abs_diff"] = None
            row["warm_speedup"] = None

        if cpu_payload["status"] == "ok" and tinygrad_payload["status"] == "ok":
            row["tinygrad_ba_abs_diff"] = abs(row["tinygrad_ba_val"] - row["cpu_ba_val"])
            row["tinygrad_contacts_ic_abs_diff"] = abs(
                row["tinygrad_contacts_ic"] - row["cpu_contacts_ic"]
            )
            row["tinygrad_warm_speedup"] = (
                row["cpu_warm_mean_seconds"] / row["tinygrad_warm_mean_seconds"]
                if row["tinygrad_warm_mean_seconds"]
                else None
            )
        else:
            row["tinygrad_ba_abs_diff"] = None
            row["tinygrad_contacts_ic_abs_diff"] = None
            row["tinygrad_warm_speedup"] = None

        if cpu_payload["status"] != "ok" or jax_payload["status"] != "ok" or tinygrad_payload["status"] not in {"ok", "skipped"}:
            failures.append(
                {
                    "pdb_id": manifest_row["pdb_id"],
                    "structure_path": str(structure_path),
                    "cpu_status": cpu_payload["status"],
                    "jax_status": jax_payload["status"],
                    "tinygrad_status": tinygrad_payload["status"],
                    "cpu_error": cpu_payload.get("error"),
                    "jax_error": jax_payload.get("error"),
                    "tinygrad_error": tinygrad_payload.get("error"),
                }
            )

        rows.append(row)

    summary = build_summary(
        rows=rows,
        failures=failures,
        repeats=repeats,
        jax_backend=get_jax_backend_name(),
        tinygrad_backend=get_tinygrad_backend_name(),
        manifest_path=manifest_path,
        structures_dir=structures_dir,
    )
    rows_path = output_dir / "cpu_vs_jax_rows.csv"
    summary_path = output_dir / "cpu_vs_jax_summary.json"
    _write_rows_csv(rows, rows_path)
    summary_path.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))

    successful_rows = _paired_success_rows(rows)
    if make_plots and successful_rows:
        jax_label = f"JAX ({summary.get('jax_backend', 'jax').upper()})"
        _plot_metric_grid(
            rows=successful_rows,
            metrics=OVERVIEW_METRICS,
            summary_metrics=summary["metrics"],
            output_path=output_dir / "cpu_vs_jax_prediction_scatter.png",
            title="CPU vs JAX prediction agreement",
            other_prefix="jax",
            other_label=jax_label,
        )
        _plot_metric_grid(
            rows=successful_rows,
            metrics=RAW_CONTACT_METRICS,
            summary_metrics=summary["metrics"],
            output_path=output_dir / "cpu_vs_jax_contact_scatter.png",
            title="CPU vs JAX contact breakdown",
            other_prefix="jax",
            other_label=jax_label,
        )
        _plot_metric_grid(
            rows=successful_rows,
            metrics=DERIVED_CONTACT_METRICS,
            summary_metrics=summary["metrics"],
            output_path=output_dir / "cpu_vs_jax_contact_summary_scatter.png",
            title="CPU vs JAX derived contact totals",
            other_prefix="jax",
            other_label=jax_label,
        )
        _plot_timing(
            rows=successful_rows,
            timing_summary=summary["timing"],
            output_path=output_dir / "cpu_vs_jax_timing.png",
            title="CPU vs JAX timing",
            other_prefix="jax",
            other_label=jax_label,
            speedup_key="warm_speedup",
        )

    tinygrad_rows = _tinygrad_success_rows(rows)
    if make_plots and tinygrad_rows:
        tinygrad_label = f"Tinygrad ({summary.get('tinygrad_backend', 'tinygrad').upper()})"
        _plot_metric_grid(
            rows=tinygrad_rows,
            metrics=OVERVIEW_METRICS,
            summary_metrics=summary["tinygrad_metrics"],
            output_path=output_dir / "cpu_vs_tinygrad_prediction_scatter.png",
            title="CPU vs Tinygrad prediction agreement",
            other_prefix="tinygrad",
            other_label=tinygrad_label,
        )
        _plot_metric_grid(
            rows=tinygrad_rows,
            metrics=RAW_CONTACT_METRICS,
            summary_metrics=summary["tinygrad_metrics"],
            output_path=output_dir / "cpu_vs_tinygrad_contact_scatter.png",
            title="CPU vs Tinygrad contact breakdown",
            other_prefix="tinygrad",
            other_label=tinygrad_label,
        )
        _plot_metric_grid(
            rows=tinygrad_rows,
            metrics=DERIVED_CONTACT_METRICS,
            summary_metrics=summary["tinygrad_metrics"],
            output_path=output_dir / "cpu_vs_tinygrad_contact_summary_scatter.png",
            title="CPU vs Tinygrad derived contact totals",
            other_prefix="tinygrad",
            other_label=tinygrad_label,
        )
        _plot_timing(
            rows=tinygrad_rows,
            timing_summary=summary["tinygrad_timing"],
            output_path=output_dir / "cpu_vs_tinygrad_timing.png",
            title="CPU vs Tinygrad timing",
            other_prefix="tinygrad",
            other_label=tinygrad_label,
            speedup_key="tinygrad_warm_speedup",
        )

    return rows_path, summary_path, summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare CPU and JAX predictions for a structure manifest.")
    parser.add_argument("--manifest", type=Path, required=True, help="Tab-separated manifest with pdb_id/chain1/chain2 columns.")
    parser.add_argument("--structures-dir", type=Path, required=True, help="Directory containing downloaded structure files.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output/compare"), help="Output artifact directory.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per backend and structure.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius.")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Interface distance cutoff in angstrom.")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Relative SASA threshold.")
    parser.add_argument("--sphere-points", type=int, default=100, help="Number of sphere points for SASA.")
    parser.add_argument("--require-gpu", action="store_true", help="Fail fast unless JAX reports a GPU-backed device.")
    parser.add_argument("--skip-plots", action="store_true", help="Write CSV/JSON summaries without generating images.")
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")
    if not args.structures_dir.exists():
        parser.error(f"Structures directory not found: {args.structures_dir}")

    try:
        rows_path, summary_path, summary = run_comparison(
            manifest_path=args.manifest,
            structures_dir=args.structures_dir,
            output_dir=args.output_dir,
            repeats=args.repeats,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            require_gpu=args.require_gpu,
            make_plots=not args.skip_plots,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error(str(exc))
        return 1

    LOGGER.info("Wrote per-structure rows to %s", rows_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))
    return 0 if summary["completed_structures"] else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
