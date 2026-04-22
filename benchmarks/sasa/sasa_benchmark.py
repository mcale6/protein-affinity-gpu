#!/usr/bin/env python3
"""Shared benchmark primitives used by the local and Modal harnesses.

Exports the backend registry, the per-structure/per-backend runner, manifest
loading / RCSB download, atom14 counting, and the single ``run_benchmark``
function that both ``benchmarks/benchmark.py`` (local M1 Max) and
``benchmarks/modal_benchmark.py`` (GPU) delegate to.

Result schema (one row per structure, columns namespaced by backend):

    pdb_id, chain1, chain2, n_atoms_atom14, device,
    <backend>_status, <backend>_error,
    <backend>_cold_seconds, <backend>_warm_mean_seconds,
    <backend>_rss_peak_mb, <backend>_jax_peak_mb, <backend>_tg_mem_used_mb,
    <backend>_ba_val, <backend>_kd,
    <backend>_sasa_sum, <backend>_contacts_*, <backend>_nis_*
"""
from __future__ import annotations

import csv
import gc
import json
import logging
import shutil
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, stdev
from typing import Any, Callable, Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from protein_affinity_gpu.utils._array import NumpyEncoder  # noqa: E402
from protein_affinity_gpu.utils.atom14 import compact_complex_atom14  # noqa: E402
from protein_affinity_gpu.utils.resources import format_duration  # noqa: E402
from protein_affinity_gpu.utils.structure import load_complex  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST = ROOT / "benchmarks/datasets/kahraman_2013_t3.tsv"
DEFAULT_STRUCTURES_DIR = ROOT / "benchmarks/downloads/kahraman_2013_t3"
DEFAULT_SPHERE_POINTS = 100
DEFAULT_REPEATS = 2

RCSB_DOWNLOAD_ROOT = "https://files.rcsb.org/download"
SUPPORTED_STRUCTURE_SUFFIXES = (".pdb", ".cif", ".ent", ".mmcif")
DOWNLOAD_SUFFIXES = (".pdb", ".cif")

CONTACT_FIELDS = {
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
SCALAR_METRICS = [
    "ba_val",
    "kd",
    "nis_aliphatic",
    "nis_charged",
    "nis_polar",
    "sasa_sum",
    *CONTACT_FIELDS.values(),
]


# --- Backend registry ---------------------------------------------------

@dataclass(frozen=True)
class BackendSpec:
    name: str
    loader: Callable[[], Callable]
    display: str


def _load_cpu():
    from protein_affinity_gpu.cpu import predict_binding_affinity

    return predict_binding_affinity


def _load_jax(mode: str):
    def _loader():
        if mode in ("block", "scan"):
            from protein_affinity_gpu.predict import predict_binding_affinity_jax

            def _run(**kw):
                return predict_binding_affinity_jax(mode=mode, **kw)
        else:
            from protein_affinity_gpu.experimental import (
                predict_binding_affinity_jax_experimental,
            )

            def _run(**kw):
                return predict_binding_affinity_jax_experimental(mode=mode, **kw)

        return _run

    return _loader


def _load_tinygrad(mode: str):
    def _loader():
        from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad

        def _run(**kw):
            return predict_binding_affinity_tinygrad(mode=mode, **kw)

        return _run

    return _loader


BACKENDS: dict[str, BackendSpec] = {
    "cpu":             BackendSpec("cpu",             _load_cpu,               "CPU (freesasa)"),
    "jax-single":      BackendSpec("jax-single",      _load_jax("single"),     "JAX (single)"),
    "jax-batch":       BackendSpec("jax-batch",       _load_jax("block"),      "JAX (batch)"),
    "jax-scan":        BackendSpec("jax-scan",        _load_jax("scan"),       "JAX (scan)"),
    "tinygrad-single": BackendSpec("tinygrad-single", _load_tinygrad("single"), "Tinygrad (single)"),
    "tinygrad-batch":  BackendSpec("tinygrad-batch",  _load_tinygrad("block"),  "Tinygrad (batch)"),
}

LOCAL_DEFAULT_TARGETS = ("cpu", "tinygrad-single", "tinygrad-batch")
GPU_DEFAULT_TARGETS = (
    "jax-single",
    "jax-batch",
    "jax-scan",
    "tinygrad-single",
    "tinygrad-batch",
)


# --- Manifest / structure resolution ------------------------------------

def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [
            {k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()}
            for row in reader
            if row.get("pdb_id")
        ]
    if not rows:
        raise ValueError(f"No rows found in manifest: {manifest_path}")
    return rows


def write_manifest_rows(rows: Iterable[dict[str, str]], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=("pdb_id", "chain1", "chain2"), delimiter="\t"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"pdb_id": row["pdb_id"], "chain1": row["chain1"], "chain2": row["chain2"]}
            )
    return manifest_path


def materialize_manifest(
    manifest_path: Path, output_dir: Path, limit: int | None = None
) -> Path:
    if limit is None or limit <= 0:
        return manifest_path
    rows = load_manifest_rows(manifest_path)[:limit]
    return write_manifest_rows(rows, output_dir / "manifest_subset.tsv")


def resolve_structure_path(structures_dir: Path, pdb_id: str) -> Path:
    normalized = pdb_id.upper()
    for suffix in SUPPORTED_STRUCTURE_SUFFIXES:
        for name in (f"{normalized}{suffix}", f"{normalized.lower()}{suffix}"):
            candidate = structures_dir / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Structure not found for {pdb_id} in {structures_dir}")


def _existing_structure_path(structures_dir: Path, pdb_id: str) -> Path | None:
    try:
        return resolve_structure_path(structures_dir, pdb_id)
    except FileNotFoundError:
        return None


def download_structure(pdb_id: str, structures_dir: Path) -> Path:
    structures_dir.mkdir(parents=True, exist_ok=True)
    normalized = pdb_id.upper()
    last_error: Exception | None = None
    for suffix in DOWNLOAD_SUFFIXES:
        destination = structures_dir / f"{normalized}{suffix}"
        tmp_path = structures_dir / f".{normalized}{suffix}.tmp"
        url = f"{RCSB_DOWNLOAD_ROOT}/{normalized}{suffix}"
        try:
            with urllib.request.urlopen(url) as response, tmp_path.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            tmp_path.replace(destination)
            return destination
        except urllib.error.HTTPError as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            if exc.code == 404:
                continue
            raise
        except urllib.error.URLError as exc:
            last_error = exc
            if tmp_path.exists():
                tmp_path.unlink()
            continue
    raise RuntimeError(f"Failed to download {normalized} from RCSB") from last_error


def ensure_manifest_structures(
    manifest_path: Path,
    structures_dir: Path,
) -> dict[str, int]:
    rows = load_manifest_rows(manifest_path)
    downloaded = 0
    existing = 0
    for row in rows:
        pdb_id = row["pdb_id"]
        if _existing_structure_path(structures_dir, pdb_id) is not None:
            existing += 1
            continue
        LOGGER.info("fetch %s", pdb_id)
        download_structure(pdb_id, structures_dir)
        downloaded += 1
    return {"requested": len(rows), "downloaded": downloaded, "existing": existing}


def count_atom14_atoms(structure_path: Path, selection: str) -> int:
    """Padded kernel atom count (``n_residues × 14``) — matches what the SASA
    kernel dispatches over, ~1.7× larger than the active ``mask.sum()``.
    """
    target, binder = load_complex(structure_path, selection=selection, sanitize=True)
    positions, _mask, _, _ = compact_complex_atom14(target, binder)
    return int(np.asarray(positions).shape[0])


# --- Cache management (tinygrad TinyJit resource pressure) --------------

def clear_accelerator_caches() -> None:
    """Drop per-shape JIT caches and force a GC pass.

    Each structure has a unique ``N`` so the prior compilation is dead
    weight for the next one. We wipe:

    * tinygrad ``_sasa_block_jit_cache`` / ``_sasa_tinygrad_jit_cache`` —
      each unique ``(block, N, M)`` pins device scratch; on Metal the
      accumulation surfaces as ``Internal Error (0000000e)`` mid-sweep.
    * JAX compilation cache via ``jax.clear_caches()`` — each XLA program
      holds a device-buffer reservation for its scratch; without clearing,
      reservations accumulate across the 16-structure sweep and OOM the
      large structures even on an A100-80GB.
    """
    try:
        from protein_affinity_gpu.sasa import (
            _sasa_block_jit_cache,
            _sasa_tinygrad_jit_cache,
        )
        _sasa_block_jit_cache.clear()
        _sasa_tinygrad_jit_cache.clear()
    except ImportError:
        pass
    try:
        from protein_affinity_gpu.sasa_experimental import (
            _sasa_tinygrad_neighbor_jit_cache,
        )
        _sasa_tinygrad_neighbor_jit_cache.clear()
    except ImportError:
        pass
    try:
        import jax  # type: ignore

        jax.clear_caches()
    except Exception:  # noqa: BLE001
        pass
    gc.collect()


clear_tinygrad_caches = clear_accelerator_caches  # legacy alias


# --- Memory profiling ---------------------------------------------------

def snapshot_memory() -> dict[str, float]:
    """Best-effort process + device memory snapshot (MB). Empty dict if none."""
    snap: dict[str, float] = {}
    try:
        import resource
        factor = 1024 * 1024 if sys.platform == "darwin" else 1024
        snap["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / factor
    except Exception:  # noqa: BLE001
        pass
    try:
        import jax  # type: ignore

        stats = jax.devices()[0].memory_stats()
        if stats:
            snap["jax_in_use_mb"] = stats.get("bytes_in_use", 0) / 1e6
            snap["jax_peak_mb"] = stats.get("peak_bytes_in_use", 0) / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        from tinygrad.helpers import GlobalCounters  # type: ignore

        used = getattr(GlobalCounters, "mem_used", 0) or 0
        if used:
            snap["tg_mem_used_mb"] = used / 1e6
    except Exception:  # noqa: BLE001
        pass
    return snap


# --- Metrics extraction --------------------------------------------------

def extract_scalar_metrics(result_dict: dict[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "ba_val": float(result_dict["ba_val"]),
        "kd": float(result_dict["kd"]),
        "nis_aliphatic": float(result_dict["nis"]["aliphatic"]),
        "nis_charged": float(result_dict["nis"]["charged"]),
        "nis_polar": float(result_dict["nis"]["polar"]),
    }
    for contact_key, field in CONTACT_FIELDS.items():
        metrics[field] = float(result_dict["contacts"][contact_key])
    sasa_data = result_dict.get("sasa_data") or []
    metrics["sasa_sum"] = float(sum(float(row.get("atom_sasa", 0.0)) for row in sasa_data))
    return metrics


# --- Per-structure/per-backend runner -----------------------------------

def run_backend_on_structure(
    predictor: Callable,
    structure_path: Path,
    selection: str,
    repeats: int,
    *,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = DEFAULT_SPHERE_POINTS,
) -> dict[str, Any]:
    """Run ``predictor`` ``repeats`` times. Capture cold/warm timings, per-run
    memory snapshots, and the final metrics. On failure returns
    ``{"status": "error", "error": ...}`` with an empty metrics dict.
    """
    timings: list[float] = []
    mem_samples: list[dict[str, float]] = []
    last_result = None

    for _ in range(repeats):
        gc.collect()
        mem_before = snapshot_memory()
        t0 = time.perf_counter()
        try:
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
        except Exception as exc:  # noqa: BLE001
            return {
                "status": "error",
                "error": f"{exc.__class__.__name__}: {exc}",
                "metrics": {},
            }
        timings.append(time.perf_counter() - t0)
        mem_after = snapshot_memory()
        sample = dict(mem_after)
        if "rss_mb" in mem_before and "rss_mb" in mem_after:
            sample["rss_delta_mb"] = mem_after["rss_mb"] - mem_before["rss_mb"]
        mem_samples.append(sample)

    cold = timings[0]
    warm = timings[1:]
    warm_mean = mean(warm) if warm else cold
    warm_std = stdev(warm) if len(warm) > 1 else 0.0
    warm_min = min(warm) if warm else cold
    warm_max = max(warm) if warm else cold
    result_dict = last_result.to_dict() if last_result is not None else None
    metrics = extract_scalar_metrics(result_dict) if result_dict is not None else {}

    def _peak(key: str) -> float | None:
        values = [s[key] for s in mem_samples if s.get(key) is not None]
        return max(values) if values else None

    return {
        "status": "ok",
        "cold_time_seconds": cold,
        "cold_time_formatted": format_duration(cold),
        "warm_times_seconds": warm,
        "warm_mean_seconds": warm_mean,
        "warm_std_seconds": warm_std,
        "warm_min_seconds": warm_min,
        "warm_max_seconds": warm_max,
        "warm_mean_formatted": format_duration(warm_mean),
        "rss_peak_mb": _peak("rss_mb"),
        "jax_peak_mb": _peak("jax_peak_mb"),
        "tg_mem_used_mb": _peak("tg_mem_used_mb"),
        "metrics": metrics,
    }


def _apply_backend_result_to_row(
    row: dict[str, Any], backend: str, result: dict[str, Any]
) -> None:
    row[f"{backend}_status"] = result["status"]
    row[f"{backend}_error"] = result.get("error")
    row[f"{backend}_cold_seconds"] = result.get("cold_time_seconds")
    row[f"{backend}_warm_mean_seconds"] = result.get("warm_mean_seconds")
    row[f"{backend}_warm_std_seconds"] = result.get("warm_std_seconds")
    row[f"{backend}_warm_min_seconds"] = result.get("warm_min_seconds")
    row[f"{backend}_warm_max_seconds"] = result.get("warm_max_seconds")
    row[f"{backend}_rss_peak_mb"] = result.get("rss_peak_mb")
    row[f"{backend}_jax_peak_mb"] = result.get("jax_peak_mb")
    row[f"{backend}_tg_mem_used_mb"] = result.get("tg_mem_used_mb")
    for metric_name, value in result.get("metrics", {}).items():
        row[f"{backend}_{metric_name}"] = value


# --- CSV / summary -------------------------------------------------------

MANIFEST_FIELDS = [
    "pdb_id", "chain1", "chain2",
    "difficulty", "n_virtual_xl",
    "best_lrmsd_no_xl", "best_lrmsd_7xl_10best", "best_lrmsd_7xl",
    "ref_table",
    "structure_path", "n_atoms_atom14", "device",
]
MEMORY_FIELDS = ("rss_peak_mb", "jax_peak_mb", "tg_mem_used_mb")


def _backend_columns(backends: Iterable[str]) -> list[str]:
    cols: list[str] = []
    for name in backends:
        cols.extend([f"{name}_status", f"{name}_error"])
        cols.extend([
            f"{name}_cold_seconds", f"{name}_warm_mean_seconds",
            f"{name}_warm_std_seconds", f"{name}_warm_min_seconds",
            f"{name}_warm_max_seconds",
        ])
        cols.extend(f"{name}_{mem}" for mem in MEMORY_FIELDS)
        cols.extend(f"{name}_{metric}" for metric in SCALAR_METRICS)
    return cols


def write_rows_csv(
    rows: list[dict[str, Any]], backends: Iterable[str], output_path: Path
) -> None:
    if not rows:
        return
    fieldnames = MANIFEST_FIELDS + _backend_columns(backends)
    extra_cols = sorted({k for row in rows for k in row if k not in fieldnames})
    fieldnames = fieldnames + extra_cols
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def build_summary(
    rows: list[dict[str, Any]],
    backends: Sequence[str],
    repeats: int,
    manifest_path: Path,
    structures_dir: Path,
    device: str,
) -> dict[str, Any]:
    per_backend: dict[str, Any] = {}
    for name in backends:
        ok_rows = [r for r in rows if r.get(f"{name}_status") == "ok"]
        failures = [r for r in rows if r.get(f"{name}_status") not in ("ok", None)]
        warm = np.asarray(
            [r[f"{name}_warm_mean_seconds"] for r in ok_rows], dtype=float
        ) if ok_rows else np.asarray([], dtype=float)
        per_backend[name] = {
            "display": BACKENDS[name].display if name in BACKENDS else name,
            "completed": len(ok_rows),
            "failed": len(failures),
            "warm_mean_seconds": float(warm.mean()) if warm.size else None,
            "warm_median_seconds": float(np.median(warm)) if warm.size else None,
            "warm_max_seconds": float(warm.max()) if warm.size else None,
            "failures": [
                {"pdb_id": r["pdb_id"], "error": r.get(f"{name}_error")}
                for r in failures
            ],
        }
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": device,
        "manifest_path": str(manifest_path),
        "structures_dir": str(structures_dir),
        "repeats": repeats,
        "backends": list(backends),
        "per_backend": per_backend,
    }


# --- Main runner ---------------------------------------------------------

def detect_device() -> str:
    try:
        import jax  # type: ignore

        platforms = {d.platform.lower() for d in jax.devices()}
        if "gpu" in platforms or "cuda" in platforms:
            return "gpu"
        if "metal" in platforms:
            return "metal"
    except Exception:  # noqa: BLE001
        pass
    return sys.platform


def run_benchmark(
    manifest_path: Path,
    structures_dir: Path,
    output_dir: Path,
    backends: Sequence[str],
    repeats: int = DEFAULT_REPEATS,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = DEFAULT_SPHERE_POINTS,
    limit: int | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Execute ``backends`` over the manifest's structures and write the unified
    ``results.csv`` + ``summary.json`` into ``output_dir``. Returns the summary
    dict.
    """
    if repeats < 1:
        raise ValueError("repeats must be at least 1")
    unknown = [name for name in backends if name not in BACKENDS]
    if unknown:
        raise ValueError(
            f"Unknown backend(s): {unknown}. Available: {sorted(BACKENDS)}"
        )

    manifest_path = Path(manifest_path)
    structures_dir = Path(structures_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_to_use = materialize_manifest(manifest_path, output_dir, limit=limit)
    ensure_manifest_structures(manifest_to_use, structures_dir)

    device_label = device or detect_device()
    LOGGER.info(
        "benchmark: device=%s backends=%s", device_label, ", ".join(backends)
    )
    predictors = {name: BACKENDS[name].loader() for name in backends}
    needs_clear = any(
        name.startswith("tinygrad") or name.startswith("jax") for name in backends
    )

    rows: list[dict[str, Any]] = []
    manifest_rows = load_manifest_rows(manifest_to_use)
    for manifest_row in manifest_rows:
        pdb_id = manifest_row["pdb_id"]
        structure_path = resolve_structure_path(structures_dir, pdb_id)
        selection = f"{manifest_row['chain1']},{manifest_row['chain2']}"
        try:
            n_atoms = count_atom14_atoms(structure_path, selection)
        except Exception as exc:  # noqa: BLE001
            LOGGER.error("%s: atom count failed: %s", pdb_id, exc)
            n_atoms = None

        row: dict[str, Any] = {
            **manifest_row,
            "structure_path": str(structure_path),
            "n_atoms_atom14": n_atoms,
            "device": device_label,
        }

        for name in backends:
            LOGGER.info("%s :: %s (repeats=%d)", pdb_id, name, repeats)
            result = run_backend_on_structure(
                predictors[name],
                structure_path=structure_path,
                selection=selection,
                repeats=repeats,
                temperature=temperature,
                distance_cutoff=distance_cutoff,
                acc_threshold=acc_threshold,
                sphere_points=sphere_points,
            )
            _apply_backend_result_to_row(row, name, result)

        rows.append(row)
        if needs_clear:
            clear_accelerator_caches()

    rows_path = output_dir / "results.csv"
    summary_path = output_dir / "summary.json"
    write_rows_csv(rows, backends, rows_path)
    summary = build_summary(
        rows, backends, repeats, manifest_to_use, structures_dir, device_label
    )
    summary["artifacts"] = {
        "rows_csv": str(rows_path),
        "summary_json": str(summary_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))
    LOGGER.info(
        "benchmark: wrote %d rows to %s (summary: %s)",
        len(rows), rows_path, summary_path,
    )
    return summary
