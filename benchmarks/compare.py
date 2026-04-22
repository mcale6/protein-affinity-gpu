#!/usr/bin/env python3
"""Multi-backend comparison harness — CSV + three-panel figure.

Runs a configurable set of PRODIGY IC-NIS backends against a manifest of
two-chain structures and writes:

- One ``comparison_rows.csv`` row per structure × all configured backends.
- ``comparison_summary.json`` with per-backend timing and metric summaries.
- ``comparison_figure.png`` with three subplots:
    1. Warm timing (seconds) vs atom14-compacted atom count, one curve per backend.
    2. Per-structure SASA-sum scatter, CPU on x-axis, each non-CPU backend a coloured series
       with its Pearson r annotated.
    3. Heatmap of per-backend Pearson r vs CPU across scalar metrics
       (ΔG, Kd, NIS channels, interface contacts).

The script also supports a ``--plot-from CSV [CSV ...]`` mode that reads
one or more CSVs produced by previous runs (e.g. CPU-only local + GPU JAX
remote), merges them on ``pdb_id``, and draws the combined figure without
re-running any prediction.

Usage
-----
``.venv/bin/python benchmarks/compare.py --manifest ... --structures-dir ... \\
    --backends cpu tinygrad-batch tinygrad-single``

``.venv/bin/python benchmarks/compare.py --plot-from local.csv gpu.csv \\
    --output-dir benchmarks/output/combined``
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from protein_affinity_gpu.cpu import predict_binding_affinity  # noqa: E402
    from protein_affinity_gpu.utils._array import NumpyEncoder  # noqa: E402
    from protein_affinity_gpu.utils.atom14 import compact_complex_atom14  # noqa: E402
    from protein_affinity_gpu.utils.resources import format_duration  # noqa: E402
    from protein_affinity_gpu.utils.structure import load_complex  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guidance
    raise SystemExit(
        f"Missing Python dependency '{exc.name}'. "
        "Use the repo virtualenv or install dependencies with "
        "\".venv/bin/python -m pip install -e .\"."
    ) from exc

LOGGER = logging.getLogger(__name__)

SUPPORTED_STRUCTURE_SUFFIXES = (".pdb", ".ent", ".cif", ".mmcif")

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
HEATMAP_METRICS = [
    "ba_val",
    "contacts_ic",
    "contacts_charged",
    "contacts_polar",
    "contacts_aliphatic",
    "nis_aliphatic",
    "nis_charged",
    "nis_polar",
]


# --- Backend registry ---------------------------------------------------

@dataclass(frozen=True)
class BackendSpec:
    name: str
    loader: Callable[[], Callable]
    display: str


def _load_cpu():
    return predict_binding_affinity


def _load_jax(mode: str):
    def _loader():
        from protein_affinity_gpu.predict import predict_binding_affinity_jax

        def _run(**kw):
            return predict_binding_affinity_jax(mode=mode, **kw)

        return _run

    return _loader


def _load_jax_single():
    def _loader():
        from protein_affinity_gpu.experimental import (
            predict_binding_affinity_jax_experimental,
        )

        def _run(**kw):
            return predict_binding_affinity_jax_experimental(mode="single", **kw)

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
    "cpu":              BackendSpec("cpu",              _load_cpu,              "CPU (freesasa)"),
    "jax-batch":        BackendSpec("jax-batch",        _load_jax("block"),     "JAX (block)"),
    "jax-scan":         BackendSpec("jax-scan",         _load_jax("scan"),      "JAX (scan)"),
    "jax-single":       BackendSpec("jax-single",       _load_jax_single(),     "JAX (single)"),
    "tinygrad-batch":   BackendSpec("tinygrad-batch",   _load_tinygrad("block"),    "Tinygrad (block)"),
    "tinygrad-single":  BackendSpec("tinygrad-single",  _load_tinygrad("single"),   "Tinygrad (single)"),
}
DEFAULT_BACKENDS = ("cpu", "tinygrad-batch", "tinygrad-single")


# --- Manifest / structure resolution ------------------------------------

def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
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


def resolve_structure_path(structures_dir: Path, pdb_id: str) -> Path:
    normalized = pdb_id.upper()
    for suffix in SUPPORTED_STRUCTURE_SUFFIXES:
        for name in (f"{normalized}{suffix}", f"{normalized.lower()}{suffix}"):
            candidate = structures_dir / name
            if candidate.exists():
                return candidate
    raise FileNotFoundError(f"Structure not found for {pdb_id} in {structures_dir}")


def count_atom14_atoms(structure_path: Path, selection: str) -> int:
    """Padded kernel atom count (``n_residues × 14``).

    This is the N the SASA kernel actually dispatches over — it matches the
    ``atoms=N`` printed by ``predict.py`` and is ~1.7× larger than
    ``mask.sum()``. The timing plot needs the padded count so the x-axis
    reflects real compute load rather than occupancy.
    """
    target, binder = load_complex(structure_path, selection=selection, sanitize=True)
    positions, _mask, _, _ = compact_complex_atom14(target, binder)
    return int(np.asarray(positions).shape[0])


def _clear_tinygrad_caches() -> None:
    """Drop every tinygrad SASA TinyJit cache and force a GC pass.

    Each unique ``(block, N, M)`` compiles a fresh TinyJit and allocates
    device-side scratch (~1–4 GB on Metal). Across a full benchmark sweep
    the caches grow monotonically because every structure has a unique
    padded N; Metal's resource heap eventually returns
    ``Internal Error (0000000e)`` mid-sweep — not at a per-structure size
    cliff, but as accumulated shader/buffer pressure.
    """
    import gc

    try:
        from protein_affinity_gpu.sasa import _sasa_block_jit_cache
        _sasa_block_jit_cache.clear()
    except ImportError:
        pass
    try:
        from protein_affinity_gpu.sasa_experimental import (
            _sasa_tinygrad_jit_cache,
            _sasa_tinygrad_neighbor_jit_cache,
        )
        _sasa_tinygrad_jit_cache.clear()
        _sasa_tinygrad_neighbor_jit_cache.clear()
    except ImportError:
        pass
    gc.collect()


# --- Per-backend runner --------------------------------------------------

def _extract_scalar_metrics(result_dict: dict[str, Any]) -> dict[str, Any]:
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
    sasa_values = [float(row.get("atom_sasa", 0.0)) for row in sasa_data]
    metrics["sasa_sum"] = float(sum(sasa_values))
    return metrics


def _run_single_backend(
    predictor,
    structure_path: Path,
    repeats: int,
    selection: str,
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    sphere_points: int,
) -> dict[str, Any]:
    timings: list[float] = []
    last_result = None
    for _ in range(repeats):
        t0 = time.perf_counter()
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
        timings.append(time.perf_counter() - t0)
    cold = timings[0]
    warm = timings[1:]
    warm_mean = mean(warm) if warm else cold
    result_dict = last_result.to_dict() if last_result is not None else None
    metrics = _extract_scalar_metrics(result_dict) if result_dict is not None else {}
    return {
        "status": "ok",
        "cold_time_seconds": cold,
        "cold_time_formatted": format_duration(cold),
        "warm_times_seconds": warm,
        "warm_mean_seconds": warm_mean,
        "warm_mean_formatted": format_duration(warm_mean),
        "metrics": metrics,
    }


def _safe_run(predictor, **kw) -> dict[str, Any]:
    try:
        return _run_single_backend(predictor, **kw)
    except Exception as exc:  # pragma: no cover - defensive CLI surface
        LOGGER.error("backend failed: %s", exc)
        return {"status": "error", "error": str(exc), "metrics": {}}


# --- Comparison run ------------------------------------------------------

def run_comparison(
    manifest_path: Path,
    structures_dir: Path,
    output_dir: Path,
    backends: list[str],
    repeats: int = 3,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    make_plot: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
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

    LOGGER.info("Loading predictors for: %s", ", ".join(backends))
    predictors = {name: BACKENDS[name].loader() for name in backends}
    has_tinygrad = any(name.startswith("tinygrad") for name in backends)

    rows: list[dict[str, Any]] = []
    manifest_rows = load_manifest(manifest_path)
    for manifest_row in manifest_rows:
        pdb_id = manifest_row["pdb_id"]
        structure_path = resolve_structure_path(structures_dir, pdb_id)
        selection = f"{manifest_row['chain1']},{manifest_row['chain2']}"
        try:
            n_atoms = count_atom14_atoms(structure_path, selection)
        except Exception as exc:  # pragma: no cover - defensive
            LOGGER.error("%s: atom count failed: %s", pdb_id, exc)
            n_atoms = None

        row: dict[str, Any] = {
            **manifest_row,
            "structure_path": str(structure_path),
            "n_atoms_atom14": n_atoms,
        }

        for name in backends:
            LOGGER.info("%s :: %s (repeats=%d)", pdb_id, name, repeats)
            payload = _safe_run(
                predictors[name],
                structure_path=structure_path,
                repeats=repeats,
                selection=selection,
                temperature=temperature,
                distance_cutoff=distance_cutoff,
                acc_threshold=acc_threshold,
                sphere_points=sphere_points,
            )
            prefix = name
            row[f"{prefix}_status"] = payload["status"]
            row[f"{prefix}_error"] = payload.get("error")
            row[f"{prefix}_cold_seconds"] = payload.get("cold_time_seconds")
            row[f"{prefix}_warm_mean_seconds"] = payload.get("warm_mean_seconds")
            for metric_name, value in payload.get("metrics", {}).items():
                row[f"{prefix}_{metric_name}"] = value

        rows.append(row)
        if has_tinygrad:
            _clear_tinygrad_caches()

    summary = _build_summary(rows, backends, repeats, manifest_path, structures_dir)

    rows_path = output_dir / "comparison_rows.csv"
    summary_path = output_dir / "comparison_summary.json"
    _write_csv(rows, backends, rows_path)
    summary_path.write_text(json.dumps(summary, indent=2, cls=NumpyEncoder))

    if make_plot:
        fig_path = output_dir / "comparison_figure.png"
        plot_figure(rows, backends, fig_path)

    return rows_path, summary_path, summary


# --- CSV / summary -------------------------------------------------------

MANIFEST_FIELDS = [
    "pdb_id", "chain1", "chain2",
    "difficulty", "n_virtual_xl",
    "best_lrmsd_no_xl", "best_lrmsd_7xl_10best", "best_lrmsd_7xl",
    "ref_table",
    "structure_path", "n_atoms_atom14",
]


def _backend_columns(backends: Iterable[str]) -> list[str]:
    cols: list[str] = []
    for name in backends:
        cols.extend([
            f"{name}_status", f"{name}_error",
            f"{name}_cold_seconds", f"{name}_warm_mean_seconds",
        ])
        cols.extend(f"{name}_{m}" for m in SCALAR_METRICS)
    return cols


def _write_csv(rows: list[dict[str, Any]], backends: Iterable[str], output_path: Path) -> None:
    if not rows:
        return
    fieldnames = MANIFEST_FIELDS + _backend_columns(backends)
    # Keep any manifest column the user added that we didn't anticipate.
    extra_cols = sorted({k for row in rows for k in row if k not in fieldnames})
    fieldnames = fieldnames + extra_cols
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})


def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _build_summary(
    rows: list[dict[str, Any]],
    backends: list[str],
    repeats: int,
    manifest_path: Path,
    structures_dir: Path,
) -> dict[str, Any]:
    per_backend: dict[str, Any] = {}
    for name in backends:
        ok_rows = [r for r in rows if r.get(f"{name}_status") == "ok"]
        failures = [r for r in rows if r.get(f"{name}_status") != "ok"]
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

    # Correlations with CPU where available.
    correlations: dict[str, dict[str, float | None]] = {}
    if "cpu" in backends:
        cpu_ok = [r for r in rows if r.get("cpu_status") == "ok"]
        for name in backends:
            if name == "cpu":
                continue
            paired = [r for r in cpu_ok if r.get(f"{name}_status") == "ok"]
            if not paired:
                continue
            per_metric: dict[str, float | None] = {}
            for metric in SCALAR_METRICS:
                xs = np.asarray(
                    [r.get(f"cpu_{metric}") for r in paired], dtype=float
                )
                ys = np.asarray(
                    [r.get(f"{name}_{metric}") for r in paired], dtype=float
                )
                per_metric[metric] = _pearson(xs, ys)
            correlations[name] = per_metric

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_path),
        "structures_dir": str(structures_dir),
        "repeats": repeats,
        "backends": list(backends),
        "per_backend": per_backend,
        "correlations_vs_cpu": correlations,
    }


# --- Figure --------------------------------------------------------------

def _import_pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Plot generation requires 'matplotlib'.") from exc
    return plt


_BACKEND_COLORS = {
    "cpu":             "#2ca02c",
    "jax-batch":       "#1f77b4",
    "jax-scan":        "#17becf",
    "jax-single":      "#9467bd",
    "tinygrad-batch":  "#ff7f0e",
    "tinygrad-single": "#d62728",
}


def _display_name(name: str) -> str:
    return BACKENDS[name].display if name in BACKENDS else name


def _ok_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get(f"{name}_status") == "ok"]


def plot_figure(rows: list[dict[str, Any]], backends: list[str], output_path: Path) -> None:
    plt = _import_pyplot()
    if not rows:
        LOGGER.warning("No rows to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    ax_timing, ax_sasa, ax_heat = axes

    # --- Subplot 1: timing vs atom count -------------------------------
    for name in backends:
        ok = _ok_rows(rows, name)
        points = [
            (r.get("n_atoms_atom14"), r.get(f"{name}_warm_mean_seconds"))
            for r in ok
            if r.get("n_atoms_atom14") is not None
            and r.get(f"{name}_warm_mean_seconds") is not None
        ]
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        color = _BACKEND_COLORS.get(name, None)
        ax_timing.plot(
            xs, ys,
            marker="o", linewidth=1.4, markersize=5,
            color=color, label=_display_name(name),
        )
    ax_timing.set_xlabel("Atom14 atoms (padded, n_residues × 14)")
    ax_timing.set_ylabel("Warm mean wall time (s)")
    ax_timing.set_yscale("log")
    ax_timing.set_title("Timing vs atom count")
    ax_timing.grid(alpha=0.25, which="both")
    ax_timing.legend(fontsize=8, loc="upper left")

    # --- Subplot 2: per-structure SASA-sum, CPU vs each backend -------
    if "cpu" in backends:
        cpu_ok = _ok_rows(rows, "cpu")
        for name in backends:
            if name == "cpu":
                continue
            paired = [r for r in cpu_ok if r.get(f"{name}_status") == "ok"]
            if not paired:
                continue
            xs = np.asarray([r.get("cpu_sasa_sum") for r in paired], dtype=float)
            ys = np.asarray([r.get(f"{name}_sasa_sum") for r in paired], dtype=float)
            color = _BACKEND_COLORS.get(name, None)
            r_value = _pearson(xs, ys)
            label = _display_name(name)
            if r_value is not None:
                label += f"  (r={r_value:.4f})"
            ax_sasa.scatter(xs, ys, s=48, alpha=0.85, color=color, label=label, zorder=2)
        # Identity line across the combined axis range.
        all_x = [r.get("cpu_sasa_sum") for r in cpu_ok if r.get("cpu_sasa_sum") is not None]
        if all_x:
            lo, hi = float(min(all_x)), float(max(all_x))
            pad = max((hi - lo) * 0.05, 1.0)
            ax_sasa.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad],
                linestyle="--", linewidth=1.0, color="#6c757d", zorder=1,
            )
    ax_sasa.set_xlabel("CPU total SASA (Å²)")
    ax_sasa.set_ylabel("Backend total SASA (Å²)")
    ax_sasa.set_title("SASA agreement (per structure)")
    ax_sasa.grid(alpha=0.25)
    ax_sasa.legend(fontsize=8, loc="upper left")

    # --- Subplot 3: heatmap of Pearson r vs CPU across metrics --------
    if "cpu" in backends and len(backends) > 1:
        non_cpu = [b for b in backends if b != "cpu"]
        matrix = np.full((len(non_cpu), len(HEATMAP_METRICS)), np.nan)
        cpu_ok = _ok_rows(rows, "cpu")
        for i, name in enumerate(non_cpu):
            paired = [r for r in cpu_ok if r.get(f"{name}_status") == "ok"]
            if not paired:
                continue
            for j, metric in enumerate(HEATMAP_METRICS):
                xs = np.asarray(
                    [r.get(f"cpu_{metric}") for r in paired], dtype=float
                )
                ys = np.asarray(
                    [r.get(f"{name}_{metric}") for r in paired], dtype=float
                )
                value = _pearson(xs, ys)
                if value is not None:
                    matrix[i, j] = value
        im = ax_heat.imshow(
            matrix,
            aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0,
        )
        ax_heat.set_xticks(range(len(HEATMAP_METRICS)))
        ax_heat.set_xticklabels(HEATMAP_METRICS, rotation=40, ha="right", fontsize=8)
        ax_heat.set_yticks(range(len(non_cpu)))
        ax_heat.set_yticklabels([_display_name(b) for b in non_cpu], fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                cell = matrix[i, j]
                if np.isnan(cell):
                    text = "n/a"
                else:
                    text = f"{cell:.3f}"
                ax_heat.text(
                    j, i, text,
                    ha="center", va="center", fontsize=8,
                    color="black" if not np.isnan(cell) and abs(cell) < 0.6 else "white",
                )
        ax_heat.set_title("Pearson r vs CPU")
        fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04)
    else:
        ax_heat.set_axis_off()
        ax_heat.text(
            0.5, 0.5,
            "No CPU baseline available\n— heatmap skipped",
            ha="center", va="center", fontsize=10, transform=ax_heat.transAxes,
        )

    fig.suptitle("Backend comparison", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote figure to %s", output_path)


# --- Plot-from-CSV mode --------------------------------------------------

def _detect_backends_from_columns(columns: Iterable[str]) -> list[str]:
    detected = []
    seen = set()
    for col in columns:
        if not col.endswith("_status"):
            continue
        name = col[: -len("_status")]
        if name in seen:
            continue
        seen.add(name)
        detected.append(name)
    # Prefer the canonical ordering when possible.
    canonical = [b for b in BACKENDS if b in seen]
    extras = [b for b in detected if b not in BACKENDS]
    return canonical + extras


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            out[key] = None
            continue
        # int-like fields
        if key == "n_atoms_atom14":
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = None
            continue
        # Numeric fields: attempt float, fall back to the string.
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def load_rows_from_csvs(csv_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    all_backends: list[str] = []
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            detected = _detect_backends_from_columns(columns)
            for name in detected:
                if name not in all_backends:
                    all_backends.append(name)
            for raw in reader:
                coerced = _coerce_row(raw)
                key = coerced.get("pdb_id")
                if not key:
                    continue
                if key not in merged:
                    merged[key] = coerced
                else:
                    for field, value in coerced.items():
                        if value is None:
                            continue
                        existing = merged[key].get(field)
                        if existing in (None, ""):
                            merged[key][field] = value
    return list(merged.values()), all_backends


# --- CLI -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Multi-backend comparison harness. "
            "Either --manifest+--structures-dir (run mode) or "
            "--plot-from CSV [CSV ...] (re-plot mode)."
        ),
    )
    parser.add_argument("--manifest", type=Path, help="TSV with pdb_id/chain1/chain2 columns.")
    parser.add_argument("--structures-dir", type=Path, help="Directory of downloaded structures.")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("benchmarks/output/compare"),
        help="Output directory for CSV/JSON/PNG.",
    )
    parser.add_argument(
        "--backends", nargs="+", default=list(DEFAULT_BACKENDS),
        choices=sorted(BACKENDS),
        help="Backends to run (default: cpu tinygrad-batch tinygrad-single).",
    )
    parser.add_argument("--repeats", type=int, default=3, help="Runs per backend per structure.")
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--distance-cutoff", type=float, default=5.5)
    parser.add_argument("--acc-threshold", type=float, default=0.05)
    parser.add_argument("--sphere-points", type=int, default=100)
    parser.add_argument("--skip-plot", action="store_true", help="Skip figure generation.")
    parser.add_argument(
        "--plot-from", type=Path, nargs="+", default=None,
        help="Read one or more CSVs (merged on pdb_id) and re-plot without running predictions.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    if args.plot_from:
        missing = [p for p in args.plot_from if not Path(p).exists()]
        if missing:
            parser.error(f"CSV not found: {missing}")
        rows, backends = load_rows_from_csvs(list(args.plot_from))
        if not rows:
            parser.error("No rows loaded from the supplied CSV(s).")
        args.output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = args.output_dir / "comparison_figure.png"
        plot_figure(rows, backends, fig_path)
        LOGGER.info("Plotted %d structures across backends: %s", len(rows), backends)
        print(str(fig_path))
        return 0

    if not args.manifest or not args.structures_dir:
        parser.error("Provide --manifest and --structures-dir (or use --plot-from).")
    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")
    if not args.structures_dir.exists():
        parser.error(f"Structures directory not found: {args.structures_dir}")

    try:
        rows_path, summary_path, summary = run_comparison(
            manifest_path=args.manifest,
            structures_dir=args.structures_dir,
            output_dir=args.output_dir,
            backends=args.backends,
            repeats=args.repeats,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            make_plot=not args.skip_plot,
        )
    except Exception as exc:  # pragma: no cover
        LOGGER.error(str(exc))
        return 1

    LOGGER.info("Wrote rows to %s", rows_path)
    LOGGER.info("Wrote summary to %s", summary_path)
    print(json.dumps(summary, indent=2, cls=NumpyEncoder))
    completed = max(
        (entry.get("completed", 0) for entry in summary["per_backend"].values()),
        default=0,
    )
    return 0 if completed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
