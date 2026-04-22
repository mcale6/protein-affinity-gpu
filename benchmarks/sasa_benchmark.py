#!/usr/bin/env python3
"""Notebook-style SASA benchmark as a regular Python script.

This is the script version of ``sasa_benchmark_colab.ipynb``:

1. Ensure Kahraman manifest structures are available locally.
2. Run the experimental multi-backend benchmark sweep.
3. Save the timing plot and two CSV summaries.

The notebook currently references experimental targets such as
``jax-single`` and ``tinygrad``. This script keeps that extended sweep
self-contained so it can run as a normal Python file or from Modal.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import shutil
import sys
import time
import traceback
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from protein_affinity_gpu.utils._array import NumpyEncoder  # noqa: E402
from protein_affinity_gpu.utils.resources import format_duration  # noqa: E402

LOGGER = logging.getLogger(__name__)

DEFAULT_MANIFEST = ROOT / "benchmarks/datasets/kahraman_2013_t3.tsv"
DEFAULT_STRUCTURES_DIR = ROOT / "benchmarks/downloads/kahraman_2013_t3"
DEFAULT_OUTPUT_DIR = ROOT / "benchmarks/output/colab_sweep"
DEFAULT_NOTEBOOK_TARGETS = (
    "cpu",
    "jax",
    "jax-single",
    "jax-scan",
    "jax-soft",
    "tinygrad",
    "tinygrad-single",
    "tinygrad-neighbor",
)
DEFAULT_GPU_TARGETS = tuple(target for target in DEFAULT_NOTEBOOK_TARGETS if target != "cpu")
DEFAULT_SPHERE_POINTS = 100
DEFAULT_REPEATS = 2
RCSB_DOWNLOAD_ROOT = "https://files.rcsb.org/download"
DOWNLOAD_SUFFIXES = (".pdb", ".cif")


def _load_jax_predictor():
    from protein_affinity_gpu.predict import predict_binding_affinity_jax

    return predict_binding_affinity_jax


def _load_cpu_predictor():
    from protein_affinity_gpu.cpu import predict_binding_affinity

    return predict_binding_affinity


def _load_jax_soft_predictor():
    from protein_affinity_gpu.experimental import predict_binding_affinity_jax_experimental

    def predictor(**kwargs):
        return predict_binding_affinity_jax_experimental(
            soft_sasa=True,
            soft_beta=10.0,
            **kwargs,
        )

    return predictor


def _load_jax_mode_predictor(mode: str):
    if mode in ("block", "scan"):
        from protein_affinity_gpu.predict import predict_binding_affinity_jax as jax_fn

        def predictor(**kwargs):
            return jax_fn(mode=mode, **kwargs)

        return predictor

    from protein_affinity_gpu.experimental import predict_binding_affinity_jax_experimental

    def predictor(**kwargs):
        return predict_binding_affinity_jax_experimental(mode=mode, **kwargs)

    return predictor


def _load_tinygrad_predictor(mode: str = "block"):
    from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad

    def predictor(**kwargs):
        return predict_binding_affinity_tinygrad(mode=mode, **kwargs)

    return predictor


def _count_atoms(structure_path: Path, selection: str) -> int:
    from protein_affinity_gpu.utils.atom14 import compact_complex_atom14
    from protein_affinity_gpu.utils.structure import load_complex

    target, binder = load_complex(structure_path, selection=selection, sanitize=True)
    positions, mask, _, _ = compact_complex_atom14(target, binder)
    del positions
    return int(mask.sum())


def cuda_available() -> bool:
    try:
        import jax
    except ImportError:
        return False

    platforms = {device.platform.lower() for device in jax.devices()}
    return "gpu" in platforms or "cuda" in platforms


def _benchmark_single(predictor, structure_path: Path, repeats: int, **kwargs):
    timings = []
    last_result = None
    for _ in range(repeats):
        start_time = time.perf_counter()
        last_result = predictor(struct_path=structure_path, **kwargs)
        timings.append(time.perf_counter() - start_time)

    cold_time = timings[0]
    warm_times = timings[1:]
    return {
        "structure_id": structure_path.stem,
        "cold_time_seconds": cold_time,
        "cold_time_formatted": format_duration(cold_time),
        "warm_times_seconds": warm_times,
        "warm_mean_seconds": (sum(warm_times) / len(warm_times)) if warm_times else cold_time,
        "result": last_result.to_dict() if last_result is not None else None,
    }


def _load_manifest_pairs(manifest_path: Path, structures_dir: Path) -> list[tuple[Path, str]]:
    pairs: list[tuple[Path, str]] = []
    for row in load_manifest_rows(manifest_path):
        path = structures_dir / f"{row['pdb_id']}.pdb"
        if not path.exists():
            path = structures_dir / f"{row['pdb_id']}.cif"
        if path.exists():
            pairs.append((path, f"{row['chain1']},{row['chain2']}"))
    return pairs


def run_experimental_benchmark(
    manifest_path: Path,
    structures_dir: Path,
    output_dir: Path,
    repeats: int,
    targets: Sequence[str],
    temperature: float,
    distance_cutoff: float,
    acc_threshold: float,
    sphere_points: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    structure_pairs = _load_manifest_pairs(manifest_path, structures_dir)
    benchmark_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": list(targets),
        "repeats": repeats,
        "results": [],
    }

    LOGGER.info(
        "benchmark: %d structures x %d targets x %d repeats",
        len(structure_pairs),
        len(targets),
        repeats,
    )

    for structure_path, selection in structure_pairs:
        try:
            n_atoms = _count_atoms(structure_path, selection)
        except Exception:  # noqa: BLE001
            n_atoms = None

        LOGGER.info("benchmark: %s N=%s sel=%s", structure_path.stem, n_atoms, selection)
        for target in targets:
            if target == "cuda":
                if not cuda_available():
                    benchmark_report["results"].append(
                        {
                            "structure_id": structure_path.stem,
                            "target": target,
                            "status": "skipped",
                            "reason": "CUDA backend not available.",
                        }
                    )
                    continue
                predictor = _load_jax_predictor()
            elif target == "jax":
                predictor = _load_jax_predictor()
            elif target == "jax-soft":
                predictor = _load_jax_soft_predictor()
            elif target == "jax-single":
                predictor = _load_jax_mode_predictor("single")
            elif target == "jax-scan":
                predictor = _load_jax_mode_predictor("scan")
            elif target == "jax-neighbor":
                predictor = _load_jax_mode_predictor("neighbor")
            elif target == "tinygrad":
                predictor = _load_tinygrad_predictor("block")
            elif target == "tinygrad-single":
                predictor = _load_tinygrad_predictor("single")
            elif target == "tinygrad-neighbor":
                predictor = _load_tinygrad_predictor("neighbor")
            else:
                predictor = _load_cpu_predictor()

            try:
                result = _benchmark_single(
                    predictor,
                    structure_path,
                    repeats=repeats,
                    selection=selection,
                    temperature=temperature,
                    distance_cutoff=distance_cutoff,
                    acc_threshold=acc_threshold,
                    sphere_points=sphere_points,
                    save_results=False,
                    quiet=True,
                )
                result["target"] = target
                result["status"] = "ok"
                result["n_atoms"] = n_atoms
                result["selection"] = selection
                benchmark_report["results"].append(result)
                LOGGER.info(
                    "  %-18s ok  cold=%.2fs warm=%.1fms",
                    target,
                    result["cold_time_seconds"],
                    result["warm_mean_seconds"] * 1000,
                )
            except Exception as exc:  # noqa: BLE001
                tb_tail = traceback.format_exc().splitlines()[-6:]
                benchmark_report["results"].append(
                    {
                        "structure_id": structure_path.stem,
                        "target": target,
                        "status": "error",
                        "reason": f"{exc.__class__.__name__}: {exc}",
                        "traceback_tail": tb_tail,
                        "n_atoms": n_atoms,
                        "selection": selection,
                    }
                )
                LOGGER.warning(
                    "  %-18s ERR %s: %s",
                    target,
                    exc.__class__.__name__,
                    str(exc)[:160],
                )

    output_path = output_dir / "benchmark_results.json"
    output_path.write_text(json.dumps(benchmark_report, indent=2, cls=NumpyEncoder))
    return output_path, benchmark_report


def parse_targets(value: str | Sequence[str] | None) -> tuple[str, ...]:
    if value is None:
        return DEFAULT_NOTEBOOK_TARGETS
    if isinstance(value, str):
        targets = [item.strip() for item in value.split(",") if item.strip()]
    else:
        targets = [str(item).strip() for item in value if str(item).strip()]
    if not targets:
        raise ValueError("At least one benchmark target is required.")
    return tuple(targets)


def load_manifest_rows(manifest_path: Path) -> list[dict[str, str]]:
    with Path(manifest_path).open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = []
        for row in reader:
            pdb_id = (row.get("pdb_id") or "").strip().upper()
            chain1 = (row.get("chain1") or "").strip()
            chain2 = (row.get("chain2") or "").strip()
            if not pdb_id or not chain1 or not chain2:
                continue
            rows.append({"pdb_id": pdb_id, "chain1": chain1, "chain2": chain2})
    if not rows:
        raise ValueError(f"No valid manifest rows found in {manifest_path}")
    return rows


def write_manifest_rows(rows: Iterable[dict[str, str]], manifest_path: Path) -> Path:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("pdb_id", "chain1", "chain2"), delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return manifest_path


def materialize_manifest(manifest_path: Path, output_dir: Path, limit: int | None = None) -> Path:
    if limit is None or limit <= 0:
        return manifest_path
    rows = load_manifest_rows(manifest_path)[:limit]
    return write_manifest_rows(rows, output_dir / "manifest_subset.tsv")


def _existing_structure_path(structures_dir: Path, pdb_id: str) -> Path | None:
    normalized = pdb_id.upper()
    for suffix in DOWNLOAD_SUFFIXES:
        candidate = structures_dir / f"{normalized}{suffix}"
        if candidate.exists():
            return candidate
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

    return {
        "requested": len(rows),
        "downloaded": downloaded,
        "existing": existing,
    }


def build_report_dataframe(report: dict):
    import pandas as pd

    rows = []
    for result in report["results"]:
        status = result.get("status")
        rows.append(
            {
                "pdb": result["structure_id"],
                "N": result.get("n_atoms"),
                "target": result["target"],
                "status": status,
                "cold_s": round(result["cold_time_seconds"], 3) if status == "ok" else None,
                "warm_ms": round(result["warm_mean_seconds"] * 1000, 1) if status == "ok" else None,
                "reason": (result.get("reason") or "")[:160] if status != "ok" else "",
            }
        )
    return pd.DataFrame(rows)


def build_warm_ms_table(df):
    if df.empty:
        return df
    ok = df[df["status"] == "ok"]
    if ok.empty:
        return ok
    return (
        ok.pivot_table(index=["pdb", "N"], columns="target", values="warm_ms")
        .sort_index(level="N")
    )


def plot_time_vs_atoms(report: dict, output_path: Path, targets: Sequence[str]) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {
        "cpu": "x",
        "jax": "o",
        "jax-single": "v",
        "jax-scan": "P",
        "jax-soft": "s",
        "tinygrad": "D",
        "tinygrad-single": "d",
        "tinygrad-neighbor": "p",
    }
    colors = {
        "cpu": "#2ca02c",
        "jax": "#1f77b4",
        "jax-single": "#9467bd",
        "jax-scan": "#8c564b",
        "jax-soft": "#aec7e8",
        "tinygrad": "#ff7f0e",
        "tinygrad-single": "#e68a00",
        "tinygrad-neighbor": "#ffb366",
    }

    by_target: dict[str, list[tuple[int, float]]] = {}
    for result in report["results"]:
        if result.get("status") != "ok" or result.get("n_atoms") is None:
            continue
        by_target.setdefault(result["target"], []).append(
            (int(result["n_atoms"]), float(result["warm_mean_seconds"]) * 1000.0)
        )

    try:
        import jax

        backend_label = jax.default_backend()
    except Exception:  # noqa: BLE001
        backend_label = "unknown"

    fig, ax = plt.subplots(figsize=(9, 6))
    for target in targets:
        points = sorted(by_target.get(target, []))
        if not points:
            continue
        xs, ys = zip(*points)
        ax.plot(
            xs,
            ys,
            marker=markers.get(target, "."),
            color=colors.get(target),
            linewidth=1.6,
            markersize=8,
            label=target,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atom14 atoms (non-padding)")
    ax.set_ylabel("warm-mean pipeline time (ms)")
    ax.set_title(f"predict_binding_affinity - JAX backend: {backend_label}")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_sasa_benchmark(
    manifest_path: Path = DEFAULT_MANIFEST,
    structures_dir: Path = DEFAULT_STRUCTURES_DIR,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    repeats: int = DEFAULT_REPEATS,
    targets: Sequence[str] = DEFAULT_NOTEBOOK_TARGETS,
    sphere_points: int = DEFAULT_SPHERE_POINTS,
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    limit: int | None = None,
) -> dict[str, object]:
    manifest_path = Path(manifest_path)
    structures_dir = Path(structures_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_to_use = materialize_manifest(manifest_path, output_dir, limit=limit)
    download_summary = ensure_manifest_structures(manifest_to_use, structures_dir)

    report_path, report = run_experimental_benchmark(
        manifest_path=manifest_to_use,
        structures_dir=structures_dir,
        output_dir=output_dir,
        repeats=repeats,
        targets=tuple(targets),
        temperature=temperature,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        sphere_points=sphere_points,
    )

    plot_path = plot_time_vs_atoms(report, output_dir / "time_vs_atoms.png", targets=targets)

    df = build_report_dataframe(report)
    rows_csv_path = output_dir / "benchmark_rows.csv"
    df.to_csv(rows_csv_path, index=False)

    wide_df = build_warm_ms_table(df)
    warm_table_path = output_dir / "benchmark_warm_ms_wide.csv"
    wide_df.to_csv(warm_table_path)

    ok = sum(1 for row in report["results"] if row.get("status") == "ok")
    err = sum(1 for row in report["results"] if row.get("status") == "error")
    skipped = sum(1 for row in report["results"] if row.get("status") == "skipped")

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest_path": str(manifest_to_use),
        "structures_dir": str(structures_dir),
        "output_dir": str(output_dir),
        "targets": list(targets),
        "repeats": repeats,
        "sphere_points": sphere_points,
        "limit": limit,
        "download": download_summary,
        "ok": ok,
        "error": err,
        "skipped": skipped,
        "artifacts": {
            "report_json": str(report_path),
            "plot_png": str(plot_path),
            "rows_csv": str(rows_csv_path),
            "warm_ms_wide_csv": str(warm_table_path),
        },
    }

    summary_path = output_dir / "benchmark_summary.json"
    summary["artifacts"]["summary_json"] = str(summary_path)
    summary_path.write_text(json.dumps(summary, indent=2))

    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Notebook-style SASA benchmark sweep as a regular Python script."
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument(
        "--targets",
        nargs="+",
        default=DEFAULT_NOTEBOOK_TARGETS,
        choices=(
            "cpu",
            "cuda",
            "jax",
            "jax-single",
            "jax-scan",
            "jax-neighbor",
            "jax-soft",
            "tinygrad",
            "tinygrad-single",
            "tinygrad-neighbor",
        ),
    )
    parser.add_argument("--sphere-points", type=int, default=DEFAULT_SPHERE_POINTS)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--distance-cutoff", type=float, default=5.5)
    parser.add_argument("--acc-threshold", type=float, default=0.05)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional manifest row limit for quick smoke runs.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="[%(name)s] %(message)s")
    logging.getLogger("protein_affinity_gpu").setLevel(level)

    summary = run_sasa_benchmark(
        manifest_path=args.manifest,
        structures_dir=args.structures_dir,
        output_dir=args.output_dir,
        repeats=args.repeats,
        targets=args.targets,
        sphere_points=args.sphere_points,
        temperature=args.temperature,
        distance_cutoff=args.distance_cutoff,
        acc_threshold=args.acc_threshold,
        limit=args.limit if args.limit > 0 else None,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
