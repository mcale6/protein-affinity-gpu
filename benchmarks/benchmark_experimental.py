#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from protein_affinity_gpu.cpu import predict_binding_affinity  # noqa: E402
    from protein_affinity_gpu.utils.resources import collect_structure_files, format_duration  # noqa: E402
    from protein_affinity_gpu.utils._array import NumpyEncoder  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guidance
    raise SystemExit(
        f"Missing Python dependency '{exc.name}'. "
        "Use the repo virtualenv or install dependencies with "
        "\".venv/bin/python -m pip install -e '.[compare]'\"."
    ) from exc

LOGGER = logging.getLogger(__name__)


def _load_jax_predictor():
    from protein_affinity_gpu.predict import predict_binding_affinity_jax

    return predict_binding_affinity_jax


def _load_jax_soft_predictor():
    from protein_affinity_gpu.experimental import (
        predict_binding_affinity_jax_experimental,
    )

    def predictor(**kwargs):
        return predict_binding_affinity_jax_experimental(
            soft_sasa=True, soft_beta=10.0, **kwargs,
        )

    return predictor


def _load_jax_mode_predictor(mode: str):
    # ``"block"`` / ``"scan"`` are in the default entry point; ``"single"`` /
    # ``"neighbor"`` require the experimental one. Route accordingly so the
    # default surface stays minimal.
    if mode in ("block", "scan"):
        from protein_affinity_gpu.predict import predict_binding_affinity_jax as jax_fn

        def predictor(**kwargs):
            return jax_fn(mode=mode, **kwargs)

        return predictor

    from protein_affinity_gpu.experimental import (
        predict_binding_affinity_jax_experimental,
    )

    def predictor(**kwargs):
        return predict_binding_affinity_jax_experimental(mode=mode, **kwargs)

    return predictor


def _load_tinygrad_predictor(mode: str = "block"):
    from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad

    def predictor(**kwargs):
        return predict_binding_affinity_tinygrad(mode=mode, **kwargs)

    return predictor


def _count_atoms(structure_path: Path, selection: str) -> int:
    """Atom14-compacted atom count (matches what the SASA kernel sees)."""
    from protein_affinity_gpu.utils.atom14 import compact_complex_atom14
    from protein_affinity_gpu.utils.structure import load_complex

    target, binder = load_complex(structure_path, selection=selection, sanitize=True)
    positions, mask, _, _ = compact_complex_atom14(target, binder)
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


def _load_manifest(manifest_path: Path, structures_dir: Path) -> list[tuple[Path, str]]:
    """Parse a Kahraman-style TSV (``pdb_id\tchain1\tchain2\t...``)."""
    pairs: list[tuple[Path, str]] = []
    with manifest_path.open() as f:
        header = f.readline()  # skip
        del header
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3 or not parts[0]:
                continue
            pdb_id = parts[0].upper()
            chain1, chain2 = parts[1], parts[2]
            path = structures_dir / f"{pdb_id}.pdb"
            if not path.exists():
                path = structures_dir / f"{pdb_id}.cif"
            if path.exists():
                pairs.append((path, f"{chain1},{chain2}"))
    return pairs


def run_benchmark(
    input_path: Path | None,
    output_dir: Path,
    repeats: int = 3,
    targets: tuple[str, ...] = ("cpu", "cuda", "tinygrad"),
    selection: str = "A,B",
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    manifest: Path | None = None,
    structures_dir: Path | None = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest is not None:
        if structures_dir is None:
            raise ValueError("--manifest requires --structures-dir")
        structure_pairs = _load_manifest(manifest, structures_dir)
    else:
        structures = collect_structure_files(input_path)
        structure_pairs = [(path, selection) for path in structures]
    benchmark_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": list(targets),
        "repeats": repeats,
        "results": [],
    }

    LOGGER.info(
        "benchmark: %d structures × %d targets × %d repeats",
        len(structure_pairs), len(targets), repeats,
    )

    for structure_path, sel in structure_pairs:
        try:
            n_atoms = _count_atoms(structure_path, sel)
        except Exception:  # noqa: BLE001
            n_atoms = None

        LOGGER.info("benchmark: %s N=%s sel=%s", structure_path.stem, n_atoms, sel)
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
                predictor = predict_binding_affinity

            try:
                result = _benchmark_single(
                    predictor,
                    structure_path,
                    repeats=repeats,
                    selection=sel,
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
                result["selection"] = sel
                benchmark_report["results"].append(result)
                LOGGER.info(
                    "  %-12s ok  cold=%.2fs warm=%.1fms",
                    target, result["cold_time_seconds"], result["warm_mean_seconds"] * 1000,
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
                        "selection": sel,
                    }
                )
                LOGGER.warning(
                    "  %-12s ERR %s: %s", target, exc.__class__.__name__, str(exc)[:160],
                )

    output_path = output_dir / "benchmark_results.json"
    output_path.write_text(json.dumps(benchmark_report, indent=2, cls=NumpyEncoder))

    ok = sum(1 for r in benchmark_report["results"] if r.get("status") == "ok")
    err = sum(1 for r in benchmark_report["results"] if r.get("status") == "error")
    skipped = sum(1 for r in benchmark_report["results"] if r.get("status") == "skipped")
    LOGGER.info("benchmark: finished ok=%d err=%d skipped=%d → %s", ok, err, skipped, output_path)
    if err:
        err_by_target: dict[str, int] = {}
        for row in benchmark_report["results"]:
            if row.get("status") == "error":
                err_by_target[row["target"]] = err_by_target.get(row["target"], 0) + 1
        for tgt, count in sorted(err_by_target.items()):
            LOGGER.warning("  %d errors on target %s", count, tgt)

    return output_path, benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CPU, CUDA (JAX), and tinygrad prediction targets.")
    parser.add_argument(
        "input_path", type=Path, nargs="?", default=None,
        help="Path to a structure file or directory (omit when using --manifest).",
    )
    parser.add_argument(
        "--manifest", type=Path, default=None,
        help="TSV with pdb_id / chain1 / chain2 columns; pairs structures with per-PDB selections.",
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=None,
        help="Directory containing PDB files referenced in the manifest.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output"), help="Benchmark artifact directory.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per target.")
    parser.add_argument("--selection", default="A,B", help="Two-chain selection, for example 'A,B'.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius.")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Interface distance cutoff in angstrom.")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Relative SASA threshold.")
    parser.add_argument("--sphere-points", type=int, default=100, help="Number of sphere points for SASA.")
    # On Mac (tinygrad Metal, no CUDA), skip JAX CPU targets by default — they
    # take an order of magnitude longer and rarely tell you anything interesting.
    # GPU hosts keep the full sweep (cold compile on jax-single / jax-scan is
    # the whole point of benchmarking those paths).
    default_mac = ("cpu", "tinygrad", "tinygrad-single", "tinygrad-neighbor")
    default_gpu = ("cpu", "jax", "jax-single", "jax-scan", "jax-neighbor", "jax-soft",
                   "tinygrad", "tinygrad-single", "tinygrad-neighbor")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("cpu", "cuda", "jax", "jax-single", "jax-scan", "jax-neighbor", "jax-soft",
                 "tinygrad", "tinygrad-single", "tinygrad-neighbor"),
        default=default_mac if sys.platform == "darwin" else default_gpu,
        help="Benchmark targets to run.",
    )
    parser.add_argument(
        "--plot", type=Path, default=None,
        help="If set, save a log-log PNG of warm-mean time vs atom count.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    return parser


def _plot_report(report: dict, out_path: Path) -> None:
    """Log-log scatter of warm-mean time vs atom count, one line per target."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    markers = {
        "cpu": "x", "cuda": "o", "jax": "o", "jax-soft": "s",
        "jax-single": "v", "jax-scan": "P", "jax-neighbor": "p",
        "tinygrad": "D", "tinygrad-single": "d", "tinygrad-neighbor": "p",
    }
    colors = {
        "cpu": "#2ca02c", "cuda": "#1f77b4", "jax": "#1f77b4",
        "jax-soft": "#aec7e8", "jax-single": "#9467bd", "jax-scan": "#8c564b",
        "jax-neighbor": "#17becf",
        "tinygrad": "#ff7f0e", "tinygrad-single": "#e68a00", "tinygrad-neighbor": "#ffb366",
    }

    by_target: dict[str, list[tuple[int, float]]] = {}
    for row in report["results"]:
        if row.get("status") != "ok" or row.get("n_atoms") is None:
            continue
        by_target.setdefault(row["target"], []).append(
            (int(row["n_atoms"]), float(row["warm_mean_seconds"]) * 1000.0)
        )

    fig, ax = plt.subplots(figsize=(8, 6))
    for target, pts in sorted(by_target.items()):
        pts.sort()
        xs, ys = zip(*pts)
        ax.plot(
            xs, ys,
            marker=markers.get(target, "."),
            color=colors.get(target),
            linewidth=1.6,
            markersize=7,
            label=target,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("atom14 atoms (non-padding)")
    ax.set_ylabel("warm-mean pipeline time (ms)")
    ax.set_title("predict_binding_affinity: time vs N atoms")
    ax.grid(True, which="both", ls=":", alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    level = logging.INFO if args.verbose else logging.WARNING
    logging.basicConfig(level=level, format="%(message)s")
    # Also route package-scoped debug output so adapter/sasa logs surface.
    logging.getLogger("protein_affinity_gpu").setLevel(level)

    if args.manifest is None and args.input_path is None:
        parser.error("Pass a structure path or --manifest PATH --structures-dir DIR.")
    if args.manifest is not None and args.structures_dir is None:
        parser.error("--manifest requires --structures-dir.")
    if args.input_path is not None and not args.input_path.exists():
        parser.error(f"Input path not found: {args.input_path}")

    try:
        output_path, report = run_benchmark(
            input_path=args.input_path,
            output_dir=args.output_dir,
            repeats=args.repeats,
            targets=tuple(args.targets),
            selection=args.selection,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            manifest=args.manifest,
            structures_dir=args.structures_dir,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error(str(exc))
        return 1

    LOGGER.info("Benchmark results written to %s", output_path)

    plot_path = args.plot if args.plot else args.output_dir / "benchmark_results.png"
    try:
        _plot_report(report, plot_path)
        LOGGER.info("Plot written to %s", plot_path)
    except Exception as exc:  # noqa: BLE001
        LOGGER.warning("Plot failed: %s", exc)

    print(json.dumps(report, indent=2, cls=NumpyEncoder))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
