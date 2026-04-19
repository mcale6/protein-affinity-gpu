#!/usr/bin/env python3

import argparse
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    from protein_affinity_gpu.cpu import predict_binding_affinity  # noqa: E402
    from protein_affinity_gpu.resources import collect_structure_files, format_duration  # noqa: E402
    from protein_affinity_gpu.results import NumpyEncoder  # noqa: E402
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guidance
    raise SystemExit(
        f"Missing Python dependency '{exc.name}'. "
        "Use the repo virtualenv or install dependencies with "
        "\".venv/bin/python -m pip install -e '.[compare]'\"."
    ) from exc

LOGGER = logging.getLogger(__name__)


def _load_jax_predictor():
    from protein_affinity_gpu.jax import predict_binding_affinity_jax

    return predict_binding_affinity_jax


def _load_tinygrad_predictor():
    from protein_affinity_gpu.tinygrad import predict_binding_affinity_tinygrad

    return predict_binding_affinity_tinygrad


def cuda_available() -> bool:
    try:
        import jax
    except ImportError:
        return False

    platforms = {device.platform.lower() for device in jax.devices()}
    return "gpu" in platforms or "cuda" in platforms


def tinygrad_available() -> bool:
    try:
        import tinygrad  # noqa: F401
    except Exception:
        return False
    return True


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


def run_benchmark(
    input_path: Path,
    output_dir: Path,
    repeats: int = 3,
    targets: tuple[str, ...] = ("cpu", "cuda", "tinygrad"),
    selection: str = "A,B",
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    structures = collect_structure_files(input_path)
    benchmark_report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "targets": list(targets),
        "repeats": repeats,
        "results": [],
    }

    for structure_path in structures:
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
            elif target == "tinygrad":
                if not tinygrad_available():
                    benchmark_report["results"].append(
                        {
                            "structure_id": structure_path.stem,
                            "target": target,
                            "status": "skipped",
                            "reason": "tinygrad backend not available.",
                        }
                    )
                    continue
                predictor = _load_tinygrad_predictor()
            else:
                predictor = predict_binding_affinity

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
            benchmark_report["results"].append(result)

    output_path = output_dir / "benchmark_results.json"
    output_path.write_text(json.dumps(benchmark_report, indent=2, cls=NumpyEncoder))
    return output_path, benchmark_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark CPU, CUDA (JAX), and tinygrad prediction targets.")
    parser.add_argument("input_path", type=Path, help="Path to a structure file or directory.")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/output"), help="Benchmark artifact directory.")
    parser.add_argument("--repeats", type=int, default=3, help="Number of runs per target.")
    parser.add_argument("--selection", default="A,B", help="Two-chain selection, for example 'A,B'.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius.")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Interface distance cutoff in angstrom.")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Relative SASA threshold.")
    parser.add_argument("--sphere-points", type=int, default=100, help="Number of sphere points for SASA.")
    parser.add_argument(
        "--targets",
        nargs="+",
        choices=("cpu", "cuda", "tinygrad"),
        default=("cpu", "cuda", "tinygrad"),
        help="Benchmark targets to run.",
    )
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if not args.input_path.exists():
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
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error(str(exc))
        return 1

    LOGGER.info("Benchmark results written to %s", output_path)
    print(json.dumps(report, indent=2, cls=NumpyEncoder))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
