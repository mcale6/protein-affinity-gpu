import argparse
import json
import logging
import sys
from pathlib import Path

from ..cpu import predict_binding_affinity
from ..resources import collect_structure_files
from ..results import NumpyEncoder

LOGGER = logging.getLogger(__name__)


def _load_jax_predictor():
    from ..jax import predict_binding_affinity_jax

    return predict_binding_affinity_jax


def _load_tinygrad_predictor():
    from ..tinygrad import predict_binding_affinity_tinygrad

    return predict_binding_affinity_tinygrad


def _resolve_predictor(backend: str):
    if backend == "jax":
        return _load_jax_predictor()
    if backend == "tinygrad":
        return _load_tinygrad_predictor()
    return predict_binding_affinity


def run_predictions(
    input_path: Path,
    backend: str,
    selection: str = "A,B",
    temperature: float = 25.0,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    sphere_points: int = 100,
    output_json: bool = False,
    output_dir: Path | None = None,
    verbose: bool = False,
):
    predictor = _resolve_predictor(backend)
    results = {}
    for structure_path in collect_structure_files(input_path):
        prediction = predictor(
            struct_path=structure_path,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
            save_results=bool(output_json and output_dir),
            output_dir=output_dir or Path("."),
            quiet=not verbose,
        )
        results[structure_path.stem] = prediction
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Predict protein binding affinity for one or more structures.")
    parser.add_argument("input_path", type=Path, help="Path to a structure file or directory.")
    parser.add_argument(
        "--backend",
        choices=("cpu", "jax", "tinygrad"),
        default="cpu",
        help="Prediction backend.",
    )
    parser.add_argument("--selection", default="A,B", help="Two-chain selection, for example 'A,B'.")
    parser.add_argument("--temperature", type=float, default=25.0, help="Temperature in Celsius.")
    parser.add_argument("--distance-cutoff", type=float, default=5.5, help="Interface distance cutoff in angstrom.")
    parser.add_argument("--acc-threshold", type=float, default=0.05, help="Relative SASA threshold.")
    parser.add_argument("--sphere-points", type=int, default=100, help="Number of sphere points for SASA.")
    parser.add_argument("--output-json", action="store_true", help="Write JSON results for each structure.")
    parser.add_argument("--output-dir", type=Path, default=Path("results"), help="Directory for JSON outputs.")
    parser.add_argument("--verbose", action="store_true", help="Enable informational logging.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")

    if not args.input_path.exists():
        parser.error(f"Input path not found: {args.input_path}")

    try:
        results = run_predictions(
            input_path=args.input_path,
            backend=args.backend,
            selection=args.selection,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            output_json=args.output_json,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error(str(exc))
        return 1

    payload = {name: result.to_dict() for name, result in results.items()}
    json.dump(payload, sys.stdout, indent=2, cls=NumpyEncoder)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
