#!/usr/bin/env python3
"""Local benchmark harness (Apple M1 Max / CPU box).

Runs the CPU and tinygrad targets (``cpu``, ``tinygrad-single``,
``tinygrad-batch``) over a Kahraman-style manifest and writes the unified
``results.csv`` / ``summary.json`` used by ``plot_results.py``.

The GPU equivalent (JAX single / batch / scan + tinygrad single / batch)
lives in ``benchmarks/modal_benchmark.py``; both runners share the inner
loop in ``benchmarks/sasa/sasa_benchmark.py`` so the CSV schema is
identical and the two outputs can be merged into a single figure.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmarks.sasa.sasa_benchmark import (  # noqa: E402
    BACKENDS,
    DEFAULT_MANIFEST,
    DEFAULT_REPEATS,
    DEFAULT_SPHERE_POINTS,
    DEFAULT_STRUCTURES_DIR,
    LOCAL_DEFAULT_TARGETS,
    run_benchmark,
)

LOGGER = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Local benchmark harness — CPU + tinygrad targets over a "
            "Kahraman-style manifest. Writes results.csv + summary.json into "
            "--output-dir."
        ),
    )
    parser.add_argument(
        "--manifest", type=Path, default=DEFAULT_MANIFEST,
        help="TSV with pdb_id/chain1/chain2 columns.",
    )
    parser.add_argument(
        "--structures-dir", type=Path, default=DEFAULT_STRUCTURES_DIR,
        help="Directory of PDB/CIF files (created + populated if missing).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmarks/output/local",
        help="Where to write results.csv, summary.json, manifest_subset.tsv.",
    )
    parser.add_argument(
        "--targets", nargs="+", default=list(LOCAL_DEFAULT_TARGETS),
        choices=sorted(BACKENDS),
        help=f"Targets to run (default: {' '.join(LOCAL_DEFAULT_TARGETS)}).",
    )
    parser.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    parser.add_argument("--sphere-points", type=int, default=DEFAULT_SPHERE_POINTS)
    parser.add_argument("--temperature", type=float, default=25.0)
    parser.add_argument("--distance-cutoff", type=float, default=5.5)
    parser.add_argument("--acc-threshold", type=float, default=0.05)
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Optional manifest row limit for quick smoke runs.",
    )
    parser.add_argument(
        "--device-label", default=None,
        help="Override the detected device label recorded in each row.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="[%(name)s] %(message)s",
    )
    logging.getLogger("protein_affinity_gpu").setLevel(
        logging.INFO if args.verbose else logging.WARNING
    )

    if not args.manifest.exists():
        parser.error(f"Manifest not found: {args.manifest}")

    try:
        summary = run_benchmark(
            manifest_path=args.manifest,
            structures_dir=args.structures_dir,
            output_dir=args.output_dir,
            backends=args.targets,
            repeats=args.repeats,
            temperature=args.temperature,
            distance_cutoff=args.distance_cutoff,
            acc_threshold=args.acc_threshold,
            sphere_points=args.sphere_points,
            limit=args.limit if args.limit > 0 else None,
            device=args.device_label,
        )
    except Exception as exc:  # pragma: no cover - CLI surface
        LOGGER.error(str(exc))
        return 1

    print(json.dumps(summary, indent=2))
    completed = max(
        (entry.get("completed", 0) for entry in summary["per_backend"].values()),
        default=0,
    )
    return 0 if completed else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
