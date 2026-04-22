#!/usr/bin/env python3
"""Modal entrypoint for the notebook-style SASA benchmark sweep."""
from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover - import-time guidance
    raise SystemExit(
        "Missing Python dependency 'modal'. Install it with "
        "\"python -m pip install -e '.[modal]'\" or \"pip install modal\"."
    ) from exc

from benchmarks.sasa_benchmark import DEFAULT_GPU_TARGETS, parse_targets, run_sasa_benchmark  # noqa: E402

LOGGER = logging.getLogger(__name__)

APP_NAME = "protein-affinity-gpu-benchmark"
GPU_TYPE = os.environ.get("MODAL_GPU", "A100-80GB")
VOLUME_NAME = os.environ.get("MODAL_BENCHMARK_VOLUME", "protein-affinity-gpu-benchmarks")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_BENCHMARK_TIMEOUT", str(8 * 60 * 60)))
VOLUME_MOUNT = "/vol"
REMOTE_REPO_ROOT = Path("/root")
REMOTE_VOLUME_ROOT = Path(VOLUME_MOUNT)
REMOTE_MANIFEST = REMOTE_REPO_ROOT / "benchmarks/datasets/kahraman_2013_t3.tsv"
REMOTE_STRUCTURES_DIR = REMOTE_VOLUME_ROOT / "datasets/kahraman_2013_t3/structures"
REMOTE_RUNS_DIR = REMOTE_VOLUME_ROOT / "runs"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "biopython",
        "numpy>=1.23,<3.0",
        "jax[cuda12]",
        "tinygrad",
        "matplotlib",
        "pandas",
    )
    .add_local_dir(
        "src",
        remote_path="/root/src",
        ignore=["**/__pycache__/**"],
    )
    .add_local_dir(
        "benchmarks",
        remote_path="/root/benchmarks",
        ignore=[
            "downloads/**",
            "output/**",
            "*.ipynb",
            "**/__pycache__/**",
        ],
    )
    .workdir("/root")
    .env({"PYTHONPATH": "/root:/root/src"})
)
app = modal.App(APP_NAME, image=image)


def _validate_modal_targets(targets: tuple[str, ...]) -> tuple[str, ...]:
    if "cpu" in targets:
        raise ValueError(
            "The Modal image is GPU-only and does not install 'prodigy-prot' or "
            "'freesasa'. Remove 'cpu' from --targets for Modal runs."
        )
    return targets


def _volume_relative(path: Path) -> str:
    return path.relative_to(REMOTE_VOLUME_ROOT).as_posix()


def _download_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(volume.read_file(remote_path))
    local_path.write_bytes(payload)


@app.function(
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
)
def run_remote_benchmark(
    repeats: int = 2,
    targets: str = ",".join(DEFAULT_GPU_TARGETS),
    sphere_points: int = 100,
    limit: int = 0,
    run_name: str = "",
) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)
    logging.getLogger("protein_affinity_gpu").setLevel(logging.INFO)

    resolved_run_name = run_name.strip() or datetime.now(timezone.utc).strftime(
        "sasa-benchmark-%Y%m%d-%H%M%S"
    )
    output_dir = REMOTE_RUNS_DIR / resolved_run_name
    resolved_targets = _validate_modal_targets(parse_targets(targets))
    summary = run_sasa_benchmark(
        manifest_path=REMOTE_MANIFEST,
        structures_dir=REMOTE_STRUCTURES_DIR,
        output_dir=output_dir,
        repeats=repeats,
        targets=resolved_targets,
        sphere_points=sphere_points,
        limit=limit if limit > 0 else None,
    )
    volume.commit()

    artifacts = {
        name: _volume_relative(Path(path))
        for name, path in summary["artifacts"].items()
    }
    return {
        **summary,
        "run_name": resolved_run_name,
        "gpu": GPU_TYPE,
        "volume_name": VOLUME_NAME,
        "volume_output_dir": _volume_relative(output_dir),
        "artifacts": artifacts,
    }


@app.local_entrypoint()
def main(
    repeats: int = 2,
    targets: str = ",".join(DEFAULT_GPU_TARGETS),
    sphere_points: int = 100,
    limit: int = 0,
    run_name: str = "",
    local_output_dir: str = "",
):
    summary = run_remote_benchmark.remote(
        repeats=repeats,
        targets=targets,
        sphere_points=sphere_points,
        limit=limit,
        run_name=run_name,
    )

    if local_output_dir.strip():
        local_dir = Path(local_output_dir).expanduser().resolve()
        for name, remote_path in summary["artifacts"].items():
            _download_file(remote_path, local_dir / Path(remote_path).name)
        summary["local_output_dir"] = str(local_dir)

    print(json.dumps(summary, indent=2))
