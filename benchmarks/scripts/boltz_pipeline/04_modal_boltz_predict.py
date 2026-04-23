#!/usr/bin/env python3
"""Modal Boltz-2 batch runner for the Kastritis 81.

Two modes per complex (``msa_only``, ``template_msa``). Defaults to a
2-prediction smoke run (``--limit 1`` x 2 modes); bump ``--limit`` or use
``--pdb-ids`` to scale.

Example:
    modal run benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py \\
      --limit 1 \\
      --modes msa_only,template_msa

    modal run benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py \\
      --pdb-ids 2OOB,3BZD \\
      --modes msa_only,template_msa

The function takes YAML text + optional template CIF bytes, writes both to a
per-job temp dir, runs ``boltz predict --use_msa_server``, and returns the
output folder as a tarball. --use_msa_server pulls paired + unpaired MSAs
from the ColabFold MMseqs2 server by default.

Outputs land in ``benchmarks/output/kastritis_81_boltz/{mode}/{pdb_id}/`` --
the tarball's top-level directory is ``{pdb_id}_{mode}`` for clean extract.

See docs/BOLTZ_PIPELINE.md for the full design.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

try:
    import modal
except ModuleNotFoundError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing 'modal'. Install via \"pip install modal\" "
        "or \"python -m pip install -e '.[modal]'\"."
    ) from exc

APP_NAME = "kastritis-boltz-predict"
GPU_TYPE = os.environ.get("MODAL_GPU", "A100-80GB")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_BOLTZ_TIMEOUT", str(20 * 60)))
BOLTZ_VERSION = os.environ.get("BOLTZ_VERSION", "2.1.1")
BOLTZ_REVISION = os.environ.get(
    "BOLTZ_REVISION", "6fdef46d763fee7fbb83ca5501ccceff43b85607"
)

boltz_vol = modal.Volume.from_name("boltz-models", create_if_missing=True)
MODELS_DIR = Path("/models/boltz")

# Boltz-2 installs torch with CUDA 13 runtime wheels but relies on
# ``libnvrtc-builtins.so.13.0`` being present at runtime for JIT compilation of
# reduction kernels. ``debian_slim`` doesn't ship it, so we pull an NVIDIA
# CUDA 13 runtime image instead (same pattern as modal_afdesign_ba_val.py,
# which uses 12.4.1 for its torch version).
image = (
    modal.Image.from_registry(
        "nvidia/cuda:13.0.0-runtime-ubuntu22.04",
        add_python="3.12",
    )
    .apt_install("gcc")   # triton's NVIDIA backend JITs host code, needs a C compiler
    .uv_pip_install(f"boltz=={BOLTZ_VERSION}")
)
download_image = (
    modal.Image.debian_slim()
    .uv_pip_install("huggingface-hub==0.36.0")
    .env({"HF_XET_HIGH_PERFORMANCE": "1"})
)

app = modal.App(APP_NAME)


@app.function(
    image=download_image,
    volumes={MODELS_DIR: boltz_vol},
    timeout=20 * 60,
)
def download_model(force_download: bool = False) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id="boltz-community/boltz-2",
        revision=BOLTZ_REVISION,
        local_dir=MODELS_DIR,
        force_download=force_download,
    )
    boltz_vol.commit()


@app.function(
    image=image,
    volumes={MODELS_DIR: boltz_vol},
    timeout=TIMEOUT_SECONDS,
    gpu=GPU_TYPE,
)
def boltz_inference(
    yaml_text: str,
    template_cif: bytes | None,
    pdb_id: str,
    mode: str,
) -> bytes:
    import io
    import subprocess
    import tarfile
    import tempfile

    workdir = Path(tempfile.mkdtemp(prefix=f"boltz-{pdb_id}-{mode}-"))
    input_yaml = workdir / "input.yaml"
    input_yaml.write_text(yaml_text)
    if template_cif is not None:
        (workdir / "template.cif").write_bytes(template_cif)

    out_dir = workdir / "out"
    cmd = [
        "boltz", "predict", str(input_yaml),
        "--use_msa_server",
        "--cache", str(MODELS_DIR),
        "--out_dir", str(out_dir),
        "--override",
    ]
    print(f"[boltz] {pdb_id}/{mode} running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True, cwd=workdir)

    # Boltz writes to out_dir/boltz_results_input/predictions/input/ by default.
    # Tar the whole out_dir under a clear name.
    tar_buf = io.BytesIO()
    arcname = f"{pdb_id}_{mode}"
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tar:
        tar.add(out_dir, arcname=arcname)
    return tar_buf.getvalue()


def _local_paths() -> dict[str, Path]:
    """Resolve local repo paths. Only called from the local entrypoint --
    NOT at module import time, because Modal imports this script inside its
    container where ``__file__`` has a different parent depth.
    """
    root = Path(__file__).resolve().parents[3]
    return {
        "root": root,
        "manifest_csv": root / "benchmarks/datasets/kastritis_81/manifest.csv",
        "yaml_root": root / "benchmarks/downloads/kastritis_81_boltz_inputs",
        "cleaned_dir": root / "benchmarks/downloads/kastritis_81/cleaned",
        "out_root": root / "benchmarks/output/kastritis_81_boltz",
    }


def _load_manifest(manifest_csv: Path) -> list[dict]:
    return list(csv.DictReader(manifest_csv.open()))


def _filter_rows(rows: list[dict], limit: int, pdb_ids: str) -> list[dict]:
    if pdb_ids.strip():
        keep = {p.strip() for p in pdb_ids.split(",") if p.strip()}
        filtered = [r for r in rows if r["pdb_id"] in keep]
        missing = keep - {r["pdb_id"] for r in filtered}
        if missing:
            print(f"[warn] requested pdb_ids not in manifest: {sorted(missing)}")
        return filtered
    return rows[:limit]


def _load_yaml(yaml_root: Path, pdb_id: str, mode: str) -> str:
    p = yaml_root / mode / f"{pdb_id}.yaml"
    if not p.exists():
        raise FileNotFoundError(
            f"missing YAML: {p}. Run 03_build_boltz_yaml.py first."
        )
    return p.read_text()


def _load_template_cif(cleaned_dir: Path, pdb_id: str) -> bytes:
    p = cleaned_dir / f"{pdb_id}_AB.cif"
    if not p.exists():
        raise FileNotFoundError(
            f"missing cleaned CIF: {p}. Run 03_build_boltz_yaml.py first."
        )
    return p.read_bytes()


@app.local_entrypoint()
def main(
    limit: int = 1,
    pdb_ids: str = "",
    modes: str = "msa_only,template_msa",
    force_download: bool = False,
):
    paths = _local_paths()
    rows = _load_manifest(paths["manifest_csv"])
    if not rows:
        print(f"[fatal] manifest is empty: {paths['manifest_csv']}", file=sys.stderr)
        sys.exit(1)
    rows = _filter_rows(rows, limit, pdb_ids)
    if not rows:
        print("[fatal] no rows selected after filter", file=sys.stderr)
        sys.exit(1)

    mode_list = [m.strip() for m in modes.split(",") if m.strip()]
    for mode in mode_list:
        if mode not in ("msa_only", "template_msa"):
            raise ValueError(f"unknown mode: {mode!r}")

    print(f"Selected {len(rows)} complexes x {len(mode_list)} modes "
          f"= {len(rows) * len(mode_list)} predictions")
    for r in rows:
        print(f"  - {r['pdb_id']:<6}  T={r['len_target']:>4}  B={r['len_binder']:>4}")

    print("\nEnsuring Boltz-2 weights are on the shared Volume...")
    download_model.remote(force_download)

    jobs: list[tuple[str, str, str, bytes | None]] = []
    for row in rows:
        pdb_id = row["pdb_id"]
        for mode in mode_list:
            yaml_text = _load_yaml(paths["yaml_root"], pdb_id, mode)
            template_cif = (
                _load_template_cif(paths["cleaned_dir"], pdb_id)
                if mode == "template_msa" else None
            )
            jobs.append((pdb_id, mode, yaml_text, template_cif))

    print(f"\nLaunching {len(jobs)} predictions on {GPU_TYPE}...")
    futures = []
    for pdb_id, mode, yaml_text, template_cif in jobs:
        fut = boltz_inference.spawn(yaml_text, template_cif, pdb_id, mode)
        futures.append((pdb_id, mode, fut))

    import tarfile

    out_root = paths["out_root"]
    out_root.mkdir(parents=True, exist_ok=True)
    for pdb_id, mode, fut in futures:
        print(f"[wait] {pdb_id} / {mode} ...")
        try:
            tar_bytes = fut.get()
        except Exception as exc:  # noqa: BLE001
            print(f"[FAIL] {pdb_id} / {mode}: {exc}")
            continue
        out_dir = out_root / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        tar_path = out_dir / f"{pdb_id}.tar.gz"
        tar_path.write_bytes(tar_bytes)
        with tarfile.open(tar_path) as tar:
            tar.extractall(out_dir)
        print(f"[done] {out_dir / f'{pdb_id}_{mode}'} (+ {tar_path.name})")

    print(f"\nAll outputs under {out_root.relative_to(paths['root'])}/")


if __name__ == "__main__":
    main()
