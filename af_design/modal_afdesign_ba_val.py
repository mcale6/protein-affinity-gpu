#!/usr/bin/env python3
"""Single-file Modal entrypoint for AfDesign binder design with a JAX SASA loss.

This rewrites the Colab-style "AF + custom ba_val loss" workflow into a
standalone Modal app:

1. Upload a local target PDB into a Modal Volume.
2. Cache AlphaFold parameters in that same Volume.
3. Run AfDesign binder hallucination on an A100-80GB GPU.
4. Save best-structure artifacts and a metrics JSON back to the Volume.

Example:
    modal run benchmarks/modal_afdesign_ba_val.py \
      --pdb-path /path/to/EGFR_oneBchain.pdb \
      --chain B \
      --binder-len 20 \
      --num-steps 80 \
      --run-name egfr-ba-val \
      --local-output-dir benchmarks/output/egfr-ba-val
"""
from __future__ import annotations

import json
import logging
import os
import sys
import tarfile
import tempfile
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

APP_NAME = "protein-affinity-gpu-afdesign-ba-val"
GPU_TYPE = os.environ.get("MODAL_GPU", "A100-80GB")
VOLUME_NAME = os.environ.get("MODAL_AFDESIGN_VOLUME", "protein-affinity-gpu-afdesign")
TIMEOUT_SECONDS = int(os.environ.get("MODAL_AFDESIGN_TIMEOUT", str(8 * 60 * 60)))
ALPHAFOLD_PARAMS_URL = (
    "https://storage.googleapis.com/alphafold/alphafold_params_2022-12-06.tar"
)
COLABDESIGN_GIT_URL = "git+https://github.com/sokrypton/ColabDesign.git@v1.1.1"

VOLUME_MOUNT = "/vol"
REMOTE_VOLUME_ROOT = Path(VOLUME_MOUNT)
REMOTE_INPUTS_DIR = REMOTE_VOLUME_ROOT / "inputs"
REMOTE_OUTPUTS_DIR = REMOTE_VOLUME_ROOT / "outputs"
REMOTE_AF_DATA_DIR = REMOTE_VOLUME_ROOT / "alphafold"
REMOTE_PARAMS_DIR = REMOTE_AF_DATA_DIR / "params"

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
    )
    .apt_install("clang", "git")
    .pip_install(
        "biopython",
        "numpy>=1.23,<3.0",
        "jax[cuda12]<0.5",
        "tinygrad",
        COLABDESIGN_GIT_URL,
    )
    .env(
        {
            "PYTHONPATH": "/root:/root/src",
            "TF_CPP_MIN_LOG_LEVEL": "3",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": "0.95",
            "JAX_DEFAULT_MATMUL_PRECISION": "highest",
        }
    )
    .workdir("/root")
    .add_local_dir(
        "src",
        remote_path="/root/src",
        ignore=["**/__pycache__/**"],
    )
)
app = modal.App(APP_NAME, image=image)

VALID_BINDER_SEQ_MODES = ("soft", "pseudo")
VALID_DESIGN_MODES = ("logits", "soft")


def _volume_relative(path: Path) -> str:
    return path.relative_to(REMOTE_VOLUME_ROOT).as_posix()


def _download_volume_file(remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    payload = b"".join(volume.read_file(remote_path))
    local_path.write_bytes(payload)


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:  # noqa: BLE001
            pass
    if hasattr(value, "tolist"):
        try:
            return value.tolist()
        except Exception:  # noqa: BLE001
            pass
    return value


def _normalize_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        available = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {available}. Got: {value!r}")
    return normalized


def _ensure_alphafold_params() -> None:
    REMOTE_PARAMS_DIR.mkdir(parents=True, exist_ok=True)
    if any(REMOTE_PARAMS_DIR.glob("*.npz")):
        return

    with tempfile.NamedTemporaryFile(suffix=".tar", delete=False) as handle:
        tmp_tar = Path(handle.name)

    try:
        logging.info("Downloading AlphaFold parameters to %s", REMOTE_PARAMS_DIR)
        urllib.request.urlretrieve(ALPHAFOLD_PARAMS_URL, tmp_tar)
        with tarfile.open(tmp_tar) as archive:
            archive.extractall(REMOTE_PARAMS_DIR)
        volume.commit()
    finally:
        tmp_tar.unlink(missing_ok=True)


@app.function(
    gpu=GPU_TYPE,
    timeout=TIMEOUT_SECONDS,
    volumes={VOLUME_MOUNT: volume},
)
def run_afdesign_binder(
    pdb_volume_path: str,
    chain: str = "B",
    binder_len: int = 20,
    binder_chain: str = "",
    run_name: str = "",
    num_steps: int = 80,
    sphere_points: int = 100,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    soft_sasa_beta: float = 10.0,
    contact_beta: float = 8.0,
    nis_beta: float = 20.0,
    rg_weight: float = 0.5,
    helix_weight: float = -0.2,
    plddt_weight: float = 0.1,
    pae_weight: float = 0.1,
    i_pae_weight: float = 0.1,
    i_con_weight: float = 2.0,
    ba_val_weight: float = 2.0,
    seed: int = 0,
    use_multimer: bool = False,
    use_soft_contacts: bool = True,
    use_soft_nis: bool = True,
    binder_seq_mode: str = "soft",
    design_mode: str = "logits",
    design_temp: float = 1.0,
) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)

    from colabdesign import clear_mem, mk_afdesign_model
    from protein_affinity_gpu.af_design import add_ba_val_loss

    binder_seq_mode = _normalize_choice(
        "binder_seq_mode",
        binder_seq_mode,
        VALID_BINDER_SEQ_MODES,
    )
    design_mode = _normalize_choice("design_mode", design_mode, VALID_DESIGN_MODES)
    _ensure_alphafold_params()

    pdb_path = REMOTE_VOLUME_ROOT / pdb_volume_path
    if not pdb_path.exists():
        raise FileNotFoundError(f"Input structure not found in volume: {pdb_path}")

    resolved_run_name = run_name.strip() or datetime.now(timezone.utc).strftime(
        "afdesign-ba-val-%Y%m%d-%H%M%S"
    )
    output_dir = REMOTE_OUTPUTS_DIR / resolved_run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    clear_mem()
    af_model = mk_afdesign_model(
        protocol="binder",
        data_dir=str(REMOTE_AF_DATA_DIR),
        use_multimer=use_multimer,
        debug=False,
    )

    prep_kwargs: dict[str, object] = {
        "pdb_filename": str(pdb_path),
        "chain": chain,
    }
    if binder_chain.strip():
        prep_kwargs["binder_chain"] = binder_chain.strip()
    else:
        prep_kwargs["binder_len"] = binder_len
    af_model.prep_inputs(**prep_kwargs)

    add_ba_val_loss(
        af_model,
        sphere_points=sphere_points,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        soft_sasa_beta=soft_sasa_beta,
        contact_beta=contact_beta,
        nis_beta=nis_beta,
        use_soft_contacts=use_soft_contacts,
        use_soft_nis=use_soft_nis,
        binder_seq_mode=binder_seq_mode,
    )

    af_model.restart(
        seed=seed,
        mode=["gumbel", "soft"],
        reset_opt=False,
    )
    af_model.opt["weights"]["rg"] = rg_weight
    af_model.opt["weights"]["helix"] = helix_weight
    af_model.opt["weights"]["plddt"] = plddt_weight
    af_model.opt["weights"]["pae"] = pae_weight
    af_model.opt["weights"]["i_pae"] = i_pae_weight
    af_model.opt["weights"]["i_con"] = i_con_weight
    af_model.opt["weights"]["ba_val"] = ba_val_weight

    import numpy as _np

    binder_len_effective = int(getattr(af_model, "_binder_len", binder_len))
    binder_ca_history: list[list[list[float]]] = []

    def _capture_binder_ca(model) -> None:
        atom_positions = model.aux.get("atom_positions")
        if atom_positions is None:
            return
        atoms = _np.asarray(atom_positions)
        # AlphaFold's 37-atom-type layout puts CA at index 1.
        binder_ca = atoms[-binder_len_effective:, 1, :].tolist()
        binder_ca_history.append(binder_ca)

    if design_mode == "soft":
        af_model.design_soft(num_steps, temp=design_temp, callback=_capture_binder_ca)
    else:
        af_model.design_logits(num_steps, callback=_capture_binder_ca)

    best_aux = af_model._tmp["best"].get("aux", af_model.aux)
    best_seq = af_model.get_seqs(get_best=True)
    best_pdb_path = output_dir / "best_design.pdb"
    last_pdb_path = output_dir / "last_design.pdb"
    af_model.save_pdb(str(best_pdb_path), get_best=True)
    af_model.save_pdb(str(last_pdb_path), get_best=False)

    trajectory = [
        {str(key): _to_serializable(val) for key, val in row.items()}
        for row in af_model._tmp["log"]
    ]
    trajectory_path = output_dir / "trajectory.json"
    trajectory_path.write_text(json.dumps(trajectory, indent=2))

    sequences_path = output_dir / "best_sequences.json"
    sequences_path.write_text(json.dumps(best_seq, indent=2))

    binder_ca_path = output_dir / "binder_ca_history.json"
    binder_ca_path.write_text(json.dumps(binder_ca_history))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU_TYPE,
        "volume_name": VOLUME_NAME,
        "run_name": resolved_run_name,
        "input_pdb": pdb_volume_path,
        "chain": chain,
        "binder_len": binder_len,
        "binder_chain": binder_chain,
        "num_steps": num_steps,
        "sphere_points": sphere_points,
        "distance_cutoff": distance_cutoff,
        "acc_threshold": acc_threshold,
        "soft_sasa_beta": soft_sasa_beta,
        "contact_beta": contact_beta,
        "nis_beta": nis_beta,
        "use_soft_contacts": use_soft_contacts,
        "use_soft_nis": use_soft_nis,
        "binder_seq_mode": binder_seq_mode,
        "design_mode": design_mode,
        "design_temp": design_temp,
        "use_multimer": use_multimer,
        "weights": {
            "rg": rg_weight,
            "helix": helix_weight,
            "plddt": plddt_weight,
            "pae": pae_weight,
            "i_pae": i_pae_weight,
            "i_con": i_con_weight,
            "ba_val": ba_val_weight,
        },
        "best_sequences": best_seq,
        "best_metrics": _to_serializable(best_aux.get("log", {})),
        "losses": _to_serializable(best_aux.get("losses", {})),
        "artifacts": {
            "best_pdb": _volume_relative(best_pdb_path),
            "last_pdb": _volume_relative(last_pdb_path),
            "trajectory_json": _volume_relative(trajectory_path),
            "best_sequences_json": _volume_relative(sequences_path),
            "binder_ca_history_json": _volume_relative(binder_ca_path),
        },
    }

    summary_path = output_dir / "summary.json"
    summary["artifacts"]["summary_json"] = _volume_relative(summary_path)
    summary_path.write_text(json.dumps(_to_serializable(summary), indent=2))

    volume.commit()
    return summary


@app.local_entrypoint()
def main(
    pdb_path: str,
    chain: str = "B",
    binder_len: int = 20,
    binder_chain: str = "",
    run_name: str = "",
    num_steps: int = 80,
    sphere_points: int = 100,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    soft_sasa_beta: float = 10.0,
    contact_beta: float = 8.0,
    nis_beta: float = 20.0,
    rg_weight: float = 0.5,
    helix_weight: float = -0.2,
    plddt_weight: float = 0.1,
    pae_weight: float = 0.1,
    i_pae_weight: float = 0.1,
    i_con_weight: float = 2.0,
    ba_val_weight: float = 2.0,
    seed: int = 0,
    use_multimer: bool = False,
    use_soft_contacts: bool = True,
    use_soft_nis: bool = True,
    binder_seq_mode: str = "soft",
    design_mode: str = "logits",
    design_temp: float = 1.0,
    local_output_dir: str = "",
):
    local_pdb_path = Path(pdb_path).expanduser().resolve()
    if not local_pdb_path.exists():
        raise FileNotFoundError(f"Local structure not found: {local_pdb_path}")

    remote_input_path = Path("inputs") / local_pdb_path.name
    with volume.batch_upload(force=True) as upload:
        upload.put_file(local_pdb_path, remote_input_path.as_posix())

    summary = run_afdesign_binder.remote(
        pdb_volume_path=remote_input_path.as_posix(),
        chain=chain,
        binder_len=binder_len,
        binder_chain=binder_chain,
        run_name=run_name,
        num_steps=num_steps,
        sphere_points=sphere_points,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        soft_sasa_beta=soft_sasa_beta,
        contact_beta=contact_beta,
        nis_beta=nis_beta,
        rg_weight=rg_weight,
        helix_weight=helix_weight,
        plddt_weight=plddt_weight,
        pae_weight=pae_weight,
        i_pae_weight=i_pae_weight,
        i_con_weight=i_con_weight,
        ba_val_weight=ba_val_weight,
        seed=seed,
        use_multimer=use_multimer,
        use_soft_contacts=use_soft_contacts,
        use_soft_nis=use_soft_nis,
        binder_seq_mode=binder_seq_mode,
        design_mode=design_mode,
        design_temp=design_temp,
    )

    if local_output_dir.strip():
        target_dir = Path(local_output_dir).expanduser().resolve()
        for remote_path in summary["artifacts"].values():
            remote_name = Path(str(remote_path)).name
            _download_volume_file(str(remote_path), target_dir / remote_name)
        summary["local_output_dir"] = str(target_dir)

    print(json.dumps(summary, indent=2))
