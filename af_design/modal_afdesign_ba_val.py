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
    hotspot: str = "",
    run_name: str = "",
    num_steps: int = 80,
    sphere_points: int = 100,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    soft_sasa_beta: float = 10.0,
    contact_beta: float = 8.0,
    nis_beta: float = 20.0,
    rg_weight: float = 0.3,
    helix_weight: float = -0.2,
    plddt_weight: float = 0.1,
    pae_weight: float = 0.1,
    i_pae_weight: float = 0.1,
    i_con_weight: float = 1.0,
    con_weight: float = 1.0,
    iptm_weight: float = 0.05,
    ba_val_weight: float = 0.3,
    seed: int = 0,
    use_multimer: bool = False,
    use_soft_contacts: bool = True,
    use_soft_nis: bool = True,
    binder_seq_mode: str = "soft",
    design_mode: str = "logits",
    design_temp: float = 1.0,
    three_stage: bool = True,
    logits_iters: int = 75,
    soft_iters: int = 45,
    hard_iters: int = 10,
    schedule_mode: str = "three_stage",
    soft_max_iters: int = 400,
    hard_max_iters: int = 100,
    iptm_target: float = 0.7,
    ba_val_target: float = -8.0,
    stability_window: int = 10,
    early_stop_patience: int = 50,
    early_stop_min_delta: float = 0.05,
    filter_plddt_min: float = 0.8,
    filter_iptm_min: float = 0.5,
    filter_ipae_max: float = 0.4,
    filter_icon_min: float = 3.5,
    filter_ipsae_min: float = 0.6,
    save_trajectory: bool = True,
) -> dict[str, object]:
    logging.basicConfig(level=logging.INFO, format="[%(name)s] %(message)s", force=True)

    import jax.numpy as jnp
    from colabdesign import clear_mem, mk_afdesign_model
    from protein_affinity_gpu.af_design import add_ba_val_loss
    from protein_affinity_gpu.sasa import (
        calculate_sasa_batch_scan,
        generate_sphere_points,
    )
    from protein_affinity_gpu.utils import residue_constants
    from protein_affinity_gpu.utils.residue_library import default_library as _residue_library

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
    if hotspot.strip():
        prep_kwargs["hotspot"] = hotspot.strip()
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
    af_model.opt["weights"]["con"] = con_weight
    af_model.opt["weights"]["i_ptm"] = iptm_weight
    af_model.opt["weights"]["ba_val"] = ba_val_weight

    import numpy as _np

    binder_len_effective = int(getattr(af_model, "_binder_len", binder_len))
    binder_ca_history: list[list[list[float]]] = []
    bsa_history: list[float] = []
    traj_positions: list[_np.ndarray] = []
    traj_aatype: list[_np.ndarray] = []
    traj_atom_mask: list[_np.ndarray] = []
    traj_plddt: list[_np.ndarray] = []

    _radii_matrix_np = _np.asarray(
        _residue_library.radii_matrix, dtype=_np.float32
    )  # [20, 37] per-restype × atom37 VdW radii
    _sphere_points_jnp = jnp.asarray(
        generate_sphere_points(sphere_points), dtype=jnp.float32
    )

    def _capture_design_state(model) -> None:
        atom_positions = model.aux.get("atom_positions")
        atom_mask = model.aux.get("atom_mask")
        if atom_positions is None or atom_mask is None:
            return

        atoms = _np.asarray(atom_positions)                 # [total_len, 37, 3]
        atom_mask_np = _np.asarray(atom_mask).astype(_np.float32)  # [total_len, 37]

        # AlphaFold's 37-atom-type layout puts CA at index 1.
        binder_ca = atoms[-binder_len_effective:, 1, :].tolist()
        binder_ca_history.append(binder_ca)

        total_len = atoms.shape[0]
        target_len = total_len - binder_len_effective
        n_atom_types = residue_constants.atom_type_num

        aatype_np = _np.asarray(
            model._inputs["batch"]["aatype"], dtype=_np.int32
        )[:total_len]
        aatype_np = _np.clip(aatype_np, 0, _radii_matrix_np.shape[0] - 1)

        # Only pay for the 3 hard-SASA kernel launches when ba_val is active.
        # Phase A of the adaptive schedule (and stage 1 of three_stage) runs
        # with weight=0 — compute is pure overhead there. Keep bsa_history
        # length aligned by appending NaN so plot step-indexing stays correct.
        ba_val_weight_now = float(model.opt["weights"].get("ba_val", 0.0))
        if ba_val_weight_now > 0.0:
            full_radii = _radii_matrix_np[aatype_np]  # [total_len, 37]

            pos_flat = jnp.asarray(atoms.reshape(-1, 3), dtype=jnp.float32)
            radii_flat = jnp.asarray(full_radii.reshape(-1), dtype=jnp.float32)
            mask_complex = jnp.asarray(atom_mask_np.reshape(-1), dtype=jnp.float32)

            target_residue_slot = _np.zeros(total_len, dtype=_np.float32)
            target_residue_slot[:target_len] = 1.0
            target_atom_slot = jnp.asarray(
                _np.repeat(target_residue_slot, n_atom_types)
            )
            mask_target_only = mask_complex * target_atom_slot
            mask_binder_only = mask_complex * (1.0 - target_atom_slot)

            block_size = max(1, min(int(pos_flat.shape[0]), 768))
            sasa_complex = calculate_sasa_batch_scan(
                pos_flat, radii_flat, mask_complex, block_size, _sphere_points_jnp
            )
            sasa_target = calculate_sasa_batch_scan(
                pos_flat, radii_flat, mask_target_only, block_size, _sphere_points_jnp
            )
            sasa_binder = calculate_sasa_batch_scan(
                pos_flat, radii_flat, mask_binder_only, block_size, _sphere_points_jnp
            )
            bsa = float(
                sasa_target.sum() + sasa_binder.sum() - sasa_complex.sum()
            )
            bsa_history.append(bsa)
        else:
            bsa_history.append(float("nan"))

        if save_trajectory:
            traj_positions.append(atoms.copy())
            traj_aatype.append(aatype_np.copy())
            traj_atom_mask.append(atom_mask_np.copy())
            plddt_arr = model.aux.get("plddt")
            if plddt_arr is not None:
                traj_plddt.append(_np.asarray(plddt_arr).copy())
            else:
                traj_plddt.append(_np.zeros(total_len, dtype=_np.float32))

    def _latest_metric(key: str, default: float) -> float:
        # ColabDesign appends the fully-formed per-step log dict to
        # ``_tmp["log"]`` inside ``_save_results`` after the forward pass.
        # That is the exact same dict that lands in trajectory.json, so
        # reading it here is both correct and self-consistent. ``aux["log"]``
        # was unreliable — an earlier adaptive run ran all 400 Phase A iters
        # without firing the patience break, which only makes sense if
        # cur was drifting in a way that kept resetting a_best_iter.
        tmp = getattr(af_model, "_tmp", None)
        if isinstance(tmp, dict):
            log_list = tmp.get("log")
            if isinstance(log_list, list) and log_list:
                row = log_list[-1]
                if isinstance(row, dict) and key in row:
                    try:
                        return float(row[key])
                    except (TypeError, ValueError):
                        pass
        return default

    if schedule_mode == "adaptive":
        # Phase A: soft with ba_val gated to 0. Exits on either
        #   (converged) i_ptm stays >= iptm_target for stability_window steps
        #   (stalled)   no improvement of early_stop_min_delta in the last
        #               early_stop_patience steps — a dead trajectory
        #   (budget)    soft_max_iters reached.
        # Phase B only runs on a converged exit; a stalled trajectory skips
        # straight to hard so the summary carries the failure signal without
        # burning budget on ba_val gradient noise.
        af_model.opt["weights"]["ba_val"] = 0.0
        phase_a_iters = 0
        a_streak = 0
        a_best = float("-inf")
        a_best_iter = 0
        a_converged = False
        a_stalled = False
        for _ in range(int(soft_max_iters)):
            af_model.design_soft(1, temp=design_temp, callback=_capture_design_state)
            phase_a_iters += 1
            cur = _latest_metric("i_ptm", 0.0)
            if cur > a_best + float(early_stop_min_delta):
                a_best = cur
                a_best_iter = phase_a_iters
            if cur >= float(iptm_target):
                a_streak += 1
                if a_streak >= int(stability_window):
                    a_converged = True
                    break
            else:
                a_streak = 0
            gap = phase_a_iters - a_best_iter
            logging.info(
                "[phase_a] step=%d cur_iptm=%.3f a_best=%.3f a_best_iter=%d gap=%d streak=%d",
                phase_a_iters, cur, a_best, a_best_iter, gap, a_streak,
            )
            if gap >= int(early_stop_patience):
                a_stalled = True
                break

        phase_b_iters = 0
        b_streak = 0
        b_best = float("inf")
        b_best_iter = 0
        b_converged = False
        b_stalled = False
        if a_converged:
            # Phase B: re-enable ba_val only on a converged Phase A. Same
            # twin exits (converged / stalled / budget). Stall skips to hard.
            af_model.opt["weights"]["ba_val"] = ba_val_weight
            remaining = max(0, int(soft_max_iters) - phase_a_iters)
            for _ in range(remaining):
                af_model.design_soft(1, temp=design_temp, callback=_capture_design_state)
                phase_b_iters += 1
                cur = _latest_metric("ba_val", 0.0)
                if cur < b_best - float(early_stop_min_delta):
                    b_best = cur
                    b_best_iter = phase_b_iters
                if cur <= float(ba_val_target):
                    b_streak += 1
                    if b_streak >= int(stability_window):
                        b_converged = True
                        break
                else:
                    b_streak = 0
                gap = phase_b_iters - b_best_iter
                logging.info(
                    "[phase_b] step=%d cur_ba_val=%.3f b_best=%.3f b_best_iter=%d gap=%d streak=%d",
                    phase_b_iters, cur, b_best, b_best_iter, gap, b_streak,
                )
                if gap >= int(early_stop_patience):
                    b_stalled = True
                    break

        af_model.design_hard(int(hard_max_iters), callback=_capture_design_state)
        effective_steps = phase_a_iters + phase_b_iters + int(hard_max_iters)
        stage_schedule = {
            "mode": "adaptive",
            "phase_a_iters": int(phase_a_iters),
            "phase_b_iters": int(phase_b_iters),
            "hard_iters": int(hard_max_iters),
            "soft_max_iters": int(soft_max_iters),
            "iptm_target": float(iptm_target),
            "ba_val_target": float(ba_val_target),
            "stability_window": int(stability_window),
            "early_stop_patience": int(early_stop_patience),
            "early_stop_min_delta": float(early_stop_min_delta),
            "phase_a_exit": (
                "converged" if a_converged else "stalled" if a_stalled else "budget"
            ),
            "phase_b_exit": (
                "skipped" if not a_converged
                else "converged" if b_converged
                else "stalled" if b_stalled
                else "budget"
            ),
        }
    elif three_stage:
        # Stage 1 runs with the ``ba_val`` PRODIGY ΔG term zeroed out: before
        # contacts actually form, PRODIGY's IC-NIS score collapses to the
        # −15.94 intercept plus regression-coefficient noise, so its gradient
        # wastes budget that the AF-native structural terms could use.
        # Re-enable it for stage 2+, once the binder has been folded and
        # approximately placed.
        af_model.opt["weights"]["ba_val"] = 0.0
        af_model.design_logits(logits_iters, callback=_capture_design_state)
        af_model.opt["weights"]["ba_val"] = ba_val_weight
        af_model.design_soft(soft_iters, temp=design_temp, callback=_capture_design_state)
        af_model.design_hard(hard_iters, callback=_capture_design_state)
        effective_steps = int(logits_iters + soft_iters + hard_iters)
        stage_schedule = {
            "mode": "three_stage",
            "logits_iters": int(logits_iters),
            "soft_iters": int(soft_iters),
            "hard_iters": int(hard_iters),
        }
    else:
        if design_mode == "soft":
            af_model.design_soft(num_steps, temp=design_temp, callback=_capture_design_state)
        else:
            af_model.design_logits(num_steps, callback=_capture_design_state)
        effective_steps = int(num_steps)
        stage_schedule = {"mode": f"single_{design_mode}", "num_steps": int(num_steps)}

    best_aux = af_model._tmp["best"].get("aux", af_model.aux)
    best_seq = af_model.get_seqs(get_best=True)

    # ---- ipSAE (Dunbrack) from best aux pAE: interface score over
    # ---- inter-chain residue pairs confident under pAE. pAE matrix is
    # ---- in Angstroms (0-31.75); normalised logs use /31 internally.
    def _compute_ipsae(pae_mat, target_len_, binder_len_, pae_cutoff: float = 10.0):
        if pae_mat is None:
            return None
        pae_arr = _np.asarray(pae_mat, dtype=_np.float32)
        if pae_arr.ndim == 3:
            pae_arr = pae_arr[0]
        total_needed = target_len_ + binder_len_
        if pae_arr.ndim != 2 or pae_arr.shape[0] < total_needed:
            return None
        # Defensive: if the matrix looks normalised, scale back to Å.
        if float(pae_arr.max()) <= 1.0 + 1e-3:
            pae_arr = pae_arr * 31.0
        upper = pae_arr[:target_len_, target_len_:total_needed]
        lower = pae_arr[target_len_:total_needed, :target_len_]
        pae_inter = 0.5 * (upper + lower.T)
        mask = pae_inter < pae_cutoff
        n_conf = int(mask.sum())
        if n_conf == 0:
            return 0.0
        d0 = max(1.24 * (n_conf - 15) ** (1.0 / 3.0) - 1.8, 0.5)
        scores = 1.0 / (1.0 + (pae_inter / d0) ** 2)
        return float((scores * mask).sum() / n_conf)

    total_len_best = None
    target_len_best = None
    try:
        atom_pos_best = best_aux.get("atom_positions")
        if atom_pos_best is not None:
            total_len_best = int(_np.asarray(atom_pos_best).shape[0])
            target_len_best = total_len_best - binder_len_effective
    except Exception:  # noqa: BLE001
        pass

    ipsae_value = None
    if target_len_best is not None and target_len_best > 0:
        ipsae_value = _compute_ipsae(
            best_aux.get("pae"), target_len_best, binder_len_effective
        )

    # NaNs in bsa_history mark steps where ba_val was inactive (Phase A /
    # logits stage); exclude them from the scalar summary fields.
    _finite_bsa = [b for b in bsa_history if b == b]  # NaN != NaN
    if _finite_bsa:
        max_bsa = float(max(_finite_bsa))
        final_bsa = float(_finite_bsa[-1])
    else:
        max_bsa = None
        final_bsa = None
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

    bsa_history_path = output_dir / "bsa_history.json"
    bsa_history_path.write_text(json.dumps(bsa_history))

    trajectory_pdb_path: Path | None = None
    if save_trajectory and traj_positions:
        from colabdesign.af.alphafold.common import protein as _af_protein

        residue_index_arr = _np.asarray(af_model._inputs["residue_index"])
        total_len_traj = traj_positions[0].shape[0]
        target_len_traj = total_len_traj - binder_len_effective
        chain_index_arr = _np.zeros(total_len_traj, dtype=_np.int32)
        chain_index_arr[target_len_traj:] = 1
        if residue_index_arr.shape[0] != total_len_traj:
            residue_index_arr = _np.arange(total_len_traj, dtype=_np.int32)

        pdb_frames: list[str] = []
        for frame_i, (pos, aat, mask) in enumerate(
            zip(traj_positions, traj_aatype, traj_atom_mask)
        ):
            plddt_frame = (
                traj_plddt[frame_i]
                if frame_i < len(traj_plddt)
                else _np.zeros(total_len_traj, dtype=_np.float32)
            )
            b_factors = 100.0 * mask * plddt_frame[..., None]
            prot = _af_protein.Protein(
                atom_positions=pos,
                aatype=aat,
                atom_mask=mask,
                residue_index=residue_index_arr,
                chain_index=chain_index_arr,
                b_factors=b_factors,
            )
            frame_pdb = _af_protein.to_pdb(prot)
            atom_lines = [
                line for line in frame_pdb.splitlines()
                if line.startswith(("ATOM", "HETATM", "TER"))
            ]
            pdb_frames.append(
                f"MODEL {frame_i + 1:>8}\n" + "\n".join(atom_lines) + "\nENDMDL\n"
            )
        trajectory_pdb_path = output_dir / "trajectory.pdb"
        trajectory_pdb_path.write_text("".join(pdb_frames) + "END\n")

    best_metrics = _to_serializable(best_aux.get("log", {}))
    if isinstance(best_metrics, dict):
        if ipsae_value is not None:
            best_metrics["ipSAE"] = ipsae_value
        if max_bsa is not None:
            best_metrics["bsa"] = max_bsa
            best_metrics["bsa_final"] = final_bsa

    def _cmp(value, threshold, mode: str) -> dict[str, object]:
        ok = False
        if value is not None:
            try:
                if mode == "min":
                    ok = float(value) >= float(threshold)
                else:
                    ok = float(value) <= float(threshold)
            except (TypeError, ValueError):
                ok = False
        return {
            "value": value,
            "threshold": threshold,
            "direction": mode,
            "pass": bool(ok),
        }

    bm = best_metrics if isinstance(best_metrics, dict) else {}
    filter_gate = {
        "plddt":  _cmp(bm.get("plddt"),  filter_plddt_min, "min"),
        "i_ptm":  _cmp(bm.get("i_ptm"),  filter_iptm_min, "min"),
        "i_pae":  _cmp(bm.get("i_pae"),  filter_ipae_max, "max"),
        "i_con":  _cmp(bm.get("i_con"),  filter_icon_min, "min"),
        "ipSAE":  _cmp(bm.get("ipSAE"),  filter_ipsae_min, "min"),
    }
    filter_gate["all_pass"] = bool(all(v["pass"] for v in filter_gate.values()))

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "gpu": GPU_TYPE,
        "volume_name": VOLUME_NAME,
        "run_name": resolved_run_name,
        "input_pdb": pdb_volume_path,
        "chain": chain,
        "binder_len": binder_len,
        "binder_chain": binder_chain,
        "hotspot": hotspot,
        "num_steps": num_steps,
        "effective_steps": effective_steps,
        "stage_schedule": stage_schedule,
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
            "con": con_weight,
            "i_ptm": iptm_weight,
            "ba_val": ba_val_weight,
        },
        "filters": filter_gate,
        "best_sequences": best_seq,
        "best_metrics": best_metrics,
        "losses": _to_serializable(best_aux.get("losses", {})),
        "artifacts": {
            "best_pdb": _volume_relative(best_pdb_path),
            "last_pdb": _volume_relative(last_pdb_path),
            "trajectory_json": _volume_relative(trajectory_path),
            "best_sequences_json": _volume_relative(sequences_path),
            "binder_ca_history_json": _volume_relative(binder_ca_path),
            "bsa_history_json": _volume_relative(bsa_history_path),
            **(
                {"trajectory_pdb": _volume_relative(trajectory_pdb_path)}
                if trajectory_pdb_path is not None else {}
            ),
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
    hotspot: str = "",
    run_name: str = "",
    num_steps: int = 80,
    sphere_points: int = 100,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    soft_sasa_beta: float = 10.0,
    contact_beta: float = 8.0,
    nis_beta: float = 20.0,
    rg_weight: float = 0.3,
    helix_weight: float = -0.2,
    plddt_weight: float = 0.1,
    pae_weight: float = 0.1,
    i_pae_weight: float = 0.1,
    i_con_weight: float = 1.0,
    con_weight: float = 1.0,
    iptm_weight: float = 0.05,
    ba_val_weight: float = 0.3,
    seed: int = 0,
    use_multimer: bool = False,
    use_soft_contacts: bool = True,
    use_soft_nis: bool = True,
    binder_seq_mode: str = "soft",
    design_mode: str = "logits",
    design_temp: float = 1.0,
    three_stage: bool = True,
    logits_iters: int = 75,
    soft_iters: int = 45,
    hard_iters: int = 10,
    schedule_mode: str = "three_stage",
    soft_max_iters: int = 400,
    hard_max_iters: int = 100,
    iptm_target: float = 0.7,
    ba_val_target: float = -8.0,
    stability_window: int = 10,
    early_stop_patience: int = 50,
    early_stop_min_delta: float = 0.05,
    filter_plddt_min: float = 0.8,
    filter_iptm_min: float = 0.5,
    filter_ipae_max: float = 0.4,
    filter_icon_min: float = 3.5,
    filter_ipsae_min: float = 0.6,
    save_trajectory: bool = True,
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
        hotspot=hotspot,
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
        con_weight=con_weight,
        iptm_weight=iptm_weight,
        ba_val_weight=ba_val_weight,
        seed=seed,
        use_multimer=use_multimer,
        use_soft_contacts=use_soft_contacts,
        use_soft_nis=use_soft_nis,
        binder_seq_mode=binder_seq_mode,
        design_mode=design_mode,
        design_temp=design_temp,
        three_stage=three_stage,
        logits_iters=logits_iters,
        soft_iters=soft_iters,
        hard_iters=hard_iters,
        schedule_mode=schedule_mode,
        soft_max_iters=soft_max_iters,
        hard_max_iters=hard_max_iters,
        iptm_target=iptm_target,
        ba_val_target=ba_val_target,
        stability_window=stability_window,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
        filter_plddt_min=filter_plddt_min,
        filter_iptm_min=filter_iptm_min,
        filter_ipae_max=filter_ipae_max,
        filter_icon_min=filter_icon_min,
        filter_ipsae_min=filter_ipsae_min,
        save_trajectory=save_trajectory,
    )

    if local_output_dir.strip():
        target_dir = Path(local_output_dir).expanduser().resolve()
        for remote_path in summary["artifacts"].values():
            remote_name = Path(str(remote_path)).name
            _download_volume_file(str(remote_path), target_dir / remote_name)
        summary["local_output_dir"] = str(target_dir)

    print(json.dumps(summary, indent=2))
