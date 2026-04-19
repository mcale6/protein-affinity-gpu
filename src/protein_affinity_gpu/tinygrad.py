"""Tinygrad implementation of the PRODIGY IC-NIS pipeline."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import numpy as np
from tinygrad import Device, Tensor

from .contacts import analyze_contacts_tinygrad, calculate_residue_contacts_tinygrad
from .logging_utils import get_logger, log_duration
from .results import ContactAnalysis, ProdigyResults, build_sasa_records
from .sasa import (
    calculate_sasa_batch_tinygrad,
    calculate_sasa_tinygrad,
    generate_sphere_points_tinygrad,
)
from .scoring import (
    calculate_nis_percentages_tinygrad,
    calculate_relative_sasa_tinygrad,
    coefficient_tensors_tinygrad,
    dg_to_kd_tinygrad,
    get_atom_radii_tinygrad,
    score_ic_nis_tinygrad,
)
from .structure import load_complex
from .utils import residue_constants
from .utils.residue_classification import ResidueClassification
from .utils.residue_library import default_library as residue_library

LOGGER = get_logger(__name__)

_ATOMS_PER_RESIDUE = residue_constants.atom_type_num
_RESIDUE_RADII_MATRIX = Tensor(np.asarray(residue_library.radii_matrix, dtype=np.float32))
_RELATIVE_SASA_ARRAY = Tensor(
    np.asarray(ResidueClassification().relative_sasa_array, dtype=np.float32)
)
_CONTACT_CLASS_MATRIX = Tensor(
    np.asarray(ResidueClassification("ic").classification_matrix, dtype=np.float32)
)
_NIS_CLASS_MATRIX = Tensor(
    np.asarray(ResidueClassification("protorp").classification_matrix, dtype=np.float32)
)
_COEFFS, _INTERCEPT = coefficient_tensors_tinygrad()


def _one_hot(indices: np.ndarray, num_classes: int) -> Tensor:
    """Numpy-built one-hot (tinygrad lacks a direct ``jax.nn.one_hot`` equivalent)."""
    indices = np.asarray(indices, dtype=np.int64)
    onehot = np.zeros((indices.shape[0], num_classes), dtype=np.float32)
    valid = (indices >= 0) & (indices < num_classes)
    onehot[np.arange(indices.shape[0])[valid], indices[valid]] = 1.0
    return Tensor(onehot)


def _device_name() -> str:
    """Resolve the tinygrad device name (user override via ``TINYGRAD_DEVICE``)."""
    override = os.environ.get("TINYGRAD_DEVICE")
    if override:
        return override.upper()
    return str(Device.DEFAULT).upper()


def estimate_optimal_block_size(n_atoms: int) -> int:
    """Memory-friendly block size for the batched SASA kernel on accelerators.

    Target: keep the per-block scratch tensor ``[block, M=100, N]`` under ~1GB
    (floats), which lets Metal/CUDA fuse one kernel per block. With the
    dot-product formulation the peak is ``block * 100 * n_atoms * 4B``.
    """
    target_elements = 250_000_000  # ~1GB of float32 scratch per block
    block_size = max(32, target_elements // (100 * max(n_atoms, 1)))
    return min(block_size, n_atoms, 1024)


def predict_binding_affinity_tinygrad(
    struct_path: str | Path,
    selection: str = "A,B",
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: Optional[str | Path] = ".",
    quiet: bool = True,
) -> ProdigyResults:
    """Run the PRODIGY IC-NIS pipeline end-to-end on tinygrad."""
    target_chain, binder_chain = [chain.strip() for chain in selection.split(",")]
    with log_duration(LOGGER, "tinygrad.load_complex", extra=str(struct_path)):
        target, binder = load_complex(struct_path, selection=selection, sanitize=True)
    device = _device_name()
    sphere_points_tensor = generate_sphere_points_tinygrad(sphere_points)

    target_positions = Tensor(np.asarray(target.atom_positions, dtype=np.float32))
    binder_positions = Tensor(np.asarray(binder.atom_positions, dtype=np.float32))
    target_mask = Tensor(np.asarray(target.atom_mask, dtype=np.float32))
    binder_mask = Tensor(np.asarray(binder.atom_mask, dtype=np.float32))

    complex_positions = Tensor.cat(
        target_positions.reshape(-1, 3),
        binder_positions.reshape(-1, 3),
        dim=0,
    )
    complex_mask = Tensor.cat(target_mask.reshape(-1), binder_mask.reshape(-1), dim=0)

    num_classes = len(residue_constants.restypes)
    target_sequence = _one_hot(np.asarray(target.aatype), num_classes)
    binder_sequence = _one_hot(np.asarray(binder.aatype), num_classes)
    total_sequence = Tensor.cat(target_sequence, binder_sequence, dim=0)
    complex_radii = Tensor.cat(
        get_atom_radii_tinygrad(target_sequence, _RESIDUE_RADII_MATRIX),
        get_atom_radii_tinygrad(binder_sequence, _RESIDUE_RADII_MATRIX),
        dim=0,
    )

    n_atoms = int(complex_positions.shape[0])
    LOGGER.info("device=%s atoms=%d selection=%s", device, n_atoms, selection)

    with log_duration(LOGGER, "tinygrad.contacts"):
        contacts = calculate_residue_contacts_tinygrad(
            target_positions,
            binder_positions,
            target_mask,
            binder_mask,
            distance_cutoff=distance_cutoff,
        )
        contact_types = analyze_contacts_tinygrad(
            contacts, target_sequence, binder_sequence, _CONTACT_CLASS_MATRIX
        )

    if device in {"METAL", "CUDA", "GPU"}:
        block_size = estimate_optimal_block_size(n_atoms)
        LOGGER.debug("sasa batched block_size=%d", block_size)
        with log_duration(LOGGER, "tinygrad.sasa_batch", extra=f"block={block_size}"):
            complex_sasa = calculate_sasa_batch_tinygrad(
                coords=complex_positions,
                vdw_radii=complex_radii,
                mask=complex_mask,
                sphere_points=sphere_points_tensor,
                block_size=block_size,
            )
    else:
        with log_duration(LOGGER, "tinygrad.sasa_full"):
            complex_sasa = calculate_sasa_tinygrad(
                coords=complex_positions,
                vdw_radii=complex_radii,
                mask=complex_mask,
                sphere_points=sphere_points_tensor,
            )

    with log_duration(LOGGER, "tinygrad.nis"):
        relative_sasa = calculate_relative_sasa_tinygrad(
            complex_sasa=complex_sasa,
            sequence_probabilities=total_sequence,
            relative_sasa_array=_RELATIVE_SASA_ARRAY,
            atoms_per_residue=_ATOMS_PER_RESIDUE,
        )
        nis_percentages = calculate_nis_percentages_tinygrad(
            sasa_values=relative_sasa,
            sequence_probabilities=total_sequence,
            character_matrix=_NIS_CLASS_MATRIX,
            threshold=acc_threshold,
        )

    contact_types_np = contact_types.numpy()
    nis_np = nis_percentages.numpy()

    with log_duration(LOGGER, "tinygrad.score"):
        dg = score_ic_nis_tinygrad(
            Tensor(float(contact_types_np[1])),
            Tensor(float(contact_types_np[3])),
            Tensor(float(contact_types_np[2])),
            Tensor(float(contact_types_np[4])),
            Tensor(float(nis_np[0])),
            Tensor(float(nis_np[1])),
            _COEFFS,
            _INTERCEPT,
        )
        kd = dg_to_kd_tinygrad(dg, temperature=temperature)

    sasa_data = build_sasa_records(
        complex_sasa=np.asarray(complex_sasa.numpy()),
        relative_sasa=np.asarray(relative_sasa.numpy()),
        target=target,
        binder=binder,
        chain_labels=(target_chain, binder_chain),
    )

    dg_value = float(np.asarray(dg.numpy()).reshape(-1)[0])
    kd_value = float(np.asarray(kd.numpy()).reshape(-1)[0])
    results = ProdigyResults(
        contact_types=ContactAnalysis(contact_types_np.tolist()),
        binding_affinity=np.float32(dg_value),
        dissociation_constant=np.float32(kd_value),
        nis_aliphatic=np.float32(nis_np[0]),
        nis_charged=np.float32(nis_np[1]),
        nis_polar=np.float32(nis_np[2]),
        structure_id=Path(struct_path).stem,
        sasa_data=sasa_data,
    )

    if save_results:
        results.save_results(output_dir or ".")
    if not quiet:
        LOGGER.info("Tinygrad prediction complete for %s", results.structure_id)
    return results
