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
    """One-hot via ``Tensor.one_hot`` — invalid (negative / OOB) indices → zero row.

    ``Tensor.one_hot`` compares each index against ``arange(num_classes)`` and
    emits ``1`` on equality, so ``-1`` padding naturally produces a zero row
    without special-casing on CPU first.
    """
    return Tensor(np.asarray(indices, dtype=np.int64)).one_hot(num_classes).float()


def _device_name() -> str:
    """Resolve the tinygrad device name (user override via ``TINYGRAD_DEVICE``)."""
    override = os.environ.get("TINYGRAD_DEVICE")
    if override:
        return override.upper()
    return str(Device.DEFAULT).upper()


def estimate_optimal_block_size(n_atoms: int) -> int:
    """Amortize per-block kernel-launch overhead on accelerators.

    Empirically on Apple Metal (M-series), throughput improves monotonically
    up to ``block ≈ 768`` for 1A2K-sized complexes: 152→9.3s, 256→6.8s,
    512→3.5s, 768→2.3s, 1024→7.3s. Past 768 Metal starts spilling the
    [block, 100, N] float32 scratch — about 5GB — out of the fast L2/MMU
    path. The cap stays at 768; for smaller proteins it clamps to N.
    """
    return min(768, n_atoms)


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

    with log_duration(LOGGER, "tinygrad.score"):
        dg = score_ic_nis_tinygrad(
            contact_types[1],
            contact_types[3],
            contact_types[2],
            contact_types[4],
            nis_percentages[0],
            nis_percentages[1],
            _COEFFS,
            _INTERCEPT,
        )
        kd = dg_to_kd_tinygrad(dg, temperature=temperature)

    # Single materialization pass — downstream is numpy-only (record assembly,
    # dataclass construction). Keeps the compute graph connected through
    # scoring so tinygrad can fuse across contacts → NIS → ΔG.
    contact_types_np = contact_types.numpy()
    nis_np = nis_percentages.numpy()
    complex_sasa_np = complex_sasa.numpy()
    relative_sasa_np = relative_sasa.numpy()
    dg_value = float(dg.numpy().reshape(-1)[0])
    kd_value = float(kd.numpy().reshape(-1)[0])

    sasa_data = build_sasa_records(
        complex_sasa=complex_sasa_np,
        relative_sasa=relative_sasa_np,
        target=target,
        binder=binder,
        chain_labels=(target_chain, binder_chain),
    )
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
