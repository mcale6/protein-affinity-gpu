import logging
import subprocess
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .contacts import analyze_contacts, calculate_residue_contacts
from .results import ContactAnalysis, ProdigyResults, build_sasa_records
from .sasa import calculate_sasa, calculate_sasa_batch, generate_sphere_points
from .scoring import (
    NIS_CONSTANTS,
    calculate_nis_percentages,
    calculate_relative_sasa,
    dg_to_kd,
    get_atom_radii,
    score_ic_nis,
)
from .structure import load_complex
from .utils import residue_constants
from .utils.residue_classification import ResidueClassification
from .utils.residue_library import default_library as residue_library

LOGGER = logging.getLogger(__name__)

_ATOMS_PER_RESIDUE = residue_constants.atom_type_num
_RESIDUE_RADII_MATRIX = jnp.array(residue_library.radii_matrix)
_RELATIVE_SASA_ARRAY = jnp.array(ResidueClassification().relative_sasa_array)
_CONTACT_CLASS_MATRIX = jnp.array(ResidueClassification("ic").classification_matrix)
_NIS_CLASS_MATRIX = jnp.array(ResidueClassification("protorp").classification_matrix)
_COEFFS = jnp.array(
    [
        NIS_CONSTANTS["ic_cc"],
        NIS_CONSTANTS["ic_ca"],
        NIS_CONSTANTS["ic_pp"],
        NIS_CONSTANTS["ic_pa"],
        NIS_CONSTANTS["p_nis_a"],
        NIS_CONSTANTS["p_nis_c"],
    ]
)
_INTERCEPT = jnp.array([NIS_CONSTANTS["intercept"]])


def estimate_optimal_block_size(n_atoms: int) -> int:
    """Estimate a memory-friendly block size for Metal devices."""
    amplitude = 6.8879e02
    decay = -2.6156e-04
    offset = 17.4525
    block_size = int(round(amplitude * np.exp(decay * n_atoms) + offset))
    max_block = min(250, int(5000 / np.sqrt(max(n_atoms, 1) / 1000)))
    return max(5, min(block_size, max_block))


def estimate_max_atoms(backend: str, safety_factor: float = 0.8, sphere_points: int = 100) -> int:
    """Estimate the maximum feasible number of atoms for the current device."""
    backend = backend.upper()
    if backend == "METAL":
        available_memory = 20_000_000_000
    else:
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,nounits,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            _, total = [int(part.strip()) for part in result.split(",")]
            available_memory = total * 1_000_000
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return 100_000

    bytes_per_float32 = 4
    memory_per_atom = (
        3 * bytes_per_float32
        + sphere_points * 3 * bytes_per_float32
        + sphere_points * bytes_per_float32 * 1000
    )
    max_atoms = int((available_memory * safety_factor) / memory_per_atom)
    rounded = int(str(max_atoms)[0] + "0" * max(len(str(max_atoms)) - 1, 0))
    return max(rounded, 1000)


def predict_binding_affinity_jax(
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
    """Run the JAX affinity prediction pipeline."""
    target_chain, binder_chain = [chain.strip() for chain in selection.split(",")]
    target, binder = load_complex(struct_path, selection=selection, sanitize=True)
    backend_name = jax.default_backend().upper()
    sphere_points_array = generate_sphere_points(sphere_points)

    complex_positions = jnp.concatenate([target.atom_positions, binder.atom_positions], axis=0).reshape(-1, 3)
    complex_mask = jnp.concatenate([target.atom_mask, binder.atom_mask], axis=0).reshape(-1)

    num_classes = len(residue_constants.restypes)
    target_sequence = jax.nn.one_hot(target.aatype, num_classes=num_classes)
    binder_sequence = jax.nn.one_hot(binder.aatype, num_classes=num_classes)
    total_sequence = jnp.concatenate([target_sequence, binder_sequence], axis=0)
    complex_radii = jnp.concatenate(
        [
            get_atom_radii(target_sequence, _RESIDUE_RADII_MATRIX),
            get_atom_radii(binder_sequence, _RESIDUE_RADII_MATRIX),
        ]
    )

    n_atoms = int(complex_positions.shape[0])
    max_atoms = estimate_max_atoms(backend_name, sphere_points=sphere_points)
    LOGGER.info("JAX backend %s with %s atoms (limit %s)", backend_name, n_atoms, max_atoms)
    if n_atoms > max_atoms and backend_name != "METAL":
        raise ValueError(f"Too many atoms for JAX backend: {n_atoms} > {max_atoms}")

    contacts = calculate_residue_contacts(
        target.atom_positions,
        binder.atom_positions,
        target.atom_mask,
        binder.atom_mask,
        distance_cutoff=distance_cutoff,
    )
    contact_types = analyze_contacts(contacts, target_sequence, binder_sequence, _CONTACT_CLASS_MATRIX)

    if backend_name == "METAL":
        block_size = estimate_optimal_block_size(n_atoms)
        complex_sasa = calculate_sasa_batch(
            coords=complex_positions,
            vdw_radii=complex_radii,
            mask=complex_mask,
            block_size=block_size,
            sphere_points=sphere_points_array,
        )
    else:
        complex_sasa = calculate_sasa(
            coords=complex_positions,
            vdw_radii=complex_radii,
            mask=complex_mask,
            sphere_points=sphere_points_array,
        )

    relative_sasa = calculate_relative_sasa(
        complex_sasa=complex_sasa,
        sequence_probabilities=total_sequence,
        relative_sasa_array=_RELATIVE_SASA_ARRAY,
        atoms_per_residue=_ATOMS_PER_RESIDUE,
    )
    nis_percentages = calculate_nis_percentages(
        sasa_values=relative_sasa,
        sequence_probabilities=total_sequence,
        character_matrix=_NIS_CLASS_MATRIX,
        threshold=acc_threshold,
    )
    dg = score_ic_nis(
        contact_types[1],
        contact_types[3],
        contact_types[2],
        contact_types[4],
        nis_percentages[0],
        nis_percentages[1],
        _COEFFS,
        _INTERCEPT,
    )
    kd = dg_to_kd(dg, temperature=temperature)
    sasa_data = build_sasa_records(
        complex_sasa=complex_sasa,
        relative_sasa=relative_sasa,
        target=target,
        binder=binder,
        chain_labels=(target_chain, binder_chain),
    )

    results = ProdigyResults(
        contact_types=ContactAnalysis(contact_types.tolist()),
        binding_affinity=np.float32(dg[0]),
        dissociation_constant=np.float32(kd[0]),
        nis_aliphatic=np.float32(nis_percentages[0]),
        nis_charged=np.float32(nis_percentages[1]),
        nis_polar=np.float32(nis_percentages[2]),
        structure_id=Path(struct_path).stem,
        sasa_data=sasa_data,
    )
    if save_results:
        results.save_results(output_dir or ".")
    if not quiet:
        LOGGER.info("JAX prediction complete for %s", results.structure_id)
    return results
