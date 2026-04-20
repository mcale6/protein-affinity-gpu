import logging
import subprocess
from pathlib import Path
from typing import Optional

import jax
import jax.numpy as jnp
import numpy as np

from .contacts import analyze_contacts, calculate_residue_contacts
from .results import ContactAnalysis, ProdigyResults, build_sasa_records
from .sasa import calculate_sasa_batch, calculate_sasa_batch_soft, generate_sphere_points
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
from .utils.atom14 import (
    compact_atom37_to_atom14_numpy,
    expand_atom14_to_atom37_numpy,
)
from .utils.residue_classification import ResidueClassification
from .utils.residue_library import default_library as residue_library

LOGGER = logging.getLogger(__name__)

_ATOMS_PER_RESIDUE_ATOM14 = 14
_RESIDUE_RADII_MATRIX_ATOM14 = jnp.array(residue_library.radii_matrix_atom14)
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


def estimate_optimal_block_size(
    n_atoms: int,
    backend: str = "CPU",
    sphere_points: int = 100,
    cpu_scratch_bytes: int = 1_000_000_000,
) -> int:
    """Pick a block size for the batched SASA kernel.

    Metal: empirical exp-decay fit — small blocks keep per-dispatch scratch
    under unified-memory pressure. CPU / CUDA: prefer large blocks (fewer
    kernel launches dominate over memory pressure); target ``cpu_scratch_bytes``
    of float32 ``[B, M, N]`` scratch.
    """
    backend = backend.upper()
    if backend == "METAL":
        amplitude = 6.8879e02
        decay = -2.6156e-04
        offset = 17.4525
        block_size = int(round(amplitude * np.exp(decay * n_atoms) + offset))
        max_block = min(250, int(5000 / np.sqrt(max(n_atoms, 1) / 1000)))
        return max(5, min(block_size, max_block))

    per_atom_bytes = sphere_points * max(n_atoms, 1) * 4
    block_size = max(32, cpu_scratch_bytes // per_atom_bytes)
    return int(min(block_size, n_atoms))


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
    soft_sasa: bool = False,
    soft_beta: float = 10.0,
) -> ProdigyResults:
    """Run the JAX affinity prediction pipeline.

    ``soft_sasa=True`` swaps the hard Shrake–Rupley threshold for a sigmoid
    of sharpness ``soft_beta`` — meaningful gradients w.r.t. coords / radii at
    the cost of some accuracy (β→∞ recovers the hard kernel). Intended for
    training / differentiable design; leave off for straight inference.
    """
    target_chain, binder_chain = [chain.strip() for chain in selection.split(",")]
    target, binder = load_complex(struct_path, selection=selection, sanitize=True)
    backend_name = jax.default_backend().upper()
    sphere_points_array = generate_sphere_points(sphere_points)

    # Compact atom37 → atom14 before the SASA kernel. Shrinks N by ~4.4× on
    # a typical complex (avg 8.35 heavy atoms/residue vs 37 padded slots).
    # Numpy path is fine here because the structure is loaded from disk —
    # for differentiable AF-training paths, swap to ``compact_atom37_to_atom14_jax``.
    target_aatype_np = np.asarray(target.aatype)
    binder_aatype_np = np.asarray(binder.aatype)
    target_pos14, target_mask14 = compact_atom37_to_atom14_numpy(
        np.asarray(target.atom_positions), np.asarray(target.atom_mask), target_aatype_np
    )
    binder_pos14, binder_mask14 = compact_atom37_to_atom14_numpy(
        np.asarray(binder.atom_positions), np.asarray(binder.atom_mask), binder_aatype_np
    )
    target_pos14_j = jnp.asarray(target_pos14)
    binder_pos14_j = jnp.asarray(binder_pos14)
    target_mask14_j = jnp.asarray(target_mask14)
    binder_mask14_j = jnp.asarray(binder_mask14)

    complex_positions = jnp.concatenate([target_pos14_j, binder_pos14_j], axis=0).reshape(-1, 3)
    complex_mask = jnp.concatenate([target_mask14_j, binder_mask14_j], axis=0).reshape(-1)

    num_classes = len(residue_constants.restypes)
    target_sequence = jax.nn.one_hot(target.aatype, num_classes=num_classes)
    binder_sequence = jax.nn.one_hot(binder.aatype, num_classes=num_classes)
    total_sequence = jnp.concatenate([target_sequence, binder_sequence], axis=0)
    complex_radii = jnp.concatenate(
        [
            get_atom_radii(target_sequence, _RESIDUE_RADII_MATRIX_ATOM14, target_mask14_j),
            get_atom_radii(binder_sequence, _RESIDUE_RADII_MATRIX_ATOM14, binder_mask14_j),
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

    block_size = estimate_optimal_block_size(n_atoms, backend_name, sphere_points)
    sasa_fn = calculate_sasa_batch_soft if soft_sasa else calculate_sasa_batch
    sasa_kwargs = {"beta": soft_beta} if soft_sasa else {}
    complex_sasa = sasa_fn(
        coords=complex_positions,
        vdw_radii=complex_radii,
        mask=complex_mask,
        block_size=block_size,
        sphere_points=sphere_points_array,
        **sasa_kwargs,
    )

    relative_sasa = calculate_relative_sasa(
        complex_sasa=complex_sasa,
        sequence_probabilities=total_sequence,
        relative_sasa_array=_RELATIVE_SASA_ARRAY,
        atoms_per_residue=_ATOMS_PER_RESIDUE_ATOM14,
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
    # Scatter atom14 SASA back to atom37 so per-atom reporting lines up with
    # the structure's original atom layout.
    all_aatype_np = np.concatenate([target_aatype_np, binder_aatype_np])
    complex_sasa_37 = expand_atom14_to_atom37_numpy(
        np.asarray(complex_sasa), all_aatype_np
    ).reshape(-1)
    sasa_data = build_sasa_records(
        complex_sasa=complex_sasa_37,
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
