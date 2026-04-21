"""Unified PRODIGY IC-NIS prediction pipeline.

One body, two backends. :func:`_run_pipeline` consumes a
:class:`~.backends.BackendAdapter` and produces :class:`ProdigyResults`;
backend-specific entry points (``predict_binding_affinity_jax``,
``predict_binding_affinity_tinygrad``) remain thin shims for the public
API. :func:`predict_binding_affinity` routes by ``backend`` string.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np

from .backends import BackendAdapter, get_adapter
from .contacts import analyze_contacts
from .results import ContactAnalysis, ProdigyResults, build_sasa_records
from .scoring import (
    calculate_nis_percentages,
    calculate_relative_sasa,
    dg_to_kd,
    get_atom_radii,
    score_ic_nis,
)
from .utils import residue_constants
from .utils.atom14 import compact_complex_atom14, expand_atom14_to_atom37
from .utils.logging_utils import get_logger, log_duration
from .utils.structure import load_complex

Backend = Literal["cpu", "jax", "tinygrad"]

LOGGER = get_logger(__name__)
_ATOMS_PER_RESIDUE_ATOM14 = 14


def _scalar(xp: np.ndarray) -> float:
    """Collapse a 0-d / 1-d / shape-[1] numpy value to a Python float."""
    return float(np.asarray(xp).reshape(-1)[0])


def _run_pipeline(
    adapter: BackendAdapter,
    struct_path: str | Path,
    selection: str,
    distance_cutoff: float,
    acc_threshold: float,
    temperature: float,
    sphere_points: int,
    save_results: bool,
    output_dir: Optional[str | Path],
    quiet: bool,
) -> ProdigyResults:
    """Backend-agnostic PRODIGY IC-NIS pipeline."""
    target_chain, binder_chain = [chain.strip() for chain in selection.split(",")]
    with log_duration(LOGGER, f"{adapter.name}.load_complex", extra=str(struct_path)):
        target, binder = load_complex(struct_path, selection=selection, sanitize=True)

    sphere_points_array = adapter.sphere_points(sphere_points)

    # Compact atom37 → atom14 (~4.4× fewer atoms) for the SASA kernel.
    positions_np, mask_np, _, (target_aatype, binder_aatype) = compact_complex_atom14(
        target, binder
    )
    complex_positions = adapter.from_numpy(positions_np)
    complex_mask = adapter.from_numpy(mask_np)

    num_classes = len(residue_constants.restypes)
    target_sequence = adapter.one_hot(target_aatype, num_classes)
    binder_sequence = adapter.one_hot(binder_aatype, num_classes)
    total_sequence = adapter.concat([target_sequence, binder_sequence], axis=0)

    # ``radii_matrix_atom14`` already zeros padding slots, so no per-atom mask needed.
    complex_radii = adapter.concat(
        [
            get_atom_radii(target_sequence, adapter.radii_matrix_atom14),
            get_atom_radii(binder_sequence, adapter.radii_matrix_atom14),
        ],
        axis=0,
    )

    n_atoms = int(complex_positions.shape[0])
    adapter.validate_size(n_atoms, sphere_points)
    LOGGER.info("device=%s atoms=%d selection=%s", adapter.name, n_atoms, selection)

    # Contacts use atom37 positions — 37×37 pair checks are cheap compared to SASA.
    target_pos_37 = adapter.from_numpy(np.asarray(target.atom_positions, dtype=np.float32))
    binder_pos_37 = adapter.from_numpy(np.asarray(binder.atom_positions, dtype=np.float32))
    target_mask_37 = adapter.from_numpy(np.asarray(target.atom_mask, dtype=np.float32))
    binder_mask_37 = adapter.from_numpy(np.asarray(binder.atom_mask, dtype=np.float32))

    with log_duration(LOGGER, f"{adapter.name}.contacts"):
        contacts = adapter.residue_contacts(
            target_pos_37, binder_pos_37, target_mask_37, binder_mask_37,
            distance_cutoff=distance_cutoff,
        )
        contact_types = analyze_contacts(
            contacts, target_sequence, binder_sequence, adapter.contact_class_matrix,
        )

    block_size = adapter.estimate_block_size(n_atoms, sphere_points)
    with log_duration(
        LOGGER,
        f"{adapter.name}.sasa",
        level=logging.INFO,
        extra=f"N={n_atoms} M={sphere_points} block={block_size}",
    ):
        complex_sasa = adapter.sasa(
            coords=complex_positions,
            vdw_radii=complex_radii,
            mask=complex_mask,
            sphere_points=sphere_points_array,
            block_size=block_size,
        )

    with log_duration(LOGGER, f"{adapter.name}.nis"):
        relative_sasa = calculate_relative_sasa(
            complex_sasa=complex_sasa,
            sequence_probabilities=total_sequence,
            relative_sasa_array=adapter.relative_sasa_array,
            atoms_per_residue=_ATOMS_PER_RESIDUE_ATOM14,
        )
        nis_percentages = calculate_nis_percentages(
            sasa_values=relative_sasa,
            sequence_probabilities=total_sequence,
            character_matrix=adapter.nis_class_matrix,
            threshold=acc_threshold,
        )

    with log_duration(LOGGER, f"{adapter.name}.score"):
        dg = score_ic_nis(
            contact_types[1], contact_types[3], contact_types[2], contact_types[4],
            nis_percentages[0], nis_percentages[1],
            adapter.coeffs, adapter.intercept,
        )
        kd = dg_to_kd(dg, temperature=temperature)

    # Single materialization pass so the compute graph stays connected through
    # scoring — backends that fuse (tinygrad JIT, jax.jit) can span contacts→ΔG.
    contact_types_np = adapter.to_numpy(contact_types)
    nis_np = adapter.to_numpy(nis_percentages)
    complex_sasa_np = adapter.to_numpy(complex_sasa)
    relative_sasa_np = adapter.to_numpy(relative_sasa)
    dg_value = _scalar(adapter.to_numpy(dg))
    kd_value = _scalar(adapter.to_numpy(kd))

    # Scatter atom14 SASA back to atom37 for per-atom reporting.
    all_aatype_np = np.concatenate([target_aatype, binder_aatype])
    complex_sasa_37 = expand_atom14_to_atom37(complex_sasa_np, all_aatype_np).reshape(-1)
    sasa_data = build_sasa_records(
        complex_sasa=complex_sasa_37,
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
        LOGGER.info("%s prediction complete for %s", adapter.name, results.structure_id)
    return results


def predict_binding_affinity(
    struct_path: str | Path,
    backend: Backend = "jax",
    selection: str = "A,B",
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: Optional[str | Path] = ".",
    quiet: bool = True,
    **backend_kwargs: Any,
) -> ProdigyResults:
    """Route PRODIGY prediction to the chosen backend.

    ``backend="cpu"`` delegates to the reference PRODIGY/freesasa pipeline
    in :mod:`.cpu`; the GPU backends share the pipeline in :func:`_run_pipeline`.
    ``backend_kwargs`` flow through to the adapter constructor (e.g.
    ``soft_sasa=True`` / ``soft_beta=...`` for the JAX adapter).
    """
    if backend == "cpu":
        from .cpu import predict_binding_affinity as cpu_predict

        return cpu_predict(
            struct_path,
            selection=selection,
            temperature=temperature,
            distance_cutoff=distance_cutoff,
            acc_threshold=acc_threshold,
            sphere_points=sphere_points,
            save_results=save_results,
            output_dir=output_dir or ".",
            quiet=quiet,
        )

    adapter = get_adapter(backend, **backend_kwargs)
    return _run_pipeline(
        adapter,
        struct_path=struct_path,
        selection=selection,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        temperature=temperature,
        sphere_points=sphere_points,
        save_results=save_results,
        output_dir=output_dir,
        quiet=quiet,
    )


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
    mode: Literal["block", "single", "scan"] = "block",
) -> ProdigyResults:
    """Run the PRODIGY IC-NIS pipeline on JAX.

    ``soft_sasa=True`` swaps the hard Shrake–Rupley threshold for a sigmoid of
    sharpness ``soft_beta`` — meaningful gradients w.r.t. coords / radii at the
    cost of some accuracy (β→∞ recovers the hard kernel). Intended for
    training / differentiable design; leave off for straight inference.

    ``mode`` selects the SASA dispatch: ``"block"`` (per-block Python loop —
    bounded scratch), ``"single"`` (one fused ``@jit``, ``[N, M, N]`` scratch),
    or ``"scan"`` (``jax.lax.scan`` over blocks, AlphaFold-style; pairs with
    ``jax.checkpoint`` for memory-efficient backprop).
    """
    from .backends._jax import JAXAdapter

    return _run_pipeline(
        JAXAdapter(soft_sasa=soft_sasa, soft_beta=soft_beta, mode=mode),
        struct_path=struct_path,
        selection=selection,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        temperature=temperature,
        sphere_points=sphere_points,
        save_results=save_results,
        output_dir=output_dir,
        quiet=quiet,
    )


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
    """Run the PRODIGY IC-NIS pipeline on tinygrad."""
    from .backends._tinygrad import TinygradAdapter

    return _run_pipeline(
        TinygradAdapter(),
        struct_path=struct_path,
        selection=selection,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        temperature=temperature,
        sphere_points=sphere_points,
        save_results=save_results,
        output_dir=output_dir,
        quiet=quiet,
    )
