"""Stable helpers for AlphaFold/ColabDesign-style differentiable design."""
from __future__ import annotations

import jax
import jax.numpy as jnp

jax.config.update("jax_default_matmul_precision", "highest")

from .contacts import analyze_contacts, calculate_residue_contacts
from .contacts_soft import calculate_residue_contacts_soft
from .sasa import generate_sphere_points
from .sasa_soft import calculate_sasa_batch_scan_soft
from .scoring import (
    NIS_COEFFICIENTS,
    calculate_nis_percentages,
    calculate_relative_sasa,
    score_ic_nis,
)
from .scoring_soft import calculate_nis_percentages_soft
from .utils import residue_constants
from .utils.residue_classification import ResidueClassification
from .utils.residue_library import default_library as residue_library

VALID_BINDER_SEQ_MODES = ("soft", "pseudo")


def _normalize_choice(name: str, value: str, choices: tuple[str, ...]) -> str:
    normalized = value.strip().lower()
    if normalized not in choices:
        available = ", ".join(choices)
        raise ValueError(f"{name} must be one of: {available}. Got: {value!r}")
    return normalized


def add_ba_val_loss(
    model,
    *,
    sphere_points: int = 100,
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    soft_sasa_beta: float = 10.0,
    contact_beta: float = 8.0,
    nis_beta: float = 20.0,
    use_soft_contacts: bool = True,
    use_soft_nis: bool = True,
    binder_seq_mode: str = "soft",
) -> None:
    """Attach a differentiable ``ba_val`` callback to an AfDesign model."""
    binder_seq_mode = _normalize_choice(
        "binder_seq_mode",
        binder_seq_mode,
        VALID_BINDER_SEQ_MODES,
    )
    model.opt["weights"].update({"ba_val": 0.0})

    sphere_points_array = jnp.asarray(generate_sphere_points(sphere_points), dtype=jnp.float32)
    residue_radii_matrix = jnp.asarray(residue_library.radii_matrix, dtype=jnp.float32)
    ic_matrix = jnp.asarray(
        ResidueClassification("ic").classification_matrix,
        dtype=jnp.float32,
    )
    protorp_matrix = jnp.asarray(
        ResidueClassification("protorp").classification_matrix,
        dtype=jnp.float32,
    )
    relative_sasa_array = jnp.asarray(
        ResidueClassification().relative_sasa_array,
        dtype=jnp.float32,
    )
    coeffs = jnp.asarray(NIS_COEFFICIENTS.as_tuple(), dtype=jnp.float32)
    intercept = jnp.asarray(NIS_COEFFICIENTS.intercept, dtype=jnp.float32)
    atoms_per_residue = residue_constants.atom_type_num

    def ba_val_loss(inputs, outputs, aux):
        pos = outputs["structure_module"]["final_atom_positions"]
        mask = outputs["structure_module"]["final_atom_mask"]

        total_len = pos.shape[0]
        binder_len = int(model._binder_len)
        target_len = total_len - binder_len

        target_pos = pos[:target_len]
        target_mask = mask[:target_len]
        binder_pos = pos[target_len:]
        binder_mask = mask[target_len:]

        target_aatype = inputs["batch"]["aatype"][:target_len]
        target_seq = jax.nn.one_hot(
            target_aatype,
            len(residue_constants.restypes),
            dtype=jnp.float32,
        )
        # During ``design_logits()``, ColabDesign feeds ``seq["pseudo"]`` into
        # AlphaFold, but ``seq["soft"]`` is the actual residue-probability
        # simplex. For sequence-dependent auxiliary losses like ``ba_val``,
        # ``soft`` is the cleaner gradient carrier; ``pseudo`` remains useful
        # when you want the loss branch to mirror the AF forward input exactly.
        binder_seq = aux["seq"][binder_seq_mode][0, -binder_len:]

        if use_soft_contacts:
            contacts = calculate_residue_contacts_soft(
                target_pos,
                binder_pos,
                target_mask,
                binder_mask,
                distance_cutoff=distance_cutoff,
                beta=contact_beta,
            )
        else:
            contacts = calculate_residue_contacts(
                target_pos,
                binder_pos,
                target_mask,
                binder_mask,
                distance_cutoff=distance_cutoff,
            )

        contact_types = analyze_contacts(
            contacts,
            target_seq,
            binder_seq,
            class_matrix=ic_matrix,
        )

        target_radii = target_seq @ residue_radii_matrix
        binder_radii = binder_seq @ residue_radii_matrix

        complex_positions = jnp.concatenate([target_pos, binder_pos], axis=0).reshape(-1, 3)
        complex_mask = jnp.concatenate([target_mask, binder_mask], axis=0).reshape(-1)
        complex_radii = jnp.concatenate([target_radii, binder_radii], axis=0).reshape(-1)
        seq_full = jnp.concatenate([target_seq, binder_seq], axis=0)

        block_size = max(1, min(int(complex_positions.shape[0]), 768))
        complex_sasa = calculate_sasa_batch_scan_soft(
            coords=complex_positions,
            vdw_radii=complex_radii,
            mask=complex_mask,
            block_size=block_size,
            sphere_points=sphere_points_array,
            beta=soft_sasa_beta,
            checkpoint_body=True,
        )
        relative_sasa = calculate_relative_sasa(
            complex_sasa=complex_sasa,
            sequence_probabilities=seq_full,
            relative_sasa_array=relative_sasa_array,
            atoms_per_residue=atoms_per_residue,
        )

        if use_soft_nis:
            nis = calculate_nis_percentages_soft(
                sasa_values=relative_sasa,
                sequence_probabilities=seq_full,
                character_matrix=protorp_matrix,
                threshold=acc_threshold,
                beta=nis_beta,
            )
        else:
            nis = calculate_nis_percentages(
                sasa_values=relative_sasa,
                sequence_probabilities=seq_full,
                character_matrix=protorp_matrix,
                threshold=acc_threshold,
            )

        dg = score_ic_nis(
            contact_types[1],
            contact_types[3],
            contact_types[2],
            contact_types[4],
            nis[0],
            nis[1],
            coeffs,
            intercept,
        )
        return {"ba_val": jnp.squeeze(dg)}

    model._callbacks["model"]["loss"].append(ba_val_loss)


__all__ = ["add_ba_val_loss"]
