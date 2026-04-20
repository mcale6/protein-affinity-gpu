"""Residue-level contact computation — shared analyzer, backend-specific
distance kernels.

``analyze_contacts`` is backend-agnostic (broadcast outer product + sum).
``calculate_residue_contacts*`` are deliberately split: the jax variant
uses a 5-D pairwise-diff tensor while the tinygrad variant rewrites it as
two matmuls to dodge Metal's unified-memory pressure on the
``[N_t, N_b, 37, 37, 3]`` intermediate.
"""
from __future__ import annotations

import jax.numpy as jnp

from .utils._array import Array, stack_scalars


def analyze_contacts(
    contacts: Array,
    target_sequence: Array,
    binder_sequence: Array,
    class_matrix: Array,
) -> Array:
    """Aggregate residue contacts into the 6-tuple ``[AA, CC, PP, AC, AP, CP]``.

    Works on jax arrays and tinygrad tensors — uses broadcast outer product
    instead of ``einsum`` (tinygrad has no einsum). For a tinygrad boolean
    ``contacts`` we multiply by ``1.0`` to promote to float for the mask.
    """
    target_classes = target_sequence @ class_matrix
    binder_classes = binder_sequence @ class_matrix
    interaction_probs = target_classes[:, None, :, None] * binder_classes[None, :, None, :]
    masked = interaction_probs * (contacts[:, :, None, None] * 1.0)
    total = masked.sum(axis=0).sum(axis=0)
    return stack_scalars(
        total[0, 0],
        total[1, 1],
        total[2, 2],
        total[0, 1] + total[1, 0],
        total[0, 2] + total[2, 0],
        total[1, 2] + total[2, 1],
    )


def calculate_residue_contacts(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    distance_cutoff: float = 5.5,
) -> jnp.ndarray:
    """JAX: 5-D pairwise diff → any-reduce. Fine on GPU/CPU with enough RAM."""
    target_mask = target_mask.reshape(target_pos.shape[0], -1)
    binder_mask = binder_mask.reshape(binder_pos.shape[0], -1)

    diff = target_pos[:, None, :, None, :] - binder_pos[None, :, None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)

    cutoff_sq = distance_cutoff ** 2
    contact_mask = (
        (dist2 <= cutoff_sq)
        & (target_mask[:, None, :, None] > 0)
        & (binder_mask[None, :, None, :] > 0)
    )
    return jnp.any(contact_mask, axis=(2, 3))


def calculate_residue_contacts_tinygrad(
    target_pos: Array,
    binder_pos: Array,
    target_mask: Array,
    binder_mask: Array,
    distance_cutoff: float = 5.5,
) -> Array:
    """Tinygrad: flatten to atom lists + dot-product distances → reshape + any-reduce.

    Skips the ``[N_t, N_b, 37, 37, 3]`` intermediate — computes
    ``|a|² + |b|² − 2⟨a,b⟩`` via one matmul, then reshapes back to
    ``[N_t, 37, N_b, 37]`` for the any-reduce over atom axes.
    """
    n_t, atoms_t, _ = target_pos.shape
    n_b, atoms_b, _ = binder_pos.shape
    target_mask = target_mask.reshape(n_t, atoms_t)
    binder_mask = binder_mask.reshape(n_b, atoms_b)

    tgt_flat = target_pos.reshape(n_t * atoms_t, 3)
    bnd_flat = binder_pos.reshape(n_b * atoms_b, 3)

    tgt_norm2 = (tgt_flat * tgt_flat).sum(axis=-1)
    bnd_norm2 = (bnd_flat * bnd_flat).sum(axis=-1)
    dot = tgt_flat @ bnd_flat.transpose(-1, -2)
    dist2 = tgt_norm2[:, None] + bnd_norm2[None, :] - 2.0 * dot

    cutoff_sq = distance_cutoff * distance_cutoff
    within = (dist2 <= cutoff_sq).reshape(n_t, atoms_t, n_b, atoms_b)
    valid = (target_mask[:, :, None, None] > 0) & (binder_mask[None, None, :, :] > 0)
    contact_mask = within & valid
    return contact_mask.max(axis=-1).max(axis=1)
