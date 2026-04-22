"""Stable differentiable contact kernels for design-time objectives."""
from __future__ import annotations

import jax
import jax.numpy as jnp


def calculate_residue_contacts_soft(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    distance_cutoff: float = 5.5,
    beta: float = 8.0,
) -> jnp.ndarray:
    """Smooth residue contact probability in ``[0, 1]`` for each residue pair.

    A residue-residue contact is modeled as ``1 - prod(1 - p_ij)`` over all
    valid atom pairs ``i, j`` where each atom-pair contact probability is a
    sigmoid around the hard distance cutoff.
    """
    target_mask = target_mask.reshape(target_pos.shape[0], -1)
    binder_mask = binder_mask.reshape(binder_pos.shape[0], -1)

    diff = target_pos[:, None, :, None, :] - binder_pos[None, :, None, :, :]
    dist2 = jnp.sum(diff**2, axis=-1)
    valid = (
        (target_mask[:, None, :, None] > 0)
        & (binder_mask[None, :, None, :] > 0)
    )

    logits = beta * ((distance_cutoff * distance_cutoff) - dist2)
    log_not_contact = -jax.nn.softplus(logits)
    log_not_contact = jnp.where(valid, log_not_contact, 0.0)
    return 1.0 - jnp.exp(log_not_contact.sum(axis=(2, 3)))


__all__ = ["calculate_residue_contacts_soft"]
