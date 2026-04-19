import jax.numpy as jnp


def calculate_residue_contacts(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    distance_cutoff: float = 5.5,
) -> jnp.ndarray:
    """Calculate residue-level contacts from atom coordinates."""
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


def analyze_contacts(
    contacts: jnp.ndarray,
    target_sequence: jnp.ndarray,
    binder_sequence: jnp.ndarray,
    class_matrix: jnp.ndarray,
) -> jnp.ndarray:
    """Compute interaction classes across contacting residues."""
    target_classes = target_sequence @ class_matrix
    binder_classes = binder_sequence @ class_matrix
    interaction_probs = jnp.einsum("ti,bj->tbij", target_classes, binder_classes)
    masked_interactions = interaction_probs * contacts[:, :, None, None]
    total = masked_interactions.sum(axis=(0, 1))
    return jnp.array(
        [
            total[0, 0],
            total[1, 1],
            total[2, 2],
            total[0, 1] + total[1, 0],
            total[0, 2] + total[2, 0],
            total[1, 2] + total[2, 1],
        ]
    )
