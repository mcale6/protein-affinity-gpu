import jax.numpy as jnp
from tinygrad import Tensor as _TGTensor


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


# --- Tinygrad variants ---------------------------------------------------

def calculate_residue_contacts_tinygrad(
    target_pos,
    binder_pos,
    target_mask,
    binder_mask,
    distance_cutoff: float = 5.5,
):
    """Residue-level boolean contact matrix (tinygrad) via dot-product distances.

    Skips the [N_t, N_b, 37, 37, 3] pairwise-diff tensor — flattens to atom
    lists, computes ``|a|² + |b|² − 2⟨a,b⟩`` via a single matmul, then reshapes
    back to [N_t, 37, N_b, 37] to any-reduce over the atom axes.
    """
    n_t, atoms_t, _ = target_pos.shape
    n_b, atoms_b, _ = binder_pos.shape
    target_mask = target_mask.reshape(n_t, atoms_t)
    binder_mask = binder_mask.reshape(n_b, atoms_b)

    tgt_flat = target_pos.reshape(n_t * atoms_t, 3)
    bnd_flat = binder_pos.reshape(n_b * atoms_b, 3)

    tgt_norm2 = (tgt_flat * tgt_flat).sum(axis=-1)              # [N_t*37]
    bnd_norm2 = (bnd_flat * bnd_flat).sum(axis=-1)              # [N_b*37]
    dot = tgt_flat @ bnd_flat.transpose(-1, -2)                 # [N_t*37, N_b*37]
    dist2 = tgt_norm2[:, None] + bnd_norm2[None, :] - 2.0 * dot

    cutoff_sq = distance_cutoff * distance_cutoff
    within = (dist2 <= cutoff_sq).reshape(n_t, atoms_t, n_b, atoms_b)
    valid = (target_mask[:, :, None, None] > 0) & (binder_mask[None, None, :, :] > 0)
    contact_mask = within & valid
    # Any-reduce over the two atom axes → [N_t, N_b] residue contact matrix.
    return contact_mask.max(axis=-1).max(axis=1)


def analyze_contacts_tinygrad(
    contacts,
    target_sequence,
    binder_sequence,
    class_matrix,
):
    """Aggregate residue contacts into the 6-tuple [AA, CC, PP, AC, AP, CP] (tinygrad)."""
    target_classes = target_sequence @ class_matrix
    binder_classes = binder_sequence @ class_matrix
    # Replaces jnp.einsum("ti,bj->tbij", ...) — outer product via broadcasts.
    interaction_probs = target_classes[:, None, :, None] * binder_classes[None, :, None, :]
    masked = interaction_probs * contacts[:, :, None, None].float()
    total = masked.sum(axis=0).sum(axis=0)

    return _TGTensor.stack(
        total[0, 0],
        total[1, 1],
        total[2, 2],
        total[0, 1] + total[1, 0],
        total[0, 2] + total[2, 0],
        total[1, 2] + total[2, 1],
    )
