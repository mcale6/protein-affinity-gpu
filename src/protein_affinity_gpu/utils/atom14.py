"""atom37 ↔ atom14 layout conversion.

atom37 pads each residue to 37 heavy-atom columns; most rows are zero
(Gly uses 4 real atoms, Ala 5, Trp 14). atom14 packs only the real atoms,
with padding beyond the residue's actual count. Compacting atom37 → atom14
before the SASA kernel shrinks ``N`` by ~4.4× (167 valid slots across the
20 standard amino acids, averaged to 8.35/residue against 37).

The gather indices live in :mod:`residue_constants` so the mapping stays a
pure function of the restype list.
"""
from __future__ import annotations

import numpy as np

from . import residue_constants


def compact_atom37_to_atom14_numpy(
    positions_37: np.ndarray,
    mask_37: np.ndarray,
    aatype: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Numpy compaction — fine for the inference path where structures are
    loaded from disk as numpy arrays. ``positions_37`` is ``[R, 37, 3]``,
    ``mask_37`` is ``[R, 37]``, ``aatype`` is ``[R]`` int.
    """
    gather = residue_constants.restype_atom14_to_atom37[aatype]  # [R, 14]
    positions_14 = np.take_along_axis(positions_37, gather[:, :, None], axis=1)
    mask_14 = residue_constants.restype_atom14_mask[aatype].astype(mask_37.dtype)
    return positions_14, mask_14


def compact_atom37_to_atom14_jax(positions_37, mask_37, aatype):
    """JAX compaction — gradient-clean via ``take_along_axis``.

    Gather indices are baked into a static ``jnp`` array the first call and
    indexed by ``aatype`` on every call; gradients flow through ``positions_37``
    but not through ``aatype`` (integer sequence isn't differentiable).
    """
    import jax.numpy as jnp

    gather_table = jnp.asarray(residue_constants.restype_atom14_to_atom37)
    mask_table = jnp.asarray(residue_constants.restype_atom14_mask)
    gather = gather_table[aatype]  # [R, 14]
    positions_14 = jnp.take_along_axis(
        positions_37,
        jnp.broadcast_to(gather[:, :, None], gather.shape + (3,)),
        axis=1,
    )
    mask_14 = mask_table[aatype].astype(mask_37.dtype)
    return positions_14, mask_14


def expand_atom14_to_atom37_numpy(
    values_14: np.ndarray,
    aatype: np.ndarray,
) -> np.ndarray:
    """Scatter per-atom14 values back to atom37 layout (zeros elsewhere).

    ``values_14`` is ``[R, 14]`` (or flat ``[R*14]``); returns ``[R, 37]``.
    Used for reassembling per-atom SASA records after the atom14 kernel.

    Padding atom14 slots have ``gather == 0`` and would otherwise clobber the
    real atom37 slot 0 (N), so the scatter is mask-gated.
    """
    gather = residue_constants.restype_atom14_to_atom37[aatype]  # [R, 14]
    valid = residue_constants.restype_atom14_mask[aatype].astype(bool)  # [R, 14]
    R = gather.shape[0]
    v14 = np.asarray(values_14).reshape(R, 14)
    values_37 = np.zeros((R, 37), dtype=v14.dtype)
    r_idx = np.broadcast_to(np.arange(R)[:, None], gather.shape)
    values_37[r_idx[valid], gather[valid]] = v14[valid]
    return values_37


def expand_atom14_to_atom37_jax(values_14, aatype):
    """JAX scatter — only needed if downstream reporting is inside the graph.

    Multiplies by atom14 mask first so padding slots contribute 0 to atom37
    slot 0 (the collision point), preserving the real N atom's value.
    """
    import jax.numpy as jnp

    gather_table = jnp.asarray(residue_constants.restype_atom14_to_atom37)
    mask_table = jnp.asarray(residue_constants.restype_atom14_mask)
    gather = gather_table[aatype]  # [R, 14]
    R = gather.shape[0]
    v14 = jnp.asarray(values_14).reshape(R, 14) * mask_table[aatype].astype(
        values_14.dtype
    )
    values_37 = jnp.zeros((R, 37), dtype=v14.dtype)
    r_idx = jnp.broadcast_to(jnp.arange(R)[:, None], gather.shape)
    return values_37.at[r_idx, gather].set(v14)
