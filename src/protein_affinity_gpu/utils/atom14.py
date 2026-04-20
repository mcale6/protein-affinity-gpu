"""atom37 ↔ atom14 layout conversion.

atom37 pads each residue to 37 heavy-atom columns; most rows are zero
(Gly uses 4 real atoms, Ala 5, Trp 14). atom14 packs only the real atoms,
with padding beyond the residue's actual count. Compacting atom37 → atom14
before the SASA kernel shrinks ``N`` by ~4.4× (167 valid slots across the
20 standard amino acids, averaged to 8.35/residue against 37).

A single ``xp`` parameter selects the array library — numpy by default
(inference path where structures come off disk), or ``jax.numpy`` when the
compaction must stay differentiable through ``positions_37``.
"""
from __future__ import annotations

import numpy as np

from . import residue_constants
from .structure import Protein


def compact_atom37_to_atom14(positions_37, mask_37, aatype, xp=np):
    """Gather atom37 → atom14 via ``take_along_axis``.

    ``positions_37`` is ``[R, 37, 3]``, ``mask_37`` is ``[R, 37]``,
    ``aatype`` is ``[R]`` int. Returns ``(positions_14, mask_14)``.
    Gradients flow through ``positions_37`` when ``xp=jnp``; ``aatype``
    is an integer sequence so no gradient.
    """
    gather = xp.asarray(residue_constants.restype_atom14_to_atom37)[aatype]  # [R, 14]
    positions_14 = xp.take_along_axis(
        positions_37,
        xp.broadcast_to(gather[:, :, None], gather.shape + (3,)),
        axis=1,
    )
    mask_14 = xp.asarray(residue_constants.restype_atom14_mask)[aatype].astype(mask_37.dtype)
    return positions_14, mask_14


def expand_atom14_to_atom37(values_14, aatype, xp=np):
    """Scatter per-atom14 values back to atom37 (zeros in padding slots).

    ``values_14`` is ``[R, 14]`` (or flat ``[R*14]``); returns ``[R, 37]``.
    Padding atom14 slots have ``gather == 0`` and would otherwise clobber
    atom37 slot 0 (N), so values are mask-gated to zero before scatter.
    """
    gather = xp.asarray(residue_constants.restype_atom14_to_atom37)[aatype]  # [R, 14]
    mask_14 = xp.asarray(residue_constants.restype_atom14_mask)[aatype]
    R = gather.shape[0]
    v14 = xp.asarray(values_14).reshape(R, 14) * mask_14.astype(values_14.dtype)
    r_idx = xp.broadcast_to(xp.arange(R)[:, None], gather.shape)

    if xp is np:
        values_37 = np.zeros((R, 37), dtype=v14.dtype)
        values_37[r_idx, gather] = v14
        return values_37

    values_37 = xp.zeros((R, 37), dtype=v14.dtype)
    return values_37.at[r_idx, gather].set(v14)


def compact_complex_atom14(
    target: Protein, binder: Protein
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Numpy-side prep for a two-chain complex → concatenated atom14 arrays.

    Returns ``(positions [N*14, 3], mask [N*14], aatype_concat [R_t+R_b],
    (target_aatype, binder_aatype))`` — the last tuple is kept for the atom37
    scatter back in the reporting step. All dtype float32 / int64.
    """
    target_aatype = np.asarray(target.aatype)
    binder_aatype = np.asarray(binder.aatype)
    target_pos14, target_mask14 = compact_atom37_to_atom14(
        np.asarray(target.atom_positions, dtype=np.float32),
        np.asarray(target.atom_mask, dtype=np.float32),
        target_aatype,
    )
    binder_pos14, binder_mask14 = compact_atom37_to_atom14(
        np.asarray(binder.atom_positions, dtype=np.float32),
        np.asarray(binder.atom_mask, dtype=np.float32),
        binder_aatype,
    )
    positions = np.concatenate(
        [target_pos14.reshape(-1, 3), binder_pos14.reshape(-1, 3)], axis=0
    )
    mask = np.concatenate([target_mask14.reshape(-1), binder_mask14.reshape(-1)], axis=0)
    aatype = np.concatenate([target_aatype, binder_aatype])
    return positions, mask, aatype, (target_aatype, binder_aatype)
