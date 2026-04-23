"""PAE-gated residue contacts — inference-side PRODIGY adaptation.

Recapitulates the structuremap primitive (Bludau et al. 2022,
https://github.com/MannLabs/structuremap_analysis) for inter-chain PRODIGY
contact counting. structuremap treats AlphaFold's Predicted Aligned Error
as additive distance uncertainty:

    # structuremap get_neighbors() / annotate_accessibility() gate:
    PAE_ij <= max_dist
    AND euclidean_dist + PAE_ij <= max_dist
    # (+ an optional Ca->Cb cone for pPSE, not applicable to inter-chain contacts)

Ported here to inter-chain residue contacts. Two gate modes:

  - ``gate_mode="confidence"`` (default): independent gates
        min_heavy_atom_dist <= distance_cutoff
        AND PAE_ij <= pae_cutoff
    PRODIGY's 5.5 A contact + a standalone AF-confidence filter. Clean to
    reason about; easy to ablate by setting ``pae_cutoff=inf``.

  - ``gate_mode="pessimistic"``: structuremap-literal additive gate
        min_heavy_atom_dist + PAE_ij <= distance_cutoff
        AND PAE_ij <= pae_cutoff
    PAE inflates effective distance; only high-confidence close contacts
    survive. Strictly stronger than ``confidence`` mode for any given
    ``pae_cutoff``.

This is a drop-in replacement for
:func:`protein_affinity_gpu.contacts.calculate_residue_contacts` with one
extra argument: the inter-chain PAE block. Downstream
(:func:`protein_affinity_gpu.contacts.analyze_contacts`, NIS, IC-NIS
linear model) is unchanged -- only the contact-counting step consumes PAE.

**NIS is intentionally not PAE-gated.** Solvent accessibility is a
geometric property of the single predicted structure, not a pairwise
inter-chain confidence, so PAE does not fit there cleanly.

Calibration workflow outline
----------------------------
1. Run stock PRODIGY on the crystal PDB -> ``dG_crystal`` (literature baseline).
2. Run stock PRODIGY on the AF-predicted complex -> ``dG_pred_nopae`` (PDB->AF drift baseline).
3. Run PRODIGY with this module swapped in -> ``dG_pred_pae`` (the experiment).
4. Fit new coefficients on the Kastritis 81 (or Vreven v5.5 207) against
   experimental dG; compare LOO-CV Pearson / RMSE across the three.

Usage
-----
>>> from protein_affinity_gpu.contacts_pae import (
...     load_pae_json, slice_pae_inter, calculate_residue_contacts_pae,
... )
>>> pae_full = load_pae_json("predicted.json")            # [L, L]
>>> pae_ab = slice_pae_inter(pae_full, n_target, n_binder) # [N_t, N_b]
>>> contacts_pae = calculate_residue_contacts_pae(
...     target_pos, binder_pos, target_mask, binder_mask,
...     pae_inter=pae_ab, distance_cutoff=5.5, pae_cutoff=10.0,
... )

For the differentiable design-loop version, see ``docs/PAE.md``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import jax.numpy as jnp
import numpy as np

GateMode = Literal["confidence", "pessimistic"]


def load_pae_json(path: str | Path) -> np.ndarray:
    """Load an AlphaFold PAE matrix from JSON.

    Handles the three schemas seen in the wild (matches the structuremap
    branch at ``download_alphafold_pae``):

      - AFDB v1-v2: flat ``"distance"`` list, reshaped to N x N.
      - AFDB v3+ / AF2-Multimer / ColabFold: nested ``"predicted_aligned_error"``.
      - AF3 server: ``"pae"`` key (note: token-indexed; caller must collapse
        multi-token residues e.g. via ``token_chain_ids``).

    Returns
    -------
    np.ndarray shape ``(L, L)`` in Angstroms, dtype float32.
    """
    data = json.loads(Path(path).read_text())
    if isinstance(data, list):                      # AFDB wraps in a single-elem list
        data = data[0]
    if "predicted_aligned_error" in data:
        pae = np.asarray(data["predicted_aligned_error"], dtype=np.float32)
    elif "pae" in data:                              # AF3 server
        pae = np.asarray(data["pae"], dtype=np.float32)
    elif "distance" in data:                         # AFDB v1-v2 flat
        flat = np.asarray(data["distance"], dtype=np.float32)
        size = int(np.sqrt(len(flat)))
        if size * size != len(flat):
            raise ValueError(
                f"AFDB v1-v2 PAE not square: len={len(flat)} (size={size})"
            )
        pae = flat.reshape(size, size)
    else:
        raise ValueError(f"Unknown PAE schema; keys={sorted(data.keys())}")
    if pae.ndim != 2 or pae.shape[0] != pae.shape[1]:
        raise ValueError(f"PAE must be square 2-D; got shape={pae.shape}")
    return pae


def slice_pae_inter(
    pae_full: np.ndarray,
    target_len: int,
    binder_len: int,
    *,
    symmetrize: bool = True,
) -> np.ndarray:
    """Extract the inter-chain PAE block.

    Assumes the convention used elsewhere in this repo (target chain first,
    binder chain second), matching the ipSAE slice at
    ``af_design/modal_afdesign_ba_val.py:402-404``. PAE is generally
    asymmetric because it conditions on the frame of residue j; setting
    ``symmetrize=True`` returns ``0.5 * (upper + lower.T)``.
    """
    tot = target_len + binder_len
    if pae_full.shape[0] < tot:
        raise ValueError(
            f"PAE matrix too small: shape={pae_full.shape}, need >= {tot}"
        )
    upper = pae_full[:target_len, target_len:tot]
    lower = pae_full[target_len:tot, :target_len]
    if symmetrize:
        return 0.5 * (upper + lower.T)
    return upper


def calculate_residue_contacts_pae(
    target_pos: jnp.ndarray,
    binder_pos: jnp.ndarray,
    target_mask: jnp.ndarray,
    binder_mask: jnp.ndarray,
    pae_inter: jnp.ndarray,
    *,
    distance_cutoff: float = 5.5,
    pae_cutoff: float = 10.0,
    gate_mode: GateMode = "confidence",
) -> jnp.ndarray:
    """Inter-chain residue contact mask with structuremap-style PAE gating.

    Mirrors :func:`protein_affinity_gpu.contacts.calculate_residue_contacts`
    (JAX 5-D pairwise diff, any-reduce over atom37 axes) with an additional
    PAE gate.

    Parameters
    ----------
    target_pos : [N_t, 37, 3] atom37 coordinates, target chain.
    binder_pos : [N_b, 37, 3] atom37 coordinates, binder chain.
    target_mask, binder_mask : [N_t, 37] / [N_b, 37] atom presence masks.
    pae_inter : [N_t, N_b] inter-chain PAE in Angstroms. Build with
        :func:`slice_pae_inter`. Residue ordering must match the positions.
    distance_cutoff : float, default 5.5 A (PRODIGY).
    pae_cutoff : float, default 10 A (matches ipSAE default in this repo).
    gate_mode :
        - ``"confidence"``: ``(dist <= cutoff) AND (PAE <= pae_cutoff)``.
        - ``"pessimistic"``: ``(dist + PAE <= cutoff) AND (PAE <= pae_cutoff)``
          (structuremap's additive-uncertainty gate).

    Returns
    -------
    [N_t, N_b] boolean -- feeds
    :func:`protein_affinity_gpu.contacts.analyze_contacts` unchanged.
    """
    n_t, atoms_t, _ = target_pos.shape
    n_b, atoms_b, _ = binder_pos.shape
    target_mask = target_mask.reshape(n_t, atoms_t)
    binder_mask = binder_mask.reshape(n_b, atoms_b)

    diff = target_pos[:, None, :, None, :] - binder_pos[None, :, None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)                     # [N_t, N_b, 37, 37]
    dist = jnp.sqrt(jnp.maximum(dist2, 0.0))

    atom_valid = (
        (target_mask[:, None, :, None] > 0)
        & (binder_mask[None, :, None, :] > 0)
    )
    dist_masked = jnp.where(atom_valid, dist, jnp.inf)
    min_dist = jnp.min(dist_masked, axis=(2, 3))            # [N_t, N_b]

    pae_ok = pae_inter <= pae_cutoff
    if gate_mode == "confidence":
        return (min_dist <= distance_cutoff) & pae_ok
    if gate_mode == "pessimistic":
        return ((min_dist + pae_inter) <= distance_cutoff) & pae_ok
    raise ValueError(
        f"gate_mode must be 'confidence' or 'pessimistic', got {gate_mode!r}"
    )


__all__ = [
    "load_pae_json",
    "slice_pae_inter",
    "calculate_residue_contacts_pae",
    "GateMode",
]
