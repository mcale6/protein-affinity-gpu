"""Stable differentiable JAX SASA kernels.

These are the sigmoid-smoothed counterparts to :mod:`protein_affinity_gpu.sasa`.
They provide gradients through the buried-point test and are intended for
training / design use cases rather than strict inference parity.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import jit

from .sasa import (
    _dispatch_blocked_jax,
    _dispatch_blocked_jax_scan,
    _log_device_memory,
    _log_single_pass_scratch,
    _precompute_sasa_inputs,
)


@jit
def _soft_sasa_block_kernel(
    block_coords: jnp.ndarray,
    block_radii: jnp.ndarray,
    block_abs_idx: jnp.ndarray,
    all_coords: jnp.ndarray,
    coords_norm2: jnp.ndarray,
    all_radii_with_probe: jnp.ndarray,
    radii_probe_sq: jnp.ndarray,
    sphere_points: jnp.ndarray,
    beta: jnp.ndarray,
) -> jnp.ndarray:
    """Differentiable per-block accessible-point count.

    Replaces ``dist² <= r²`` with ``sigmoid(beta * (r² - dist²))`` and uses
    ``log(1 - sigmoid(x)) = -softplus(x)`` for numerical stability.
    """
    n_atoms = all_coords.shape[0]

    block_norm2 = jnp.sum(block_coords * block_coords, axis=-1)
    dot_bn = block_coords @ all_coords.T
    dist2_bn = block_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_bn
    radsum = block_radii[:, None] + all_radii_with_probe[None, :]
    within = dist2_bn <= (radsum * radsum)
    atom_idx = jnp.arange(n_atoms, dtype=jnp.int32)
    not_self = atom_idx[None, :] != block_abs_idx[:, None]
    block_inter = within & not_self

    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]
    scaled_norm2 = jnp.sum(scaled**2, axis=-1)
    dot = jnp.einsum("bms,ns->bmn", scaled, all_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
    margin = radii_probe_sq[None, None, :] - dist2
    log_not_occluded = -jax.nn.softplus(beta * margin)
    log_not_occluded = jnp.where(block_inter[:, None, :], log_not_occluded, 0.0)
    log_not_buried = log_not_occluded.sum(axis=-1)
    return jnp.exp(log_not_buried).sum(axis=-1)


def calculate_sasa_batch_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable blocked JAX SASA. Approaches the hard kernel as beta grows."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords,
        vdw_radii,
        mask,
        probe_radius,
    )
    beta_array = jnp.asarray(beta, dtype=jnp.float32)
    accessible_counts = _dispatch_blocked_jax(
        _soft_sasa_block_kernel,
        masked_coords,
        radii_with_probe,
        coords_norm2,
        radii_probe_sq,
        sphere_points,
        block_size,
        beta_array,
    )
    n_points = sphere_points.shape[0]
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (accessible_counts / n_points)
    block_until_ready = getattr(out, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
        _log_device_memory("jax.sasa.block_soft")
    return out


def calculate_sasa_batch_scan_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable blocked JAX SASA via ``jax.lax.scan``."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords,
        vdw_radii,
        mask,
        probe_radius,
    )
    beta_array = jnp.asarray(beta, dtype=jnp.float32)
    accessible_counts = _dispatch_blocked_jax_scan(
        _soft_sasa_block_kernel,
        masked_coords,
        radii_with_probe,
        coords_norm2,
        radii_probe_sq,
        sphere_points,
        block_size,
        beta_array,
    )
    n_points = sphere_points.shape[0]
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (accessible_counts / n_points)
    block_until_ready = getattr(out, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
        _log_device_memory("jax.sasa.scan_soft")
    return out


@jit
def _calculate_sasa_jax_soft_impl(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    beta: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Differentiable single-pass SASA."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords,
        vdw_radii,
        mask,
        probe_radius,
    )
    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]

    dot_nn = masked_coords @ masked_coords.T
    dist2_inter = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = ~jnp.eye(n_atoms, dtype=jnp.bool_)
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    scaled_norm2 = jnp.sum(scaled**2, axis=-1)
    dot = jnp.einsum("nms,ks->nmk", scaled, masked_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    margin = radii_probe_sq[None, None, :] - dist2
    log_not_occluded = -jax.nn.softplus(beta * margin)
    log_not_occluded = jnp.where(interaction_matrix[:, None, :], log_not_occluded, 0.0)
    log_not_buried = log_not_occluded.sum(axis=-1)
    n_accessible = jnp.exp(log_not_buried).sum(axis=-1)
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (n_accessible / n_points)


def calculate_sasa_jax_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable fully-fused JAX SASA."""
    _log_single_pass_scratch(coords.shape[0], sphere_points.shape[0], soft=True)
    out = _calculate_sasa_jax_soft_impl(
        coords,
        vdw_radii,
        mask,
        sphere_points,
        jnp.asarray(beta, dtype=jnp.float32),
        probe_radius,
    )
    block_until_ready = getattr(out, "block_until_ready", None)
    if callable(block_until_ready):
        block_until_ready()
        _log_device_memory("jax.sasa.single_soft")
    return out


__all__ = [
    "calculate_sasa_batch_soft",
    "calculate_sasa_batch_scan_soft",
    "calculate_sasa_jax_soft",
]
