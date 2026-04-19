import math

import jax.numpy as jnp
import numpy as np
from jax import jit
from tinygrad import Tensor as _TGTensor
from tinygrad import TinyJit as _TGTinyJit
from tinygrad import dtypes as _tg_dtypes


def generate_sphere_points(n: int) -> jnp.ndarray:
    """Generate approximately even sphere points with a golden spiral."""
    if n <= 0:
        return jnp.zeros((0, 3))
    if n == 1:
        return jnp.array([[0.0, 1.0, 0.0]])

    indices = jnp.arange(n, dtype=jnp.float32)
    y = 1.0 - (2.0 * indices) / (n - 1)
    radius = jnp.sqrt(1 - y**2)
    theta = jnp.pi * (3.0 - jnp.sqrt(5.0)) * indices
    return jnp.stack([radius * jnp.cos(theta), y, radius * jnp.sin(theta)], axis=1)

@jit
def calculate_sasa(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4
) -> jnp.ndarray:
    """
    Calculate the solvent-accessible surface area (SASA).

    Args:
        coords: [N, 3] Array of atom coordinates.
        vdw_radii: [N] Array of van der Waals radii.
        mask: [N] Binary mask (1 for valid atoms, 0 for masked).
        sphere_points: [M, 3] Predefined sphere points.
        probe_radius: Probe radius for SASA calculation.

    Returns:
        sasa: [N] Solvent-accessible surface area for each atom.
    """
    # Apply mask to coordinates and radii
    masked_coords = coords * mask[:, None]  # [N, 3]
    masked_radii = vdw_radii * mask         # [N]
    radii_with_probe = (masked_radii + probe_radius) * mask

    # Interaction matrix: check for overlapping atoms
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]  # Pairwise differences
    dist2 = jnp.sum(diff ** 2, axis=-1)  # Squared distances
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    # SASA calculation
    scaled_points = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    diff = scaled_points[:, :, None, :] - masked_coords[None, None, :, :]  # [N, M, N, 3] # TO DO too much memory
    dist2 = jnp.sum(diff ** 2, axis=-1)  # [N, M, N]
    is_buried = (dist2 <= radii_with_probe[None, None, :] ** 2) & interaction_matrix[:, None, :]
    buried_points = jnp.any(is_buried, axis=-1)  # [N, M]
    n_accessible = sphere_points.shape[0] - jnp.sum(buried_points, axis=-1)  # [N]

    # Surface area per atom
    areas = 4.0 * jnp.pi * (radii_with_probe ** 2)
    sasa = areas * (n_accessible / sphere_points.shape[0])

    return sasa

def calculate_sasa_batch(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """
    Calculate the solvent-accessible surface area (SASA).
    """
    # Apply mask to coordinates and radii
    masked_coords = coords * mask[:, None]  # [N, 3]
    masked_radii = vdw_radii * mask        # [N]
    radii_with_probe = (masked_radii + probe_radius) * mask  # [N]

    # Interaction matrix: check for overlapping atoms
    diff = masked_coords[:, None, :] - masked_coords[None, :, :]  # [N, N, 3]
    dist2 = jnp.sum(diff ** 2, axis=-1)  # [N, N]
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    buried_points = jnp.zeros((n_atoms, n_points), dtype=bool)

    # Process atoms in blocks to reduce peak memory usage
    for start_idx in range(0, n_atoms, block_size):
        end_idx = min(start_idx + block_size, n_atoms)

        # Calculate scaled points for this block
        block_scaled_points = (sphere_points[None, :, :] *
                             radii_with_probe[start_idx:end_idx, None, None] +
                             masked_coords[start_idx:end_idx, None, :])  # [block, M, 3]

        # Calculate distances to all atoms using a more memory-efficient formulation
        # |a-b|² = |a|² + |b|² - 2⟨a,b⟩
        scaled_points_norm2 = jnp.sum(block_scaled_points**2, axis=-1)  # [block, M]
        coords_norm2 = jnp.sum(masked_coords**2, axis=-1)  # [N]

        # Compute dot product term efficiently
        dot_prod = jnp.einsum('bms,ns->bmn',
                            block_scaled_points,  # [block, M, 3]
                            masked_coords)        # [N, 3]

        # Calculate distances
        dist2 = (scaled_points_norm2[:, :, None] +
                coords_norm2[None, None, :] -
                2 * dot_prod)  # [block, M, N]

        # Check which points are buried
        is_buried = (dist2 <= radii_with_probe[None, None, :]**2) & \
                   interaction_matrix[start_idx:end_idx, None, :]
        block_buried = jnp.any(is_buried, axis=-1)  # [block, M]

        # Update buried points for this block
        buried_points = buried_points.at[start_idx:end_idx].set(block_buried)

    # Calculate final SASA
    n_accessible = n_points - jnp.sum(buried_points, axis=-1)
    areas = 4.0 * jnp.pi * (radii_with_probe ** 2)
    sasa = areas * (n_accessible / n_points)

    return sasa


# --- Tinygrad variants ---------------------------------------------------

def generate_sphere_points_tinygrad(n: int):
    """Golden-spiral sphere points as a tinygrad Tensor."""
    if n <= 0:
        return _TGTensor.zeros((0, 3))
    if n == 1:
        return _TGTensor([[0.0, 1.0, 0.0]])

    indices = np.arange(n, dtype=np.float32)
    y = 1.0 - (2.0 * indices) / (n - 1)
    radius = np.sqrt(1.0 - y * y)
    theta = math.pi * (3.0 - math.sqrt(5.0)) * indices
    points = np.stack([radius * np.cos(theta), y, radius * np.sin(theta)], axis=1).astype(np.float32)
    return _TGTensor(points)


def _calculate_sasa_tinygrad_impl(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    probe_radius: float = 1.4,
):
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask

    diff_inter = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2_inter = (diff_inter ** 2).sum(axis=-1)
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = _TGTensor.eye(coords.shape[0], dtype=_tg_dtypes.bool) == 0
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled_points = (
        sphere_points[None, :, :] * radii_with_probe[:, None, None]
        + masked_coords[:, None, :]
    )
    diff = scaled_points[:, :, None, :] - masked_coords[None, None, :, :]
    dist2 = (diff ** 2).sum(axis=-1)
    is_buried = (dist2 <= radii_with_probe[None, None, :] ** 2) & interaction_matrix[:, None, :]
    buried_points = is_buried.max(axis=-1)
    n_accessible = sphere_points.shape[0] - buried_points.sum(axis=-1)

    areas = 4.0 * math.pi * (radii_with_probe ** 2)
    return (areas * (n_accessible / sphere_points.shape[0])).realize()


calculate_sasa_tinygrad = _TGTinyJit(_calculate_sasa_tinygrad_impl)


def calculate_sasa_batch_tinygrad(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    block_size: int,
    probe_radius: float = 1.4,
):
    """Blocked SASA — compiles one kernel per block to keep the graph small.

    Uses the dot-product formulation ``|a-b|² = |a|² + |b|² - 2⟨a,b⟩`` to
    avoid materializing the ``[block, M, N, 3]`` pairwise-diff tensor.
    """
    masked_coords = (coords * mask[:, None]).realize()
    masked_radii = (vdw_radii * mask).realize()
    radii_with_probe = ((masked_radii + probe_radius) * mask).realize()

    diff_inter = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2_inter = (diff_inter ** 2).sum(axis=-1)
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = _TGTensor.eye(coords.shape[0], dtype=_tg_dtypes.bool) == 0
    interaction_matrix = ((dist2_inter <= radsum2) & not_eye).realize()

    coords_norm2 = (masked_coords * masked_coords).sum(axis=-1).realize()
    radii_probe_sq = (radii_with_probe * radii_with_probe).realize()

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))

    # Per-block SASA into a numpy buffer — avoids chaining huge lazy graphs
    # through ``Tensor.cat`` (tinygrad hits "graph_rewrite stack too big" on
    # aggregated graphs of this shape).
    n_accessible = np.empty(n_atoms, dtype=np.float32)
    for start in range(0, n_atoms, block_size):
        end = min(start + block_size, n_atoms)
        block_radii = radii_with_probe[start:end]
        block_coords = masked_coords[start:end]
        block_inter = interaction_matrix[start:end]

        # scaled: [block, M, 3]
        scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]
        scaled_flat = scaled.reshape(-1, 3)
        scaled_norm2 = (scaled * scaled).sum(axis=-1)  # [block, M]

        # [block*M, N] dot products — O(N²M) instead of O(N²M·3 materialized)
        dot = scaled_flat @ masked_coords.transpose(0, 1)
        dot = dot.reshape(end - start, n_points, n_atoms)

        dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
        is_buried = (dist2 <= radii_probe_sq[None, None, :]) & block_inter[:, None, :]
        buried_points = is_buried.max(axis=-1).sum(axis=-1)
        n_accessible[start:end] = (n_points - buried_points).numpy().astype(np.float32)

    areas = (4.0 * math.pi) * (radii_with_probe ** 2)
    return (areas * _TGTensor(n_accessible) / n_points).realize()
