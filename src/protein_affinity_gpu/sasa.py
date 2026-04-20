import math

import jax
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

    # Midpoint Fibonacci spacing along the Z axis — matches freesasa's
    # sasa_sr.c test_points(). Endpoint spacing collapses two indices onto
    # the poles, and Y-axis ordering rotates the spiral relative to the
    # molecule versus freesasa's Z ordering; either divergence shifts
    # per-atom SASA enough to flip residues across the NIS threshold.
    indices = jnp.arange(n, dtype=jnp.float32)
    z = 1.0 - (2.0 * indices + 1.0) / n
    radius = jnp.sqrt(1.0 - z * z)
    theta = jnp.pi * (3.0 - jnp.sqrt(5.0)) * indices
    return jnp.stack([radius * jnp.cos(theta), radius * jnp.sin(theta), z], axis=1)


@jit
def _sasa_block_kernel(
    block_coords: jnp.ndarray,      # [B, 3]
    block_radii: jnp.ndarray,       # [B]
    block_inter: jnp.ndarray,       # [B, N]
    all_coords: jnp.ndarray,        # [N, 3]
    coords_norm2: jnp.ndarray,      # [N]
    radii_probe_sq: jnp.ndarray,    # [N]
    sphere_points: jnp.ndarray,     # [M, 3]
) -> jnp.ndarray:
    """Per-block buried-point count via |a-b|² = |a|² + |b|² - 2⟨a,b⟩."""
    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]  # [B,M,3]
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)  # [B, M]
    dot = jnp.einsum("bms,ns->bmn", scaled, all_coords)  # [B, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & block_inter[:, None, :]
    return jnp.any(is_buried, axis=-1).sum(axis=-1)  # [B]


def calculate_sasa_batch(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Blocked Shrake–Rupley SASA on JAX.

    Dispatches a @jit'd per-block kernel. Uniform block shape is preserved
    across iterations (the tail block pulls its window back so it still has
    ``block_size`` atoms) so the kernel compiles exactly once per call-site.
    """
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask

    diff = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    coords_norm2 = jnp.sum(masked_coords ** 2, axis=-1)
    radii_probe_sq = radii_with_probe ** 2

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))

    block_counts = []
    for start in range(0, n_atoms, block_size):
        end = min(start + block_size, n_atoms)
        effective_start = min(start, n_atoms - block_size)
        counts = _sasa_block_kernel(
            masked_coords[effective_start:effective_start + block_size],
            radii_with_probe[effective_start:effective_start + block_size],
            interaction_matrix[effective_start:effective_start + block_size],
            masked_coords,
            coords_norm2,
            radii_probe_sq,
            sphere_points,
        )
        write_offset = start - effective_start
        block_counts.append(counts[write_offset:write_offset + (end - start)])

    buried_counts = jnp.concatenate(block_counts)
    n_accessible = n_points - buried_counts
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (n_accessible / n_points)


@jit
def _soft_sasa_block_kernel(
    block_coords: jnp.ndarray,      # [B, 3]
    block_radii: jnp.ndarray,       # [B]
    block_inter: jnp.ndarray,       # [B, N]
    all_coords: jnp.ndarray,        # [N, 3]
    coords_norm2: jnp.ndarray,      # [N]
    radii_probe_sq: jnp.ndarray,    # [N]
    sphere_points: jnp.ndarray,     # [M, 3]
    beta: jnp.ndarray,              # scalar sharpness
) -> jnp.ndarray:
    """Differentiable per-block accessible-point count.

    Replaces ``buried = dist² ≤ r²`` with ``sigmoid(β·(r² − dist²))``. Uses the
    identity ``log(1 − sigmoid(x)) = −softplus(x)`` so the per-point accessible
    probability is computed in log-space to stay numerically stable as β → ∞.
    Non-interacting / self pairs contribute 0 to the log-sum via ``where``.
    """
    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]  # [B,M,3]
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)  # [B, M]
    dot = jnp.einsum("bms,ns->bmn", scaled, all_coords)  # [B, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
    margin = radii_probe_sq[None, None, :] - dist2  # >0 if occluded
    log_not_occluded = -jax.nn.softplus(beta * margin)  # log P(point not buried by atom j)
    log_not_occluded = jnp.where(block_inter[:, None, :], log_not_occluded, 0.0)
    log_not_buried = log_not_occluded.sum(axis=-1)  # [B, M]
    accessible_per_point = jnp.exp(log_not_buried)  # [B, M] in [0, 1]
    return accessible_per_point.sum(axis=-1)  # [B]


def calculate_sasa_batch_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable sigmoid-smoothed SASA. Approaches ``calculate_sasa_batch`` as β→∞."""
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask

    diff = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2 = jnp.sum(diff ** 2, axis=-1)
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    interaction_matrix = (dist2 <= radsum2) & ~jnp.eye(coords.shape[0], dtype=bool)

    coords_norm2 = jnp.sum(masked_coords ** 2, axis=-1)
    radii_probe_sq = radii_with_probe ** 2

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))
    beta_array = jnp.asarray(beta, dtype=jnp.float32)

    block_counts = []
    for start in range(0, n_atoms, block_size):
        end = min(start + block_size, n_atoms)
        effective_start = min(start, n_atoms - block_size)
        counts = _soft_sasa_block_kernel(
            masked_coords[effective_start:effective_start + block_size],
            radii_with_probe[effective_start:effective_start + block_size],
            interaction_matrix[effective_start:effective_start + block_size],
            masked_coords,
            coords_norm2,
            radii_probe_sq,
            sphere_points,
            beta_array,
        )
        write_offset = start - effective_start
        block_counts.append(counts[write_offset:write_offset + (end - start)])

    accessible_counts = jnp.concatenate(block_counts)
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (accessible_counts / n_points)


# --- Tinygrad variants ---------------------------------------------------

def generate_sphere_points_tinygrad(n: int):
    """Golden-spiral sphere points as a tinygrad Tensor — computed on-device."""
    if n <= 0:
        return _TGTensor.zeros((0, 3))

    indices = _TGTensor.arange(n, dtype=_tg_dtypes.float32)
    z = 1.0 - (2.0 * indices + 1.0) / n
    radius = (1.0 - z * z).sqrt()
    theta = math.pi * (3.0 - math.sqrt(5.0)) * indices
    return _TGTensor.stack(radius * theta.cos(), radius * theta.sin(), z, dim=1)


def _calculate_sasa_tinygrad_impl(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    probe_radius: float = 1.4,
):
    """Fully vectorized Shrake–Rupley SASA — single TinyJit-fused pass.

    Uses the dot-product identity ``|a−b|² = |a|² + |b|² − 2⟨a,b⟩`` so the hot
    tensor is [N, M, N] float32 instead of [N, M, N, 3]. Cuts peak scratch by
    3× relative to the diff form; for N beyond ~5k the batched variant is
    still the right call, but this path is now the fast one on Metal for
    typical binary complexes.
    """
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask

    diff_inter = masked_coords[:, None, :] - masked_coords[None, :, :]
    dist2_inter = (diff_inter * diff_inter).sum(axis=-1)
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = _TGTensor.eye(coords.shape[0], dtype=_tg_dtypes.bool) == 0
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled_points = (
        sphere_points[None, :, :] * radii_with_probe[:, None, None]
        + masked_coords[:, None, :]
    )  # [N, M, 3]

    scaled_norm2 = (scaled_points * scaled_points).sum(axis=-1)      # [N, M]
    coords_norm2 = (masked_coords * masked_coords).sum(axis=-1)      # [N]
    dot = scaled_points @ masked_coords.transpose(-1, -2)            # [N, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    radii_probe_sq = radii_with_probe * radii_with_probe
    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & interaction_matrix[:, None, :]
    buried_points = is_buried.max(axis=-1).sum(axis=-1)
    n_accessible = sphere_points.shape[0] - buried_points

    areas = 4.0 * math.pi * radii_probe_sq
    return (areas * (n_accessible / sphere_points.shape[0])).realize()


calculate_sasa_tinygrad = _TGTinyJit(_calculate_sasa_tinygrad_impl)


def _sasa_block_tinygrad_impl(
    block_coords,          # [B, 3]    realized leaf
    block_radii,           # [B]       realized leaf
    block_abs_idx,         # [B]       int32; absolute atom index for each block row
    all_coords,            # [N, 3]    realized leaf
    coords_norm2,          # [N]       realized leaf
    all_radii_with_probe,  # [N]       realized leaf
    radii_probe_sq,        # [N]       realized leaf
    sphere_points,         # [M, 3]    realized leaf
):
    """Per-block buried-point count — TinyJit-wrapped for kernel reuse.

    Graph is bounded: 1 pair-distance matmul + 1 probe-scattered matmul + masks
    + one reduce. Inter mask is computed inline (no 270 MB upfront realize).
    ``block_abs_idx`` is a per-call [B] index buffer (scalars get const-folded
    by TinyJit on first trace and wouldn't update on later calls).
    """
    n_atoms = all_coords.shape[0]
    n_points = sphere_points.shape[0]

    # Block-vs-all pair distances via dot-product identity — [B, N].
    block_norm2 = (block_coords * block_coords).sum(axis=-1)              # [B]
    dot_bn = block_coords @ all_coords.transpose(-1, -2)                  # [B, N]
    dist2_bn = block_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_bn

    radsum = block_radii[:, None] + all_radii_with_probe[None, :]         # [B, N]
    within = dist2_bn <= (radsum * radsum)
    atom_idx = _TGTensor.arange(n_atoms, dtype=_tg_dtypes.int32)          # [N]
    not_self = atom_idx[None, :] != block_abs_idx[:, None]                # [B, N]
    block_inter = within & not_self                                       # [B, N]

    # Probe-centered sphere points vs all atoms — [B, M, N] via matmul.
    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]
    scaled_norm2 = (scaled * scaled).sum(axis=-1)                         # [B, M]
    dot = scaled @ all_coords.transpose(-1, -2)                           # [B, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & block_inter[:, None, :]
    buried_points = is_buried.max(axis=-1).sum(axis=-1)                   # [B]
    return (n_points - buried_points).cast(_tg_dtypes.float32).realize()


# TinyJit captures each input tensor's shape on its first trace and errors
# on subsequent calls whose shapes differ. For the batched SASA kernel the
# signature (B, N, M) is fixed per structure/block-size but varies across
# structures in a benchmark loop, so we cache one JIT per shape triple.
_sasa_block_jit_cache: dict[tuple[int, int, int], "_TGTinyJit"] = {}


def _get_sasa_block_jit(block_size: int, n_atoms: int, n_sphere_points: int):
    key = (block_size, n_atoms, n_sphere_points)
    jit = _sasa_block_jit_cache.get(key)
    if jit is None:
        jit = _TGTinyJit(_sasa_block_tinygrad_impl)
        _sasa_block_jit_cache[key] = jit
    return jit


def calculate_sasa_batch_tinygrad(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    block_size: int,
    probe_radius: float = 1.4,
):
    """Blocked SASA on tinygrad — TinyJit'd per-block kernel, pipelined cat.

    Three moves that matter for Metal perf:
    - The per-block body is ``_TGTinyJit``-wrapped, so the compiled kernel is
      cached after the first block; the remaining blocks hit the cache.
      A separate JIT is cached per (block, N, M) triple so repeated calls
      across structures with different atom counts each get their own
      compiled kernel instead of tripping TinyJit's shape-mismatch check.
    - No per-block ``.realize()``. Each JIT call already returns a realized
      buffer, so the outer accumulator builds a shallow 1-level graph over
      those buffers — cat of leaves, safe vs ``graph_rewrite stack too big``.
    - The 16k×16k boolean interaction matrix is gone. Each JIT'd call derives
      its own [B, N] inter mask in-kernel, freeing ~270 MB of unified memory.

    The tail block uses the ``effective_start`` trick so every JIT call sees
    the same shape (``block_size``) — avoids triggering a second kernel
    compile for the shorter final block.
    """
    masked_coords = (coords * mask[:, None]).realize()
    masked_radii = (vdw_radii * mask).realize()
    radii_with_probe = ((masked_radii + probe_radius) * mask).realize()
    coords_norm2 = (masked_coords * masked_coords).sum(axis=-1).realize()
    radii_probe_sq = (radii_with_probe * radii_with_probe).realize()

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))
    sasa_block_jit = _get_sasa_block_jit(block_size, n_atoms, n_points)

    block_slices = []
    for start in range(0, n_atoms, block_size):
        end = min(start + block_size, n_atoms)
        effective_start = min(start, n_atoms - block_size)
        # ``.contiguous().realize()`` detaches the slice from the parent
        # buffer; TinyJit rejects duplicate buffers (slice + full tensor share
        # storage) and also rejects const inputs — both need their own
        # materialized buffer.
        block_coords = masked_coords[effective_start:effective_start + block_size].contiguous().realize()
        block_radii = radii_with_probe[effective_start:effective_start + block_size].contiguous().realize()
        block_abs_idx = _TGTensor(
            np.arange(effective_start, effective_start + block_size, dtype=np.int32)
        ).realize()

        block_out = sasa_block_jit(
            block_coords,
            block_radii,
            block_abs_idx,
            masked_coords,
            coords_norm2,
            radii_with_probe,
            radii_probe_sq,
            sphere_points,
        )  # realized [block_size]
        write_offset = start - effective_start
        block_slices.append(block_out[write_offset:write_offset + (end - start)])

    n_accessible = block_slices[0] if len(block_slices) == 1 else _TGTensor.cat(*block_slices, dim=0)
    areas = (4.0 * math.pi) * radii_probe_sq
    return (areas * n_accessible / n_points).realize()
