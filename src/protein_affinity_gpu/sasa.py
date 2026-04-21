import math
from functools import partial
from typing import Callable, Iterator

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tinygrad import Tensor as _TGTensor
from tinygrad import TinyJit as _TGTinyJit
from tinygrad import dtypes as _tg_dtypes

from .utils.logging_utils import get_logger

LOGGER = get_logger(__name__)


def _snapshot_device_memory() -> dict[str, float]:
    """Best-effort device/process memory snapshot (MB). Empty dict if nothing probeable.

    Sources (additive, each try/except-guarded):
    - Process RSS via ``resource.getrusage`` — units differ by OS (bytes on
      macOS, KB on Linux), so we normalize both to MB.
    - ``jax.devices()[0].memory_stats()`` — returns ``None`` on CPU; on GPU
      (CUDA/ROCm/Metal) it exposes ``bytes_in_use`` and ``peak_bytes_in_use``
      cumulative since process start. Peak is the signal we want for sizing.
    - ``tinygrad.helpers.GlobalCounters.mem_used`` — live buffer bytes;
      post-kernel this reflects persistent buffers only (intermediates freed
      on ``.realize()``), so treat alongside the pre-call scratch estimate.
    """
    snap: dict[str, float] = {}
    try:
        import resource
        import sys
        factor = 1024 * 1024 if sys.platform == "darwin" else 1024
        snap["rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / factor
    except Exception:  # noqa: BLE001
        pass
    try:
        stats = jax.devices()[0].memory_stats()
        if stats:
            snap["jax_in_use_mb"] = stats.get("bytes_in_use", 0) / 1e6
            snap["jax_peak_mb"] = stats.get("peak_bytes_in_use", 0) / 1e6
    except Exception:  # noqa: BLE001
        pass
    try:
        from tinygrad.helpers import GlobalCounters
        used = getattr(GlobalCounters, "mem_used", 0) or 0
        if used:
            snap["tg_mem_used_mb"] = used / 1e6
    except Exception:  # noqa: BLE001
        pass
    return snap


def _log_device_memory(tag: str) -> None:
    """Log ``_snapshot_device_memory`` in a single-line ``key=value`` format."""
    snap = _snapshot_device_memory()
    if not snap:
        return
    LOGGER.info("%s: mem %s", tag, " ".join(f"{k}={v:.0f}" for k, v in snap.items()))


def generate_sphere_points(n: int) -> np.ndarray:
    """Golden-spiral sphere points as ``[n, 3]`` float32 numpy. Adapters wrap.

    Midpoint Fibonacci spacing along the Z axis — matches freesasa's
    ``sasa_sr.c::test_points()``. Endpoint spacing collapses two indices onto
    the poles, and Y-axis ordering rotates the spiral relative to the
    molecule versus freesasa's Z ordering; either divergence shifts per-atom
    SASA enough to flip residues across the NIS threshold.
    """
    if n <= 0:
        return np.zeros((0, 3), dtype=np.float32)

    indices = np.arange(n, dtype=np.float32)
    z = 1.0 - (2.0 * indices + 1.0) / n
    radius = np.sqrt(1.0 - z * z)
    theta = np.pi * (3.0 - np.sqrt(5.0)) * indices
    return np.stack([radius * np.cos(theta), radius * np.sin(theta), z], axis=1).astype(np.float32)


def _iter_blocks(n_atoms: int, block_size: int) -> Iterator[tuple[int, int, int]]:
    """Yield ``(start, end, effective_start)`` so every block keeps ``block_size`` atoms.

    The tail block's window is pulled back (``effective_start = n_atoms -
    block_size``) to preserve a uniform kernel shape — compiles once, avoids
    a second kernel for the shorter final block. Callers slice the kernel
    output by ``start - effective_start``.
    """
    for start in range(0, n_atoms, block_size):
        end = min(start + block_size, n_atoms)
        effective_start = min(start, n_atoms - block_size)
        yield start, end, effective_start


def _precompute_sasa_inputs(coords, vdw_radii, mask, probe_radius):
    """Masked coords/radii + cached norms shared by every SASA path.

    Works on JAX arrays and tinygrad tensors — all tensor methods. Callers
    that need realized tinygrad buffers ``.realize()`` the outputs themselves.
    """
    masked_coords = coords * mask[:, None]
    masked_radii = vdw_radii * mask
    radii_with_probe = (masked_radii + probe_radius) * mask
    coords_norm2 = (masked_coords * masked_coords).sum(axis=-1)
    radii_probe_sq = radii_with_probe * radii_with_probe
    return masked_coords, radii_with_probe, coords_norm2, radii_probe_sq


# --- JAX -----------------------------------------------------------------

@jit
def _sasa_block_kernel(
    block_coords: jnp.ndarray,           # [B, 3]
    block_radii: jnp.ndarray,            # [B]
    block_abs_idx: jnp.ndarray,          # [B]  int32 — absolute atom index per block row
    all_coords: jnp.ndarray,             # [N, 3]
    coords_norm2: jnp.ndarray,           # [N]
    all_radii_with_probe: jnp.ndarray,   # [N]
    radii_probe_sq: jnp.ndarray,         # [N]
    sphere_points: jnp.ndarray,          # [M, 3]
) -> jnp.ndarray:
    """Per-block buried-point count — inline ``[B, N]`` inter-mask.

    Dot-product identity ``|a−b|² = |a|² + |b|² − 2⟨a,b⟩`` for both the
    ``[B, N]`` pair-distance pass and the ``[B, M, N]`` sphere-point pass.
    No upfront ``[N, N]`` interaction matrix, mirroring the tinygrad kernel.
    """
    n_atoms = all_coords.shape[0]

    block_norm2 = jnp.sum(block_coords * block_coords, axis=-1)
    dot_bn = block_coords @ all_coords.T
    dist2_bn = block_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_bn
    radsum = block_radii[:, None] + all_radii_with_probe[None, :]
    within = dist2_bn <= (radsum * radsum)
    atom_idx = jnp.arange(n_atoms, dtype=jnp.int32)
    not_self = atom_idx[None, :] != block_abs_idx[:, None]
    block_inter = within & not_self  # [B, N]

    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)
    dot = jnp.einsum("bms,ns->bmn", scaled, all_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & block_inter[:, None, :]
    return jnp.any(is_buried, axis=-1).sum(axis=-1)  # [B]


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
    """Differentiable per-block accessible-point count — inline ``[B, N]`` inter.

    Replaces ``buried = dist² ≤ r²`` with ``sigmoid(β·(r² − dist²))``. Uses
    ``log(1 − sigmoid(x)) = −softplus(x)`` so the per-point accessible
    probability is computed in log-space and stays stable as β → ∞.
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
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)
    dot = jnp.einsum("bms,ns->bmn", scaled, all_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot
    margin = radii_probe_sq[None, None, :] - dist2
    log_not_occluded = -jax.nn.softplus(beta * margin)
    log_not_occluded = jnp.where(block_inter[:, None, :], log_not_occluded, 0.0)
    log_not_buried = log_not_occluded.sum(axis=-1)
    return jnp.exp(log_not_buried).sum(axis=-1)  # [B]


def _dispatch_blocked_jax(
    kernel: Callable,
    masked_coords: jnp.ndarray,
    radii_with_probe: jnp.ndarray,
    coords_norm2: jnp.ndarray,
    radii_probe_sq: jnp.ndarray,
    sphere_points: jnp.ndarray,
    block_size: int,
    *extra_kernel_args,
) -> jnp.ndarray:
    """Iterate ``_iter_blocks`` over ``kernel``; ``extra_kernel_args`` pass-through (e.g. ``beta``)."""
    n_atoms = masked_coords.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))

    block_counts = []
    for start, end, effective_start in _iter_blocks(n_atoms, block_size):
        block_abs_idx = jnp.arange(
            effective_start, effective_start + block_size, dtype=jnp.int32
        )
        counts = kernel(
            masked_coords[effective_start:effective_start + block_size],
            radii_with_probe[effective_start:effective_start + block_size],
            block_abs_idx,
            masked_coords,
            coords_norm2,
            radii_with_probe,
            radii_probe_sq,
            sphere_points,
            *extra_kernel_args,
        )
        write_offset = start - effective_start
        block_counts.append(counts[write_offset:write_offset + (end - start)])
    return jnp.concatenate(block_counts)


def calculate_sasa_batch(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Blocked Shrake–Rupley SASA on JAX.

    Dispatches a ``@jit``'d per-block kernel that computes its ``[B, N]``
    inter-mask inline — no upfront ``[N, N]`` realization. Uniform block
    shape across iterations (tail window pulled back) so the kernel compiles
    exactly once per call-site.
    """
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    buried_counts = _dispatch_blocked_jax(
        _sasa_block_kernel,
        masked_coords, radii_with_probe, coords_norm2, radii_probe_sq,
        sphere_points, block_size,
    )
    n_points = sphere_points.shape[0]
    n_accessible = n_points - buried_counts
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (n_accessible / n_points)
    out.block_until_ready()
    _log_device_memory("jax.sasa.block")
    return out


def calculate_sasa_batch_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable sigmoid-smoothed SASA. Approaches :func:`calculate_sasa_batch` as β→∞."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    beta_array = jnp.asarray(beta, dtype=jnp.float32)
    accessible_counts = _dispatch_blocked_jax(
        _soft_sasa_block_kernel,
        masked_coords, radii_with_probe, coords_norm2, radii_probe_sq,
        sphere_points, block_size,
        beta_array,
    )
    n_points = sphere_points.shape[0]
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (accessible_counts / n_points)
    out.block_until_ready()
    _log_device_memory("jax.sasa.block_soft")
    return out


# --- JAX single-pass + scan variants -------------------------------------
#
# These are alternatives to the per-block Python-dispatch path above:
#
# - ``_calculate_sasa_jax_impl`` is a fully-fused single-pass kernel — direct
#   analog of ``_calculate_sasa_tinygrad_impl``. One ``@jit``, one XLA program;
#   peak scratch is ``[N, M, N]`` so this is the fast path only when that fits
#   on the device. On GPU it lets XLA pick its own tiling.
# - ``_dispatch_blocked_jax_scan`` uses ``jax.lax.scan`` so the entire blocked
#   sweep compiles as one program — same memory profile as the Python-loop
#   dispatcher but no per-block Python latency, and the pattern AlphaFold uses
#   (``layer_stack`` over Evoformer blocks) so ``jax.checkpoint`` plugs in
#   cleanly for memory-efficient backprop.

@jit
def _calculate_sasa_jax_impl(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Fully vectorized Shrake–Rupley SASA — single ``@jit`` pass, no block loop."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]

    dot_nn = masked_coords @ masked_coords.T
    dist2_inter = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = ~jnp.eye(n_atoms, dtype=jnp.bool_)
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)
    dot = jnp.einsum("nms,ks->nmk", scaled, masked_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & interaction_matrix[:, None, :]
    buried = jnp.any(is_buried, axis=-1).sum(axis=-1)
    n_accessible = n_points - buried
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (n_accessible / n_points)


@jit
def _calculate_sasa_jax_soft_impl(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    beta: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Differentiable single-pass SASA. β→∞ recovers :func:`_calculate_sasa_jax_impl`."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]

    dot_nn = masked_coords @ masked_coords.T
    dist2_inter = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = ~jnp.eye(n_atoms, dtype=jnp.bool_)
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    scaled_norm2 = jnp.sum(scaled ** 2, axis=-1)
    dot = jnp.einsum("nms,ks->nmk", scaled, masked_coords)
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    margin = radii_probe_sq[None, None, :] - dist2
    log_not_occluded = -jax.nn.softplus(beta * margin)
    log_not_occluded = jnp.where(interaction_matrix[:, None, :], log_not_occluded, 0.0)
    log_not_buried = log_not_occluded.sum(axis=-1)
    n_accessible = jnp.exp(log_not_buried).sum(axis=-1)
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (n_accessible / n_points)


def _log_single_pass_scratch(n_atoms: int, n_points: int, soft: bool) -> None:
    """Warn when fused [N, M, N] scratch gets large — leading cause of OOM."""
    scratch_gb = n_atoms * n_points * n_atoms * 4 / 1e9
    level = "warning" if scratch_gb > 4.0 else "info"
    getattr(LOGGER, level)(
        "jax.sasa.single%s: N=%d M=%d → peak [N,M,N] scratch ~%.2f GB",
        "_soft" if soft else "", n_atoms, n_points, scratch_gb,
    )


def calculate_sasa_jax(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Single-pass JAX SASA — fastest when ``[N, M, N]`` scratch fits on device."""
    _log_single_pass_scratch(coords.shape[0], sphere_points.shape[0], soft=False)
    out = _calculate_sasa_jax_impl(coords, vdw_radii, mask, sphere_points, probe_radius)
    out.block_until_ready()
    _log_device_memory("jax.sasa.single")
    return out


def calculate_sasa_jax_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable single-pass JAX SASA."""
    _log_single_pass_scratch(coords.shape[0], sphere_points.shape[0], soft=True)
    out = _calculate_sasa_jax_soft_impl(
        coords, vdw_radii, mask, sphere_points,
        jnp.asarray(beta, dtype=jnp.float32), probe_radius,
    )
    out.block_until_ready()
    _log_device_memory("jax.sasa.single_soft")
    return out


# --- JAX neighbor-cutoff SASA -------------------------------------------
#
# Direct port of :func:`_calculate_sasa_tinygrad_neighbor_impl`:
# ``jax.lax.top_k`` on ``-dist²`` picks the K nearest atoms per row, so
# the inner buried-check tensor is ``[N, M, K]`` instead of ``[N, M, N]``
# — same 80× scratch reduction at N=5000 M=100 K=64.
#
# Correctness: lossless when K covers the worst-case neighbor count
# within the physical occlusion radius ``(r_i + r_j + probe)``. For
# atom14 protein spheres K=64 comfortably covers the densest interior.
# Any topk-selected neighbor outside that radius gives ``dist² > r²``
# and falls out of the buried check naturally — no false occlusion.
#
# ``k_neighbors`` is ``static_argnames``'d so XLA const-folds the K
# dimension through ``lax.top_k`` and the gathers. Mode is inference-
# only: ``top_k`` is not usefully differentiable w.r.t. coordinates, so
# no soft variant is provided (use ``mode="scan"`` for training).

@partial(jit, static_argnames=("k_neighbors",))
def _calculate_sasa_jax_neighbor_impl(
    coords: jnp.ndarray,        # [N, 3]
    vdw_radii: jnp.ndarray,     # [N]
    mask: jnp.ndarray,          # [N]
    sphere_points: jnp.ndarray, # [M, 3]
    k_neighbors: int,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Single-pass JAX SASA on K nearest neighbors — ``[N, M, K]`` peak scratch."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]

    # Pairwise distances; push self onto the diagonal so top-K skips it.
    dot_nn = masked_coords @ masked_coords.T                               # [N, N]
    dist2_nn = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    eye = jnp.eye(n_atoms, dtype=jnp.float32)
    dist2_no_self = dist2_nn + eye * 1e10

    # ``top_k`` returns the *largest*; negate distance to get K nearest.
    _, neighbor_idx = jax.lax.top_k(-dist2_no_self, k_neighbors)            # [N, K]

    nb_coords = masked_coords[neighbor_idx]                                 # [N, K, 3]
    nb_radii_sq = radii_probe_sq[neighbor_idx]                              # [N, K]
    nb_norm2 = jnp.sum(nb_coords * nb_coords, axis=-1)                      # [N, K]

    scaled = sphere_points[None, :, :] * radii_with_probe[:, None, None] + masked_coords[:, None, :]
    sp_norm2 = jnp.sum(scaled ** 2, axis=-1)                                # [N, M]

    # ``[N, M, 3] ⊗ [N, 3, K] → [N, M, K]`` — the only big tensor here.
    dot = jnp.einsum("nms,nks->nmk", scaled, nb_coords)                     # [N, M, K]
    dist2 = sp_norm2[:, :, None] + nb_norm2[:, None, :] - 2.0 * dot         # [N, M, K]

    is_buried = dist2 <= nb_radii_sq[:, None, :]
    buried = jnp.any(is_buried, axis=-1).sum(axis=-1)                       # [N]
    n_accessible = n_points - buried
    areas = 4.0 * jnp.pi * radii_probe_sq
    return areas * (n_accessible / n_points)


def calculate_sasa_jax_neighbor(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    k_neighbors: int = 64,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Neighbor-cutoff JAX SASA — direct analog of :func:`calculate_sasa_tinygrad_neighbor`."""
    n_atoms = int(coords.shape[0])
    k = int(min(k_neighbors, max(1, n_atoms - 1)))
    n_points = int(sphere_points.shape[0])
    scratch_mb = n_atoms * n_points * k * 4 / 1e6
    dense_mb = n_atoms * n_points * n_atoms * 4 / 1e6
    LOGGER.info(
        "jax.sasa.neighbor: N=%d M=%d K=%d → scratch ~%.1f MB vs dense ~%.1f MB (%.0f× smaller)",
        n_atoms, n_points, k, scratch_mb, dense_mb, dense_mb / max(scratch_mb, 0.01),
    )
    out = _calculate_sasa_jax_neighbor_impl(
        coords, vdw_radii, mask, sphere_points, k, probe_radius,
    )
    out.block_until_ready()
    _log_device_memory("jax.sasa.neighbor")
    return out


def _scan_block_starts(n_atoms: int, block_size: int) -> tuple[np.ndarray, int]:
    """``effective_starts`` array for ``lax.scan`` over uniform block windows."""
    block_size = max(1, min(int(block_size), n_atoms))
    n_blocks = (n_atoms + block_size - 1) // block_size
    starts = np.array(
        [min(i * block_size, n_atoms - block_size) for i in range(n_blocks)],
        dtype=np.int32,
    )
    return starts, block_size


def _dispatch_blocked_jax_scan(
    kernel: Callable,
    masked_coords: jnp.ndarray,
    radii_with_probe: jnp.ndarray,
    coords_norm2: jnp.ndarray,
    radii_probe_sq: jnp.ndarray,
    sphere_points: jnp.ndarray,
    block_size: int,
    *extra_kernel_args,
) -> jnp.ndarray:
    """Same blocked sweep as :func:`_dispatch_blocked_jax`, fused into one ``lax.scan``.

    Compiles once per ``(N, block_size, M)``, no per-block Python dispatch.
    Differentiable; wrap the ``body`` with ``jax.checkpoint`` for ``O(carry)``
    memory training (AlphaFold pattern).
    """
    n_atoms = masked_coords.shape[0]
    starts_np, block_size = _scan_block_starts(n_atoms, block_size)
    starts = jnp.asarray(starts_np)
    n_blocks = starts.shape[0]
    arange_b = jnp.arange(block_size, dtype=jnp.int32)
    LOGGER.info(
        "jax.sasa.scan: N=%d block=%d blocks=%d M=%d",
        n_atoms, block_size, n_blocks, sphere_points.shape[0],
    )

    def body(_carry, eff_start):
        block_coords = jax.lax.dynamic_slice_in_dim(masked_coords, eff_start, block_size, axis=0)
        block_radii = jax.lax.dynamic_slice_in_dim(radii_with_probe, eff_start, block_size, axis=0)
        block_abs_idx = eff_start + arange_b
        counts = kernel(
            block_coords, block_radii, block_abs_idx,
            masked_coords, coords_norm2, radii_with_probe, radii_probe_sq,
            sphere_points, *extra_kernel_args,
        )
        return _carry, counts

    _, all_counts = jax.lax.scan(body, None, starts)  # [n_blocks, block_size]

    # Stitch valid windows. Only the last block can overlap (effective_start
    # was pulled back); take all rows of the leading n_blocks-1 blocks plus
    # the unique tail of the last.
    if n_blocks == 1:
        return all_counts[0, :n_atoms]
    last_eff = int(starts_np[-1])
    last_start = (n_blocks - 1) * block_size
    write_offset = last_start - last_eff
    head = all_counts[:-1].reshape(-1)
    tail = all_counts[-1, write_offset:]
    return jnp.concatenate([head, tail])


def calculate_sasa_batch_scan(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Blocked Shrake–Rupley SASA — :func:`calculate_sasa_batch` via ``lax.scan``."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    buried_counts = _dispatch_blocked_jax_scan(
        _sasa_block_kernel,
        masked_coords, radii_with_probe, coords_norm2, radii_probe_sq,
        sphere_points, block_size,
    )
    n_points = sphere_points.shape[0]
    n_accessible = n_points - buried_counts
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (n_accessible / n_points)
    out.block_until_ready()
    _log_device_memory("jax.sasa.scan")
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
    """Differentiable blocked SASA via ``lax.scan``."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    beta_array = jnp.asarray(beta, dtype=jnp.float32)
    accessible_counts = _dispatch_blocked_jax_scan(
        _soft_sasa_block_kernel,
        masked_coords, radii_with_probe, coords_norm2, radii_probe_sq,
        sphere_points, block_size,
        beta_array,
    )
    n_points = sphere_points.shape[0]
    areas = 4.0 * jnp.pi * radii_probe_sq
    out = areas * (accessible_counts / n_points)
    out.block_until_ready()
    _log_device_memory("jax.sasa.scan_soft")
    return out


# --- Tinygrad ------------------------------------------------------------

def _calculate_sasa_tinygrad_impl(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    probe_radius: float = 1.4,
):
    """Fully vectorized Shrake–Rupley SASA — single TinyJit-fused pass.

    Dot-product identity ``|a−b|² = |a|² + |b|² − 2⟨a,b⟩`` for both the
    ``[N, N]`` interaction mask and the ``[N, M, N]`` sphere-point pass —
    mask scratch is ``[N, N]`` float32 instead of ``[N, N, 3]``.
    """
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )

    dot_nn = masked_coords @ masked_coords.transpose(-1, -2)              # [N, N]
    dist2_inter = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    radsum2 = (radii_with_probe[:, None] + radii_with_probe[None, :]) ** 2
    not_eye = _TGTensor.eye(coords.shape[0], dtype=_tg_dtypes.bool) == 0
    interaction_matrix = (dist2_inter <= radsum2) & not_eye

    scaled_points = (
        sphere_points[None, :, :] * radii_with_probe[:, None, None]
        + masked_coords[:, None, :]
    )  # [N, M, 3]
    scaled_norm2 = (scaled_points * scaled_points).sum(axis=-1)           # [N, M]
    dot = scaled_points @ masked_coords.transpose(-1, -2)                 # [N, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & interaction_matrix[:, None, :]
    buried_points = is_buried.max(axis=-1).sum(axis=-1)
    n_accessible = sphere_points.shape[0] - buried_points

    areas = 4.0 * math.pi * radii_probe_sq
    return (areas * (n_accessible / sphere_points.shape[0])).realize()


# TinyJit is shape-captured, so a naked singleton
# ``_TGTinyJit(_calculate_sasa_tinygrad_impl)`` raises ``JitError: args
# mismatch`` the second time it's called with a different N or M. Cache
# one JIT per ``(N, M)`` shape tuple — matches the block-kernel strategy
# below.
_sasa_tinygrad_jit_cache: dict[tuple[int, int], "_TGTinyJit"] = {}


def calculate_sasa_tinygrad(coords, vdw_radii, mask, sphere_points, probe_radius: float = 1.4):
    """Full-graph tinygrad SASA — compiled once per ``(N, M)`` shape tuple."""
    key = (int(coords.shape[0]), int(sphere_points.shape[0]))
    jit = _sasa_tinygrad_jit_cache.get(key)
    if jit is None:
        scratch_mb = key[0] * key[1] * key[0] * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa.single: compiling new TinyJit for N=%d M=%d "
            "(scratch ~%.1f MB, cache size %d → %d)",
            key[0], key[1], scratch_mb,
            len(_sasa_tinygrad_jit_cache), len(_sasa_tinygrad_jit_cache) + 1,
        )
        jit = _TGTinyJit(_calculate_sasa_tinygrad_impl)
        _sasa_tinygrad_jit_cache[key] = jit
    else:
        LOGGER.debug("tinygrad.sasa.single: TinyJit cache hit N=%d M=%d", *key)
    out = jit(coords, vdw_radii, mask, sphere_points, probe_radius)
    _log_device_memory("tinygrad.sasa.single")
    return out


# --- Tinygrad neighbor-cutoff SASA --------------------------------------
#
# Same algorithm as ``_calculate_sasa_tinygrad_impl`` but with the inner
# ``[N, M, N]`` buried-check shrunk to ``[N, M, K]`` by selecting the K
# nearest neighbors of each atom. The dense ``[N, N]`` interaction mask
# is implicitly the cutoff in the original kernel; we just *use* it to
# skip work instead of broadcasting it. ``Tensor.topk`` and fancy
# indexing both fuse inside ``TinyJit`` so the topk + gather + buried
# pass compile as one program — no intermediate ``[N, N]`` materialized
# beyond the ``[N, N]`` distance matrix needed for topk itself.
#
# Memory: ``[N, M, K]`` scratch instead of ``[N, M, N]`` — at N=5000
# M=100 K=64 that's 130 MB instead of 10 GB, ~80× reduction.
#
# Correctness vs. the dense kernel: lossless when K covers the worst-
# case neighbor count within the physical occlusion radius
# ``(r_i + r_j + probe)``. Protein interiors max out around 40-60
# atoms within that radius for atom14 spheres; K=64 is a safe default.
# The buried check itself is the physical cutoff (``dist² ≤ r²``),
# so any neighbor selected by topk that's beyond that radius simply
# contributes zero — no false occlusion.

def _calculate_sasa_tinygrad_neighbor_impl(
    coords,                 # [N, 3]
    vdw_radii,              # [N]
    mask,                   # [N]
    sphere_points,          # [M, 3]
    k_neighbors: int = 64,
    probe_radius: float = 1.4,
):
    """Single-pass tinygrad SASA on K nearest neighbors. ``K`` is const-folded."""
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]

    # Pairwise atom distances; drop self by adding a large constant on the
    # diagonal so topk picks K *other* atoms.
    dot_nn = masked_coords @ masked_coords.transpose(-1, -2)               # [N, N]
    dist2_nn = coords_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_nn
    eye = _TGTensor.eye(n_atoms, dtype=_tg_dtypes.float32)
    dist2_no_self = dist2_nn + eye * 1e10

    # K nearest neighbor indices per atom. ``-dist2`` because ``topk`` returns
    # *largest*; we want smallest distance.
    _, neighbor_idx = (-dist2_no_self).topk(k_neighbors, dim=-1)            # [N, K]

    nb_coords = masked_coords[neighbor_idx]                                 # [N, K, 3]
    nb_radii_sq = radii_probe_sq[neighbor_idx]                              # [N, K]
    nb_norm2 = (nb_coords * nb_coords).sum(axis=-1)                         # [N, K]

    # Sphere points around each atom centre.
    scaled_points = (
        sphere_points[None, :, :] * radii_with_probe[:, None, None]
        + masked_coords[:, None, :]
    )                                                                       # [N, M, 3]
    sp_norm2 = (scaled_points * scaled_points).sum(axis=-1)                 # [N, M]

    # ``[N, M, 3] @ [N, 3, K] → [N, M, K]`` — the only big tensor here.
    dot = scaled_points @ nb_coords.transpose(-1, -2)                       # [N, M, K]
    dist2 = sp_norm2[:, :, None] + nb_norm2[:, None, :] - 2.0 * dot         # [N, M, K]

    # Buried check vs neighbor radii. No explicit interaction mask: any
    # neighbor outside (r_i + r_j + probe) gives ``dist² > r²`` and falls
    # out naturally.
    is_buried = dist2 <= nb_radii_sq[:, None, :]                            # [N, M, K]
    buried_points = is_buried.max(axis=-1).sum(axis=-1)                     # [N]
    n_accessible = n_points - buried_points

    areas = 4.0 * math.pi * radii_probe_sq
    return (areas * (n_accessible / n_points)).realize()


_sasa_tinygrad_neighbor_jit_cache: dict[tuple[int, int, int], "_TGTinyJit"] = {}


def calculate_sasa_tinygrad_neighbor(
    coords, vdw_radii, mask, sphere_points,
    k_neighbors: int = 64,
    probe_radius: float = 1.4,
):
    """Neighbor-cutoff tinygrad SASA — JIT cached per ``(N, M, K)``."""
    n_atoms = int(coords.shape[0])
    k_neighbors = int(min(k_neighbors, max(1, n_atoms - 1)))
    key = (n_atoms, int(sphere_points.shape[0]), k_neighbors)
    jit = _sasa_tinygrad_neighbor_jit_cache.get(key)
    if jit is None:
        scratch_mb = key[0] * key[1] * key[2] * 4 / 1e6
        dense_mb = key[0] * key[1] * key[0] * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa.neighbor: compiling new TinyJit for N=%d M=%d K=%d "
            "(scratch ~%.1f MB vs dense ~%.1f MB → %.0f× smaller, cache size %d → %d)",
            key[0], key[1], key[2], scratch_mb, dense_mb, dense_mb / max(scratch_mb, 0.01),
            len(_sasa_tinygrad_neighbor_jit_cache), len(_sasa_tinygrad_neighbor_jit_cache) + 1,
        )
        jit = _TGTinyJit(_calculate_sasa_tinygrad_neighbor_impl)
        _sasa_tinygrad_neighbor_jit_cache[key] = jit
    else:
        LOGGER.debug("tinygrad.sasa.neighbor: TinyJit cache hit N=%d M=%d K=%d", *key)
    out = jit(coords, vdw_radii, mask, sphere_points, k_neighbors, probe_radius)
    _log_device_memory("tinygrad.sasa.neighbor")
    return out


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

    Same shape as the JAX blocked kernel: ``[B, N]`` inter-mask computed
    inline via dot-product identity. ``block_abs_idx`` is a per-call ``[B]``
    buffer (scalars get const-folded by TinyJit on first trace and wouldn't
    update on later calls).
    """
    n_atoms = all_coords.shape[0]
    n_points = sphere_points.shape[0]

    block_norm2 = (block_coords * block_coords).sum(axis=-1)              # [B]
    dot_bn = block_coords @ all_coords.transpose(-1, -2)                  # [B, N]
    dist2_bn = block_norm2[:, None] + coords_norm2[None, :] - 2.0 * dot_bn

    radsum = block_radii[:, None] + all_radii_with_probe[None, :]         # [B, N]
    within = dist2_bn <= (radsum * radsum)
    atom_idx = _TGTensor.arange(n_atoms, dtype=_tg_dtypes.int32)
    not_self = atom_idx[None, :] != block_abs_idx[:, None]                # [B, N]
    block_inter = within & not_self

    scaled = sphere_points[None, :, :] * block_radii[:, None, None] + block_coords[:, None, :]
    scaled_norm2 = (scaled * scaled).sum(axis=-1)                         # [B, M]
    dot = scaled @ all_coords.transpose(-1, -2)                           # [B, M, N]
    dist2 = scaled_norm2[:, :, None] + coords_norm2[None, None, :] - 2.0 * dot

    is_buried = (dist2 <= radii_probe_sq[None, None, :]) & block_inter[:, None, :]
    buried_points = is_buried.max(axis=-1).sum(axis=-1)                   # [B]
    return (n_points - buried_points).cast(_tg_dtypes.float32).realize()


# TinyJit captures each input tensor's shape on its first trace and errors
# on subsequent calls whose shapes differ. We cache one JIT per (B, N, M)
# triple so repeated calls across differently-sized structures each reuse
# their own compiled kernel.
_sasa_block_jit_cache: dict[tuple[int, int, int], "_TGTinyJit"] = {}


def _get_sasa_block_jit(block_size: int, n_atoms: int, n_sphere_points: int):
    key = (block_size, n_atoms, n_sphere_points)
    jit = _sasa_block_jit_cache.get(key)
    if jit is None:
        scratch_mb = block_size * n_sphere_points * n_atoms * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa: compiling new TinyJit for block=%d N=%d M=%d "
            "(scratch ~%.1f MB, cache size %d → %d)",
            block_size, n_atoms, n_sphere_points, scratch_mb,
            len(_sasa_block_jit_cache), len(_sasa_block_jit_cache) + 1,
        )
        jit = _TGTinyJit(_sasa_block_tinygrad_impl)
        _sasa_block_jit_cache[key] = jit
    else:
        LOGGER.debug(
            "tinygrad.sasa: TinyJit cache hit block=%d N=%d M=%d",
            block_size, n_atoms, n_sphere_points,
        )
    return jit


def calculate_sasa_batch_tinygrad(
    coords,
    vdw_radii,
    mask,
    sphere_points,
    block_size: int,
    probe_radius: float = 1.4,
):
    """Blocked SASA on tinygrad — TinyJit'd per-block kernel, numpy accumulate.

    - Per-block body is ``_TGTinyJit``-wrapped (cache hit after block 1); one
      JIT per ``(block, N, M)`` shape triple so different structures don't
      trip TinyJit's shape-mismatch check.
    - ``[B, N]`` inter-mask computed inline (no 270 MB upfront realize).
    - Tail block uses ``effective_start`` so every JIT call sees the same
      shape — no second compile for the shorter final block.
    - Each block's output is copied out to a numpy slab via ``.numpy()``
      before the next JIT call. TinyJit reuses the output buffer across
      calls — keeping raw Tensor handles from an earlier call means the
      next call's write overwrites those earlier "slices" in place,
      silently zeroing large chunks of SASA when there are 3+ blocks.
      (Lazy ``.contiguous().realize()``, ``+0``, or ``Tensor.cat`` on the
      handles don't copy, so the numpy detach is the only path that's
      correct here.)
    """
    masked_coords, radii_with_probe, coords_norm2, radii_probe_sq = _precompute_sasa_inputs(
        coords, vdw_radii, mask, probe_radius
    )
    masked_coords = masked_coords.realize()
    radii_with_probe = radii_with_probe.realize()
    coords_norm2 = coords_norm2.realize()
    radii_probe_sq = radii_probe_sq.realize()

    n_atoms = coords.shape[0]
    n_points = sphere_points.shape[0]
    block_size = max(1, min(int(block_size), n_atoms))
    sasa_block_jit = _get_sasa_block_jit(block_size, n_atoms, n_points)

    n_accessible_np = np.empty(n_atoms, dtype=np.float32)
    for start, end, effective_start in _iter_blocks(n_atoms, block_size):
        # ``.contiguous().realize()`` detaches the slice from the parent
        # buffer — TinyJit rejects duplicate buffers (slice shares storage
        # with full tensor) and also rejects const scalar inputs.
        block_coords = masked_coords[effective_start:effective_start + block_size].contiguous().realize()
        block_radii = radii_with_probe[effective_start:effective_start + block_size].contiguous().realize()
        block_abs_idx = _TGTensor(
            np.arange(effective_start, effective_start + block_size, dtype=np.int32)
        ).realize()

        block_out = sasa_block_jit(
            block_coords, block_radii, block_abs_idx,
            masked_coords, coords_norm2, radii_with_probe, radii_probe_sq,
            sphere_points,
        )
        write_offset = start - effective_start
        n_accessible_np[start:end] = block_out.numpy()[write_offset:write_offset + (end - start)]

    n_accessible = _TGTensor(n_accessible_np)
    areas = (4.0 * math.pi) * radii_probe_sq
    out = (areas * n_accessible / n_points).realize()
    _log_device_memory("tinygrad.sasa.block")
    return out
