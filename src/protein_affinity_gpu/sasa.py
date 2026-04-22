"""Default SASA kernels — JAX (blocked, ``lax.scan``, single-pass) and the
blocked tinygrad dispatch used by the tinygrad adapter.

Kernels kept here cover the paths we expect to run in production: the JAX
blocked / scan / single-pass families and the tinygrad blocked kernel.
Experimental variants (soft/differentiable, neighbor-cutoff, the fully-fused
tinygrad single-pass) still live in :mod:`.sasa_experimental` and import the
shared primitives below.
"""
import math
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


# --- JAX blocked kernel --------------------------------------------------

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
    No upfront ``[N, N]`` interaction matrix.
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


# --- JAX lax.scan variant -------------------------------------------------
#
# Same per-block kernel as :func:`calculate_sasa_batch`, dispatched via
# ``jax.lax.scan`` so the whole sweep compiles as one program (no per-block
# Python latency). Matches AlphaFold's ``layer_stack`` pattern, so
# ``jax.checkpoint`` slots in cleanly for memory-efficient backprop.


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
    checkpoint_body: bool = False,
) -> jnp.ndarray:
    """Same blocked sweep as :func:`_dispatch_blocked_jax`, fused into one ``lax.scan``.

    Compiles once per ``(N, block_size, M)``, no per-block Python dispatch.
    Differentiable; set ``checkpoint_body=True`` to wrap the body with
    ``jax.checkpoint`` for ``O(carry)`` memory training (AlphaFold pattern) —
    needed when the whole sweep sits inside a backward pass and per-block
    activations would otherwise dominate peak memory.
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

    scan_body = jax.checkpoint(body) if checkpoint_body else body
    _, all_counts = jax.lax.scan(scan_body, None, starts)  # [n_blocks, block_size]

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


# --- JAX single-pass kernel ----------------------------------------------
#
# Fully-fused single-pass SASA — one ``@jit``, one XLA program. Peak scratch
# ``[N, M, N]`` so this is the fast path only when that fits on the device;
# on GPU XLA picks its own tiling. Soft/differentiable variant lives in
# :mod:`.sasa_experimental` and reuses :func:`_log_single_pass_scratch`.

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


# --- Tinygrad blocked kernel --------------------------------------------

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
    jit_fn = _sasa_block_jit_cache.get(key)
    if jit_fn is None:
        scratch_mb = block_size * n_sphere_points * n_atoms * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa: compiling new TinyJit for block=%d N=%d M=%d "
            "(scratch ~%.1f MB, cache size %d → %d)",
            block_size, n_atoms, n_sphere_points, scratch_mb,
            len(_sasa_block_jit_cache), len(_sasa_block_jit_cache) + 1,
        )
        jit_fn = _TGTinyJit(_sasa_block_tinygrad_impl)
        _sasa_block_jit_cache[key] = jit_fn
    else:
        LOGGER.debug(
            "tinygrad.sasa: TinyJit cache hit block=%d N=%d M=%d",
            block_size, n_atoms, n_sphere_points,
        )
    return jit_fn


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


# --- Tinygrad single-pass kernel ----------------------------------------

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


# TinyJit is shape-captured, so one singleton ``_TGTinyJit(…)`` would raise
# ``JitError: args mismatch`` the second time it's called with a different N
# or M. Cache one JIT per ``(N, M)`` shape tuple — matches the block-kernel
# strategy in :func:`calculate_sasa_batch_tinygrad`.
_sasa_tinygrad_jit_cache: dict[tuple[int, int], "_TGTinyJit"] = {}


def calculate_sasa_tinygrad(
    coords, vdw_radii, mask, sphere_points, probe_radius: float = 1.4,
):
    """Full-graph tinygrad SASA — compiled once per ``(N, M)`` shape tuple."""
    key = (int(coords.shape[0]), int(sphere_points.shape[0]))
    jit_fn = _sasa_tinygrad_jit_cache.get(key)
    if jit_fn is None:
        scratch_mb = key[0] * key[1] * key[0] * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa.single: compiling new TinyJit for N=%d M=%d "
            "(scratch ~%.1f MB, cache size %d → %d)",
            key[0], key[1], scratch_mb,
            len(_sasa_tinygrad_jit_cache), len(_sasa_tinygrad_jit_cache) + 1,
        )
        jit_fn = _TGTinyJit(_calculate_sasa_tinygrad_impl)
        _sasa_tinygrad_jit_cache[key] = jit_fn
    else:
        LOGGER.debug("tinygrad.sasa.single: TinyJit cache hit N=%d M=%d", *key)
    out = jit_fn(coords, vdw_radii, mask, sphere_points, probe_radius)
    _log_device_memory("tinygrad.sasa.single")
    return out


__all__ = [
    "calculate_sasa_batch",
    "calculate_sasa_batch_scan",
    "calculate_sasa_batch_tinygrad",
    "calculate_sasa_jax",
    "calculate_sasa_tinygrad",
    "generate_sphere_points",
]
