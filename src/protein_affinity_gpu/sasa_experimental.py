"""Experimental SASA kernels — soft/differentiable, neighbor-cutoff, the
fully-fused single-pass tinygrad kernel, and bucketed-padding wrappers.

The default path is :func:`.sasa.calculate_sasa_batch` /
:func:`.sasa.calculate_sasa_batch_scan`. The non-differentiable single-pass
JAX kernel (:func:`.sasa.calculate_sasa_jax`) and the blocked tinygrad kernel
(:func:`.sasa.calculate_sasa_batch_tinygrad`) also live next to those
defaults. Everything here is opt-in via
:mod:`.backends._jax_experimental` / :mod:`.backends._tinygrad` and the
experimental predictor entry points in :mod:`.experimental`.
"""
import math
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit
from tinygrad import Tensor as _TGTensor
from tinygrad import TinyJit as _TGTinyJit
from tinygrad import dtypes as _tg_dtypes

from .sasa import (
    LOGGER,
    _dispatch_blocked_jax,
    _dispatch_blocked_jax_scan,
    _log_device_memory,
    _log_single_pass_scratch,
    _precompute_sasa_inputs,
    calculate_sasa_batch,
    calculate_sasa_batch_tinygrad,
)


# --- JAX soft / differentiable blocked kernels ---------------------------

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


def calculate_sasa_batch_soft(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    beta: float = 10.0,
) -> jnp.ndarray:
    """Differentiable sigmoid-smoothed SASA. Approaches the hard block kernel as β→∞."""
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


# --- JAX single-pass soft/differentiable kernel --------------------------
#
# Hard single-pass kernel lives in :mod:`.sasa`; only the soft variant stays
# in the experimental surface.

@jit
def _calculate_sasa_jax_soft_impl(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    sphere_points: jnp.ndarray,
    beta: jnp.ndarray,
    probe_radius: float = 1.4,
) -> jnp.ndarray:
    """Differentiable single-pass SASA. β→∞ recovers :func:`sasa._calculate_sasa_jax_impl`."""
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
# ``jax.lax.top_k`` on ``-dist²`` picks the K nearest atoms per row, so
# the inner buried-check tensor is ``[N, M, K]`` instead of ``[N, M, N]``
# — ~80× scratch reduction at N=5000 M=100 K=64. Lossless when K covers
# the worst-case neighbor count within the physical occlusion radius
# (``r_i + r_j + probe``); for atom14 protein spheres K=64 is safe.
# ``top_k`` is not usefully differentiable, so no soft variant is provided
# — use :func:`calculate_sasa_batch_scan_soft` for training.

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


# TinyJit is shape-captured, so a naked singleton
# ``_TGTinyJit(_calculate_sasa_tinygrad_impl)`` raises ``JitError: args
# mismatch`` the second time it's called with a different N or M. Cache
# one JIT per ``(N, M)`` shape tuple — matches the block-kernel strategy
# below.
_sasa_tinygrad_jit_cache: dict[tuple[int, int], "_TGTinyJit"] = {}


def calculate_sasa_tinygrad(coords, vdw_radii, mask, sphere_points, probe_radius: float = 1.4):
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


# --- Tinygrad neighbor-cutoff SASA --------------------------------------
#
# Same algorithm as ``_calculate_sasa_tinygrad_impl`` but with the inner
# ``[N, M, N]`` buried-check shrunk to ``[N, M, K]`` by selecting the K
# nearest neighbors of each atom. ``Tensor.topk`` and fancy indexing both
# fuse inside ``TinyJit`` so the topk + gather + buried pass compile as one
# program — ~80× scratch reduction at K=64, lossless when K covers the
# worst-case occlusion-neighbor count (K=64 safe for atom14 protein spheres).

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
    jit_fn = _sasa_tinygrad_neighbor_jit_cache.get(key)
    if jit_fn is None:
        scratch_mb = key[0] * key[1] * key[2] * 4 / 1e6
        dense_mb = key[0] * key[1] * key[0] * 4 / 1e6
        LOGGER.info(
            "tinygrad.sasa.neighbor: compiling new TinyJit for N=%d M=%d K=%d "
            "(scratch ~%.1f MB vs dense ~%.1f MB → %.0f× smaller, cache size %d → %d)",
            key[0], key[1], key[2], scratch_mb, dense_mb, dense_mb / max(scratch_mb, 0.01),
            len(_sasa_tinygrad_neighbor_jit_cache), len(_sasa_tinygrad_neighbor_jit_cache) + 1,
        )
        jit_fn = _TGTinyJit(_calculate_sasa_tinygrad_neighbor_impl)
        _sasa_tinygrad_neighbor_jit_cache[key] = jit_fn
    else:
        LOGGER.debug("tinygrad.sasa.neighbor: TinyJit cache hit N=%d M=%d K=%d", *key)
    out = jit_fn(coords, vdw_radii, mask, sphere_points, k_neighbors, probe_radius)
    _log_device_memory("tinygrad.sasa.neighbor")
    return out


# --- Bucketed-padding wrappers -----------------------------------------
#
# The per-shape JIT caches (both tinygrad TinyJit and JAX XLA) recompile on
# every distinct ``N``. With atom14 structures each complex has a unique
# padded count, so every sweep pays ``n_structures`` compile passes. On
# Apple Metal that compile storm is the dominant cost and also the source of
# ``Internal Error (0000000e)`` under sustained pressure.
#
# Bucketed padding rounds ``N`` up to the next multiple of ``bucket_step``
# (e.g. 2048) *before* invoking the kernel. Padded atoms carry
# ``mask=0, coords=0, vdw_radii=0`` — ``_precompute_sasa_inputs`` zeros them
# out, so they contribute ``area=0`` per atom and never register as
# occluders for real atoms (``scaled_norm2 > 0`` for any real atom not at
# the origin, so the inner ``dist² <= 0`` buried check fails). Output for
# real atoms is numerically identical; padded rows are sliced off before
# return.
#
# The JIT cache keys on padded ``N``. Kahraman T3 padded N ∈ [2.5k, 12.8k]
# collapses to ~5 distinct buckets at step=2048, so one compile serves
# several structures.


def bucket_padded_size(n: int, step: int) -> int:
    """Round ``n`` up to the next multiple of ``step``; ``step<=1`` is a no-op.

    Callers pass the original atom count and receive the bucketed count.
    Pair with :func:`calculate_sasa_batch_tinygrad_bucketed` /
    :func:`calculate_sasa_batch_bucketed` to pad before kernel dispatch.
    """
    n = int(n)
    step = int(step)
    if step <= 1 or n <= 0:
        return n
    return ((n + step - 1) // step) * step


def _pad_tinygrad_inputs(
    coords: "_TGTensor",
    vdw_radii: "_TGTensor",
    mask: "_TGTensor",
    pad_n: int,
) -> tuple["_TGTensor", "_TGTensor", "_TGTensor"]:
    """Append ``pad_n`` zero rows to coords ``[N, 3]`` / radii ``[N]`` / mask ``[N]``.

    Padded atoms' ``mask=0`` route through ``_precompute_sasa_inputs`` as
    zero-coords / zero-radii, so they contribute nothing to real atoms.
    """
    coords_pad = _TGTensor(np.zeros((pad_n, 3), dtype=np.float32))
    radii_pad = _TGTensor(np.zeros(pad_n, dtype=np.float32))
    mask_pad = _TGTensor(np.zeros(pad_n, dtype=np.float32))
    return (
        _TGTensor.cat(coords, coords_pad, dim=0),
        _TGTensor.cat(vdw_radii, radii_pad, dim=0),
        _TGTensor.cat(mask, mask_pad, dim=0),
    )


def calculate_sasa_batch_tinygrad_bucketed(
    coords: "_TGTensor",
    vdw_radii: "_TGTensor",
    mask: "_TGTensor",
    sphere_points: "_TGTensor",
    block_size: int,
    probe_radius: float = 1.4,
    bucket_step: int = 2048,
) -> "_TGTensor":
    """Blocked tinygrad SASA with padded-to-bucket ``N`` for JIT cache reuse.

    Pads ``N`` up to the next multiple of ``bucket_step``, dispatches the
    existing blocked kernel (whose TinyJit cache keys on the bucketed shape),
    then slices padded tail rows off the result.
    """
    n_orig = int(coords.shape[0])
    n_padded = bucket_padded_size(n_orig, bucket_step)
    if n_padded == n_orig:
        LOGGER.info(
            "tinygrad.sasa.bucketed: N=%d already multiple of step=%d — bypassing padding",
            n_orig, bucket_step,
        )
        return calculate_sasa_batch_tinygrad(
            coords, vdw_radii, mask, sphere_points, block_size, probe_radius,
        )

    pad = n_padded - n_orig
    LOGGER.info(
        "tinygrad.sasa.bucketed: N=%d → %d (step=%d, +%d padded atoms)",
        n_orig, n_padded, bucket_step, pad,
    )
    coords_p, vdw_p, mask_p = _pad_tinygrad_inputs(coords, vdw_radii, mask, pad)
    out_padded = calculate_sasa_batch_tinygrad(
        coords_p, vdw_p, mask_p, sphere_points, block_size, probe_radius,
    )
    # Slice the real-atom prefix; realize to detach from the padded buffer so
    # the caller sees a clean ``[n_orig]`` tensor.
    return out_padded[:n_orig].contiguous().realize()


def calculate_sasa_batch_bucketed(
    coords: jnp.ndarray,
    vdw_radii: jnp.ndarray,
    mask: jnp.ndarray,
    block_size: int,
    sphere_points: jnp.ndarray,
    probe_radius: float = 1.4,
    bucket_step: int = 2048,
) -> jnp.ndarray:
    """JAX blocked SASA with padded-to-bucket ``N`` — analog of the tinygrad path.

    XLA keys its compile cache on shapes too, so this is the same "one compile
    per bucket" trick but for :func:`.sasa.calculate_sasa_batch`.
    """
    n_orig = int(coords.shape[0])
    n_padded = bucket_padded_size(n_orig, bucket_step)
    if n_padded == n_orig:
        return calculate_sasa_batch(coords, vdw_radii, mask, block_size, sphere_points, probe_radius)

    pad = n_padded - n_orig
    LOGGER.info(
        "jax.sasa.bucketed: N=%d → %d (step=%d, +%d padded atoms)",
        n_orig, n_padded, bucket_step, pad,
    )
    coords_p = jnp.concatenate([coords, jnp.zeros((pad, 3), dtype=coords.dtype)], axis=0)
    vdw_p = jnp.concatenate([vdw_radii, jnp.zeros(pad, dtype=vdw_radii.dtype)], axis=0)
    mask_p = jnp.concatenate([mask, jnp.zeros(pad, dtype=mask.dtype)], axis=0)
    out_padded = calculate_sasa_batch(coords_p, vdw_p, mask_p, block_size, sphere_points, probe_radius)
    return out_padded[:n_orig]


__all__ = [
    "bucket_padded_size",
    "calculate_sasa_batch_bucketed",
    "calculate_sasa_batch_soft",
    "calculate_sasa_batch_scan_soft",
    "calculate_sasa_batch_tinygrad_bucketed",
    "calculate_sasa_jax_soft",
    "calculate_sasa_jax_neighbor",
    "calculate_sasa_tinygrad",
    "calculate_sasa_tinygrad_neighbor",
]
