"""Tests for the bucketed-padding SASA wrappers in ``sasa_experimental``.

Validates numerical equivalence vs the unbucketed kernels, and confirms that
the underlying TinyJit cache is keyed on the bucketed ``N`` (so structures
with different original ``N`` but the same bucket reuse one compile).
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest


def test_bucket_padded_size_rounds_up_to_step():
    from protein_affinity_gpu.sasa_experimental import bucket_padded_size

    assert bucket_padded_size(0, 2048) == 0
    assert bucket_padded_size(1, 2048) == 2048
    assert bucket_padded_size(2048, 2048) == 2048
    assert bucket_padded_size(2049, 2048) == 4096
    assert bucket_padded_size(3661, 2048) == 4096
    assert bucket_padded_size(12810, 2048) == 14336
    # step <= 1 is a no-op (disables bucketing).
    assert bucket_padded_size(1234, 1) == 1234
    assert bucket_padded_size(1234, 0) == 1234


def test_tinygrad_bucketed_matches_block_on_1a2k():
    """Bucketed output must agree with the unbucketed blocked kernel per-atom."""
    from tinygrad import Tensor

    from protein_affinity_gpu.sasa import calculate_sasa_batch_tinygrad, generate_sphere_points
    from protein_affinity_gpu.sasa_experimental import calculate_sasa_batch_tinygrad_bucketed
    from protein_affinity_gpu.scoring import get_atom_radii
    from protein_affinity_gpu.utils import residue_constants
    from protein_affinity_gpu.utils.atom14 import compact_complex_atom14
    from protein_affinity_gpu.utils.residue_library import default_library as residue_library
    from protein_affinity_gpu.utils.structure import load_complex

    fixture = Path("benchmarks/fixtures/1A2K.pdb")
    target, binder = load_complex(fixture, selection="A,B", sanitize=True)
    positions_np, mask_np, _, (t_aatype, b_aatype) = compact_complex_atom14(target, binder)

    coords = Tensor(np.ascontiguousarray(positions_np))
    mask = Tensor(np.ascontiguousarray(mask_np).astype(np.float32))
    num_classes = len(residue_constants.restypes)
    radii_matrix = Tensor(np.asarray(residue_library.radii_matrix_atom14, dtype=np.float32))
    t_seq = Tensor(np.asarray(t_aatype, dtype=np.int64)).one_hot(num_classes).float()
    b_seq = Tensor(np.asarray(b_aatype, dtype=np.int64)).one_hot(num_classes).float()
    vdw = Tensor.cat(
        get_atom_radii(t_seq, radii_matrix),
        get_atom_radii(b_seq, radii_matrix),
        dim=0,
    )
    sphere = Tensor(generate_sphere_points(100))

    n_atoms = int(coords.shape[0])
    block = min(768, n_atoms)

    ref = calculate_sasa_batch_tinygrad(coords, vdw, mask, sphere, block).numpy()
    bucketed = calculate_sasa_batch_tinygrad_bucketed(
        coords, vdw, mask, sphere, block, bucket_step=1024,
    ).numpy()

    assert bucketed.shape == ref.shape
    # Real atoms must match the unpadded kernel to within float-roundoff of the
    # bigger reduction dimension — tolerances here track `test_tinygrad_smoke`.
    max_abs = float(np.max(np.abs(bucketed - ref)))
    assert max_abs < 1e-2, f"bucketed vs block max|Δ| = {max_abs:.3e}"


def test_tinygrad_bucketed_jit_cache_keys_on_bucketed_shape():
    """Two structures in the same bucket must share one compiled TinyJit."""
    from tinygrad import Tensor

    from protein_affinity_gpu.sasa import (
        _sasa_block_jit_cache,
        calculate_sasa_batch_tinygrad,
        generate_sphere_points,
    )
    from protein_affinity_gpu.sasa_experimental import calculate_sasa_batch_tinygrad_bucketed

    sphere = Tensor(generate_sphere_points(100))
    block = 64
    bucket_step = 256

    def _fake_inputs(n: int):
        rng = np.random.default_rng(seed=n)
        coords = Tensor(rng.standard_normal((n, 3), dtype=np.float32) * 10.0)
        vdw = Tensor(np.full(n, 1.7, dtype=np.float32))
        mask = Tensor(np.ones(n, dtype=np.float32))
        return coords, vdw, mask

    # Clear cache first so we start from a known state.
    _sasa_block_jit_cache.clear()
    # Two different original N's that share the 256-step bucket (both → 512).
    for n in (300, 400):
        coords, vdw, mask = _fake_inputs(n)
        out = calculate_sasa_batch_tinygrad_bucketed(
            coords, vdw, mask, sphere, block, bucket_step=bucket_step,
        )
        assert int(out.shape[0]) == n
        assert math.isfinite(float(out.numpy().sum()))

    # Exactly one cache entry — both calls hit the same (block, bucketed_n, M).
    bucketed_keys = [
        key for key in _sasa_block_jit_cache
        if key[0] == block and key[2] == int(sphere.shape[0])
    ]
    assert len(bucketed_keys) == 1, bucketed_keys
    assert bucketed_keys[0][1] == 512  # padded to bucket_step=256 → 512

    # Sanity: a third N that lands in a different bucket adds a new entry.
    coords, vdw, mask = _fake_inputs(600)
    calculate_sasa_batch_tinygrad_bucketed(
        coords, vdw, mask, sphere, block, bucket_step=bucket_step,
    )
    bucketed_keys = sorted(
        key[1] for key in _sasa_block_jit_cache
        if key[0] == block and key[2] == int(sphere.shape[0])
    )
    assert bucketed_keys == [512, 768], bucketed_keys


def test_predict_binding_affinity_tinygrad_bucketed_mode():
    """End-to-end: bucketed mode yields a finite ΔG close to the block mode."""
    from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad

    fixture = Path("benchmarks/fixtures/1A2K.pdb")
    ref = predict_binding_affinity_tinygrad(fixture, selection="A,B", mode="block")
    bucketed = predict_binding_affinity_tinygrad(
        fixture, selection="A,B", mode="bucketed", bucket_step=1024,
    )

    dg_ref = float(ref.binding_affinity)
    dg_bucketed = float(bucketed.binding_affinity)
    assert math.isfinite(dg_bucketed)
    assert abs(dg_bucketed - dg_ref) < 0.25, (dg_ref, dg_bucketed)
