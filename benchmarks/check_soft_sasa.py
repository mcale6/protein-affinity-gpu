#!/usr/bin/env python3
"""Sanity check: soft-SASA → hard-SASA as β grows, and has non-zero gradients.

    .venv/bin/python benchmarks/check_soft_sasa.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from protein_affinity_gpu.sasa import (
    calculate_sasa_batch,
    calculate_sasa_batch_soft,
    generate_sphere_points,
)


def main() -> None:
    # Tiny synthetic system: 4 atoms in a tetrahedron.
    rng = np.random.default_rng(0)
    coords = jnp.asarray(rng.normal(size=(4, 3)) * 2.0, dtype=jnp.float32)
    radii = jnp.asarray([1.7, 1.7, 1.55, 1.52], dtype=jnp.float32)
    mask = jnp.ones(4, dtype=jnp.float32)
    sp = generate_sphere_points(100)

    hard = calculate_sasa_batch(coords, radii, mask, 4, sp)
    print(f"hard-SASA: {np.asarray(hard)}")
    for beta in (1.0, 10.0, 100.0, 1000.0):
        soft = calculate_sasa_batch_soft(coords, radii, mask, 4, sp, beta=beta)
        diff = float(jnp.max(jnp.abs(soft - hard)))
        print(f"  soft β={beta:>6}: max|soft − hard| = {diff:.3f}")

    # Gradient w.r.t. radii at β=50 — should be non-zero for non-fully-buried atoms.
    def loss(r):
        return calculate_sasa_batch_soft(coords, r, mask, 4, sp, beta=50.0).sum()

    g = jax.grad(loss)(radii)
    print(f"dSASA/dradii (β=50): {np.asarray(g)}")
    assert float(jnp.max(jnp.abs(g))) > 1e-3, "soft-SASA gradient is suspiciously zero"
    print("OK — soft-SASA is differentiable.")


if __name__ == "__main__":
    main()
