#!/usr/bin/env python3
"""Profile the JAX batched SASA kernel vs `estimate_optimal_block_size`.

Sweeps a range of block sizes on a single complex, reports per-call compile
cost, steady-state timing, scratch memory, and numerical drift, then flags
whether the estimator's pick is near the observed optimum.

    .venv/bin/python benchmarks/profile_sasa.py [--struct PATH] [--selection A,B]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np

from protein_affinity_gpu.jax import (
    estimate_max_atoms,
    estimate_optimal_block_size,
    predict_binding_affinity_jax,
)
from protein_affinity_gpu.sasa import calculate_sasa_batch, generate_sphere_points
from protein_affinity_gpu.scoring import get_atom_radii
from protein_affinity_gpu.structure import load_complex
from protein_affinity_gpu.utils import residue_constants
from protein_affinity_gpu.utils.residue_library import default_library as residue_library

_RADII = jnp.array(residue_library.radii_matrix)


def build_inputs(struct: Path, selection: str, sphere_points: int):
    t, b = load_complex(struct, selection=selection, sanitize=True)
    pos = jnp.concatenate([t.atom_positions, b.atom_positions], axis=0).reshape(-1, 3)
    mask = jnp.concatenate([t.atom_mask, b.atom_mask], axis=0).reshape(-1)
    nc = len(residue_constants.restypes)
    seq_t = jax.nn.one_hot(t.aatype, nc)
    seq_b = jax.nn.one_hot(b.aatype, nc)
    radii = jnp.concatenate(
        [get_atom_radii(seq_t, _RADII, t.atom_mask),
         get_atom_radii(seq_b, _RADII, b.atom_mask)]
    )
    return pos, mask, radii, generate_sphere_points(sphere_points)


def bench(pos, mask, radii, sp, block_size: int, runs: int):
    """Return (first_call_s, median_steady_s, output) — first call includes compile."""
    t0 = time.perf_counter()
    out = calculate_sasa_batch(pos, radii, mask, block_size, sp).block_until_ready()
    first = time.perf_counter() - t0
    steady = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = calculate_sasa_batch(pos, radii, mask, block_size, sp).block_until_ready()
        steady.append(time.perf_counter() - t0)
    return first, float(np.median(steady)), out


def scratch_mb(block_size: int, n_atoms: int, m_points: int) -> float:
    """Peak per-block scratch: [B, M, N] float32."""
    return 4 * block_size * m_points * n_atoms / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", type=Path, default=ROOT / "benchmarks/fixtures/1A2K.pdb")
    ap.add_argument("--selection", default="A,B")
    ap.add_argument("--sphere-points", type=int, default=100)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--skip-full", action="store_true", help="skip full predict timing")
    args = ap.parse_args()

    pos, mask, radii, sp = build_inputs(args.struct, args.selection, args.sphere_points)
    n_atoms = int(pos.shape[0])
    backend = jax.default_backend().upper()
    est_block = estimate_optimal_block_size(n_atoms, backend, args.sphere_points)
    max_atoms = estimate_max_atoms(backend, sphere_points=args.sphere_points)

    print(f"struct={args.struct.name}  backend={backend}  atoms={n_atoms}  M={args.sphere_points}", flush=True)
    print(f"estimate_max_atoms({backend}) = {max_atoms:,}", flush=True)
    print(f"estimate_optimal_block_size({n_atoms}) = {est_block}  (scratch {scratch_mb(est_block, n_atoms, args.sphere_points):.0f} MB)", flush=True)
    print(f"interaction_matrix = {n_atoms * n_atoms / 1e6:.1f} MB (bool)", flush=True)
    print(flush=True)

    # Keep the sweep small — each new block size triggers a fresh JAX compile.
    candidates = sorted({est_block, 25, 512, n_atoms})
    candidates = [b for b in candidates if 1 <= b <= n_atoms]

    results: dict[int, tuple[float, float, np.ndarray]] = {}
    ref = None
    print(f"{'block':>6} | {'scratch':>9} | {'compile':>8} | {'median':>7} | {'max|Δ|':>8} | note", flush=True)
    print("-" * 65, flush=True)
    for b in candidates:
        first, median, out = bench(pos, mask, radii, sp, b, args.runs)
        arr = np.asarray(out)
        diff = 0.0 if ref is None else float(np.max(np.abs(arr - ref)))
        if ref is None:
            ref = arr
        results[b] = (first, median, arr)
        note = "est" if b == est_block else ("B=N" if b == n_atoms else "")
        print(f"{b:>6} | {scratch_mb(b, n_atoms, args.sphere_points):>7.0f} MB | "
              f"{first*1000:>6.0f} ms | {median*1000:>5.0f} ms | {diff:>8.1e} | {note}",
              flush=True)

    best_block = min(results, key=lambda k: results[k][1])
    best_time = results[best_block][1]
    est_time = results[est_block][1]
    print()
    print(f"observed optimum : block={best_block}  median={best_time*1000:.0f} ms")
    print(f"estimator pick   : block={est_block}  median={est_time*1000:.0f} ms  "
          f"({est_time / best_time:.2f}× optimum)")

    if not args.skip_full:
        print()
        print("full predict_binding_affinity_jax (single call, includes load + compile):")
        t0 = time.perf_counter()
        res = predict_binding_affinity_jax(args.struct, selection=args.selection)
        full = time.perf_counter() - t0
        print(f"  total={full:.2f}s  ΔG={float(res.binding_affinity):.2f}  Kd={float(res.dissociation_constant):.3e}")


if __name__ == "__main__":
    main()
