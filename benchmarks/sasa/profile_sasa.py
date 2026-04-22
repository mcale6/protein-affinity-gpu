#!/usr/bin/env python3
"""Profile JAX block vs tinygrad block vs tinygrad bucketed on one complex.

For a single structure:
- Sweeps a range of block sizes on the JAX blocked kernel and reports per-call
  compile cost, steady-state timing, scratch memory, and numerical drift.
- Times the tinygrad blocked kernel at its default ``block = min(768, N)``.
- Times the tinygrad bucketed kernel at ``bucket_step={1024, 2048}`` so the
  compile-amortisation benefit shows up on repeat calls with different N.

    .venv/bin/python benchmarks/sasa/profile_sasa.py [--struct PATH] [--selection A,B]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import jax
import jax.numpy as jnp
import numpy as np
from tinygrad import Tensor

from protein_affinity_gpu.backends._jax import JAXAdapter
from protein_affinity_gpu.predict import predict_binding_affinity_jax
from protein_affinity_gpu.sasa import (
    _sasa_block_jit_cache,
    calculate_sasa_batch,
    calculate_sasa_batch_tinygrad,
    generate_sphere_points,
)
from protein_affinity_gpu.sasa_experimental import (
    bucket_padded_size,
    calculate_sasa_batch_tinygrad_bucketed,
)
from protein_affinity_gpu.scoring import get_atom_radii
from protein_affinity_gpu.utils import residue_constants
from protein_affinity_gpu.utils.residue_library import default_library as residue_library
from protein_affinity_gpu.utils.structure import load_complex

_JAX_ADAPTER = JAXAdapter()

_RADII_JAX = jnp.array(residue_library.radii_matrix_atom14)
_RADII_TG = Tensor(np.asarray(residue_library.radii_matrix_atom14, dtype=np.float32))


def _build_inputs(struct: Path, selection: str, sphere_points: int):
    """Shared atom14 inputs for both backends. Returns JAX + tinygrad copies."""
    from protein_affinity_gpu.utils.atom14 import compact_complex_atom14

    target, binder = load_complex(struct, selection=selection, sanitize=True)
    positions_np, mask_np, _, (t_aatype, b_aatype) = compact_complex_atom14(target, binder)

    pos_jax = jnp.asarray(positions_np)
    mask_jax = jnp.asarray(mask_np)
    nc = len(residue_constants.restypes)
    t_seq_jax = jax.nn.one_hot(t_aatype, nc)
    b_seq_jax = jax.nn.one_hot(b_aatype, nc)
    radii_jax = jnp.concatenate(
        [get_atom_radii(t_seq_jax, _RADII_JAX),
         get_atom_radii(b_seq_jax, _RADII_JAX)]
    )
    sp_jax = jnp.asarray(generate_sphere_points(sphere_points))

    pos_tg = Tensor(np.ascontiguousarray(positions_np))
    mask_tg = Tensor(np.ascontiguousarray(mask_np).astype(np.float32))
    t_seq_tg = Tensor(np.asarray(t_aatype, dtype=np.int64)).one_hot(nc).float()
    b_seq_tg = Tensor(np.asarray(b_aatype, dtype=np.int64)).one_hot(nc).float()
    radii_tg = Tensor.cat(
        get_atom_radii(t_seq_tg, _RADII_TG),
        get_atom_radii(b_seq_tg, _RADII_TG),
        dim=0,
    )
    sp_tg = Tensor(generate_sphere_points(sphere_points))

    return {
        "jax": (pos_jax, mask_jax, radii_jax, sp_jax),
        "tg": (pos_tg, mask_tg, radii_tg, sp_tg),
        "n_atoms": int(positions_np.shape[0]),
    }


def _bench_jax(pos, mask, radii, sp, block_size: int, runs: int):
    """Return (cold_s, median_steady_s, output) — cold includes JAX compile."""
    t0 = time.perf_counter()
    out = calculate_sasa_batch(pos, radii, mask, block_size, sp).block_until_ready()
    cold = time.perf_counter() - t0
    steady = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = calculate_sasa_batch(pos, radii, mask, block_size, sp).block_until_ready()
        steady.append(time.perf_counter() - t0)
    return cold, float(np.median(steady)), out


def _bench_tg_block(pos, mask, radii, sp, block_size: int, runs: int):
    """Tinygrad blocked — cold includes TinyJit compile for each new shape."""
    t0 = time.perf_counter()
    out = calculate_sasa_batch_tinygrad(pos, radii, mask, sp, block_size)
    _ = out.numpy()  # force realize
    cold = time.perf_counter() - t0
    steady = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = calculate_sasa_batch_tinygrad(pos, radii, mask, sp, block_size)
        _ = out.numpy()
        steady.append(time.perf_counter() - t0)
    return cold, float(np.median(steady)), out


def _bench_tg_bucketed(pos, mask, radii, sp, block_size: int, bucket_step: int, runs: int):
    t0 = time.perf_counter()
    out = calculate_sasa_batch_tinygrad_bucketed(
        pos, radii, mask, sp, block_size, bucket_step=bucket_step,
    )
    _ = out.numpy()
    cold = time.perf_counter() - t0
    steady = []
    for _ in range(runs):
        t0 = time.perf_counter()
        out = calculate_sasa_batch_tinygrad_bucketed(
            pos, radii, mask, sp, block_size, bucket_step=bucket_step,
        )
        _ = out.numpy()
        steady.append(time.perf_counter() - t0)
    return cold, float(np.median(steady)), out


def _scratch_mb(block_size: int, n_atoms: int, m_points: int) -> float:
    """Peak per-block scratch for the [B, M, N] buried check (float32)."""
    return 4 * block_size * m_points * n_atoms / 1e6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--struct", type=Path, default=ROOT / "benchmarks/fixtures/1A2K.pdb")
    ap.add_argument("--selection", default="A,B")
    ap.add_argument("--sphere-points", type=int, default=100)
    ap.add_argument("--runs", type=int, default=2)
    ap.add_argument("--bucket-steps", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--skip-jax", action="store_true")
    ap.add_argument("--skip-tinygrad", action="store_true")
    ap.add_argument("--skip-full", action="store_true", help="skip full predict timing")
    args = ap.parse_args()

    inputs = _build_inputs(args.struct, args.selection, args.sphere_points)
    n_atoms = inputs["n_atoms"]
    backend = jax.default_backend().upper()
    est_block = _JAX_ADAPTER.estimate_block_size(n_atoms, args.sphere_points)
    max_atoms = _JAX_ADAPTER._estimate_max_atoms(sphere_points=args.sphere_points)

    print(f"struct={args.struct.name}  backend={backend}  atoms={n_atoms}  M={args.sphere_points}",
          flush=True)
    print(f"max_atoms({backend}) = {max_atoms:,}", flush=True)
    print(f"estimate_block_size({n_atoms}) = {est_block}  "
          f"(scratch {_scratch_mb(est_block, n_atoms, args.sphere_points):.0f} MB)", flush=True)
    print(f"interaction_matrix = {n_atoms * n_atoms / 1e6:.1f} MB (bool)", flush=True)
    print(flush=True)

    jax_results: dict[int, tuple[float, float, np.ndarray]] = {}

    if not args.skip_jax:
        print("== JAX blocked kernel sweep ==")
        candidates = sorted({est_block, 25, 512, n_atoms})
        candidates = [b for b in candidates if 1 <= b <= n_atoms]

        ref = None
        print(f"{'block':>6} | {'scratch':>9} | {'compile':>8} | {'median':>7} | {'max|Δ|':>8} | note",
              flush=True)
        print("-" * 65, flush=True)
        pos_jax, mask_jax, radii_jax, sp_jax = inputs["jax"]
        for b in candidates:
            cold, median, out = _bench_jax(pos_jax, mask_jax, radii_jax, sp_jax, b, args.runs)
            arr = np.asarray(out)
            diff = 0.0 if ref is None else float(np.max(np.abs(arr - ref)))
            if ref is None:
                ref = arr
            jax_results[b] = (cold, median, arr)
            note = "est" if b == est_block else ("B=N" if b == n_atoms else "")
            print(f"{b:>6} | {_scratch_mb(b, n_atoms, args.sphere_points):>7.0f} MB | "
                  f"{cold * 1000:>6.0f} ms | {median * 1000:>5.0f} ms | {diff:>8.1e} | {note}",
                  flush=True)

        best_block = min(jax_results, key=lambda k: jax_results[k][1])
        best_time = jax_results[best_block][1]
        est_time = jax_results[est_block][1]
        print()
        print(f"observed optimum : block={best_block}  median={best_time * 1000:.0f} ms")
        print(f"estimator pick   : block={est_block}  median={est_time * 1000:.0f} ms  "
              f"({est_time / best_time:.2f}× optimum)")

    if not args.skip_tinygrad:
        print()
        print("== Tinygrad blocked kernel (ref) vs bucketed ==")
        pos_tg, mask_tg, radii_tg, sp_tg = inputs["tg"]
        tg_block = min(768, n_atoms)

        # Clear the cache so the first call measures a true cold compile.
        _sasa_block_jit_cache.clear()
        cold_b, med_b, ref_tg = _bench_tg_block(
            pos_tg, mask_tg, radii_tg, sp_tg, tg_block, args.runs,
        )
        ref_arr = ref_tg.numpy()
        print(f"{'mode':>12} | {'N_padded':>9} | {'compile':>8} | {'median':>7} | {'max|Δ|':>8}",
              flush=True)
        print("-" * 62, flush=True)
        print(f"{'block':>12} | {n_atoms:>9} | {cold_b * 1000:>6.0f} ms | "
              f"{med_b * 1000:>5.0f} ms | {'—':>8}", flush=True)

        for step in args.bucket_steps:
            padded = bucket_padded_size(n_atoms, step)
            _sasa_block_jit_cache.clear()
            cold_p, med_p, out_p = _bench_tg_bucketed(
                pos_tg, mask_tg, radii_tg, sp_tg, tg_block, step, args.runs,
            )
            diff = float(np.max(np.abs(out_p.numpy() - ref_arr)))
            print(f"{'bucketed/' + str(step):>12} | {padded:>9} | {cold_p * 1000:>6.0f} ms | "
                  f"{med_p * 1000:>5.0f} ms | {diff:>8.1e}", flush=True)

    if not args.skip_full:
        print()
        print("full predict_binding_affinity_jax (single call, includes load + compile):")
        t0 = time.perf_counter()
        res = predict_binding_affinity_jax(args.struct, selection=args.selection)
        full = time.perf_counter() - t0
        print(f"  total={full:.2f}s  ΔG={float(res.binding_affinity):.2f}  "
              f"Kd={float(res.dissociation_constant):.3e}")


if __name__ == "__main__":
    main()
