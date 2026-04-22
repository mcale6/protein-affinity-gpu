# protein-affinity-gpu — Experimental Surface

> Companion to [INDEX.md](INDEX.md). This document covers the experimental
> entry points and kernels that still live behind
> `protein_affinity_gpu.experimental` and `protein_affinity_gpu.sasa_experimental`.
> Stable differentiable AFDesign helpers are documented separately in
> [AF_DESIGN.md](AF_DESIGN.md).

The default predictor surface exposes only two SASA kernels
(`calculate_sasa_batch`, `calculate_sasa_batch_scan`) wired through
`predict_binding_affinity_jax` with `mode ∈ {"block", "scan"}`. The stable
differentiable soft-SASA kernels now live in `protein_affinity_gpu.sasa_soft`;
this document covers the experimental entry points plus the neighbor-cutoff
and tinygrad kernels, and `sasa_experimental.py` remains a compatibility
re-export layer for the soft JAX kernels.

---

## 1. Entry points — `protein_affinity_gpu.experimental`

```python
from protein_affinity_gpu.experimental import (
    predict_binding_affinity_jax_experimental,
    predict_binding_affinity_tinygrad,
)
```

### 1.1 JAX (experimental)

```python
predict_binding_affinity_jax_experimental(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    soft_sasa=False, soft_beta=10.0,
    mode="block",            # "block" | "single" | "scan" | "neighbor"
    k_neighbors=64,          # only used when mode="neighbor"
) -> ProdigyResults
```

- Constructs `JAXExperimentalAdapter(mode=…, soft_sasa=…, soft_beta=…,
  k_neighbors=…)` and delegates to the shared `_run_pipeline` body.
- `"single"` — fully-fused `@jit` SASA, `[N, M, N]` scratch; fastest path
  when it fits.
- `"neighbor"` — `jax.lax.top_k` on `-dist²` keeps the K nearest atoms per
  row, buried-check scratch `[N, M, K]` instead of `[N, M, N]` (~80×
  smaller at K=64). Inference-only (`top_k` isn't usefully
  differentiable).
- `soft_sasa=True` swaps in the sigmoid-smoothed SASA kernel (applies to
  `"block"`, `"scan"`, `"single"` modes only). Approaches the hard kernel
  as β→∞; provides non-zero gradients through the atom-buried check.

### 1.2 Tinygrad

```python
predict_binding_affinity_tinygrad(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    mode="block",            # "block" | "single" | "neighbor"
    k_neighbors=64,          # only used when mode="neighbor"
) -> ProdigyResults
```

Device selection comes from `Device.DEFAULT`, overridable via
`TINYGRAD_DEVICE=CPU|METAL|CUDA`. Tinygrad kernels are recompiled on the
first call per shape and cached thereafter.

`protein-affinity-predict --backend tinygrad` lazy-loads the same
implementation through `cli/predict.py`, so the CLI continues to work
without a package-root re-export.

---

## 2. Adapters — `backends/`

| Adapter | Module | Notable behavior |
|---------|--------|------------------|
| `JAXExperimentalAdapter` | [`backends/_jax_experimental.py`](../src/protein_affinity_gpu/backends/_jax_experimental.py) | Subclasses `JAXAdapter`. Extends `mode` with `"single"` and `"neighbor"`; accepts `soft_sasa` / `soft_beta` (routes through `calculate_sasa_batch_soft` / `calculate_sasa_batch_scan_soft` / `calculate_sasa_jax_soft`); accepts `k_neighbors` for the neighbor mode. |
| `TinygradAdapter` | [`backends/_tinygrad.py`](../src/protein_affinity_gpu/backends/_tinygrad.py) | `mode={"block","single","neighbor"}` (default `"block"`). `METAL` / `CUDA` / `GPU` → batched SASA with `block=min(768, N)`; any other device (incl. `CPU` / `CLANG`) → full `calculate_sasa_tinygrad` kernel (`block_size=None`). `"neighbor"` keeps only the K nearest atoms per row (`k_neighbors=64` default) to shrink scratch from `[N, M, N]` to `[N, M, K]`. |

The backends router accepts `"jax-experimental"` alongside `"jax"` and
`"tinygrad"`:

```python
from protein_affinity_gpu.backends import get_adapter

adapter = get_adapter("jax-experimental", mode="single", soft_sasa=True, soft_beta=20.0)
```

---

## 3. SASA kernels — `sasa_experimental.py`

The neighbor-cutoff JAX kernels, tinygrad kernels, and compatibility re-exports
live in [`src/protein_affinity_gpu/sasa_experimental.py`](../src/protein_affinity_gpu/sasa_experimental.py).
The stable differentiable soft JAX kernels themselves live in
[`src/protein_affinity_gpu/sasa_soft.py`](../src/protein_affinity_gpu/sasa_soft.py).

> The non-differentiable single-pass JAX kernel (`calculate_sasa_jax`) and
> the blocked tinygrad kernel (`calculate_sasa_batch_tinygrad`) live in the
> default [`sasa.py`](../src/protein_affinity_gpu/sasa.py). The stable soft
> JAX kernels now live in [`sasa_soft.py`](../src/protein_affinity_gpu/sasa_soft.py)
> and are re-exported from
> [`sasa_experimental.py`](../src/protein_affinity_gpu/sasa_experimental.py)
> for compatibility.

### 3.1 JAX — differentiable / neighbor

| Function | Notes |
|----------|-------|
| `calculate_sasa_batch_soft(..., beta=10.0)` | Differentiable sigmoid-smoothed variant of `calculate_sasa_batch`; shares the `_dispatch_blocked_jax` loop. Uses `log(1 − σ) = −softplus(x)` for numerical stability. Approaches the hard kernel as β→∞. |
| `calculate_sasa_batch_scan_soft(...)` | Same kernel as above, dispatched via `jax.lax.scan` (AlphaFold `layer_stack` pattern; wrap the body in `jax.checkpoint` for memory-efficient backprop). |
| `calculate_sasa_jax_soft(...)` | Differentiable single-pass JAX SASA (sigmoid kernel); β→∞ recovers the hard `calculate_sasa_jax` in [`sasa.py`](../src/protein_affinity_gpu/sasa.py). |
| `calculate_sasa_jax_neighbor(..., k_neighbors=64)` | Single-pass JAX port of `calculate_sasa_tinygrad_neighbor` — `jax.lax.top_k` on `-dist²` keeps the K nearest atoms per row. `k_neighbors` is a `static_argnames` so XLA const-folds the K dimension. Inference-only. |

### 3.2 Tinygrad

| Function | Notes |
|----------|-------|
| `calculate_sasa_tinygrad` | `TinyJit`-wrapped full kernel for CPU / CLANG devices (dot-product identity for both `[N, N]` and `[N, M, N]` passes). |
| `calculate_sasa_tinygrad_neighbor(..., k_neighbors=64)` | Single TinyJit kernel that uses `Tensor.topk` to keep only the K nearest atoms per row, shrinking the buried-check scratch from `[N, M, N]` to `[N, M, K]` (~80× at K=64). Lossless when K covers the worst-case occlusion-neighbor count; trades scratch for `topk` compute, so on Apple Metal it is **slower** than the blocked kernel — its win is memory headroom on size-constrained devices. |

See also: `calculate_sasa_batch_tinygrad` lives in
[`sasa.py`](../src/protein_affinity_gpu/sasa.py) — it's the default path on
accelerator tinygrad devices. The
[TINYGRAD_SASA_OPTIMIZATION.md](TINYGRAD_SASA_OPTIMIZATION.md) write-up
walks through the three changes that took it from 126s to 2.4s on 1A2K
Apple Metal.

Module-level JIT caches: `_sasa_tinygrad_jit_cache`,
`_sasa_tinygrad_neighbor_jit_cache` in this module, `_sasa_block_jit_cache`
in [`sasa.py`](../src/protein_affinity_gpu/sasa.py) — all keyed by
tensor-shape so multi-structure runs reuse the compiled kernel.

### 3.3 Device-memory logging

Every wrapper calls `_log_device_memory(tag)` after realization (JAX paths
also `block_until_ready()` first so the reading reflects the actual
compute). Log tags emitted from the experimental surface:

- `jax.sasa.block_soft`, `jax.sasa.scan_soft`, `jax.sasa.single_soft`,
  `jax.sasa.neighbor`
- `tinygrad.sasa.single`, `tinygrad.sasa.neighbor`

Hard-path tags (`jax.sasa.block`, `jax.sasa.scan`, `jax.sasa.single`,
`tinygrad.sasa.block`) come from [`sasa.py`](../src/protein_affinity_gpu/sasa.py).

Enable via `setup_logging("INFO")` or `--verbose` in the CLIs.

### 3.4 Sanity-check helper

```bash
.venv/bin/python benchmarks/check_soft_sasa.py
```

Verifies that the soft kernel → hard kernel as β grows and that the
gradient w.r.t. atomic radii is non-zero on a tiny tetrahedron.

---

## 4. Experimental benchmark harness

[`benchmarks/benchmark_experimental.py`](../benchmarks/benchmark_experimental.py)
is the original full-sweep harness. It understands every target below:

| Target | Loader | Backend kwarg(s) |
|--------|--------|------------------|
| `cpu` | `predict_binding_affinity` | — |
| `cuda` | GPU variant (auto-skipped without a CUDA device) | — |
| `jax` | `predict_binding_affinity_jax(mode="block")` | — |
| `jax-scan` | `predict_binding_affinity_jax(mode="scan")` | — |
| `jax-single` | `predict_binding_affinity_jax_experimental(mode="single")` | — |
| `jax-neighbor` | `predict_binding_affinity_jax_experimental(mode="neighbor")` | `k_neighbors` |
| `jax-soft` | `predict_binding_affinity_jax_experimental(soft_sasa=True)` | `soft_beta` |
| `tinygrad` | `predict_binding_affinity_tinygrad` | — |
| `tinygrad-neighbor` | `predict_binding_affinity_tinygrad(mode="neighbor")` | `k_neighbors` |

Usage mirrors the default harness:

```bash
.venv/bin/python benchmarks/benchmark_experimental.py benchmarks/fixtures \
    --output-dir benchmarks/output \
    --targets cpu jax jax-scan jax-single jax-neighbor tinygrad
```

Writes `benchmark_results.json` with per-target cold + warm timings, plus
the same per-run memory snapshot (`rss_peak_mb`, `jax_peak_mb`,
`jax_in_use_mb`, `jax_peak_delta_mb`) that the default harness captures.

Use the default harness ([`benchmarks/benchmark.py`](../benchmarks/benchmark.py))
when you only need the supported CPU / JAX (`block`, `scan`) targets —
that file's `--targets` is deliberately restricted and will refuse
experimental target names with a clear error pointing at this harness.

---

## 5. Related documents

- [INDEX.md](INDEX.md) — default surface and project overview.
- [TINYGRAD_SASA_OPTIMIZATION.md](TINYGRAD_SASA_OPTIMIZATION.md) — the
  three-step optimization story behind the current batched tinygrad
  kernel (126s → 2.4s on 1A2K Apple Metal).
