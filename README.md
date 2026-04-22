# protein-affinity-gpu

`protein-affinity-gpu` is a research-friendly Python package for protein-protein binding affinity prediction, solvent-accessible surface area (SASA) analysis, and reproducible CPU/JAX benchmarking.

Three first-class backends — CPU (freesasa via PRODIGY), JAX (blocked
and `lax.scan`-fused Shrake–Rupley), and tinygrad (per-shape `TinyJit`
block kernel on METAL / CUDA / GPU; full fused kernel on CPU / CLANG).
Soft/differentiable SASA and the extra JAX modes (single-pass,
neighbor-cutoff) are documented separately in
[docs/EXPERIMENTAL.md](docs/EXPERIMENTAL.md).

## Installation

End-user install from PyPI — the core deps already cover CPU
(`prodigy-prot`, `freesasa`), JAX (`jax`, `jaxlib`), tinygrad, and the
plot stack (`matplotlib`, `pandas`):

```bash
pip install "protein-affinity-gpu==1.6.9"
```

Contributor / local dev — clone the repo and sync with
[uv](https://docs.astral.sh/uv/) for a reproducible environment pinned
against `uv.lock`:

```bash
uv sync                # core deps into .venv/, honouring uv.lock
uv sync --extra dev    # adds pytest, ruff, build (for tests + release)
uv sync --extra modal  # adds modal for the GPU benchmark entrypoint
```

The three dependency files have distinct jobs: `pyproject.toml` declares
unpinned ranges and is what PyPI publishes, `uv.lock` is the exact
pinned resolution, and `.venv/` is a local (gitignored) virtualenv `uv`
materialises from the lock.

## CLI

The package installs one console script — `protein-affinity-predict`.
It runs one structure or a whole folder through any backend:

```bash
protein-affinity-predict benchmarks/fixtures --backend cpu --output-json
protein-affinity-predict benchmarks/fixtures --backend jax --output-json
protein-affinity-predict benchmarks/fixtures --backend tinygrad --output-json
```

| Flag | Default | Description |
|------|---------|-------------|
| `input_path` | — | File or directory of `.pdb` / `.ent` / `.cif` / `.mmcif`. |
| `--backend {cpu,jax,tinygrad}` | `cpu` | Prediction backend. |
| `--selection` | `A,B` | Comma-separated two-chain selection. |
| `--temperature` | `25.0` | Temperature in °C (affects Kd). |
| `--distance-cutoff` | `5.5` | Å cutoff for interface contacts. |
| `--acc-threshold` | `0.05` | Relative SASA threshold for NIS. |
| `--sphere-points` | `100` | Shrake–Rupley sphere resolution. |
| `--output-json` | off | Also write `<stem>_results.json` per structure. |
| `--output-dir` | `results/` | Destination when `--output-json` is set. |
| `--verbose` | off | `DEBUG`-level logging with per-phase timings (stderr, colored on TTY). |

Each run prints the same summary `str(result)` produces — ΔG, Kd,
contact breakdown, NIS breakdown. `--output-json` persists the full
per-atom result; `--verbose` streams phase timings to stderr.

Benchmarking is deliberately kept out of the installed CLI — the harness
scripts live in [`benchmarks/`](benchmarks) and are invoked directly:

```bash
# Default CPU / JAX harness with memory profiling
.venv/bin/python benchmarks/benchmark.py benchmarks/fixtures --output-dir benchmarks/output

# Multi-backend comparison (cpu + tinygrad + jax), CSV + three-panel figure
.venv/bin/python benchmarks/compare.py --manifest benchmarks/datasets/kahraman_2013_t3.tsv \
    --structures-dir benchmarks/downloads/kahraman_2013_t3 \
    --backends cpu tinygrad-batch tinygrad-single
```

## Python API

```python
from pathlib import Path

from protein_affinity_gpu import (
    load_complex,
    predict,
    predict_binding_affinity,
    predict_binding_affinity_jax,
)

structure = Path("benchmarks/fixtures/1A2K.pdb")
target, binder = load_complex(structure, selection="A,B")

# Default backend-specific entry points:
cpu_result = predict_binding_affinity(structure, selection="A,B")
jax_result = predict_binding_affinity_jax(structure, selection="A,B")                # mode="block"
jax_scan   = predict_binding_affinity_jax(structure, selection="A,B", mode="scan")

# Or route through the unified predictor:
result = predict(structure, backend="jax", selection="A,B")

# Experimental (tinygrad / single / neighbor / soft) — see docs/EXPERIMENTAL.md:
from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad
tg_result = predict_binding_affinity_tinygrad(structure, selection="A,B")
```

## Result

Every backend returns the same `ProdigyResults` dataclass. Call
`result.to_dict()` (or `result.save_results(output_dir)`) to get a
stable JSON-serialisable view — the CLI writes exactly this shape when
given `--output-json`:

| Field | Meaning |
|-------|---------|
| `ba_val` | Predicted ΔG of binding in kcal/mol (PRODIGY IC-NIS). |
| `kd` | Dissociation constant in molar units — `dg_to_kd(ba_val, temperature)`. |
| `contacts` | Interface residue–residue contact counts by **A**liphatic / **C**harged / **P**olar pair (`AA`, `CC`, `PP`, `AC`, `AP`, `CP`), plus derived totals (`IC`, `chargedC`, `polarC`, `aliphaticC`). |
| `nis` | Percentage of the non-interacting surface per character class. |
| `sasa_data` | Per-atom SASA after the NIS mask, with chain / residue / atom metadata. |

```json
{
  "structure_id": "1A2K",
  "ba_val": -9.42,
  "kd": 1.23e-07,
  "contacts": {"AA": 12, "CC": 3, "PP": 5, "AC": 4, "AP": 6, "CP": 2,
               "IC": 32, "chargedC": 9, "polarC": 13, "aliphaticC": 22},
  "nis": {"aliphatic": 41.2, "charged": 24.1, "polar": 34.7},
  "sasa_data": [{"chain": "A", "resname": "ALA", "resindex": 1,
                 "atomname": "CA", "atom_sasa": 12.5, "relative_sasa": 0.83}]
}
```

## Backends and Devices

| Backend | Entry point | Requires | Device selection |
|---------|-------------|----------|------------------|
| CPU (PRODIGY) | `predict_binding_affinity` | `prodigy-prot`, `freesasa` | n/a |
| JAX | `predict_binding_affinity_jax` (`mode="block"`/`"scan"`) | `jax`, `jaxlib` | `jax.default_backend()` |
| tinygrad | `predict_binding_affinity_tinygrad` (`mode="block"`/`"single"`/`"neighbor"`) | `tinygrad` | `Device.DEFAULT`, override via `TINYGRAD_DEVICE` |

Every backend shares one pipeline in `protein_affinity_gpu.predict`
parametrized by a `BackendAdapter` (see `protein_affinity_gpu.backends`).
Each adapter owns device resolution, lazy constant materialization, SASA
kernel dispatch, and the block-size heuristic:

- **JAX / CUDA** — `JAXAdapter.validate_size()` reads total GPU memory via
  `nvidia-smi` and raises `ValueError` if a complex exceeds the estimated
  ceiling; block size targets ~1 GB float32 scratch.
- **JAX / Apple Metal** — skips the size check (unified memory); block size
  comes from an empirical exp-decay fit of throughput vs atom count.
- **JAX / CPU** — conservative 100k-atom ceiling.
- **tinygrad / METAL, CUDA, GPU** — batched SASA with `block = min(768, N)`
  via a per-shape `TinyJit` cache.
- **tinygrad / CPU, CLANG** — full (non-batched) SASA kernel, also
  `TinyJit`-wrapped.

Force a JAX device with standard JAX environment variables, e.g.
`JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`. Set
`TINYGRAD_DEVICE=CPU|METAL|CUDA` to override tinygrad's device choice.

The tinygrad backend exposes three SASA kernels via the `mode` kwarg on
`predict_binding_affinity_tinygrad`:

| `mode` | Scratch | When to use |
|--------|---------|-------------|
| `"block"` (default) | `[block, M, N]`, `block ≤ 768` | Bounded scratch, per-shape `TinyJit` cache. Clear the cache (`sasa._sasa_block_jit_cache.clear()` + `gc.collect()`) between unrelated structures on Metal — each new shape pins ~1–4 GB and they accumulate across long sweeps. |
| `"single"` | `[N, M, N]` | Fully fused single kernel — fastest when it fits. On Apple Metal handles up to ~12k atom14-padded atoms in isolation; Metal returns `Internal Error (0000000e)` under sustained compile pressure without cache clearing. |
| `"neighbor"` | `[N, M, K]`, K=64 default | Memory-constrained GPUs — ~80× less scratch than `single` via `Tensor.topk` on negative squared distances. **Slower** than `block` on Apple Metal (`topk` dominates). Lossless when K covers the worst-case occlusion-neighbor count. |

On the Kahraman 2013 T3 set (16 two-chain complexes, padded N ∈ [2.5k, 12.8k])
the tinygrad block kernel runs at ~0.69 s warm-mean vs ~0.49 s for
CPU freesasa — within 1.5× of CPU — with Pearson `r > 0.9998` against CPU on
per-structure SASA totals and `r = 1.000` on ΔG, Kd, NIS and contact metrics.

## CPU vs JAX Dataset Comparison

For the Kahraman 2013 T3 set included in [benchmarks/datasets/kahraman_2013_t3.tsv](benchmarks/datasets/kahraman_2013_t3.tsv), you can fetch the listed structures and run a CPU vs JAX comparison with:

```bash
.venv/bin/python -m pip install -e .
REQUIRE_GPU=1 bash benchmarks/run_kahraman_compare.sh
```

The script downloads the required PDB files into `benchmarks/downloads/kahraman_2013_t3/` and writes CSV, JSON, and plot artifacts to `benchmarks/output/kahraman_2013_t3/`.

## Modal Benchmark

The Colab notebook has also been refactored into:

- [benchmarks/sasa_benchmark.py](benchmarks/sasa_benchmark.py) for a normal local Python run
- [benchmarks/modal_benchmark.py](benchmarks/modal_benchmark.py) for a Modal GPU run

Setup:

```bash
python3 -m pip install -e ".[modal]"
modal setup
```

The Modal image is GPU-only: it installs `biopython`, `numpy`, `jax[cuda12]`,
`tinygrad`, `matplotlib`, and `pandas`, but intentionally skips
`freesasa` and `prodigy-prot`. Because of that, the Modal entrypoint does
not support the `cpu` benchmark target.

Run the benchmark on Modal:

```bash
# Default GPU is A100-80GB.
modal run benchmarks/modal_benchmark.py --repeats 2 --run-name kahraman-a100

# Or set MODAL_GPU explicitly if you want to override it.
MODAL_GPU=A100-80GB modal run benchmarks/modal_benchmark.py --repeats 2 --run-name kahraman-a100

# Optional quick smoke run over just the first 10 manifest rows.
modal run benchmarks/modal_benchmark.py --limit 10 --run-name smoke-10

# Download the output artifacts if you did not pass --local-output-dir.
modal volume get protein-affinity-gpu-benchmarks runs/kahraman-a100 benchmarks/output/modal-kahraman-a100
```

If you pass `--local-output-dir benchmarks/output/modal-kahraman-a100`, the Modal entrypoint will also download `benchmark_results.json`, `benchmark_summary.json`, `benchmark_rows.csv`, `benchmark_warm_ms_wide.csv`, and `time_vs_atoms.png` back to your machine after the remote run completes.

## Development

```bash
python3 -m pip install -e ".[dev]"
python3.11 -m pytest
```
