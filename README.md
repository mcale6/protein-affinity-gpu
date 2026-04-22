# protein-affinity-gpu

`protein-affinity-gpu` is a research-friendly Python package for protein-protein binding affinity prediction, solvent-accessible surface area (SASA) analysis, and reproducible CPU/JAX benchmarking.

The default surface is CPU + JAX (blocked and `lax.scan`-fused Shrake–Rupley).
Tinygrad, single-pass / neighbor-cutoff JAX, and differentiable soft-SASA
live behind `protein_affinity_gpu.experimental` — see
[docs/EXPERIMENTAL.md](docs/EXPERIMENTAL.md).

## Installation

```bash
python3 -m pip install "protein-affinity-gpu==1.6.9"
```

A single install pulls in everything — CPU (`prodigy-prot`, `freesasa`),
JAX (`jax`, `jaxlib`), tinygrad, and the benchmarking plot stack
(`matplotlib`, `pandas`).

## CLI

Predict a structure or folder:

```bash
protein-affinity-predict benchmarks/fixtures --backend cpu --output-json
protein-affinity-predict benchmarks/fixtures --backend jax --output-json
protein-affinity-predict benchmarks/fixtures --backend tinygrad --output-json
```

Run the benchmark harness:

```bash
# Default: CPU / JAX (block) / JAX (scan) with memory profiling
.venv/bin/python benchmarks/benchmark.py benchmarks/fixtures --output-dir benchmarks/output

# Notebook-style Kahraman sweep as a regular Python script
.venv/bin/python benchmarks/sasa_benchmark.py --output-dir benchmarks/output/colab_sweep
```

### `protein-affinity-predict` flags

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

The CLI prints a concise summary per structure to stdout (ΔG, Kd, contact
analysis, NIS breakdown — the same block you get from `str(result)`).
`--verbose` additionally streams colored phase timings to stderr, useful
for profiling the tinygrad / JAX pipelines. Use `--output-json` to persist
the full per-atom JSON to disk.

### `protein-affinity-benchmark` flags

| Flag | Default | Description |
|------|---------|-------------|
| `input_path` | — | File or directory of structures. |
| `--output-dir` | `benchmarks/output` | Destination for `benchmark_results.json`. |
| `--repeats` | `3` | Runs per target; first is cold, rest averaged. |
| `--targets` | `cpu cuda jax jax-scan` | Subset of `{cpu, cuda, jax, jax-scan}` for the default harness. For the extended sweep (`jax-single`, `jax-neighbor`, `jax-soft`, `tinygrad`, `tinygrad-neighbor`), use `benchmarks/sasa_benchmark.py`. |
| `--selection`, `--temperature`, `--distance-cutoff`, `--acc-threshold`, `--sphere-points` | — | Same meaning as `predict`. |
| `--verbose` | off | `INFO`-level logging. |

`cuda` is automatically reported as `skipped` when no CUDA device is
detected, so the harness is safe to run unconditionally on hosts without
a GPU.

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

## Result Schema

Both backends return the same `ProdigyResults` dataclass, with a stable
`to_dict()` representation that is also what the CLI writes to JSON:

```json
{
  "structure_id": "1A2K",
  "ba_val": -9.42,
  "kd": 1.23e-07,
  "contacts": {
    "AA": 12.0, "CC": 3.0, "PP": 5.0,
    "AC": 4.0, "AP": 6.0, "CP": 2.0,
    "IC": 32.0,
    "chargedC": 9.0, "polarC": 13.0, "aliphaticC": 22.0
  },
  "nis": { "aliphatic": 41.2, "charged": 24.1, "polar": 34.7 },
  "sasa_data": [
    { "chain": "A", "resname": "ALA", "resindex": 1,
      "atomname": "CA", "atom_sasa": 12.5, "relative_sasa": 0.83 }
  ]
}
```

- `ba_val` — predicted ΔG of binding in kcal/mol (PRODIGY IC-NIS).
- `kd` — dissociation constant in molar units (`dg_to_kd(ba_val, temperature)`).
- `contacts` — interface residue–residue contact counts by character pair
  (**A**liphatic / **C**harged / **P**olar), plus derived totals (`IC`,
  `chargedC`, `polarC`, `aliphaticC`).
- `nis` — percentage of the non-interacting surface per character class.
- `sasa_data` — per-atom SASA after masking, with chain / residue metadata.

Save directly to disk with `results.save_results(output_dir)`.

## Backends and Devices

| Backend | Entry point | Requires | Device selection |
|---------|-------------|----------|------------------|
| CPU (PRODIGY) | `predict_binding_affinity` | `prodigy-prot`, `freesasa` | n/a |
| JAX | `predict_binding_affinity_jax` (`mode="block"`/`"scan"`) | `jax`, `jaxlib` | `jax.default_backend()` |
| tinygrad *(experimental)* | `experimental.predict_binding_affinity_tinygrad` | `tinygrad` | `Device.DEFAULT`, override via `TINYGRAD_DEVICE` |

GPU backends share a single pipeline in `protein_affinity_gpu.predict`,
parametrized by a `BackendAdapter` (see `protein_affinity_gpu.backends`).
Each adapter owns its device resolution, lazy constant materialization,
SASA kernel dispatch, and block-size heuristic:

- **JAX / CUDA** — `JAXAdapter.validate_size()` reads total GPU memory via
  `nvidia-smi` and raises `ValueError` if a complex exceeds the estimated
  ceiling; block size targets ~1 GB float32 scratch.
- **JAX / Apple Metal** — assumes ~20 GB of unified memory and skips the
  size check; block size comes from an empirical exp-decay fit.
- **JAX / CPU** — conservative 100k-atom ceiling.
- **tinygrad / METAL, CUDA, GPU** — batched SASA capped at `block=768`.
- **tinygrad / CPU** — full (non-batched) SASA path.

Force a JAX device with standard JAX environment variables, e.g.
`JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`.

The experimental tinygrad backend (imported from
`protein_affinity_gpu.experimental`) exposes three SASA kernels via the
`mode` kwarg on `predict_binding_affinity_tinygrad`:

| `mode` | Scratch | When to use |
|--------|---------|-------------|
| `"block"` (default) | `[block, M, N]`, block≤768 | Default; bounded scratch, per-shape `TinyJit` cache. |
| `"single"` | `[N, M, N]` | Fastest if it fits — single fused kernel; OOMs past ~5k atoms on 16 GB devices. |
| `"neighbor"` | `[N, M, K]` (K=64 default) | Memory-constrained GPUs. ~80× less scratch than `single`; **slower** than `block` on Apple Metal (topk dominates). Lossless when K covers the worst-case occlusion-neighbor count. |

Set `TINYGRAD_DEVICE=CPU|METAL|CUDA` to override device selection. Expect
~10–30× the CPU-freesasa wall time on large complexes — tinygrad kernels are
recompiled on first call and then cached per shape. For the full
experimental surface (including differentiable soft-SASA and the extended
JAX modes), see [docs/EXPERIMENTAL.md](docs/EXPERIMENTAL.md).

## Benchmark Fixtures

The repository tracks a tiny canonical fixture set under [benchmarks/fixtures/1A2K.pdb](benchmarks/fixtures/1A2K.pdb) — it ships with the package so tests and examples work out of the box. Only the generated `benchmarks/output/` and `benchmarks/downloads/` directories are ignored by git.

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

Run the benchmark on Modal:

```bash
# Pick a GPU type with MODAL_GPU if you want something other than the default L4.
MODAL_GPU=L4 modal run benchmarks/modal_benchmark.py --repeats 2 --run-name kahraman-l4

# Optional quick smoke run over just the first 10 manifest rows.
MODAL_GPU=L4 modal run benchmarks/modal_benchmark.py --limit 10 --run-name smoke-10

# Download the output artifacts if you did not pass --local-output-dir.
modal volume get protein-affinity-gpu-benchmarks runs/kahraman-l4 benchmarks/output/modal-kahraman-l4
```

If you pass `--local-output-dir benchmarks/output/modal-kahraman-l4`, the Modal entrypoint will also download `benchmark_results.json`, `benchmark_summary.json`, `benchmark_rows.csv`, `benchmark_warm_ms_wide.csv`, and `time_vs_atoms.png` back to your machine after the remote run completes.

## Development

```bash
python3 -m pip install -e ".[dev]"
python3.11 -m pytest
```
