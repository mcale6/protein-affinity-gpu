# protein-affinity-gpu

`protein-affinity-gpu` is a research-friendly Python package for protein-protein binding affinity prediction, solvent-accessible surface area (SASA) analysis, and reproducible CPU/JAX benchmarking.

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
.venv/bin/python benchmarks/benchmark.py benchmarks/fixtures --output-dir benchmarks/output
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
| `--targets` | `cpu cuda tinygrad` | Subset of `{cpu, cuda, tinygrad}` to benchmark. |
| `--selection`, `--temperature`, `--distance-cutoff`, `--acc-threshold`, `--sphere-points` | — | Same meaning as `predict`. |
| `--verbose` | off | `INFO`-level logging. |

`cuda` is automatically reported as `skipped` when no CUDA device is
detected. The harness is safe to run unconditionally in CI.

## Python API

```python
from pathlib import Path

from protein_affinity_gpu import (
    load_complex,
    predict_binding_affinity,
    predict_binding_affinity_jax,
    predict_binding_affinity_tinygrad,
)

structure = Path("benchmarks/fixtures/1A2K.pdb")
target, binder = load_complex(structure, selection="A,B")

cpu_result = predict_binding_affinity(structure, selection="A,B")
jax_result = predict_binding_affinity_jax(structure, selection="A,B")
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
| JAX | `predict_binding_affinity_jax` | `jax`, `jaxlib` | `jax.default_backend()` |
| tinygrad | `predict_binding_affinity_tinygrad` | `tinygrad` | `Device.DEFAULT`, override via `TINYGRAD_DEVICE` |

The JAX backend auto-tunes memory based on the detected device and always
routes through `calculate_sasa_batch`, which dispatches a jitted per-block
kernel (compiled once per call-site) to cap peak memory:

- **CUDA** — `estimate_max_atoms()` reads total GPU memory via `nvidia-smi`
  and raises `ValueError` if a complex exceeds the estimated ceiling.
- **Apple Metal** — assumes ~20 GB of unified memory and skips the size check.
- **CPU (JAX)** — conservative 100k-atom ceiling.

Block size comes from `estimate_optimal_block_size(n_atoms)`.

Force a device with standard JAX environment variables, e.g.
`JAX_PLATFORMS=cpu` or `JAX_PLATFORMS=cuda`.

The tinygrad backend routes METAL/CUDA/GPU through `calculate_sasa_batch_tinygrad`
(dot-product pairwise distances, one realized kernel per block) and falls back
to the full `calculate_sasa_tinygrad` on CPU. Set `TINYGRAD_DEVICE=CPU|METAL|CUDA`
to override device selection. Expect ~10–30× the CPU-freesasa wall time on
large complexes — tinygrad kernels are recompiled on first call and then cached.

## Benchmark Fixtures

The repository tracks a tiny canonical fixture set under [benchmarks/fixtures/1A2K.pdb](benchmarks/fixtures/1A2K.pdb) — it ships with the package so tests and examples work out of the box. Only the generated `benchmarks/output/` and `benchmarks/downloads/` directories are ignored by git.

## CPU vs JAX Dataset Comparison

For the Kahraman 2013 T3 set included in [benchmarks/datasets/kahraman_2013_t3.tsv](benchmarks/datasets/kahraman_2013_t3.tsv), you can fetch the listed structures and run a CPU vs JAX comparison with:

```bash
.venv/bin/python -m pip install -e .
REQUIRE_GPU=1 bash benchmarks/run_kahraman_compare.sh
```

The script downloads the required PDB files into `benchmarks/downloads/kahraman_2013_t3/` and writes CSV, JSON, and plot artifacts to `benchmarks/output/kahraman_2013_t3/`.

## Development

```bash
python3 -m pip install -e ".[dev]"
python3.11 -m pytest
```
