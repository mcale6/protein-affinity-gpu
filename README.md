# protein-affinity-gpu

`protein-affinity-gpu` is a research-friendly Python package for protein-protein binding affinity prediction, solvent-accessible surface area (SASA) analysis, and reproducible CPU/JAX benchmarking.

## Installation

```bash
python3 -m pip install "protein-affinity-gpu==1.6.9"
```

For JAX support:

```bash
python3 -m pip install "protein-affinity-gpu[jax]"
```

For benchmarking and plots:

```bash
python3 -m pip install "protein-affinity-gpu[bench]"
```

## What Changed

- The package import root is now `protein_affinity_gpu`.
- The CLI is now split into `protein-affinity-predict` and `protein-affinity-benchmark`.
- Benchmarking is scriptable and fixture-driven instead of notebook-led.
- Generated benchmark artifacts are no longer tracked in git.

## Python API

```python
from pathlib import Path

from protein_affinity_gpu import (
    load_complex,
    predict_binding_affinity,
    predict_binding_affinity_jax,
)

structure = Path("benchmarks/fixtures/1A2K.pdb")
target, binder = load_complex(structure, selection="A,B")

cpu_result = predict_binding_affinity(structure, selection="A,B")
jax_result = predict_binding_affinity_jax(structure, selection="A,B")
```

## CLI

Predict a structure or folder:

```bash
protein-affinity-predict benchmarks/fixtures --backend cpu --output-json
protein-affinity-predict benchmarks/fixtures --backend jax --output-json
```

Run the benchmark harness:

```bash
python3 benchmarks/run.py benchmarks/fixtures --output-dir benchmarks/output
```

Or via the installed CLI:

```bash
protein-affinity-benchmark benchmarks/fixtures --output-dir benchmarks/output
```

## Benchmark Fixtures

The repository keeps a tiny canonical fixture set under [benchmarks/fixtures/1A2K.pdb](benchmarks/fixtures/1A2K.pdb). Generated benchmark JSON files should go to `benchmarks/output/`, which is ignored by git.

## Development

```bash
python3 -m pip install -e ".[dev]"
python3.11 -m pytest
```
