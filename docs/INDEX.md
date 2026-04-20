# protein-affinity-gpu — Project Index

> Generated project knowledge index for `protein-affinity-gpu` v1.6.9.
> Source of truth: [README.md](../README.md), [pyproject.toml](../pyproject.toml).

`protein-affinity-gpu` is a Python package for protein–protein binding
affinity prediction with three interchangeable backends:

- **CPU** — a thin wrapper around [`prodigy-prot`] + [`freesasa`] that reproduces
  the PRODIGY IC-NIS model.
- **JAX** — a vectorized re-implementation of the same pipeline that runs on
  CPU, CUDA, or Apple Metal via [JAX].
- **tinygrad** — a port of the JAX pipeline onto [tinygrad] that runs on
  METAL, CUDA, CLANG, or the tinygrad CPU device via
  `predict_binding_affinity_tinygrad`.

All backends share one data model ([`Protein`](../src/protein_affinity_gpu/utils/structure.py),
[`ProdigyResults`](../src/protein_affinity_gpu/results.py)) so results are
directly comparable. The JAX and tinygrad backends share a single pipeline
in [`predict.py`](../src/protein_affinity_gpu/predict.py), parametrized by a
[`BackendAdapter`](../src/protein_affinity_gpu/backends/_adapter.py).

---

## 1. Navigation

| Area | Entry Point |
|------|-------------|
| Install / quickstart | [README.md](../README.md) |
| Package metadata | [pyproject.toml](../pyproject.toml) |
| Python API root | [src/protein_affinity_gpu/__init__.py](../src/protein_affinity_gpu/__init__.py) |
| Unified predictor | [src/protein_affinity_gpu/predict.py](../src/protein_affinity_gpu/predict.py) |
| CPU predictor | [src/protein_affinity_gpu/cpu.py](../src/protein_affinity_gpu/cpu.py) |
| Backend adapters | [src/protein_affinity_gpu/backends/](../src/protein_affinity_gpu/backends/) |
| Logging helpers | [src/protein_affinity_gpu/utils/logging_utils.py](../src/protein_affinity_gpu/utils/logging_utils.py) |
| Structure loader | [src/protein_affinity_gpu/utils/structure.py](../src/protein_affinity_gpu/utils/structure.py) |
| CLI — predict | [src/protein_affinity_gpu/cli/predict.py](../src/protein_affinity_gpu/cli/predict.py) |
| CLI — benchmark | [src/protein_affinity_gpu/cli/benchmark.py](../src/protein_affinity_gpu/cli/benchmark.py) |
| Benchmark harness | [benchmarks/run.py](../benchmarks/run.py) |
| Test suite | [tests/](../tests) |
| CI | [.github/workflows/ci.yml](../.github/workflows/ci.yml) |
| Release script | [update_pkg.sh](../update_pkg.sh) |

---

## 2. Repository layout

```
protein-affinity-gpu/
├── pyproject.toml             # Hatchling build, entry points, optional extras
├── README.md                  # User-facing quickstart
├── LICENSE
├── update_pkg.sh              # Bump version + build sdist/wheel
├── .github/workflows/ci.yml   # pytest + build on push/PR (Python 3.11)
├── benchmarks/
│   ├── run.py                 # Standalone entry into the benchmark CLI
│   └── fixtures/1A2K.pdb      # Canonical two-chain complex used by tests
├── src/protein_affinity_gpu/
│   ├── __init__.py            # Public API (lazy-loads impls)
│   ├── version.py             # __version__ (read by Hatch)
│   ├── predict.py             # Unified pipeline + `predict(backend=…)` router + jax/tinygrad entry points
│   ├── cpu.py                 # PRODIGY + freesasa CPU pipeline
│   ├── sasa.py                # SASA kernels (JAX + tinygrad, full + blocked)
│   ├── contacts.py            # Residue contacts + interaction class counts
│   ├── scoring.py             # NISCoefficients + backend-agnostic scoring primitives
│   ├── results.py             # ProdigyResults / ContactAnalysis, JSON writer
│   ├── backends/
│   │   ├── _adapter.py        # BackendAdapter Protocol
│   │   ├── _jax.py            # JAXAdapter
│   │   └── _tinygrad.py       # TinygradAdapter
│   ├── cli/
│   │   ├── predict.py         # `protein-affinity-predict`
│   │   └── benchmark.py       # `protein-affinity-benchmark`
│   ├── data/                  # naccess.config, vdw.radii, thomson*.xyz
│   └── utils/
│       ├── _array.py                  # Array TypeAlias, NumpyEncoder, concat/stack/exp shims
│       ├── atom14.py                  # atom37 ↔ atom14 gather/scatter
│       ├── logging_utils.py           # setup_logging, get_logger, log_duration
│       ├── resources.py               # Packaged-data helpers, file collection
│       ├── structure.py               # Protein dataclass, load_complex/load_structure
│       ├── residue_constants.py       # AlphaFold-derived atom/residue tables
│       ├── residue_classification.py  # IC / PROTORP character matrices, ASA refs
│       └── residue_library.py         # VdW radii per (residue, atom)
└── tests/                     # pytest suite — see §6
```

---

## 3. Public Python API

All symbols below are re-exported from the package root:

```python
from protein_affinity_gpu import (
    __version__,
    Protein,
    ProdigyResults,
    ContactAnalysis,
    load_structure,
    load_complex,
    predict,                           # unified router: backend="cpu"|"jax"|"tinygrad"
    predict_binding_affinity,          # CPU-only (legacy alias)
    predict_binding_affinity_jax,
    predict_binding_affinity_tinygrad,
)
```

Every backend entry point is lazy-loaded at first call so importing the
package doesn't pull JAX *and* tinygrad into memory.

### 3.1 Structure I/O — `utils/structure.py`

| Symbol | Purpose |
|--------|---------|
| `Protein` | Frozen dataclass with `atom_positions`, `aatype`, `atom_mask`, `residue_index`, `chain_index`, `b_factors`. |
| `load_structure(path, chain_id=None, sanitize=True)` | Parse a PDB / mmCIF file into a single-chain `Protein`. |
| `load_complex(path, selection="A,B", sanitize=True)` | Parse a two-chain complex into `(target, binder)`. |
| `from_bio_structure`, `from_pdb_string`, `from_mmcif_string` | Lower-level constructors. |
| `sanitize_structure` | Strips waters, hetero, hydrogens, insertion codes, extra models, and non-selected chains. |
| `to_pdb(Protein)` | Round-trip a `Protein` back to a PDB string. |
| `from_prediction(features, result, ...)` | Build a `Protein` from AlphaFold-style model outputs. |

### 3.2 CPU pipeline — `cpu.py`

```python
predict_binding_affinity(
    struct_path, selection=None,
    temperature=25.0, distance_cutoff=5.5,
    acc_threshold=0.05, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
) -> ProdigyResults
```

- Requires the optional `prodigy-prot` and `freesasa` dependencies.
- `execute_freesasa(structure, sphere_points)` is exposed for custom flows.
- Uses the packaged `data/naccess.config` classifier.

### 3.3 Unified pipeline — `predict.py`

```python
from protein_affinity_gpu import predict

predict(
    struct_path, backend="jax",       # "cpu" | "jax" | "tinygrad"
    selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    # backend-specific extras flow through **backend_kwargs, e.g. soft_sasa=True
) -> ProdigyResults
```

`_run_pipeline(adapter, …)` is the shared body consumed by both the router
and the backend-specific shims below. Each phase (`load_complex`,
`contacts`, `sasa`, `nis`, `score`) is wrapped in `log_duration`, so enabling
debug logging prints a start / done-in-X.XXXs line per phase.

### 3.3a Backend adapters — `backends/`

The adapter Protocol ([`_adapter.py`](../src/protein_affinity_gpu/backends/_adapter.py))
names the surface the pipeline calls. Concrete adapters own their
device resolution, lazy constants, and kernel dispatch:

| Adapter | Notable behavior |
|---------|------------------|
| `JAXAdapter` | `soft_sasa=True` swaps in the sigmoid SASA kernel; `validate_size` calls `nvidia-smi` on CUDA; block size uses an exp-decay fit on Metal, ~1 GB scratch target otherwise. |
| `TinygradAdapter` | `METAL` / `CUDA` / `GPU` → batched SASA with `block=min(768, N)`; any other device (incl. `CPU` / `CLANG`) → full `calculate_sasa_tinygrad` kernel (`block_size=None`). |

### 3.3b Backend entry points (in `predict.py`)

```python
predict_binding_affinity_jax(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    soft_sasa=False, soft_beta=10.0,
) -> ProdigyResults

predict_binding_affinity_tinygrad(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
) -> ProdigyResults
```

Each constructs the matching adapter (`JAXAdapter` / `TinygradAdapter`)
and delegates to `_run_pipeline`. tinygrad is a core dependency; its
device selection is `Device.DEFAULT`, overridable via
`TINYGRAD_DEVICE=CPU|METAL|CUDA`.

### 3.4 SASA kernels — `sasa.py`

| Function | Notes |
|----------|-------|
| `generate_sphere_points(n)` | Golden-spiral sphere point distribution (JAX). |
| `generate_sphere_points_tinygrad(n)` | Same distribution, returns a tinygrad `Tensor`. |
| `calculate_sasa_batch(coords, vdw_radii, mask, block_size, sphere_points, probe_radius=1.4)` | Blocked Shrake–Rupley (JAX): Python dispatcher over a `@jit`'d per-block kernel using `|a−b|² = a² + b² − 2⟨a,b⟩`. Tail block reuses the last full window so the kernel compiles once. |
| `calculate_sasa_tinygrad` | `TinyJit`-wrapped full kernel for CPU / CLANG devices. |
| `calculate_sasa_batch_tinygrad(..., block_size=...)` | Per-block-realized dot-product kernel; each block is `.numpy()`-reduced into a buffer to keep tinygrad's graph-rewrite stack bounded. |

### 3.5 Contact analysis — `contacts.py`

- `calculate_residue_contacts(target_pos, binder_pos, target_mask, binder_mask, distance_cutoff=5.5)` — pairwise residue contact mask; 5-D diff variant (JAX / numpy).
- `calculate_residue_contacts_tinygrad(…)` — matmul-reshape variant that sidesteps Metal's unified-memory pressure on the ``[N_t, N_b, 37, 37, 3]`` intermediate.
- `analyze_contacts(contacts, target_seq, binder_seq, class_matrix)` — backend-agnostic broadcast outer product, returns the 6-tuple `[AA, CC, PP, AC, AP, CP]`.

### 3.6 Scoring — `scoring.py`

PRODIGY IC-NIS coefficients live in the `NISCoefficients` frozen dataclass
(singleton: `NIS_COEFFICIENTS`). Backend-agnostic primitives operate on
duck-typed tensor methods (`@`, `.sum`, `.reshape`, `.clip`) so the same
body runs on numpy, jax, and tinygrad:

- `get_atom_radii(seq_one_hot, radii_matrix, atom_mask=None)` — optional mask for atom37 padding.
- `calculate_relative_sasa(complex_sasa, seq_probs, relative_sasa_array, atoms_per_residue)`
- `calculate_nis_percentages(sasa_values, seq_probs, character_matrix, threshold=0.05)`
- `score_ic_nis(ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c, coeffs, intercept)` — returns ΔG in kcal/mol.
- `dg_to_kd(dg, temperature=25.0)` — dissociation constant in M.
- `coefficient_tensors_tinygrad(coefficients=NIS_COEFFICIENTS)` — coefficient vector + intercept as tinygrad `Tensor`s (used by `TinygradAdapter`).

### 3.7 Results — `results.py`

| Symbol | Purpose |
|--------|---------|
| `ContactAnalysis(values)` | Wrap 6-tuple of contact counts; `.to_dict()` adds totals and grouped counts (`IC`, `chargedC`, `polarC`, `aliphaticC`). |
| `ProdigyResults` | Dataclass with ΔG, Kd, NIS percentages, contacts, and a structured `sasa_data` array. `to_dict()` / `save_results(output_dir)` / `__str__` for reports. |
| `build_sasa_records(complex_sasa, relative_sasa, target, binder, chain_labels)` | Build the structured `sasa_data` array; materializes jax/tinygrad inputs via `to_numpy`. |

### 3.8 Utilities — `utils/`

- `_array` — `Array` TypeAlias (numpy | jax | tinygrad), `NumpyEncoder`, `to_numpy`, `concat` / `stack_scalars` / `exp` dispatch shims.
- `atom14` — `compact_atom37_to_atom14`, `expand_atom14_to_atom37`, `compact_complex_atom14`; all accept `xp=np|jnp` for differentiable paths.
- `resources` — `data_path`, `read_text_resource`, `collect_structure_files`, `format_duration`.
- `structure` — `Protein` dataclass, `load_complex`, `load_structure`, sanitizers.
- `residue_constants` — AlphaFold-derived lookups (`restypes`, `restype_1to3`, `atom_types`, `atom_type_num`, `STANDARD_ATOM_MASK`, chi definitions, rigid groups, …).
- `ResidueClassification(kind="protorp" | "ic")` — character matrix, cached indices, reference relative SASA array.
- `ResidueLibrary` — parses `data/vdw.radii`, exposes `get_radius`, `is_polar`, `[n_restypes, 37]` radii matrix, and `[n_restypes, 14]` atom14 variant. Module-level `default_library` is pre-built.

### 3.9 Logging helpers — `utils/logging_utils.py`

- `setup_logging(level="INFO", *, propagate=True)` — attach a `StreamHandler`
  to the package logger (`protein_affinity_gpu`) with an
  `HH:MM:SS [LEVEL] name: message` format. Idempotent — safe to call from
  notebooks or the CLI.
- `get_logger(name)` — returns a namespaced child of the package logger, e.g.
  `get_logger(__name__)` inside a submodule.
- `log_duration(logger, label, *, level=logging.DEBUG, extra=None)` — context
  manager that emits `"<label>: start"` and `"<label>: done in X.XXXs"` at
  the given level. Wraps every phase of the unified pipeline
  (`<device>.load_complex`, `<device>.contacts`, `<device>.sasa`,
  `<device>.nis`, `<device>.score`) where `<device>` is the adapter's
  resolved device name (e.g. `METAL`, `CUDA`, `CPU`).

---

## 4. Command-line interface

Both CLIs are registered as console scripts in [pyproject.toml](../pyproject.toml).

### 4.1 `protein-affinity-predict`

```bash
protein-affinity-predict <input_path> \
    [--backend cpu|jax|tinygrad] \
    [--selection A,B] \
    [--temperature 25.0] \
    [--distance-cutoff 5.5] \
    [--acc-threshold 0.05] \
    [--sphere-points 100] \
    [--output-json] [--output-dir results/] [--verbose]
```

- `input_path` may be a file or directory; a directory is walked through
  `collect_structure_files`.
- Prints a single JSON document combining every structure to stdout.
- When both `--output-json` and `--output-dir` are set, each structure is also
  written to `<output-dir>/<stem>_results.json`.

### 4.2 `protein-affinity-benchmark`

```bash
protein-affinity-benchmark <input_path> \
    [--output-dir benchmarks/output] \
    [--repeats 3] \
    [--targets cpu cuda tinygrad] \
    [--selection A,B] \
    [--temperature 25.0] [--distance-cutoff 5.5] \
    [--acc-threshold 0.05] [--sphere-points 100] [--verbose]
```

- Runs `repeats` iterations per target per structure, reporting a cold run and
  the mean of the warm runs.
- `cuda` is auto-skipped if no GPU/CUDA device is found.
- Writes `<output-dir>/benchmark_results.json` and echoes the report to stdout.
- `benchmarks/run.py` is a shim so the harness runs from a source checkout
  without installing the package.

---

## 5. Data and scoring model

### 5.1 IC-NIS coefficients (`scoring.NIS_COEFFICIENTS`)

| Feature | Coefficient |
|---------|-------------|
| `ic_cc` (charged-charged contacts) | −0.09459 |
| `ic_ca` (charged-aliphatic) | −0.10007 |
| `ic_pp` (polar-polar) | 0.19577 |
| `ic_pa` (polar-aliphatic) | −0.22671 |
| `p_nis_a` (% NIS aliphatic) | 0.18681 |
| `p_nis_c` (% NIS charged) | 0.13810 |
| `intercept` | −15.9433 |

`score_ic_nis` returns ΔG in kcal/mol; `dg_to_kd` converts to Kd using
R = 1.9858775 × 10⁻³ kcal/(mol·K).

### 5.2 Packaged data (`src/protein_affinity_gpu/data/`)

| File | Use |
|------|-----|
| `naccess.config` | Classifier passed to freesasa in the CPU path. |
| `vdw.radii` | Per-residue/per-atom radii parsed by `ResidueLibrary`. |
| `thomson100.xyz`, `thomson1000.xyz`, `thomson15092.xyz` | Even sphere-point tables. |

### 5.3 Residue character models

Two classification schemes are provided. The JAX pipeline uses `"ic"` for
contact classification and `"protorp"` for NIS:

| Scheme | Notable differences |
|--------|--------------------|
| `ic` | CYS→Aliphatic, HIS→Charged, TRP/TYR→Aliphatic. |
| `protorp` | CYS/HIS/TRP/TYR→Polar. |

---

## 6. Tests

Located under [`tests/`](../tests). Common fixture: `benchmarks/fixtures/1A2K.pdb`.

| Test | Scope |
|------|-------|
| `test_imports.py` | Top-level re-exports and CLI modules import. |
| `test_structure.py` | `load_complex` sanitizes H, water, and non-selected chains. |
| `test_regression.py` | CPU vs JAX prediction stay within `|ΔΔG| < 0.75` and `|ΔIC| < 10`. Skips if JAX / prodigy-prot / freesasa are missing. |
| `test_tinygrad_smoke.py` | Tinygrad prediction returns finite ΔG, within `|ΔΔG| < 0.75` and `|ΔIC| < 10` of the CPU reference. |
| `test_benchmark_smoke.py` | Benchmark harness runs with mocked predictor; CUDA is reported as skipped when unavailable. |
| `test_resources.py` | Packaged `vdw.radii` is accessible via `read_text_resource`. |
| `test_residue_library.py` | `ResidueLibrary.radii_matrix` has the expected shape. |
| `test_results.py` | `ProdigyResults.save_results` round-trips through JSON. |

Run with:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

CI (GitHub Actions, Python 3.11) runs `pytest` and `python -m build` on every
push and pull request.

---

## 7. Install & release

### 7.1 Dependencies (`pyproject.toml`)

- Core: `biopython`, `prodigy-prot`, `freesasa`, `numpy>=1.23,<3.0`, `jax`,
  `jaxlib`, `tinygrad`, `matplotlib`, `pandas`. A single
  `pip install protein-affinity-gpu` brings in every backend and the
  benchmarking plot stack.
- `[dev]` extra: `build`, `pytest>=8.0`, `ruff>=0.6`.

Build system: **Hatchling**, with the version read dynamically from
`src/protein_affinity_gpu/version.py`.

### 7.2 Release flow (`update_pkg.sh`)

```bash
./update_pkg.sh [major|minor|patch|none]
```

Bumps the semantic version in `version.py`, wipes `build/` and `dist/`, then
runs `python3 -m build` (sdist + wheel).

---

## 8. End-to-end example

```python
from pathlib import Path

from protein_affinity_gpu import load_complex, predict
from protein_affinity_gpu.utils.logging_utils import setup_logging

setup_logging("DEBUG")  # per-phase timings from the unified pipeline

structure = Path("benchmarks/fixtures/1A2K.pdb")

target, binder = load_complex(structure, selection="A,B")
print(target.atom_positions.shape, binder.atom_positions.shape)

cpu = predict(structure, backend="cpu", selection="A,B")
jax_result = predict(structure, backend="jax", selection="A,B")
tg_result = predict(structure, backend="tinygrad", selection="A,B")

print(cpu)
print(f"ΔΔG (CPU vs JAX)      = {cpu.binding_affinity - jax_result.binding_affinity:+.3f}")
print(f"ΔΔG (CPU vs tinygrad) = {cpu.binding_affinity - tg_result.binding_affinity:+.3f}")
```

The `ProdigyResults` returned by any backend serializes to the same JSON
schema, keyed by `structure_id`, with top-level fields `ba_val`, `kd`,
`contacts`, `nis`, and a per-atom `sasa_data` list.

[`prodigy-prot`]: https://github.com/haddocking/prodigy
[`freesasa`]: https://freesasa.github.io/
[JAX]: https://github.com/jax-ml/jax
[tinygrad]: https://github.com/tinygrad/tinygrad
