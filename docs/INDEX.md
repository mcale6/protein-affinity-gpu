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

All backends share one data model ([`Protein`](../src/protein_affinity_gpu/structure.py),
[`ProdigyResults`](../src/protein_affinity_gpu/results.py)) so results are
directly comparable.

---

## 1. Navigation

| Area | Entry Point |
|------|-------------|
| Install / quickstart | [README.md](../README.md) |
| Package metadata | [pyproject.toml](../pyproject.toml) |
| Python API root | [src/protein_affinity_gpu/__init__.py](../src/protein_affinity_gpu/__init__.py) |
| CPU predictor | [src/protein_affinity_gpu/cpu.py](../src/protein_affinity_gpu/cpu.py) |
| JAX predictor | [src/protein_affinity_gpu/jax.py](../src/protein_affinity_gpu/jax.py) |
| tinygrad predictor | [src/protein_affinity_gpu/tinygrad.py](../src/protein_affinity_gpu/tinygrad.py) |
| Logging helpers | [src/protein_affinity_gpu/logging_utils.py](../src/protein_affinity_gpu/logging_utils.py) |
| Structure loader | [src/protein_affinity_gpu/structure.py](../src/protein_affinity_gpu/structure.py) |
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
│   ├── __init__.py            # Public API (lazy-loads CPU / JAX impls)
│   ├── version.py             # __version__ (read by Hatch)
│   ├── structure.py           # Protein dataclass, load_complex/load_structure
│   ├── cpu.py                 # PRODIGY + freesasa CPU pipeline
│   ├── jax.py                 # JAX pipeline, device/memory estimators
│   ├── tinygrad.py            # tinygrad pipeline end-to-end
│   ├── sasa.py                # SASA kernels (JAX + tinygrad, full + blocked)
│   ├── contacts.py            # Residue contact + interaction class counts (JAX + tinygrad)
│   ├── scoring.py             # PRODIGY IC-NIS coefficients & scoring (JAX + tinygrad)
│   ├── results.py             # ProdigyResults / ContactAnalysis, JSON writer
│   ├── resources.py           # Packaged-data helpers, file collection
│   ├── logging_utils.py       # setup_logging, get_logger, log_duration
│   ├── cli/
│   │   ├── predict.py         # `protein-affinity-predict`
│   │   └── benchmark.py       # `protein-affinity-benchmark`
│   ├── data/                  # naccess.config, vdw.radii, thomson*.xyz
│   └── utils/
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
    predict_binding_affinity,
    predict_binding_affinity_jax,
    predict_binding_affinity_tinygrad,
)
```

The `*_tinygrad` entry point is lazy-loaded at first call.

### 3.1 Structure I/O — `structure.py`

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

### 3.3 JAX pipeline — `jax.py`

```python
predict_binding_affinity_jax(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
) -> ProdigyResults
```

Helpers used to auto-tune memory on Metal / CUDA:

- `estimate_optimal_block_size(n_atoms)` — pick a block size for the batched
  SASA kernel on Apple Metal.
- `estimate_max_atoms(backend, safety_factor=0.8, sphere_points=100)` —
  predict the device ceiling (CUDA uses `nvidia-smi`; Metal is hard-coded).

### 3.3b tinygrad pipeline — `tinygrad.py`

```python
predict_binding_affinity_tinygrad(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
) -> ProdigyResults
```

- Imports `tinygrad` unconditionally — it's a core dependency of the package.
- Device selection: `Device.DEFAULT`, with `TINYGRAD_DEVICE=CPU|METAL|CUDA`
  environment override. `METAL` / `CUDA` / `GPU` devices route through
  `calculate_sasa_batch_tinygrad`; any other device (including `CPU` /
  `CLANG`) falls back to the full `calculate_sasa_tinygrad` kernel.
- `estimate_optimal_block_size(n_atoms)` picks a block size targeting ~1 GB
  of float32 scratch per block, clamped to `[32, min(n_atoms, 1024)]`.
- Each phase (`load_complex`, `contacts`, `sasa_batch`/`sasa_full`, `nis`,
  `score`) is wrapped in `log_duration` so enabling debug logging prints a
  start / done-in-X.XXXs line per phase.

### 3.4 SASA kernels — `sasa.py`

| Function | Notes |
|----------|-------|
| `generate_sphere_points(n)` | Golden-spiral sphere point distribution (JAX). |
| `generate_sphere_points_tinygrad(n)` | Same distribution, returns a tinygrad `Tensor`. |
| `calculate_sasa(coords, vdw_radii, mask, sphere_points, probe_radius=1.4)` | `@jit` full Shrake–Rupley (JAX). |
| `calculate_sasa_batch(..., block_size=...)` | Memory-efficient blocked JAX variant using `|a−b|² = a² + b² − 2⟨a,b⟩`. |
| `calculate_sasa_tinygrad` | `TinyJit`-wrapped full kernel for CPU / CLANG devices. |
| `calculate_sasa_batch_tinygrad(..., block_size=...)` | Per-block-realized dot-product kernel; each block is `.numpy()`-reduced into a buffer to keep tinygrad's graph-rewrite stack bounded. |

### 3.5 Contact analysis — `contacts.py`

- `calculate_residue_contacts(target_pos, binder_pos, target_mask, binder_mask, distance_cutoff=5.5)` — pairwise residue contact mask (JAX).
- `analyze_contacts(contacts, target_seq, binder_seq, class_matrix)` — projects residues onto (Aliphatic, Charged, Polar) and returns the 6-tuple `[AA, CC, PP, AC, AP, CP]`.
- `calculate_residue_contacts_tinygrad`, `analyze_contacts_tinygrad` — tinygrad equivalents (broadcast max/any replacements for the JAX reductions).

### 3.6 Scoring — `scoring.py`

PRODIGY IC-NIS constants live in `NIS_CONSTANTS`. Key functions:

- `get_atom_radii(seq_one_hot, radii_matrix)`
- `calculate_relative_sasa(complex_sasa, seq_probs, relative_sasa_array, atoms_per_residue)`
- `calculate_nis_percentages(sasa_values, seq_probs, character_matrix, threshold=0.05)`
- `score_ic_nis(ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c, coeffs, intercept)` — returns ΔG in kcal/mol.
- `dg_to_kd(dg, temperature=25.0)` — dissociation constant in M.
- `get_atom_radii_tinygrad`, `calculate_relative_sasa_tinygrad`, `calculate_nis_percentages_tinygrad`, `score_ic_nis_tinygrad`, `dg_to_kd_tinygrad`, `coefficient_tensors_tinygrad` — tinygrad equivalents operating on `Tensor`s.

### 3.7 Results — `results.py`

| Symbol | Purpose |
|--------|---------|
| `ContactAnalysis(values)` | Wrap 6-tuple of contact counts; `.to_dict()` adds totals and grouped counts (`IC`, `chargedC`, `polarC`, `aliphaticC`). |
| `ProdigyResults` | Dataclass with ΔG, Kd, NIS percentages, contacts, and a structured `sasa_data` array. `to_dict()` / `save_results(output_dir)` / `__str__` for reports. |
| `build_sasa_records(...)` | Build the structured `sasa_data` array from JAX outputs. |
| `NumpyEncoder` | JSON encoder tolerant of numpy scalars/arrays. |

### 3.8 Resources — `resources.py`

- `data_path(filename)` — context manager yielding a concrete path to a packaged data file.
- `read_text_resource(filename)` — read a packaged file as text.
- `collect_structure_files(path)` — return a sorted list of supported structure files from a file or directory (accepts `.pdb`, `.ent`, `.cif`, `.mmcif`).
- `format_duration(seconds)` — short human-readable duration.

### 3.9 Utilities — `utils/`

- `residue_constants` — AlphaFold-derived lookups (`restypes`, `restype_1to3`, `atom_types`, `atom_type_num`, `STANDARD_ATOM_MASK`, chi definitions, rigid groups, …).
- `ResidueClassification(kind="protorp" | "ic")` — character matrix, cached indices, reference relative SASA array.
- `ResidueLibrary` — parses `data/vdw.radii`, exposes `get_radius`, `is_polar`, and a `[n_restypes, n_atoms]` radii matrix. A module-level `default_library` is pre-built.

### 3.10 Logging helpers — `logging_utils.py`

- `setup_logging(level="INFO", *, propagate=True)` — attach a `StreamHandler`
  to the package logger (`protein_affinity_gpu`) with an
  `HH:MM:SS [LEVEL] name: message` format. Idempotent — safe to call from
  notebooks or the CLI.
- `get_logger(name)` — returns a namespaced child of the package logger, e.g.
  `get_logger(__name__)` inside a submodule.
- `log_duration(logger, label, *, level=logging.DEBUG, extra=None)` — context
  manager that emits `"<label>: start"` and `"<label>: done in X.XXXs"` at
  the given level. Used to tag every phase of the tinygrad pipeline
  (`tinygrad.load_complex`, `tinygrad.contacts`, `tinygrad.sasa_batch`,
  `tinygrad.nis`, `tinygrad.score`).

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

### 5.1 IC-NIS coefficients (`scoring.NIS_CONSTANTS`)

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

from protein_affinity_gpu import (
    load_complex,
    predict_binding_affinity,
    predict_binding_affinity_jax,
    predict_binding_affinity_tinygrad,
)
from protein_affinity_gpu.logging_utils import setup_logging

setup_logging("DEBUG")  # per-phase timings from the JAX / tinygrad pipelines

structure = Path("benchmarks/fixtures/1A2K.pdb")

target, binder = load_complex(structure, selection="A,B")
print(target.atom_positions.shape, binder.atom_positions.shape)

cpu = predict_binding_affinity(structure, selection="A,B")
jax_result = predict_binding_affinity_jax(structure, selection="A,B")
tg_result = predict_binding_affinity_tinygrad(structure, selection="A,B")

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
