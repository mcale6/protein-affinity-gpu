# protein-affinity-gpu — Project Index

> Generated project knowledge index for `protein-affinity-gpu` v1.6.9.
> Source of truth: [README.md](../README.md), [pyproject.toml](../pyproject.toml).

`protein-affinity-gpu` is a Python package for protein–protein binding
affinity prediction with two default backends:

- **CPU** — a thin wrapper around [`prodigy-prot`] + [`freesasa`] that reproduces
  the PRODIGY IC-NIS model.
- **JAX** — a vectorized re-implementation of the same pipeline that runs on
  CPU, CUDA, or Apple Metal via [JAX]. The default surface exposes a blocked
  Shrake–Rupley kernel (`mode="block"`) and a `lax.scan`-fused variant
  (`mode="scan"`) — both share the same `@jit` per-block body.

Both backends share one data model ([`Protein`](../src/protein_affinity_gpu/utils/structure.py),
[`ProdigyResults`](../src/protein_affinity_gpu/results.py)) so results are
directly comparable. The JAX backend lives in
[`predict.py`](../src/protein_affinity_gpu/predict.py), parametrized by a
[`BackendAdapter`](../src/protein_affinity_gpu/backends/_adapter.py).

Experimental kernels (tinygrad, single-pass / neighbor-cutoff JAX,
differentiable soft-SASA) and their benchmark harness live behind
`protein_affinity_gpu.experimental` — see
[EXPERIMENTAL.md](EXPERIMENTAL.md).

---

## 1. Navigation

| Area | Entry Point |
|------|-------------|
| Install / quickstart | [README.md](../README.md) |
| Package metadata | [pyproject.toml](../pyproject.toml) |
| Python API root | [src/protein_affinity_gpu/__init__.py](../src/protein_affinity_gpu/__init__.py) |
| Unified predictor | [src/protein_affinity_gpu/predict.py](../src/protein_affinity_gpu/predict.py) |
| CPU predictor | [src/protein_affinity_gpu/cpu.py](../src/protein_affinity_gpu/cpu.py) |
| Default SASA kernels | [src/protein_affinity_gpu/sasa.py](../src/protein_affinity_gpu/sasa.py) |
| Backend adapters | [src/protein_affinity_gpu/backends/](../src/protein_affinity_gpu/backends/) |
| Experimental surface | [src/protein_affinity_gpu/experimental.py](../src/protein_affinity_gpu/experimental.py) · [docs](EXPERIMENTAL.md) |
| Logging helpers | [src/protein_affinity_gpu/utils/logging_utils.py](../src/protein_affinity_gpu/utils/logging_utils.py) |
| Structure loader | [src/protein_affinity_gpu/utils/structure.py](../src/protein_affinity_gpu/utils/structure.py) |
| CLI — predict | [src/protein_affinity_gpu/cli/predict.py](../src/protein_affinity_gpu/cli/predict.py) |
| Default benchmark | [benchmarks/benchmark.py](../benchmarks/benchmark.py) |
| Experimental benchmark | [benchmarks/benchmark_experimental.py](../benchmarks/benchmark_experimental.py) |
| Test suite | [tests/](../tests) |
| Release script | [update_pkg.sh](../update_pkg.sh) |

---

## 2. Repository layout

```
protein-affinity-gpu/
├── pyproject.toml             # Hatchling build, entry points, optional extras
├── README.md                  # User-facing quickstart
├── LICENSE
├── update_pkg.sh              # Bump version + build sdist/wheel
├── benchmarks/
│   ├── benchmark.py              # Default harness: CPU / JAX (block, scan) + memory profiling
│   ├── benchmark_experimental.py # Full sweep incl. tinygrad / single / neighbor / soft
│   ├── run.py                    # Standalone entry into the default benchmark
│   └── fixtures/1A2K.pdb         # Canonical two-chain complex used by tests
├── src/protein_affinity_gpu/
│   ├── __init__.py            # Public API (lazy-loads impls)
│   ├── version.py             # __version__ (read by Hatch)
│   ├── predict.py             # Unified pipeline + `predict(backend=…)` router + jax entry point
│   ├── experimental.py        # Experimental entry points (tinygrad, jax-experimental)
│   ├── cpu.py                 # PRODIGY + freesasa CPU pipeline
│   ├── sasa.py                # Default SASA kernels (JAX block + scan)
│   ├── sasa_experimental.py   # Experimental SASA kernels (tinygrad, soft, single, neighbor)
│   ├── contacts.py            # Residue contacts + interaction class counts
│   ├── scoring.py             # NISCoefficients + backend-agnostic scoring primitives
│   ├── results.py             # ProdigyResults / ContactAnalysis, JSON writer
│   ├── backends/
│   │   ├── _adapter.py            # BackendAdapter Protocol
│   │   ├── _jax.py                # JAXAdapter (block / scan)
│   │   ├── _jax_experimental.py   # JAXExperimentalAdapter (+ soft / single / neighbor)
│   │   └── _tinygrad.py           # TinygradAdapter (experimental)
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
    predict,                           # unified router: backend="cpu"|"jax"
    predict_binding_affinity,          # CPU-only (legacy alias)
    predict_binding_affinity_jax,      # JAX: mode="block" | "scan"
)
```

The JAX entry point is lazy-loaded at first call so importing the package
doesn't pull JAX into memory.

Tinygrad and the extended JAX modes (single-pass, neighbor-cutoff, soft)
live on the experimental surface — import them from
`protein_affinity_gpu.experimental`. See [EXPERIMENTAL.md](EXPERIMENTAL.md).

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
    struct_path, backend="jax",       # "cpu" | "jax"
    selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    # backend-specific extras flow through **backend_kwargs, e.g. mode="scan"
) -> ProdigyResults
```

`_run_pipeline(adapter, …)` is the shared body consumed by both the router
and the backend-specific shim below. Each phase (`load_complex`,
`contacts`, `sasa`, `nis`, `score`) is wrapped in `log_duration`, so enabling
debug logging prints a start / done-in-X.XXXs line per phase.

### 3.3a Backend adapter — `backends/_jax.py`

The adapter Protocol ([`_adapter.py`](../src/protein_affinity_gpu/backends/_adapter.py))
names the surface the pipeline calls. The default JAX adapter owns its
device resolution, lazy constants, and kernel dispatch:

| Adapter | Notable behavior |
|---------|------------------|
| `JAXAdapter` | `mode={"block","scan"}` selects the SASA dispatch (default `"block"`). `"block"` runs the `@jit`'d per-block kernel in a Python loop. `"scan"` compiles the whole blocked sweep as one `lax.scan` program (AlphaFold `layer_stack` pattern, pairs with `jax.checkpoint`). `validate_size` calls `nvidia-smi` on CUDA; block size uses an exp-decay fit on Metal, ~1 GB scratch target otherwise. |

For tinygrad, single-pass, neighbor-cutoff, and soft-SASA adapters, see
[EXPERIMENTAL.md §2](EXPERIMENTAL.md).

### 3.3b JAX entry point (in `predict.py`)

```python
predict_binding_affinity_jax(
    struct_path, selection="A,B",
    distance_cutoff=5.5, acc_threshold=0.05,
    temperature=25.0, sphere_points=100,
    save_results=False, output_dir=".", quiet=True,
    mode="block",            # "block" | "scan"
) -> ProdigyResults
```

Constructs `JAXAdapter(mode=mode)` and delegates to `_run_pipeline`.

### 3.4 SASA kernels — `sasa.py`

| Function | Notes |
|----------|-------|
| `generate_sphere_points(n)` | Golden-spiral sphere point distribution as `[n, 3]` float32 numpy. Adapters wrap with their native tensor type. |
| `calculate_sasa_batch(coords, vdw_radii, mask, block_size, sphere_points, probe_radius=1.4)` | Blocked Shrake–Rupley (JAX): Python dispatcher over a `@jit`'d per-block kernel using `|a−b|² = a² + b² − 2⟨a,b⟩`. `[B, N]` inter-mask computed inline via `block_abs_idx` — no upfront `[N, N]` realize. Tail block uses `effective_start` so the kernel compiles once. |
| `calculate_sasa_batch_scan(...)` | Same blocked kernel as `calculate_sasa_batch`, dispatched via `jax.lax.scan` so the whole sweep compiles as one program; body is wrappable with `jax.checkpoint` for AlphaFold-style memory-efficient backprop. |
| `calculate_sasa_jax(coords, vdw_radii, mask, sphere_points, probe_radius=1.4)` | Fully-fused single-pass `@jit` SASA — `[N, M, N]` peak scratch. Emits an `info`/`warning` log (via `_log_single_pass_scratch`) estimating the scratch size so OOMs are obvious. Reached through `JAXExperimentalAdapter(mode="single")`. |
| `calculate_sasa_batch_tinygrad(..., block_size=...)` | Per-block `TinyJit` kernel with per-`(block, N, M)` cache; per-block output is detached to numpy on each iteration to dodge TinyJit's persistent output-buffer aliasing. Default path on accelerator tinygrad devices via `TinygradAdapter(mode="block")`. |

Each wrapper calls `_log_device_memory(tag)` after `block_until_ready()` so
the reading reflects the actual compute. The log line is a single
`key=value` string — `rss_mb` from `resource.getrusage` on any platform,
`jax_in_use_mb` / `jax_peak_mb` from `jax.devices()[0].memory_stats()` on
GPU. Tags: `jax.sasa.block`, `jax.sasa.scan`, `jax.sasa.single`,
`tinygrad.sasa.block`. Enable via `setup_logging("INFO")` or `--verbose` in
the CLIs.

Experimental kernels (`calculate_sasa_jax_soft`, `calculate_sasa_jax_neighbor`,
`calculate_sasa_batch_soft`, `calculate_sasa_batch_scan_soft`,
`calculate_sasa_tinygrad`, `calculate_sasa_tinygrad_neighbor`) are
documented in [EXPERIMENTAL.md §3](EXPERIMENTAL.md).

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
- `--backend tinygrad` lazy-loads the experimental tinygrad adapter; the
  default-surface backends are `cpu` and `jax`.

### 4.2 `protein-affinity-benchmark`

```bash
protein-affinity-benchmark <input_path> \
    [--output-dir benchmarks/output] \
    [--repeats 3] \
    [--targets cpu cuda jax jax-scan] \
    [--selection A,B] \
    [--temperature 25.0] [--distance-cutoff 5.5] \
    [--acc-threshold 0.05] [--sphere-points 100] [--verbose]
```

- The default harness ([`benchmarks/benchmark.py`](../benchmarks/benchmark.py))
  runs the `cpu`, `jax` (blocked), and `jax-scan` targets and additionally
  records peak process RSS and (on GPU) JAX `peak_bytes_in_use` per run.
- `cuda` is auto-skipped if no GPU/CUDA device is found.
- Writes `<output-dir>/benchmark_results.json` and echoes the report to stdout.
- For tinygrad / single-pass / neighbor-cutoff / soft-SASA targets, use the
  experimental harness in
  [`benchmarks/benchmark_experimental.py`](../benchmarks/benchmark_experimental.py)
  (see [EXPERIMENTAL.md §4](EXPERIMENTAL.md)).

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
| `test_imports.py` | Top-level re-exports and CLI modules import; experimental surface callables. |
| `test_structure.py` | `load_complex` sanitizes H, water, and non-selected chains. |
| `test_regression.py` | CPU vs JAX prediction stay within `|ΔΔG| < 0.75` and `|ΔIC| < 10`. Skips if JAX / prodigy-prot / freesasa are missing. |
| `test_tinygrad_smoke.py` | Tinygrad (experimental) prediction returns finite ΔG, within `|ΔΔG| < 0.75` and `|ΔIC| < 10` of the CPU reference. |
| `test_benchmark_smoke.py` | Benchmark harness runs with mocked predictor; CUDA is reported as skipped when unavailable. |
| `test_resources.py` | Packaged `vdw.radii` is accessible via `read_text_resource`. |
| `test_residue_library.py` | `ResidueLibrary.radii_matrix` has the expected shape. |
| `test_results.py` | `ProdigyResults.save_results` round-trips through JSON. |

Run with:

```bash
python3 -m pip install -e ".[dev]"
python3 -m pytest
```

---

## 7. Install & release

### 7.1 Dependencies (`pyproject.toml`)

- Core: `biopython`, `prodigy-prot`, `freesasa`, `numpy>=1.23,<3.0`, `jax`,
  `jaxlib`, `tinygrad`, `matplotlib`, `pandas`. A single
  `pip install protein-affinity-gpu` brings in every backend (including the
  experimental tinygrad surface) and the benchmarking plot stack.
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
jax_scan = predict(structure, backend="jax", selection="A,B", mode="scan")

print(cpu)
print(f"ΔΔG (CPU vs JAX block) = {cpu.binding_affinity - jax_result.binding_affinity:+.3f}")
print(f"ΔΔG (CPU vs JAX scan)  = {cpu.binding_affinity - jax_scan.binding_affinity:+.3f}")
```

The `ProdigyResults` returned by any backend serializes to the same JSON
schema, keyed by `structure_id`, with top-level fields `ba_val`, `kd`,
`contacts`, `nis`, and a per-atom `sasa_data` list.

[`prodigy-prot`]: https://github.com/haddocking/prodigy
[`freesasa`]: https://freesasa.github.io/
[JAX]: https://github.com/jax-ml/jax
[tinygrad]: https://github.com/tinygrad/tinygrad
