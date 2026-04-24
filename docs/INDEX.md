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

Stable differentiable AFDesign helpers live in
[`af_design.py`](../src/protein_affinity_gpu/af_design.py),
[`contacts_soft.py`](../src/protein_affinity_gpu/contacts_soft.py),
[`scoring_soft.py`](../src/protein_affinity_gpu/scoring_soft.py), and
[`sasa_soft.py`](../src/protein_affinity_gpu/sasa_soft.py) — see
[AF_DESIGN.md](AF_DESIGN.md). Experimental kernels and entry points
(tinygrad, single-pass / neighbor-cutoff JAX, bucketed tinygrad) still live
behind `protein_affinity_gpu.experimental` — see [EXPERIMENTAL.md](EXPERIMENTAL.md).

PAE-gated residue contacts (structuremap-style, for scoring AlphaFold /
Boltz-2 predicted complexes) live in
[`contacts_pae.py`](../src/protein_affinity_gpu/contacts_pae.py) — a drop-in
replacement for `calculate_residue_contacts` that takes an inter-chain PAE
block. See [PAE.md](PAE.md) for the three-phase calibration plan and
[BOLTZ_PIPELINE.md](BOLTZ_PIPELINE.md) for the Kastritis-81 Boltz-2 driver.

---

## 1. Navigation

| Area | Entry Point |
|------|-------------|
| Install / quickstart | [README.md](../README.md) |
| Package metadata | [pyproject.toml](../pyproject.toml) |
| Python API root | [src/protein_affinity_gpu/__init__.py](../src/protein_affinity_gpu/__init__.py) |
| Unified predictor | [src/protein_affinity_gpu/predict.py](../src/protein_affinity_gpu/predict.py) |
| AFDesign helper | [src/protein_affinity_gpu/af_design.py](../src/protein_affinity_gpu/af_design.py) · [docs](AF_DESIGN.md) |
| AFDesign Modal entrypoint + plots | [af_design/](../af_design/) · [docs](AF_DESIGN.md) |
| AFDesign April 2026 runs | [benchmarks/output/afdesign_april2026/](../benchmarks/output/afdesign_april2026/) |
| CPU predictor | [src/protein_affinity_gpu/cpu.py](../src/protein_affinity_gpu/cpu.py) |
| Default SASA kernels | [src/protein_affinity_gpu/sasa.py](../src/protein_affinity_gpu/sasa.py) |
| Stable soft kernels | [src/protein_affinity_gpu/sasa_soft.py](../src/protein_affinity_gpu/sasa_soft.py) |
| PAE-gated contacts | [src/protein_affinity_gpu/contacts_pae.py](../src/protein_affinity_gpu/contacts_pae.py) · [docs](PAE.md) |
| Backend adapters | [src/protein_affinity_gpu/backends/](../src/protein_affinity_gpu/backends/) |
| Experimental surface | [src/protein_affinity_gpu/experimental.py](../src/protein_affinity_gpu/experimental.py) · [docs](EXPERIMENTAL.md) |
| Logging helpers | [src/protein_affinity_gpu/utils/logging_utils.py](../src/protein_affinity_gpu/utils/logging_utils.py) |
| Structure loader | [src/protein_affinity_gpu/utils/structure.py](../src/protein_affinity_gpu/utils/structure.py) |
| CLI — predict | [src/protein_affinity_gpu/cli/predict.py](../src/protein_affinity_gpu/cli/predict.py) |
| Local benchmark harness (M1 Max / CPU) | [benchmarks/benchmark.py](../benchmarks/benchmark.py) |
| Modal GPU benchmark harness | [benchmarks/modal_benchmark.py](../benchmarks/modal_benchmark.py) |
| Benchmark plot merger | [benchmarks/plot_results.py](../benchmarks/plot_results.py) |
| Shared benchmark helpers | [benchmarks/sasa/sasa_benchmark.py](../benchmarks/sasa/sasa_benchmark.py) |
| JAX block-size profiler | [benchmarks/sasa/profile_sasa.py](../benchmarks/sasa/profile_sasa.py) |
| Kastritis-81 Boltz-2 pipeline | [benchmarks/scripts/boltz_pipeline/](../benchmarks/scripts/boltz_pipeline/) · [docs](BOLTZ_PIPELINE.md) |
| PAE calibration scripts | [benchmarks/scripts/pae_calibration/](../benchmarks/scripts/pae_calibration/) · [docs](PAE.md) |
| ProteinBase pipeline | [benchmarks/scripts/proteinbase_pipeline/](../benchmarks/scripts/proteinbase_pipeline/) · [docs](PROTEINBASE_BENCHMARK.md) |
| Benchmark dataset — Kastritis 81 | [benchmarks/datasets/kastritis_81/](../benchmarks/datasets/kastritis_81/) |
| Benchmark dataset — Vreven BM5.5 (106 ΔG) | [benchmarks/datasets/vreven_bm55/](../benchmarks/datasets/vreven_bm55/) |
| Benchmark dataset — ProteinBase | [benchmarks/datasets/proteinbase/](../benchmarks/datasets/proteinbase/) |
| US-align binary (TM-score) | [benchmarks/tools/USalign](../benchmarks/tools/USalign) |
| Tinygrad SASA perf notes | [docs/TINYGRAD_SASA_OPTIMIZATION.md](TINYGRAD_SASA_OPTIMIZATION.md) |
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
│   ├── benchmark.py              # Local harness (Apple M1 Max / CPU): cpu + tinygrad single/batch
│   ├── modal_benchmark.py        # Modal GPU harness: jax single/batch/scan + tinygrad single/batch
│   ├── plot_results.py           # Merge N results.csv files + render 3-panel comparison figure
│   ├── sasa/
│   │   ├── sasa_benchmark.py     # Shared helpers (BACKENDS, run_benchmark, manifest/download)
│   │   └── profile_sasa.py       # Single-complex JAX block-size sweep (estimator vs observed)
│   ├── datasets/                 # Committed manifests / metadata per benchmark family
│   │   ├── kahraman_2013_t3.tsv
│   │   ├── kastritis_81/         # PRODIGY calibration set (81 complexes, experimental ΔG)
│   │   ├── vreven_bm55/          # Docking BM5.5 (257 rows; 106 ΔG-annotated subset)
│   │   └── proteinbase/          # ProteinBase snapshot (5,361 design-target rows)
│   ├── downloads/                # Git-ignored raw structures / CSVs fetched per pipeline
│   ├── scripts/                  # Multi-step research pipelines (one dir per family)
│   │   ├── boltz_pipeline/       # Kastritis-81 / Vreven Boltz-2 runner + MM-align + PRODIGY-on-Boltz
│   │   ├── pae_calibration/      # PAE-aware PRODIGY refits (union, stratified, ElasticNet, XGBoost…)
│   │   └── proteinbase_pipeline/ # ProteinBase: plot Kd↔ipTM, download CIFs, score PRODIGY + CAD-score-LT
│   ├── tools/
│   │   └── USalign              # Single-file C++ TM-score binary (rebuild via g++ -O3 USalign.cpp)
│   ├── output/                   # Benchmark outputs (mostly git-ignored — see §4.2)
│   └── fixtures/1A2K.pdb         # Canonical two-chain complex used by tests
├── af_design/
│   ├── modal_afdesign_ba_val.py       # Modal entrypoint: AfDesign binder hallucination + ba_val loss
│   ├── extract_interface_hotspots.py  # Crystal-interface → AFDesign --hotspot string (per-chain, top-k)
│   ├── plot_afdesign.py               # Unified plot CLI: traces | rmsd | animate | compare
│   └── input/                         # Input PDBs for Modal runs (e.g. 8hgo_AB.pdb)
├── src/protein_affinity_gpu/
│   ├── __init__.py            # Public API (lazy-loads impls)
│   ├── version.py             # __version__ (read by Hatch)
│   ├── predict.py             # Unified pipeline + `predict(backend=…)` router + jax entry point
│   ├── experimental.py        # Experimental entry points (tinygrad, jax-experimental)
│   ├── cpu.py                 # PRODIGY + freesasa CPU pipeline
│   ├── sasa.py                # Benchmarked SASA kernels: JAX single/batch/scan + tinygrad single/batch
│   ├── sasa_soft.py           # Stable differentiable JAX SASA kernels
│   ├── sasa_experimental.py   # Experimental SASA kernels (neighbor-cutoff, bucketed) + soft re-exports
│   ├── contacts.py            # Residue contacts + interaction class counts
│   ├── contacts_soft.py       # Stable differentiable residue-contact probabilities
│   ├── contacts_pae.py        # PAE-gated residue contacts (structuremap-style, for AF/Boltz complexes)
│   ├── scoring.py             # NISCoefficients + backend-agnostic scoring primitives
│   ├── scoring_soft.py        # Stable differentiable NIS thresholding
│   ├── af_design.py           # Stable AfDesign / ColabDesign loss helpers
│   ├── results.py             # ProdigyResults / ContactAnalysis, JSON writer
│   ├── backends/
│   │   ├── _adapter.py            # BackendAdapter Protocol
│   │   ├── _jax.py                # JAXAdapter (block / scan)
│   │   ├── _jax_experimental.py   # JAXExperimentalAdapter (+ soft / single / neighbor)
│   │   └── _tinygrad.py           # TinygradAdapter (experimental)
│   ├── cli/
│   │   └── predict.py         # `protein-affinity-predict` (sole installed console script)
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

Stable soft helpers are importable directly:

```python
from protein_affinity_gpu.af_design import add_ba_val_loss
from protein_affinity_gpu.contacts_soft import calculate_residue_contacts_soft
from protein_affinity_gpu.scoring_soft import calculate_nis_percentages_soft
from protein_affinity_gpu.sasa_soft import calculate_sasa_batch_scan_soft
```

Experimental entry points and kernels still live on the experimental surface —
import them from `protein_affinity_gpu.experimental`. See
[EXPERIMENTAL.md](EXPERIMENTAL.md).

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
| `calculate_sasa_jax(coords, vdw_radii, mask, sphere_points, probe_radius=1.4)` | Fully-fused single-pass `@jit` SASA — `[N, M, N]` peak scratch. Emits an `info`/`warning` log (via `_log_single_pass_scratch`) estimating the scratch size so OOMs are obvious. Reached through `JAXExperimentalAdapter(mode="single")`. |
| `calculate_sasa_batch(coords, vdw_radii, mask, block_size, sphere_points, probe_radius=1.4)` | Blocked Shrake–Rupley (JAX): Python dispatcher over a `@jit`'d per-block kernel using `|a−b|² = a² + b² − 2⟨a,b⟩`. `[B, N]` inter-mask computed inline via `block_abs_idx` — no upfront `[N, N]` realize. Tail block uses `effective_start` so the kernel compiles once. |
| `calculate_sasa_batch_scan(...)` | Same blocked kernel as `calculate_sasa_batch`, dispatched via `jax.lax.scan` so the whole sweep compiles as one program; body is wrappable with `jax.checkpoint` for AlphaFold-style memory-efficient backprop. |
| `calculate_sasa_tinygrad(coords, vdw_radii, mask, sphere_points, probe_radius=1.4)` | Tinygrad analog of `calculate_sasa_jax` — fully-fused single TinyJit pass, `[N, M, N]` peak scratch; JIT cached per `(N, M)` shape tuple. Reached through `TinygradAdapter(mode="single")`. |
| `calculate_sasa_batch_tinygrad(..., block_size=...)` | Per-block `TinyJit` kernel with per-`(block, N, M)` cache; per-block output is detached to numpy on each iteration to dodge TinyJit's persistent output-buffer aliasing. Default path on accelerator tinygrad devices via `TinygradAdapter(mode="block")`. |

Each wrapper calls `_log_device_memory(tag)` after `block_until_ready()` so
the reading reflects the actual compute. The log line is a single
`key=value` string — `rss_mb` from `resource.getrusage` on any platform,
`jax_in_use_mb` / `jax_peak_mb` from `jax.devices()[0].memory_stats()` on
GPU, `tg_mem_used_mb` from `tinygrad.helpers.GlobalCounters`. Tags:
`jax.sasa.single`, `jax.sasa.block`, `jax.sasa.scan`,
`tinygrad.sasa.single`, `tinygrad.sasa.block`. Enable via
`setup_logging("INFO")` or `--verbose` in the CLIs.

Stable differentiable SASA kernels
(`calculate_sasa_jax_soft`, `calculate_sasa_batch_soft`,
`calculate_sasa_batch_scan_soft`) and the reusable AFDesign `ba_val` helper
are documented in [AF_DESIGN.md](AF_DESIGN.md). Experimental kernels and
entry points are documented in [EXPERIMENTAL.md §3](EXPERIMENTAL.md).

### 3.5 Contact analysis — `contacts.py`

- `calculate_residue_contacts(target_pos, binder_pos, target_mask, binder_mask, distance_cutoff=5.5)` — pairwise residue contact mask; 5-D diff variant (JAX / numpy).
- `calculate_residue_contacts_tinygrad(…)` — matmul-reshape variant that sidesteps Metal's unified-memory pressure on the ``[N_t, N_b, 37, 37, 3]`` intermediate.
- `analyze_contacts(contacts, target_seq, binder_seq, class_matrix)` — backend-agnostic broadcast outer product, returns the 6-tuple `[AA, CC, PP, AC, AP, CP]`.

### 3.5a PAE-gated contacts — `contacts_pae.py`

Drop-in replacement for `calculate_residue_contacts` that folds an AlphaFold /
Boltz-2 inter-chain PAE block into the contact gate. Downstream
(`analyze_contacts`, NIS, IC-NIS linear model) is unchanged — **NIS is
intentionally not PAE-gated**.

| Symbol | Purpose |
|--------|---------|
| `load_pae_json(path)` | Parse AFDB v1–v2 (`distance`), AFDB v3+ / AF2-Multimer (`predicted_aligned_error`), or AF3 (`pae`) schemas into a square `[L, L]` fp32 array. |
| `slice_pae_inter(pae_full, target_len, binder_len, symmetrize=True)` | Extract the `[N_t, N_b]` inter-chain block; symmetrizes `0.5 * (upper + lower.T)` by default. |
| `calculate_residue_contacts_pae(..., pae_inter, pae_cutoff=10.0, gate_mode="confidence"\|"pessimistic")` | Mirrors the JAX 5-D diff kernel in `contacts.py` with two extra gates. `"confidence"` (default): independent `(dist ≤ cutoff) ∧ (PAE ≤ τ)`. `"pessimistic"`: structuremap-literal additive `(dist + PAE ≤ cutoff)`. |

See [PAE.md](PAE.md) for the three-phase plan (Phase 1 ✓ inference, Phase 2
Kastritis/Vreven calibration, Phase 3 AFDesign design-loop integration) and
[BOLTZ_PIPELINE.md](BOLTZ_PIPELINE.md) for the Boltz-2 driver that feeds
Phase 2. Calibration scripts live in
[`benchmarks/scripts/pae_calibration/`](../benchmarks/scripts/pae_calibration/).

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

Only `protein-affinity-predict` is registered as a console script in
[pyproject.toml](../pyproject.toml). Benchmarking and the research pipelines
(Boltz-2, PAE calibration, ProteinBase) are intentionally out of the
installed CLI — invoke them as scripts under `benchmarks/`.

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

### 4.2 Benchmark harnesses (`benchmarks/`)

Benchmarking is deliberately kept out of the installed CLI. Two runners
cover the `(local CPU)` + `(remote GPU)` split, and a third merges both
outputs into a single figure:

```bash
# Local — Apple M1 Max / any CPU box. cpu needs prodigy-prot + freesasa.
python benchmarks/benchmark.py \
    --manifest benchmarks/datasets/kahraman_2013_t3.tsv \
    --structures-dir benchmarks/downloads/kahraman_2013_t3 \
    --output-dir benchmarks/output/local \
    --targets cpu tinygrad-single tinygrad-batch

# Remote — Modal GPU (A100-80GB by default; set MODAL_GPU to override).
modal run benchmarks/modal_benchmark.py \
    --targets jax-single,jax-batch,jax-scan,tinygrad-single,tinygrad-batch \
    --local-output-dir benchmarks/output/gpu

# Merge + plot — takes N results.csv files, merges on pdb_id.
python benchmarks/plot_results.py \
    benchmarks/output/local/results.csv \
    benchmarks/output/gpu/results.csv \
    --output-dir benchmarks/output/combined
```

Both runners share
[`benchmarks/sasa/sasa_benchmark.py`](../benchmarks/sasa/sasa_benchmark.py),
so the CSV schema is identical and the two outputs merge cleanly. Each
row is `{pdb_id, chain1, chain2, n_atoms_atom14, device}` plus one
`{backend}_status / _error / _cold_seconds / _warm_mean_seconds /
_rss_peak_mb / _jax_peak_mb / _tg_mem_used_mb / _ba_val / _kd /
_sasa_sum / _contacts_* / _nis_*` column group per backend. Modal rejects
the `cpu` target because the image does not install `prodigy-prot` or
`freesasa`; run `cpu` locally.

#### Benchmark target modes

All six targets benchmark the same Shrake–Rupley algorithm through
different kernels and dispatchers. The two-axis view (algorithm × block
dispatch) reads across this table:

| Target | Kernel in `sasa.py` | Algorithm | Block dispatch | Peak scratch | Backprop-friendly |
|--------|---------------------|-----------|----------------|--------------|-------------------|
| `cpu` | — (`freesasa` via `prodigy-prot`) | Shrake–Rupley reference | N/A | — | ❌ |
| `jax-single` | `calculate_sasa_jax` | Fully fused, no blocking | — (no loop) | `[N, M, N]` fp32 (~57 GB at N=12 k, M=100) | ✅ |
| `jax-batch` | `calculate_sasa_batch` | Blocked | **Python** loop over a `@jit`'d per-block kernel | `[B, M, N]` per call | ✅ |
| `jax-scan` | `calculate_sasa_batch_scan` | Blocked (same kernel as `jax-batch`) | **`jax.lax.scan`** fuses the loop into one XLA program | `[B, M, N]` per block, one program | ✅ + `jax.checkpoint` (AlphaFold `layer_stack` pattern) |
| `tinygrad-single` | `calculate_sasa_tinygrad` | Fully fused, no blocking | — (no loop) | `[N, M, N]` fp32 | ❌ |
| `tinygrad-batch` | `calculate_sasa_batch_tinygrad` | Blocked | **Python** loop over a `TinyJit`'d per-block kernel | `[B, M, N]` per call | ❌ |

- **Cross-backend pairs** (same algorithm on different engines): `jax-single` ↔ `tinygrad-single`, `jax-batch` ↔ `tinygrad-batch`.
- **Algorithm comparison** (fused vs blocked, same engine): `jax-single` vs `jax-batch` / `jax-scan`; `tinygrad-single` vs `tinygrad-batch`.
- **Dispatcher comparison** (within JAX): `jax-batch` (Python-driven) vs `jax-scan` (XLA-fused). Tinygrad has no `lax.scan` equivalent, so the batched tinygrad kernel mirrors `jax-batch`.

The tinygrad block kernel caches one `TinyJit` per unique `(B, N, M)`
triple, and the tinygrad single-pass kernel caches per `(N, M)`; across a
sweep these pin ~1–4 GB of Metal scratch each. The shared harness calls
`sasa_benchmark.clear_tinygrad_caches()` between structures so Metal
does not return `Internal Error (0000000e)` under accumulated pressure.

See [TINYGRAD_SASA_OPTIMIZATION.md](TINYGRAD_SASA_OPTIMIZATION.md) for the
per-shape `TinyJit` caching trick that drove the ~53× Metal speed-up on 1A2K
and notes on a parity pitfall (don't fuse the final scaling into the JIT
kernel).

### 4.3 Research pipelines (`benchmarks/scripts/`)

Pipelines are ordered scripts — numbered prefixes signal run order. Inputs
are committed under `benchmarks/datasets/`, raw downloads go to git-ignored
`benchmarks/downloads/`, and artifacts to `benchmarks/output/<family>/`.

| Pipeline | Purpose | Driver docs |
|----------|---------|-------------|
| [`boltz_pipeline/`](../benchmarks/scripts/boltz_pipeline/) | Predict Kastritis-81 / Vreven BM5.5 complexes with Boltz-2 on Modal (A100-80GB, CUDA 13 base). Runs 01_prep → 03_build_boltz_yaml → 04_modal_boltz_predict → 05_mmalign_tm (US-align) → 05b_prodigy_on_boltz → 06_plot_boltz_eval. | [BOLTZ_PIPELINE.md](BOLTZ_PIPELINE.md) |
| [`pae_calibration/`](../benchmarks/scripts/pae_calibration/) | PAE-aware PRODIGY refits on K81 / Vreven 106 / unified 287. Scripts span union refits, threshold/stratified PAE calibrations, interaction-ablation, ElasticNet priors, XGBoost residuals, entropy surrogates, plDDT-NIS, and CAD-score-LT scoring. | [PAE.md](PAE.md) |
| [`proteinbase_pipeline/`](../benchmarks/scripts/proteinbase_pipeline/) | ProteinBase: plot Kd↔Boltz ipTM, download ModelCIF + PAE, score PRODIGY via tinygrad, CAD-score-LT (binder-only ESMFold ↔ Boltz complex-bound chain). | [PROTEINBASE_BENCHMARK.md](PROTEINBASE_BENCHMARK.md) |

Outputs consumed by downstream pipelines land in:

```
benchmarks/output/
├── kahraman_2013/{local,gpu}/results.csv       # Shrake–Rupley sweep
├── kastritis_81_boltz/                         # Boltz-2 msa_only / template_msa
├── vreven_bm55_boltz/                          # same shape as K81
├── proteinbase/                                # Kd↔ipTM, PRODIGY + CAD-score-LT
├── unified/                                    # K81 ∪ V106 ∪ PB feature tables
├── union_k81_v106/pae_calibration/             # Refitted PAE-aware coefficients
└── afdesign_april2026/                         # AFDesign hallucination runs (see AF_DESIGN.md)
```

Core calibration datasets are committed:

- [`benchmarks/datasets/kastritis_81/`](../benchmarks/datasets/kastritis_81/) — 81 complexes with experimental ΔG + PRODIGY baselines (`dataset.json`).
- [`benchmarks/datasets/vreven_bm55/`](../benchmarks/datasets/vreven_bm55/) — 257-row BM5.5 manifest + 106-row ΔG subset (joined from Kastritis 81 + Pierce Ab–Ag).
- [`benchmarks/datasets/proteinbase/`](../benchmarks/datasets/proteinbase/) — 5,361-row design-target manifest from the 28-Jan-2026 ProteinBase snapshot.

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
| `conftest.py` | Session-wide guard: forces `DEBUG=0` when the shell exports `DEBUG=release` (tinygrad requires integer). |
| `test_imports.py` | Top-level re-exports and CLI modules import; experimental surface callables. |
| `test_structure.py` | `load_complex` sanitizes H, water, and non-selected chains. |
| `test_cpu_selection.py` | `_select_structure_chains` detaches unselected chains before freesasa runs. |
| `test_regression.py` | CPU vs JAX prediction stay within `|ΔΔG| < 0.75` and `|ΔIC| < 10`. Skips if JAX / prodigy-prot / freesasa are missing. |
| `test_tinygrad_smoke.py` | Tinygrad (experimental) prediction returns finite ΔG, within `|ΔΔG| < 0.75` and `|ΔIC| < 10` of the CPU reference. |
| `test_bucketed_sasa.py` | Bucketed-padding SASA wrappers in `sasa_experimental` — numerical parity vs unbucketed, plus confirms the TinyJit cache keys on bucket `N` (not raw `N`). |
| `test_af_design_soft.py` | Soft SASA/contact/NIS helpers stay numerically close to the hard kernels; `add_ba_val_loss` is wired into AFDesign's loss registry. |
| `test_af_design_bsa.py` | AFDesign BSA logging: synthetic two-monomer JAX SASA recovers positive BSA in contact, ≈0 when far; `plot_afdesign rmsd --metric {rmsd,bsa,both}` accepts all three. |
| `test_benchmark_smoke.py` | Local benchmark runner produces `results.csv` + `summary.json` with the unified schema; unknown backend names are rejected. |
| `test_plot_results.py` | Multiple `results.csv` files merge on `pdb_id`; figure render path writes a non-empty PNG. |
| `test_sasa_benchmark.py` | Shared benchmark helpers: backend registry, manifest round-trip, atom-count + memory snapshot + metrics extraction. |
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
  `pip install protein-affinity-gpu` (or `uv sync`) brings in every backend
  (including the experimental tinygrad surface) and the benchmarking plot
  stack.
- `[dev]` extra: `build`, `pytest>=8.0`, `ruff>=0.6`.
- `[modal]` extra: `modal` — only needed for the GPU-remote entry points
  (`benchmarks/modal_benchmark.py`, `benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py`,
  `af_design/modal_afdesign_ba_val.py`).

The repo pins an exact resolution via `uv.lock`; `uv sync` (or
`uv sync --extra modal`) materializes `.venv/` from the lock.

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
