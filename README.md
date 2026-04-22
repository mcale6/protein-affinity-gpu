# protein-affinity-gpu

`protein-affinity-gpu` is a research-friendly Python package for protein-protein binding affinity prediction, solvent-accessible surface area (SASA) analysis, and reproducible CPU/JAX/tinygrad benchmarking.

Three first-class backends — CPU (freesasa via PRODIGY), JAX (blocked
and `lax.scan`-fused Shrake–Rupley), and tinygrad (per-shape `TinyJit`
block kernel on METAL / CUDA / GPU; full fused kernel on CPU / CLANG).
Stable differentiable helpers for AFDesign-style losses live in
[`protein_affinity_gpu.sasa_soft`](src/protein_affinity_gpu/sasa_soft.py),
[`protein_affinity_gpu.contacts_soft`](src/protein_affinity_gpu/contacts_soft.py),
[`protein_affinity_gpu.scoring_soft`](src/protein_affinity_gpu/scoring_soft.py),
and [`protein_affinity_gpu.af_design`](src/protein_affinity_gpu/af_design.py).
Experimental JAX modes (single-pass, neighbor-cutoff) remain documented in
[docs/EXPERIMENTAL.md](docs/EXPERIMENTAL.md). See
[docs/AF_DESIGN.md](docs/AF_DESIGN.md) for the soft-vs-hard design notes.

## Installation

The package is currently installed from source. Clone the repo and sync
with [uv](https://docs.astral.sh/uv/) for a reproducible environment
pinned against `uv.lock`:

```bash
uv sync                # core deps into .venv/, honouring uv.lock
uv sync --extra modal  # adds modal for the GPU benchmark entrypoint
```

The core deps already cover CPU (`prodigy-prot`, `freesasa`), JAX (`jax`,
`jaxlib`), tinygrad, and the plot stack (`matplotlib`, `pandas`).
`pyproject.toml` declares unpinned ranges, `uv.lock` is the exact pinned
resolution, and `.venv/` is a local (gitignored) virtualenv `uv`
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
scripts live in [`benchmarks/`](benchmarks) and are invoked directly.
See [Modal Benchmark](#modal-benchmark) below.

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

# Stable differentiable helpers for design-time losses:
from protein_affinity_gpu.af_design import add_ba_val_loss
from protein_affinity_gpu.sasa_soft import calculate_sasa_batch_scan_soft

# Experimental (tinygrad / single / neighbor entry points) — see docs/EXPERIMENTAL.md:
from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad
tg_result = predict_binding_affinity_tinygrad(structure, selection="A,B")
```

The public surface exported from `protein_affinity_gpu.__init__` is:
`__version__`, `ContactAnalysis`, `Protein`, `ProdigyResults`,
`load_complex`, `load_structure`, `predict`, `predict_binding_affinity`,
`predict_binding_affinity_jax`.

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

## SASA Algorithm — Shrake–Rupley vs Lee–Richards

All three backends implement the **Shrake–Rupley** rolling-probe algorithm
(1973). For each atom, a fixed set of points is distributed on the
expanded van der Waals sphere (radius `r_atom + r_probe`, `r_probe = 1.4 Å`
by default); each point is tested against every neighbouring atom, and
the accessible area is the fraction of non-occluded points times the
sphere area. The CPU path delegates to
[freesasa](https://freesasa.github.io/) through PRODIGY; the JAX and
tinygrad paths reimplement the same algorithm as a vectorised kernel
over an `[N_atoms, N_sphere_points]` tensor so it runs on GPU.

The classical alternative is **Lee–Richards** (1971), which computes the
exact accessible surface analytically by constructing and clipping arcs
where the rolling-probe sphere slides across neighbour contacts. For a
given probe radius the Lee–Richards answer is mathematically exact (no
sphere-point discretisation error), whereas Shrake–Rupley error scales
as ~`1/sqrt(N_sphere_points)`. We use Shrake–Rupley anyway because it is
embarrassingly parallel — the entire computation reduces to elementwise
distance ops, an occlusion comparison, and a sum over the sphere-point
axis — which maps cleanly onto `jnp.einsum` / tinygrad `Tensor` ops and
onto the blocked and fused kernels documented above. Lee–Richards
requires per-atom branching, 2D arc bookkeeping, and neighbour-graph
traversal, none of which vectorise well on GPU. At the default
`--sphere-points 100` the Shrake–Rupley estimate is already within
Pearson `r = 0.9999` of freesasa's own Shrake–Rupley reference and
downstream PRODIGY metrics (ΔG, Kd, NIS, contact classes) match to
`r = 1.000`, so the integration error is well below the noise floor of
the scoring function.

## Van der Waals Radii

SASA computation uses a NACCESS-style van der Waals radii library
shipped at [`src/protein_affinity_gpu/data/vdw.radii`](src/protein_affinity_gpu/data/vdw.radii)
and loaded by `protein_affinity_gpu.utils.residue_library`. The file is a
plain text per-atom table — per-residue `ATOM <name> <radius> <polar>`
lines — that can be swapped for any NACCESS-formatted library (e.g. a
Bondi set, or a user-patched radius for a non-standard residue) by
editing the file in place before installing, or by overriding
`residue_library.default_library` at import time.

## Modal Benchmark

The comparison figure below is committed at
[`docs/assets/comparison_figure.png`](docs/assets/comparison_figure.png).
It is the merged output of one local Apple M1 Max run and one Modal A100-80GB
run, plotted with `benchmarks/plot_results.py` (which writes into the
gitignored `benchmarks/output/combined/` and is copied into
`docs/assets/` for the README).

![Backend comparison on Kahraman 2013 T3](docs/assets/comparison_figure.png)

### What was compared

Six SASA backends over the 16-complex Kahraman 2013 T3 manifest
([`benchmarks/datasets/kahraman_2013_t3.tsv`](benchmarks/datasets/kahraman_2013_t3.tsv)),
padded N ∈ [2.5k, 12.8k] atom14-atoms:

| Backend | Device | Kernel |
|---------|--------|--------|
| `cpu` | M1 Max CPU | freesasa via PRODIGY |
| `tinygrad-block` | M1 Max Metal | per-shape `TinyJit` blocked Shrake–Rupley |
| `tinygrad-single` | A100-80GB | fully fused `[N, M, N]` kernel |
| `tinygrad-batch` | A100-80GB | blocked kernel with `block = min(768, N)` |
| `jax-block` | A100-80GB | blocked Shrake–Rupley |
| `jax-scan` | A100-80GB | `lax.scan`-fused variant |
| `jax-single` | A100-80GB | fully fused single-pass kernel |

### Commands

```bash
# Local M1 Max: cpu + tinygrad-batch + tinygrad-single
.venv/bin/python benchmarks/benchmark.py \
    --manifest benchmarks/datasets/kahraman_2013_t3.tsv \
    --structures-dir benchmarks/downloads/kahraman_2013_t3 \
    --output-dir benchmarks/output/local \
    --targets cpu tinygrad-batch tinygrad-single

# Remote A100-80GB: jax-single,batch,scan + tinygrad-single,batch
modal run benchmarks/modal_benchmark.py \
    --repeats 2 --run-name kahraman-a100 \
    --targets jax-single,jax-batch,jax-scan,tinygrad-single,tinygrad-batch \
    --local-output-dir benchmarks/output/gpu

# Merge the two CSVs into one figure (earlier CSVs win on shared columns —
# passing the GPU file first keeps A100 numbers and fills only CPU-only
# columns from the local run).
.venv/bin/python benchmarks/plot_results.py \
    benchmarks/output/gpu/results.csv \
    benchmarks/output/local/results.csv \
    --output-dir benchmarks/output/combined \
    --figure-name comparison_figure.png
```

### Observed differences and why

- **All 6 GPU backends agree with CPU to Pearson `r = 0.9999`** on
  per-structure SASA totals once TF32 is disabled. Without the fix, JAX
  on A100 drifts per-structure because the Shrake–Rupley kernels use the
  `dist² = ‖a‖² + ‖b‖² − 2·⟨a,b⟩` identity — a classic catastrophic
  cancellation trap. TF32's ~10-bit mantissa in `@`/`einsum` flips
  buried/not-buried sphere-point votes near the threshold. We set
  `JAX_DEFAULT_MATMUL_PRECISION=highest` in the Modal image
  ([`benchmarks/modal_benchmark.py`](benchmarks/modal_benchmark.py))
  and as a module-level `jax.config.update(...)` in
  [`src/protein_affinity_gpu/af_design.py`](src/protein_affinity_gpu/af_design.py)
  so the design loss inherits it locally too. Tinygrad-Metal has no TF32
  path and was already correct.

- **JAX compile caches accumulate across distinct shapes** and pin device
  scratch, so a 16-structure sweep can OOM on 80 GB even when each
  individual structure fits. The sweep loop calls
  `clear_accelerator_caches()` (in
  [`benchmarks/sasa/sasa_benchmark.py`](benchmarks/sasa/sasa_benchmark.py)),
  which runs `jax.clear_caches()` + tinygrad `TinyJit` cache drops
  + `gc.collect()` between structures.

- **Two largest structures still OOM `jax-single`** (N = 12810 and
  N = 10738 → fused scratch of 66 GB / 46 GB). That is a physics limit of
  the fused kernel, not a cache issue — use `jax-block` or `jax-scan` for
  those.

- **tinygrad CUDA needs a real CUDA base image** on Modal. The slim image
  with pip-only CUDA wheels does not provide the `libcuda.so` /
  `libnvrtc.so` symlinks that tinygrad's lazy `ctypes` bindings expect,
  and every call fails with `cuInit` missing. Both Modal scripts build on
  `nvidia/cuda:12.4.1-runtime-ubuntu22.04` with `add_python="3.11"`.

### Setup

```bash
uv sync --extra modal
modal setup
```

### Artifact download

If you did not pass `--local-output-dir`, download the volume contents
with:

```bash
modal volume get protein-affinity-gpu-benchmarks \
    runs/kahraman-a100 benchmarks/output/modal-kahraman-a100
```

## References

- **PRODIGY** — Xue, L.C., Rodrigues, J.P., Kastritis, P.L., Bonvin,
  A.M.J.J., Vangone, A. *PRODIGY: a web server for predicting the binding
  affinity of protein-protein complexes.* Bioinformatics 32(23), 3676–3678
  (2016). <https://doi.org/10.1093/bioinformatics/btw514>. The IC-NIS
  scoring model and the (aliphatic/charged/polar) × contact-class scheme
  implemented in `protein_affinity_gpu.scoring` / `.contacts` follow this
  paper.
- **freesasa** — Mitternacht, S. *FreeSASA: An open source C library for
  solvent accessible surface area calculations.* F1000Research 5:189
  (2016). <https://doi.org/10.12688/f1000research.7931.1>. The CPU
  backend calls freesasa through PRODIGY and is the per-atom SASA
  ground truth the JAX / tinygrad kernels are validated against
  (`r = 0.9999` on the Kahraman 2013 T3 set).
- **Shrake–Rupley algorithm** — Shrake, A., Rupley, J.A. *Environment
  and exposure to solvent of protein atoms. Lysozyme and insulin.*
  J. Mol. Biol. 79(2), 351–371 (1973).
  <https://doi.org/10.1016/0022-2836(73)90011-9>. Rolling-probe
  sphere-point integration — the algorithm implemented in
  `protein_affinity_gpu.sasa`.
- **Lee–Richards algorithm (alternative, analytical)** — Lee, B.,
  Richards, F.M. *The interpretation of protein structures: Estimation
  of static accessibility.* J. Mol. Biol. 55(3), 379–400 (1971).
  <https://doi.org/10.1016/0022-2836(71)90324-X>. Exact arc-clipping
  analytical SASA. Not used here because the per-atom arc bookkeeping
  is hard to vectorise on GPU; listed for completeness as the classical
  alternative.
- **EDTSurf (alternative, grid / Euclidean Distance Transform)** — Xu,
  D., Zhang, Y. *Generating triangulated macromolecular surfaces by
  Euclidean Distance Transform.* PLoS ONE 4(12), e8140 (2009).
  <https://doi.org/10.1371/journal.pone.0008140>. Computes
  solvent-accessible / solvent-excluded / van der Waals surfaces on a
  regular 3D grid via EDT rather than per-atom sphere integration.
  Another candidate swap-in; not used here because a grid-resolution
  discretisation with per-grid-cell occlusion tests does not map as
  cleanly onto the existing `[N_atoms, N_sphere_points]` tensor kernels
  as Shrake–Rupley.
- **AlphaFold2** — Jumper, J. et al. *Highly accurate protein structure
  prediction with AlphaFold.* Nature 596, 583–589 (2021).
  <https://doi.org/10.1038/s41586-021-03819-2>. The `atom14` padded
  representation used throughout the JAX / tinygrad kernels and the
  AFDesign integration in `protein_affinity_gpu.af_design` follow the
  AlphaFold2 residue layout.
- **ColabDesign / AfDesign** — Krypton, S. et al. *ColabDesign:
  Making protein design accessible to all via Google Colab.*
  <https://github.com/sokrypton/ColabDesign>. The `add_ba_val_loss`
  helper in `protein_affinity_gpu.af_design` plugs into ColabDesign's
  AfDesign binder-hallucination protocol as an auxiliary loss; the
  Modal entrypoint at
  [`benchmarks/modal_afdesign_ba_val.py`](benchmarks/modal_afdesign_ba_val.py)
  installs `ColabDesign@v1.1.1` directly from the upstream repo.
- **NACCESS / Van der Waals radii** — Hubbard, S.J., Thornton, J.M.
  *NACCESS (computer program).* Department of Biochemistry and Molecular
  Biology, University College London (1993). The radii file at
  [`src/protein_affinity_gpu/data/vdw.radii`](src/protein_affinity_gpu/data/vdw.radii)
  uses the NACCESS standard residue library format; earlier source for
  the numerical values: Bondi, A. *van der Waals Volumes and Radii.*
  J. Phys. Chem. 68(3), 441–451 (1964).
  <https://doi.org/10.1021/j100785a001>.
- **Fibonacci / golden-spiral sphere** — Swinbank, R., Purser, R.J.
  *Fibonacci grids: A novel approach to global modelling.* Quarterly
  Journal of the Royal Meteorological Society 132, 1769–1793 (2006).
  <https://doi.org/10.1256/qj.05.227>. The quasi-uniform sphere-point
  generator in `protein_affinity_gpu.sasa.generate_sphere_points` uses
  the midpoint Fibonacci spacing to match freesasa's
  `sasa_sr.c::test_points()`.
- **Thomson-sphere / exact uniform sphere (alternative)** —
  Ribeiro-Filho, N. et al. *dr_sasa: Accurate algorithms for deriving
  surface areas and contacts in biomolecular assemblies.* J. Comp. Chem.
  (2019). <https://doi.org/10.1002/jcc.26049>. `dr_sasa` uses an exact
  Thomson-sphere (electrostatic-equilibrium) point set rather than the
  Fibonacci spiral; this is the more uniform alternative and a candidate
  swap-in for `generate_sphere_points` if per-atom sphere uniformity
  matters more than closed-form generation speed.
