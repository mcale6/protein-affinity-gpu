# Kastritis 81 Boltz-2 pipeline — scripts

Run in order. See [docs/BOLTZ_PIPELINE.md](../../../docs/BOLTZ_PIPELINE.md) for the full design.

| # | Script | Purpose | Status |
|---|---|---|---|
| 1 | `01_prep_kastritis.py` | Rechain + renumber + extract sequences → `manifest.csv` + `cleaned/*.pdb` | **Ready** |
| 2 | `02_fetch_msas.py` | Pre-fetch A3M per unique sequence | **Skipped** — using `--use_msa_server` at runtime |
| 3 | `03_build_boltz_yaml.py` | `manifest.csv` → Boltz YAMLs (2 modes × 81 = 162) + CIF templates (gemmi, with populated `_entity_poly_seq`) | **Ready** |
| 4 | `04_modal_boltz_predict.py` | Modal Boltz-2 runner on A100-80GB (CUDA 13 base + gcc for triton JIT); `--limit N` / `--pdb-ids X,Y` | **Ready** |
| 5 | `05_mmalign_tm.py` | US-align predicted vs. crystal → `tm_scores.csv` (also parses Boltz confidence JSON for ipTM/pTM/pLDDT) | **Ready** |
| 5b | `05b_prodigy_on_boltz.py` | Standard PRODIGY on each Boltz structure → `prodigy_scores.csv` (ΔG_pred, IC counts, NIS%) | **Ready** |
| 6 | `06_plot_boltz_eval.py` | 3-panel figure: (1) ipTM vs TM, (2) ΔG_pred vs ΔG_exp (std PRODIGY), (3) ΔG_pred vs ΔG_exp (PAE-aware — Phase 2 placeholder) | **Ready** |

## Tooling

- `benchmarks/tools/USalign` — single-file C++ binary compiled from
  [zhanggroup.org/US-align](https://zhanggroup.org/US-align/), multimer-capable,
  modern successor to MM-align. Run `g++ -O3 -ffast-math -o USalign USalign.cpp`
  on `USalign.cpp` to rebuild.

## Quick start

```bash
# Step 1 — prep (once): rechain + renumber + manifest.
python benchmarks/scripts/boltz_pipeline/01_prep_kastritis.py

# Step 3 — build YAMLs + CIFs (once).
python benchmarks/scripts/boltz_pipeline/03_build_boltz_yaml.py

# Step 4 — Modal smoke run (2 predictions: 1 complex x 2 modes, default).
modal run benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py

# Scale up: pick specific PDBs or bump --limit.
modal run benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py --pdb-ids 2OOB,3BZD
modal run benchmarks/scripts/boltz_pipeline/04_modal_boltz_predict.py --limit 81

# Step 5: TM-score (predicted vs crystal) + parse Boltz confidence.
python benchmarks/scripts/boltz_pipeline/05_mmalign_tm.py

# Step 5b: run standard PRODIGY on each Boltz-predicted structure.
python benchmarks/scripts/boltz_pipeline/05b_prodigy_on_boltz.py

# Step 6: 3-panel evaluation plot.
python benchmarks/scripts/boltz_pipeline/06_plot_boltz_eval.py
```

## Output layout

```
benchmarks/output/kastritis_81_boltz/
├── msa_only/
│   ├── {PDB}.tar.gz
│   └── {PDB}_msa_only/boltz_results_input/predictions/input/
│       ├── input_model_0.cif         # predicted structure
│       ├── confidence_input_model_0.json  # ipTM, pTM, pLDDT, ...
│       ├── pae_input_model_0.npz     # feeds Phase 2 PRODIGY-PAE
│       ├── plddt_input_model_0.npz
│       └── pde_input_model_0.npz
├── template_msa/...
├── tm_scores.csv       # step 5
├── prodigy_scores.csv  # step 5b
└── boltz_eval.{png,pdf}  # step 6
```

## Why `--use_msa_server`

Boltz calls the ColabFold MMseqs2 server, which returns paired + unpaired
MSAs for multimers by default. No local MSA caching — quicker to iterate
now; can swap to pre-fetched A3M files later if run-to-run reproducibility
becomes a problem.

## Template convention (template_msa mode)

YAMLs reference `template.cif` by basename. The Modal runner writes the
cleaned crystal CIF to `template.cif` next to the YAML in a per-job temp
dir, so Boltz resolves it relatively. This is **self-templating** —
upper-bound sanity check, not a realistic-deployment signal.
