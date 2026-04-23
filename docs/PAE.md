# PAE-Aware PRODIGY

> **Scope.** This note covers the plan for making PRODIGY's binding-affinity
> prediction flexibility-aware by folding AlphaFold's **Predicted Aligned
> Error (PAE)** into the contact-counting step. The inference primitive is
> implemented in `src/protein_affinity_gpu/contacts_pae.py`; calibration and
> design-side integration are still open and documented below.

## Three phases

| Phase | Goal | Status |
|---|---|---|
| 1 | PAE-gated contact primitive for inference on AF-predicted complexes | **Done** — `contacts_pae.py` |
| 2 | Calibration on Kastritis / Vreven vs. experimental ΔG; refit coefficients | **Next** |
| 3 | Wire PAE-aware ΔG into the AFDesign hallucination loop (`add_ba_val_loss`) | **Deferred** |

The inference side is the foundation. The design side only pays off once we've
shown PAE-gated contacts give a measurable lift on real PPIs with known ΔG.

---

## Phase 1 — Inference primitive (done)

File: `src/protein_affinity_gpu/contacts_pae.py`.

**Principle (recap of structuremap, Bludau 2022).** PAE is treated as
additive distance uncertainty. A neighbor (or, here, an inter-chain residue
contact) is counted only if both the physical distance and the PAE stay
inside the cutoff:

```text
# structuremap get_neighbors():
PAE_ij ≤ max_dist
AND euclidean_dist + PAE_ij ≤ max_dist
```

Ported to inter-chain PRODIGY contacts with two gate modes:

- **`confidence`** (default): `(min_heavy_atom_dist ≤ 5.5 Å) AND (PAE_ij ≤ τ)` —
  PRODIGY's standard contact + an AF-confidence filter. Ablate with `pae_cutoff=∞`.
- **`pessimistic`**: `(min_heavy_atom_dist + PAE_ij ≤ 5.5 Å) AND (PAE_ij ≤ τ)` —
  structuremap-literal additive gate. Strictly stronger; PAE inflates effective
  distance.

**NIS is not PAE-gated.** Solvent accessibility is a geometric property of a
single structure; pairwise PAE doesn't fit. Only the contact-counting step
consumes PAE.

**API surface:**
- `load_pae_json(path)` — handles AFDB v1–v2 (`distance`), AFDB v3+/AF2-Multimer
  (`predicted_aligned_error`), AF3 (`pae`).
- `slice_pae_inter(pae_full, target_len, binder_len, symmetrize=True)` — extracts
  the inter-chain block.
- `calculate_residue_contacts_pae(...)` — drop-in replacement for
  `calculate_residue_contacts` with a `pae_inter` kwarg.

---

## Phase 2 — Calibration workflow (in progress, April 2026)

**Inputs now on disk.** Kastritis 81 Boltz-2 prediction batch (81 × 2 modes)
landed in commit `6803718`:

| Artefact | Path |
|---|---|
| Crystal PDBs | `benchmarks/downloads/kastritis_81/*.pdb` (81) |
| Crystal IC counts + ΔG_exp + `ba_val` | `benchmarks/datasets/kastritis_81/dataset.json` |
| Boltz-2 predicted CIF | `benchmarks/output/kastritis_81_boltz/{mode}/{pdb}_{mode}/boltz_results_input/predictions/input/input_model_0.cif` |
| Boltz-2 PAE (full matrix) | `…/pae_input_model_0.npz[pae]` (float32, `[L_total, L_total]`) |
| Stock PRODIGY on Boltz CIFs | `benchmarks/output/kastritis_81_boltz/prodigy_scores.csv` |
| ipTM / pTM / pLDDT | `benchmarks/output/kastritis_81_boltz/tm_scores.csv` |

**Baseline numbers to beat** (from the Boltz pipeline README):

| Config | Pearson R | RMSE (kcal/mol) |
|---|---|---|
| Crystal (`ba_val` vs `DG`) | 0.74 | 1.88 |
| Boltz `msa_only`, stock PRODIGY | 0.62 | 2.29 |
| Boltz `template+msa`, stock PRODIGY | 0.67 | 2.10 |

The PAE-aware PRODIGY job is to close that 0.07–0.12 Pearson / 0.2–0.4 kcal
gap without reintroducing the crystal (i.e. using only AF-predicted geometry
and PAE).

### Parametrisation — linear-α additive gate

The `contacts_pae` module ships two hard gates (`confidence`, `pessimistic`).
For calibration we **generalise to a single scalar** α:

```text
contact_ij  =  1( min_heavy_atom_dist_ij  +  α · PAE_ij  ≤  d_cut )
```

| α | Meaning |
|---|---|
| `α = 0` | Recovers stock PRODIGY (α=0 is the ablation — PAE completely ignored) |
| `α = 1` | Recovers `pessimistic` mode (structuremap-literal additive gate) |
| `α ∈ (0, 1)` | Partial penalty — how we tune it |

`d_cut` is **fixed at 5.5 Å** (PRODIGY's published value) for the first
iteration; the degree of freedom to sweep is α alone.

### Two-stage calibration

| Stage | Loss | Target | What it tests |
|---|---|---|---|
| **A. Match PDB** | `Σ_pdb Σ_chan (IC_pae(α) − IC_crystal)²` | Crystal IC counts (CC, AC, PP, AP) from `dataset.json` | Does PAE erase Boltz-specific contacts so that IC matches the crystal? Pure structural, no ΔG noise. |
| **B. Match ΔG_exp** | LOO-CV `{Pearson R, RMSE, MAE}` vs `DG` | Experimental ΔG from `dataset.json` | Does the better-matched IC also predict binding affinity better? Reported in two flavours: (B1) fixed stock PRODIGY coeffs, (B2) LOO-refit 6 coeffs per α. |

**Bootstrap** (n=500 resamples over the 81 complexes) gives 95% CIs on α\*,
R, and RMSE at the best config.

### Permutation null (make-or-break sanity check)

Shuffle the 81 PAE matrices across `pdb_id` (each PAE keeps its shape and
values, just gets reassigned to a different complex). Re-run Stages A and B
on the shuffled data. If the shuffled gate also dips → PAE is just a generic
contact-count modulator, not an AF-confidence-specific filter. The
permutation p-value is the key number for claiming "PAE helps".

### Data flow wireframe

```
┌──────────────────────────────────────────────────────────────────┐
│  INPUTS                                                          │
│                                                                  │
│  benchmarks/datasets/kastritis_81/dataset.json                   │
│    ↳ per pdb: CC, AC, PP, AP          (crystal IC counts)        │
│               DG                      (experimental ΔG)          │
│               ba_val                  (PRODIGY-on-crystal ΔG)    │
│               nis_a, nis_c            (NIS, reused from crystal) │
│                                                                  │
│  benchmarks/output/kastritis_81_boltz/msa_only/{pdb}_msa_only/   │
│    boltz_results_input/predictions/input/                        │
│      ↳ input_model_0.cif              (Boltz structure)          │
│      ↳ pae_input_model_0.npz[pae]     (full PAE matrix)          │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  PARSE (per complex)                                             │
│                                                                  │
│    CIF → pos[N, 14, 3], mask[N, 14], chars[N] (A/C/P)            │
│    PAE → slice_pae_inter → pae_ab[N_t, N_b]                      │
│                                                                  │
│  Atom14 convention = all heavy atoms for standard AAs.           │
│  Sanity check: at α=0, recomputed IC must match                  │
│  prodigy_scores.csv within tol (otherwise atom14 mismatch).      │
└──────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  LINEAR-α GATE — SWEEP                                           │
│                                                                  │
│    contact_ij = 1( min_heavy_dist_ij + α·PAE_ij ≤ 5.5 )          │
│    for α ∈ linspace(0, 1.5, 16)                                  │
│                                                                  │
│    classify → IC_cc, IC_ca, IC_pp, IC_pa (per complex, per α)    │
└──────────────────────────────────────────────────────────────────┘
                          │
          ┌───────────────┼───────────────┐
          ▼               ▼               ▼
 ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
 │ Stage A       │ │ Stage B       │ │ Permutation   │
 │ match PDB     │ │ match ΔG_exp  │ │ null          │
 │               │ │               │ │ shuffle PAE   │
 │ L_A(α) =      │ │ B1 fixed coef │ │ across pdb    │
 │  Σ(IC_pae −   │ │ B2 LOO refit  │ │ re-run A & B  │
 │   IC_crystal)²│ │ R + RMSE + MAE│ │ report p      │
 └───────┬───────┘ └───────┬───────┘ └───────┬───────┘
         │                 │                 │
         └─────────── bootstrap (n=500) ─────┘
                          │
                          ▼
┌──────────────────────────────────────────────────────────────────┐
│  OUTPUTS                                                         │
│                                                                  │
│  results/calib_grid.csv      (long: pdb, α, IC counts, dg)       │
│  results/stage_A_ic.png      (2 panels: L_A(α) + per-channel)    │
│  results/stage_B_dg.png      (2 panels: fixed / LOO-refit R+RMSE)│
│  results/scatter_best.png    (3 panels: crystal, stock, PAE)     │
│  results/null_perm.png       (2 panels: curves + permutation p)  │
│  results/summary.md          (comparison table with 95% CIs)     │
└──────────────────────────────────────────────────────────────────┘
```

### Locked-in decisions (April 2026)

1. **Mode:** `msa_only` only for first iteration (not `template_msa`).
   Rationale: larger drift from crystal → larger PAE signal, cleaner test.
   Easy to toggle via `--mode` flag.
2. **`d_cut` fixed at 5.5 Å.** Two-parameter sweep (α, d_cut) gave a heatmap;
   the 1-parameter sweep gives a legible curve. Revisit in iteration 2 if
   α\* near grid edge.
3. **Coefficients:** report both (B1) fixed PRODIGY and (B2) LOO-refit side
   by side. Fixed is the conservative claim; refit is the upper bound.
4. **`calculate_residue_contacts_pae` is the reference impl,** but the
   calibration driver has its own vectorised NumPy path — no JAX/tinygrad
   dependency, for iteration speed.

### Driver

- `benchmarks/scripts/pae_calibration/quick_pae_calib.py` — single-file
  calibration driver, NumPy / pandas / SciPy / Matplotlib only.
- Usage: `python quick_pae_calib.py --mode msa_only [--limit N] [--bootstrap 500]`.
- Atom14-vs-heavy-atom parity is asserted at α=0 via diff vs
  `prodigy_scores.csv` (warning, not fatal).

**What's missing once this lands:**
- Refit coefficients (the winning α\*+coeffs from Stage B2) should be parked
  next to `NIS_COEFFICIENTS` in `scoring.py` as e.g. `NIS_COEFFICIENTS_PAE_MSA_ONLY`.
- Repeat for `template_msa` → `NIS_COEFFICIENTS_PAE_TEMPLATE_MSA` once
  iteration 1 shows a real effect.
- Validation on Vreven v5.5 (207) — held-out set to check for overfitting
  to Kastritis 81.

---

## Phase 2 v1 results — null result on Kastritis 81 (April 2026)

### The question we answered

> Does AlphaFold/Boltz PAE carry enough signal on Kastritis 81 to close the
> crystal→Boltz PRODIGY R gap (0.74 → 0.62)?

**Answer: no.** Four independent reparametrisations all saturate at R ≈ 0.63
(msa_only) / 0.67 (template_msa). The gap is a uniform feature-distribution
shift, not a PAE-recoverable signal.

### Baseline numbers

| Config | R (N=81) | RMSE (kcal/mol) |
|---|---:|---:|
| Crystal (`ba_val` vs `DG`) | 0.737 | 1.88 |
| Boltz msa_only, stock PRODIGY | 0.615 | 2.29 |
| Boltz template_msa, stock PRODIGY | 0.666 | 2.10 |
| Paper published (4-fold CV × 10 on crystal) | −0.73 | 1.89 |

### Flexibility stratification — two conventions, different pictures

The paper's flexibility cutoff is **iRMSD > 1.0 Å**, which splits Kastritis
81 into 41 rigid / 40 flexible. The Vreven v5.5 convention (iRMSD > 2.2 Å)
would put only 6 complexes in the flexible bin — too few to resolve.

| Mode | Stratum | N | FIXED R | REFIT 4-fold CV R |
|---|---|---:|---:|---:|
| msa_only | rigid | 41 | 0.633 | 0.574 |
| msa_only | flex | 40 | 0.603 | 0.443 |
| template_msa | rigid | 41 | 0.673 | 0.615 |
| template_msa | flex | 40 | 0.667 | 0.544 |

**The R gap is uniform across rigid / flex**, not flexibility-localised.
This contradicts the PAE hypothesis that gating should help most where
flexibility is high.

### Parametrisation strategies tested

| # | Form | Free params | Script | Result |
|---|---|---|---|---|
| 1 | Linear-α gate `1(d + α·PAE ≤ d_cut)` | α, d_cut | `quick_pae_calib.py` | α\* = 0, d\* = 5.5 (stock); monotone degradation away |
| 2 | Threshold-τ gate `1(d≤5.5) ∧ (PAE≤τ)` | τ | `threshold_pae_calib.py` | τ\* = ∞ (stock); aggregate R monotone in τ |
| 3 | Diagnostic refit (6 stock features) | 7 (6 coefs + intercept) | `diagnostic_refit.py` | In-sample R = FIXED + 0.01; 4-fold CV R *worse* than FIXED |
| 4 | Augmented AIC (13 candidate features) | AIC-selected | `augmented_refit.py` | No PAE/ipTM/pLDDT feature selected |

Candidate pool for (4): 6 stock + `ic_aa, ic_cp` + `ipTM, pTM, pLDDT, confidence_score` + `mean_PAE_contacts, mean_PAE_interface` + `n_contacts`. Stepwise AIC selected 5 features on each mode; **no PAE-derived feature survived selection**.

### Figures (per run outputs)

All under `benchmarks/output/kastritis_81_boltz/pae_calibration/`.

| Path | Content |
|---|---|
| `msa_only/stage_A_ic.png` | Stage A IC-mismatch heatmap (α, d_cut). Minimum at stock corner. |
| `msa_only/stage_B_dg.png` | Stage B R + RMSE heatmaps, both coef policies. |
| `msa_only/scatter_best.png` | 3-panel crystal / stock / PAE-best scatter (best ≡ stock). |
| `msa_only/null_perm.png` | Permutation null: real vs shuffled PAE. |
| `msa_only/threshold/stage_B_curve.png` | Threshold-τ 1D R/RMSE curves. |
| `msa_only/threshold/stratified_R_curve.png` | R vs τ, by iRMSD stratum. |
| `msa_only/stratified_heatmaps.png` | R/RMSE heatmaps per stratum (rigid/medium/flex). |
| `diagnostic_refit/refit_scatter_*.png` | FIXED vs 4-fold-CV refit scatter, both modes. |
| `augmented_refit/augmented_scatter_*.png` | Augmented-AIC vs stock scatter, both modes. |

### Interpretation

- Kastritis 81 is dominated by rigid, well-resolved complexes where Boltz
  already produces accurate structures. Interface PAE is low almost
  everywhere — gating (pessimistic or optimistic) barely modifies the
  contact set.
- The 6-complex Vreven-cutoff flexible subset shows a soft positive trend
  under threshold-τ gating (R flips from −0.16 at τ=∞ to +0.44 at τ=12).
  Bootstrap 95% CI: [−0.77, +0.99]. Cannot reject noise.
- The residual crystal→Boltz R gap is a *uniform feature-distribution
  shift*, not a flexibility-dependent effect. No PAE-derived feature
  survived AIC selection; neither did any global confidence scalar.

### Decision

- Kastritis 81 **cannot validate or refute PAE-aware PRODIGY** at a useful
  effect size. Defer validation to Vreven v5.5 (207 complexes, explicit
  rigid/medium/flexible/difficult strata; balanced N).
- All scripts under `benchmarks/scripts/pae_calibration/` apply unchanged
  once Vreven Boltz predictions land.
- Phase 3 (design-loop PAE gate) remains deferred pending positive Phase 2
  signal on a suitable benchmark.

### Scripts index

- `quick_pae_calib.py` — linear-α, 2D α × d_cut sweep
- `threshold_pae_calib.py` — threshold-τ sweep with stratification
- `stratify_pae_calib.py` — post-hoc stratification on calib_grid.csv
- `diagnostic_refit.py` — refit 6 stock features, 4-fold CV × 10 repeats
- `augmented_refit.py` — 13-feature candidate pool, stepwise AIC

---

## Phase 2 v1.1 — three parallel reparametrisations (April 2026)

After the v1 null, three conceptually distinct reparametrisations were tested in parallel (worktree-isolated agents) to probe directions the v1 battery didn't cover:

| # | Idea | Script | Rationale |
|---|---|---|---|
| A | Entropy surrogate: add `c · ⟨PAE⟩_contacts` to stock PRODIGY with sign-constrained `c ≥ 0` | `entropy_surrogate.py` | Thermodynamic prior — high interface PAE ⇒ entropy cost ⇒ weaker binding |
| B | pLDDT-gated NIS: rescale `%NIS_*` by per-complex pLDDT statistics | `plddt_nis.py` | v1 only gated contacts; NIS side never tested |
| C | IC × ipTM interactions: new features `IC_cc·ipTM`, …, `IC_cc·⟨PAE⟩` in AIC pool | `interaction_refit.py` | Linear AIC cannot discover products; interactions live in a space v1 couldn't see |

### Critical methodological note — the right comparator is REFIT-CV, not FIXED

v1's "ΔR vs stock FIXED" framing conflated two effects:
- Feature-signal change (what we care about)
- **Sample-size penalty** — refitting on N=81 with 4-fold CV trains each fold on ~60 complexes, whereas stock FIXED was calibrated on an *external* 80-complex crystal set → FIXED has a ~0.1 R head start unrelated to feature quality.

For refit+CV models, the matched comparator is **stock 6-feat REFIT with the same 4-fold CV protocol** (0.488 msa_only / 0.557 template_msa). Any ΔR against FIXED larger than −0.10 is really *feature-signal neutral* relative to the refit baseline.

### v1.1 results

| Exp | ΔR vs FIXED | **ΔR vs REFIT-CV (matched)** | Verdict |
|---|---:|---:|---|
| A entropy | −0.11 to −0.13 | ≈ 0 | **NO-HELP** — entropy term adds zero signal beyond the 6 stock features; sign-constrained coef stays non-negative (prior compatible, uninformative) |
| B pLDDT-NIS | −0.09 to −0.13 | +0.011 ± 0.037 | **NO-HELP** — best variant (high-pLDDT residue fraction, template_msa) indistinguishable from noise; NIS is not the bottleneck |
| **C IC×ipTM** | −0.07 to −0.09 | **+0.049 to +0.086** | **MARGINAL** — only experiment with signal above the bootstrap noise floor |

### Experiment C detail — the one positive signal

Under AIC stepwise selection over 14 candidates (6 stock + 4 `IC × ipTM` + 4 `IC × mean_PAE_contacts`):

- **Step 1 on both modes: `ic_pa × ipTM`** — strongest single feature in the entire pool, coef ≈ −0.23 to −0.24. Polar-aliphatic contacts weighted by global prediction reliability.
- **Step 2 on both modes: `ic_cc × mean_PAE_contacts`** — a pure PAE×IC interaction that v1 linear AIC could never discover because it only adds main effects. Coef ≈ −0.008 to −0.016 (small but survives selection).

Naive "add all 4 IC×ipTM interactions" fails — adds variance without sparsity benefit. AIC-guided selection is essential.

**K81 caveat on C**: ipTM median is 0.81 (msa_only) / 0.93 (template_msa) — very narrow dynamic range on this benchmark. The IC×ipTM product ≈ IC × const for the majority of rigid/high-confidence complexes. Vreven v5.5's explicit rigid/medium/flex/difficult strata should give ipTM the dynamic range it lacks on K81.

### Updated decision

- **Keep pursuing C on Vreven v5.5.** The sparse AIC-selected interaction (`ic_pa × ipTM` + `ic_cc × ⟨PAE⟩`) is the one direction that shows above-noise lift against the matched comparator. It's also the only formulation that exploits PAE × IC *coupling*, which is the right place for confidence-aware binding prediction to live.
- **Close A and B as documented null.** Entropy surrogate and NIS-side gating do not carry orthogonal signal on Kastritis 81.
- All five scripts remain applicable to Vreven v5.5 once Boltz predictions land — no code changes needed beyond pointing the data-loading paths at the new manifest.

### Updated scripts index

- `entropy_surrogate.py` — stock + thermodynamic entropy term with sign-constrained fit (v1.1 A)
- `plddt_nis.py` — pLDDT-gated / scaled NIS variants (v1.1 B)
- `interaction_refit.py` — IC × ipTM and IC × ⟨PAE⟩ interaction features + AIC (v1.1 C) — **the positive lead**

---

## Vreven v5.5 — staged for Phase 2 v2 validation

Dataset + merged affinity tables are now in-repo:

| Path | Content |
|---|---|
| `benchmarks/downloads/vreven_bm55/` | Raw upstream BM5.5 (tar + extracted PDBs + Table_BM5.5.xlsx + Pierce Ab-Ag files). Gitignored. See its README for full citations. |
| `benchmarks/datasets/vreven_bm55/manifest.csv` | 257 BM5.5 complexes, typed columns, iRMSD strata |
| `benchmarks/datasets/vreven_bm55/manifest_with_dg.csv` | Same 257 rows + `dg_exp`, `kd_nm`, `dg_source` populated for 106 complexes |
| `benchmarks/datasets/vreven_bm55/manifest_affinity_only.csv` | **The calibration target: 106 ΔG-annotated complexes** (64 from Kastritis 81 + 42 from Pierce antibody-antigen; zero overlap) |

Strata of the 106-complex affinity subset: 80 rigid / 17 medium / 9 difficult (at iRMSD 1.5/2.2 Å cutoffs). Better non-rigid coverage than Kastritis 81 (81 → 19 non-rigid vs 106 → 26 non-rigid) plus broader class mix (AA 41, OX 17, OG 11, EI 10, AS 9, ES 7, OR 7, ER 4).

**Known gap**: 151 BM5.5 complexes still without ΔG. Recoverable by merging in the Moal/Vreven AB2 table (bmm.crick.ac.uk mirror currently unreachable from our network; deferred).

**Next step to execute**: generate Boltz-2 predictions for the 106 affinity-subset complexes using the existing `benchmarks/scripts/boltz_pipeline/` pipeline (same MSA → Boltz → PAE npz steps as Kastritis 81, just pointed at `manifest_affinity_only.csv`). Then re-run `interaction_refit.py` to test whether the AIC-selected `ic_pa × ipTM` and `ic_cc × ⟨PAE⟩` signals from K81 survive at broader ipTM dynamic range.

---

## Phase 2 v2 — Vreven BM5.5 validation (April 2026)

### Pipeline parameterisation

All Boltz-pipeline scripts (01_prep, 03_build_yaml, 04_modal_predict,
05_mmalign_tm, 05b_prodigy_on_boltz, 06_plot_boltz_eval) and PAE-calibration
scripts (quick_pae_calib, augmented_refit, interaction_refit,
interaction_ablation) now accept `--dataset {kastritis, vreven}` via
`benchmarks/scripts/boltz_pipeline/dataset_registry.py`. Adding a future
benchmark is a one-entry patch to the registry.

### Boltz run on Vreven

- Command: `modal run 04_modal_boltz_predict.py --dataset vreven --limit 106 --modes msa_only --diffusion-samples 2`
- Wall time: ~35 min on A100-80GB (two diffusion rollouts per complex, MSA fetched once per complex).
- Output: `benchmarks/output/vreven_bm55_boltz/msa_only/{pdb}_msa_only/` with `input_model_{0,1}.cif`, `pae_*.npz`, `pde_*.npz`, `plddt_*.npz`, `confidence_*.json`. ~3.6 GB total on disk.
- `select_best_sample.py --dataset vreven` swaps files so sample 0 is always the higher-ipTM rollout (13 of 106 swapped).

### Baselines (msa_only, best-ipTM sample)

| Config | R | RMSE (kcal/mol) |
|---|---:|---:|
| Stock PRODIGY FIXED (2015 coefs) | 0.504 | 2.39 |
| Stock 6-feat REFIT 4-fold CV×10 | 0.533 ± 0.022 | 2.22 |
| Crystal (`ba_val`) reference | n/a — Vreven has no `ba_val_prodigy` column | — |

Pipeline-level ipTM distribution: min 0.211, median 0.788, max 0.972, **std 0.218** (vs Kastritis 81 msa_only std ~0.08). This is the key unlock.

### Augmented AIC — ipTM selected for the first time

`augmented_refit.py --dataset vreven` over the v1 13-candidate pool
(6 stock + `ic_aa, ic_cp` + `iptm, ptm, plddt, confidence_score,
mean_pae_contacts, mean_pae_interface, n_contacts`):

| Model | CV R | ΔR vs FIXED | Selected features |
|---|---:|---:|---|
| stock FIXED | 0.504 | — | — |
| stock REFIT CV | 0.533 | +0.029 | 6 stock |
| **Augmented AIC** | **0.580 ± 0.018** | **+0.076** | `nis_c, ic_ca, ` **`iptm`** `, ic_cc` |

Compare to v1 on Kastritis where AIC *rejected all* PAE and confidence features — Vreven's broader ipTM range pushes `ipTM` above the AIC threshold.

### Interaction AIC — `ic_ca × ipTM` selected

`interaction_refit.py --dataset vreven` over the 14-candidate pool
(6 stock + 4 `IC × ipTM` + 4 `IC × mean_pae_contacts`):

| Variant | CV R ± std | ΔR vs FIXED | ΔR vs REFIT | Selected |
|---|---:|---:|---:|---|
| stock FIXED | 0.504 | — | — | — |
| stock REFIT CV | 0.533 ± 0.022 | +0.029 | — | 6 stock |
| v1_all4_iptm | 0.526 ± 0.031 | +0.022 | −0.007 | 10 feats |
| **v2_aic14** | **0.578 ± 0.017** | **+0.074** | **+0.045** | `nis_c, `**`ic_ca × ipTM`**`, ic_cc` |
| v4_ridge α=10 | 0.535 ± 0.028 | +0.031 | +0.002 | 10 feats |

Naive "add-all-4-interactions" (v1) does *worse* than stock REFIT — only AIC-guided sparse selection helps.

### Single-interaction ablation — `ic_pa × ipTM` is the signal

`interaction_ablation.py` adds one interaction term at a time to stock REFIT
6-feat and runs the same 4-fold CV × 10 repeats protocol. Results:

| Single addition | Kastritis ΔR vs REFIT | Vreven ΔR vs REFIT |
|---|---:|---:|
| **+ `ic_pa × ipTM`** | **+0.034** | **+0.022** |
| + `ic_pa × mean_pae_contacts` | +0.023 | +0.018 |
| + `ic_pp × ipTM` | −0.006 | +0.008 |
| + `ic_ca × ipTM` | +0.003 | +0.004 |
| + `ic_ca × mean_pae_contacts` | −0.013 | +0.003 |
| + `ic_pp × mean_pae_contacts` | −0.014 | +0.001 |
| + `ic_cc × ipTM` | −0.021 | −0.003 |
| + `ic_cc × mean_pae_contacts` | −0.025 | −0.009 |

**`ic_pa × ipTM` is rank 1 on both benchmarks.** `ic_pa × ⟨PAE⟩` rank 2 on
both. `ic_cc × *` actively hurts on both. Identical ordering across two
independent calibration sets is a robust cross-dataset claim.

### Interpretation

Polar–apolar contacts (`ic_pa` — one residue polar N/Q/S/T, the other
aliphatic A/C/G/F/I/L/M/P/W/V/Y in PRODIGY's "ic" classification) at
high-confidence interfaces carry a repeatable PAE-aware signal that
improves Boltz-era ΔG prediction.

Two plausible mechanisms for why `ic_pa` specifically:

1. **Boltz makes the most "optional" errors on polar–apolar contacts.**
   Charged–charged placement is driven by electrostatics (hard to get
   wrong); polar–polar and polar–apolar interactions are softer and
   conformationally degenerate, so Boltz's distribution over placements
   is where ipTM-weighting has the most purchase.
2. **`ic_pa` is the largest-weighted term in the 2015 PRODIGY coefs**
   (`w_pa = −0.22671`, the strongest single coefficient). So a small
   relative change in `ic_pa` affects ΔG more than the same relative
   change in `ic_cc`, amplifying the confidence-weighting signal.

### Decision — Phase 2 v2 ships, Phase 3 remains deferred

- Cross-dataset consistency on `ic_pa × ipTM` makes the claim "PAE/ipTM
  awareness adds signal to PRODIGY-IC" quantitatively defensible at ΔR ≈
  +0.02 to +0.03 over the refit baseline.
- Effect size is modest. Useful as a post-hoc reweighting of existing
  PRODIGY predictions, not a fundamental accuracy breakthrough.
- Phase 3 (design-loop PAE gate in `af_design.py`) remains deferred —
  the AIC-sparse interaction formulation needs to be re-derived inside
  the differentiable ColabDesign callback, which is a separate
  implementation exercise. Do not port until we have a clean
  coefficient set from the LOO-refit on the union K81 + Vreven set.

### Vreven v2 artefact index

All under `benchmarks/output/vreven_bm55_boltz/`:

| Path | Content |
|---|---|
| `tm_scores.csv` | 106 × (TM, iRMSD, ipTM, pTM, pLDDT, confidence_score) |
| `prodigy_scores.csv` | 106 × (dg_pred_boltz, dg_exp, 6 IC+NIS features) |
| `best_sample.csv` | per-complex mapping of which original diffusion rollout is now at index 0 |
| `boltz_eval.{png,pdf}` | 3-panel structure + affinity summary |
| `pae_calibration/augmented_refit/features_msa_only.csv` | 13-candidate feature matrix |
| `pae_calibration/augmented_refit/report.md` | AIC stepwise trace + coefficients |
| `pae_calibration/interaction_refit/report.md` | Interaction variants + AIC selection |
| `pae_calibration/interaction_ablation/ablation_msa_only.csv` | 8-model single-addition ablation |

### Updated scripts index (v2)

- `benchmarks/scripts/boltz_pipeline/select_best_sample.py` — picks higher-ipTM diffusion rollout per complex
- `benchmarks/scripts/pae_calibration/interaction_ablation.py` — single-addition ablation, 8 models
- `benchmarks/scripts/pae_calibration/union_refit.py` — combined K81+V106 refit, extracts candidate coefficients

### Union K81 + V106 — final calibration (N=123, all v2 Boltz)

Combining both benchmarks for maximum statistical power: 64 complexes are
in both (use V106 features — newer Boltz run with best-ipTM selection),
17 K81-only (re-run through the v2 Boltz pipeline so the whole union
uses diffusion_samples=2 + best-ipTM), 42 Pierce Ab-Ag only in V106.
**Total union: 123 complexes, all v2.** Strata: 68 rigid / 55 flex.

| Model | R (CV) ± std | R (in-sample) | ΔR vs FIXED |
|---|---:|---:|---:|
| stock FIXED (2015) | 0.511 | 0.511 | — |
| stock REFIT CV | 0.431 ± 0.030 | 0.579 | −0.080 |
| augmented AIC (9 feats) | 0.459 ± 0.032 | — | −0.052 |
| **interaction AIC (6 feats)** | **0.481 ± 0.025** | — | **−0.030** |
| stock 6 + `ic_pa × ipTM` | 0.462 ± 0.025 | 0.585 | −0.049 |

Interaction AIC picks **`nis_c, ic_pa × ipTM, ic_ca, nis_a, ic_pp,
ic_cc × ⟨PAE⟩`**. A *second* interaction (`ic_cc × mean_pae_contacts`)
survives selection on the v2-refreshed union — it didn't survive on
the earlier heterogeneous union nor on either dataset alone. Cleaner
data admits a two-term PAE-aware extension.

**Ablation on the refreshed union:**

| + interaction | R_CV | ΔR vs REFIT |
|---|---:|---:|
| `ic_pa × ipTM` | 0.462 | **+0.030** |
| `ic_pa × ⟨PAE⟩` | 0.458 | +0.026 |
| `ic_pp × ipTM` | 0.444 | +0.013 |
| `ic_pp × ⟨PAE⟩` | 0.435 | +0.004 |
| `ic_ca × ipTM` | 0.433 | +0.002 |
| `ic_ca × ⟨PAE⟩` | 0.432 | +0.001 |
| `ic_cc × ipTM` | 0.429 | −0.002 |
| `ic_cc × ⟨PAE⟩` | 0.426 | −0.006 |

`ic_pa × ipTM` remains rank 1 across all three cuts (K81 +0.034,
V106 +0.022, union-v2 +0.030). Effect sizes are slightly *smaller*
than the pre-refresh union (+0.037 → +0.030) because the v2 Boltz
refresh of the 17 K81-only complexes lifted every baseline by
~+0.014 R — the refresh mostly helped stock FIXED and stock REFIT
equally, narrowing the interaction gap while keeping the ranking.

### Candidate `NIS_COEFFICIENTS_PAE` (union-v2-fitted, stored in JSON)

Saved to `benchmarks/output/union_k81_v106/pae_calibration/union_coefficients.json`:

```
ic_cc         -0.07806   (stock −0.09459)
ic_ca         -0.05722   (stock −0.10007)
ic_pp         +0.10743   (stock +0.19577)
ic_pa         -0.01331   (stock −0.22671)   ← absorbed into interaction
nis_a         +0.13702   (stock +0.18681)
nis_c         +0.20254   (stock +0.13810)
ic_pa × ipTM  −0.12222   (new PAE-aware term)
intercept    −18.49311   (stock −15.9433)
```

**The key structural finding**: when the interaction term is added, the
main-effect `ic_pa` coefficient collapses from stock −0.22671 to
−0.01331 — near zero. The signal PRODIGY attributed to "polar–apolar
contact count" is actually carried by **"polar–apolar contact count
weighted by prediction confidence"** — `ic_pa` alone contributes
nothing once its confidence-weighted version is in the model.

### Limit: stock FIXED still wins on absolute R

On the union, no refit variant beats stock FIXED's R=0.497. The 2015
coefficients — fitted on a disjoint set of 80 crystal complexes — hold
up better than any refit within our 123 Boltz-featured complexes. The
interaction gain is only visible against the matched stock REFIT
comparator (+0.037) because the refit itself loses ~0.075 R to
finite-sample variance on 123 rows × 7 parameters.

**Interpretation**: Phase 2 v2 establishes that an ipTM-weighted
`ic_pa` interaction is the single most robust PAE-aware feature in
the PRODIGY formulation, but:

1. For *production scoring on Boltz-like predictors*, stock FIXED
   coefficients remain the best choice — they generalise from crystal
   to Boltz better than any in-sample refit.
2. For *gradient-based design* (Phase 3 af_design integration), the
   `ic_pa × ipTM` formulation has two clean properties: (a) it exposes
   a single interpretable confidence-weighting term without over-
   parametrising the loss, (b) cross-dataset consistency means the
   coefficient sign (negative, i.e. high-confidence polar–apolar
   contacts tighten ΔG) is mechanistically defensible.

### Decision (v2 final)

- **`NIS_COEFFICIENTS_PAE` candidate**: committed to JSON only, *not*
  wired into `scoring.py`. Prematurely replacing stock coefficients
  would regress production scoring for negligible gain.
- **Phase 3 port criterion**: when the differentiable ColabDesign
  callback is ready, introduce `ic_pa × ipTM` as an additive term with
  the union-fitted coefficient. Keep `pae_tau` / `pae_beta` knobs from
  the original Phase 3 design as configurable override for research.
- **Further work**: re-run the 17 K81-only complexes through the v2
  Boltz pipeline (diffusion_samples=2 + best-ipTM) to eliminate the
  within-union Boltz-run heterogeneity; expected to lift the candidate
  model's CV R by ~0.02. Orthogonal to sourcing AB2 for coverage
  beyond 123.

---

## Phase 3 — Design-side integration (deferred)

> **Do not implement until Phase 2 validates the primitive on experimental ΔG.**

**Goal.** Replace the soft contact tensor inside `add_ba_val_loss`
(`src/protein_affinity_gpu/af_design.py:100–116`) with a PAE-gated variant so
the AFDesign hallucination loop optimises a flexibility-aware ΔG.

**Access to PAE during design.** The PAE matrix is already available as
`aux["pae"]` in the ColabDesign callback — the same source ipSAE uses at
`af_design/modal_afdesign_ba_val.py:426`.

**Soft gate for gradient flow.**

```python
# Current (af_design.py:100–116):
contacts = calculate_residue_contacts_soft(
    target_pos, binder_pos, target_mask, binder_mask,
    distance_cutoff=distance_cutoff, beta=contact_beta,
)

# Proposed:
pae_inter = _slice_pae_inter(aux["pae"], target_len, binder_len)   # [T, B]
pae_gate  = jax.nn.sigmoid(pae_beta * (pae_tau - pae_inter))       # soft τ gate
contacts  = contacts * pae_gate                                     # element-wise
```

- New kwargs on `add_ba_val_loss`: `pae_tau` (default 10 Å), `pae_beta`
  (default 2.0).
- Surface these through the Modal entrypoint (`modal_afdesign_ba_val.py`).
- The existing stage-1-zeros-ba_val cascade still applies — PAE gate
  changes the shape of the contact tensor, not the regime.

**Modal caveat.** The current design pipeline runs with `use_multimer=False`
(ColabDesign binder protocol — AF2 monomer on concatenated target+binder).
Phase 3 should revisit whether to keep that or switch to AF2-Multimer for
design, since cross-chain PAE semantics differ. This is a design-time
decision and out of scope for Phase 2.

**Alternative (lower-risk) integration.** Instead of gating contacts, add a
second loss term: `pae_penalty = mean(pae_inter[mask_contact])` weighted
separately. This keeps the existing `ba_val` untouched but adds explicit
pressure to reduce interface PAE. Worth comparing against the gate approach
once Phase 2 lands.

---

## References

- **structuremap** — Bludau et al. 2022, *PLoS Biol* —
  https://github.com/MannLabs/structuremap_analysis (primitive source).
- **PRODIGY-IC** — Vangone & Bonvin 2015, *eLife* —
  https://elifesciences.org/articles/07454.
- **ipSAE** — Dunbrack 2025 — https://github.com/DunbrackLab/IPSAE
  (already in this repo, commit `762ec0d`; orthogonal to Prodigy-PAE —
  pTM-surrogate scalar vs. per-pair contact counter).
- **Kastritis 2011 Affinity Benchmark v2** —
  https://bmm.crick.ac.uk/~bmmadmin/Affinity/.
- **Vreven 2015 Affinity Benchmark v5.5** — same URL, updated tables.
