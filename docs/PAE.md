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

## Phase 2 — Calibration workflow (next)

**Inputs.**

| Artefact | Source | Notes |
|---|---|---|
| Crystal PDB | Kastritis 2011 Affinity Benchmark v2 (81 complexes) | Prodigy's original set |
| Experimental ΔG (kcal/mol) | Same | Ground truth |
| Predicted structure + PAE | AF2-Multimer (run **outside** this repo, standard structure prediction) | Used as the PAE source — this is where the predicted PPI geometry comes from |
| Optional superset | Vreven 2015 Affinity Benchmark v5.5 (207) | For rigid/medium/flexible stratification |

**Protocol.**

For each complex `(pdb, dG_exp)`:

1. Load crystal PDB → run stock PRODIGY → `dG_crystal` (literature baseline).
2. Load predicted structure + PAE → run stock PRODIGY → `dG_pred_nopae`
   (measures PDB→AF drift without PAE correction).
3. Load predicted structure + PAE → run PRODIGY with `contacts_pae` swapped
   in → `dG_pred_pae` (the experiment).

Evaluate:

- Pearson R and RMSE (kcal/mol) vs. `dG_exp` for each of the three.
- Per-complex residuals — expect `dG_pred_pae` to narrow the gap vs.
  `dG_pred_nopae`, particularly on the **flexible** subset.
- Sweep `pae_cutoff τ ∈ {5, 10, 15} Å`.
- Ablate `gate_mode ∈ {confidence, pessimistic}`.

**Coefficient refit.** Once gated contacts produce sensible intermediate
values, refit the 6 IC-NIS coefficients on the Kastritis 81 with leave-one-out
cross-validation. Park new coefficients next to `NIS_COEFFICIENTS` in
`scoring.py` as e.g. `NIS_COEFFICIENTS_PAE`.

**What's missing in this repo for Phase 2:**
- The Kastritis 81 PDB list + ΔG CSV. Likely source:
  [HADDOCK Affinity Benchmark v2](https://bmm.crick.ac.uk/~bmmadmin/Affinity/).
- An AF2-Multimer prediction batch. Structure prediction is **external** to
  this repo — any standard pipeline (ColabFold, AlphaPulldown, local
  AlphaFold-Multimer, AF3 server) produces the needed CIF + PAE JSON, which
  `contacts_pae.load_pae_json` will consume.
- A calibration driver (notebook or script) that loops over the benchmark and
  runs the three-way comparison above.

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
