# PAE- and CAD-aware PRODIGY — research notes

Public summary of what this repo's calibration experiments established,
what did and didn't work, and what the shippable recommendation is.

Sensitive implementation scripts and full experiment write-ups live
outside the public tree (`benchmarks/scripts/` and the phase-specific
design docs are kept locally only). This note is the
externally-citable summary of findings.

## What we asked

Can AlphaFold / Boltz-2 confidence signals (PAE, pLDDT, ipTM) and local
structural-agreement measures (CAD-score-LT) lift the 2015 PRODIGY IC-NIS
linear model above its published R ≈ 0.73 when the model is applied to
*predicted* structures instead of crystals?

## Benchmarks used

| Benchmark | N | Affinity source | Role |
|---|---:|---|---|
| Kastritis Affinity Benchmark v2 | 81 | ITC (gold standard) | PRODIGY's original training set |
| Vreven BM5.5 affinity subset | 106 | Mixed: ITC + SPR + BLI; adds Pierce-lab antibody-antigen cases | Primary validation |
| ProteinBase kinetic subset | 100 | SPR / BLI (designed antibody-antigen binders) | External / domain-shift test |
| Union (dedup) | **287** | — | Cross-source modelling |

Stock PRODIGY's 6 features (`ic_cc, ic_ca, ic_pp, ic_pa, nis_a, nis_c`)
are computed identically on all three benchmarks via the tinygrad
backend applied to Boltz-2 predictions.

## Headlines

### Phase 2 — PAE gating the contact step

- **Linear-α gate** `1(d_min + α·PAE_ij ≤ 5.5)` (a generalisation of
  structuremap's pessimistic gate): `α* = 0` at `d_cut = 5.5 Å` across
  every dataset — **no PAE gating improves the IC counts on their own**.
- **Threshold gate** `1(d ≤ 5.5) ∧ (PAE ≤ τ)`: also `τ* = ∞` (= stock).
- **Coefficient refit** of the 6 stock features under grouped 4-fold
  CV × 10 repeats lifts R by < +0.02 in-sample on Boltz features, and
  the stock 2015 coefficients **beat refits at out-of-sample grouped
  CV** on all three benchmarks.
- **The one exception:** an interaction term `ic_pa × ipTM` (or the
  PAE-weighted variant `ic_pa × ⟨PAE⟩_interface`) survives AIC
  stepwise selection on all three benchmarks with the same sign and
  contributes +0.02 to +0.03 R over stock-REFIT-CV.

### Phase 3 — Adding CAD-score-LT features

CAD-score-LT (Olechnovic & Venclovas) measures contact-area agreement
between two structures. We extract both global scores and 60+ per-residue,
per-atom, per-contact distribution summaries (mean, p10/25/50/75/90,
fraction thresholds, TP/FP/FN areas, backbone vs side-chain, etc.).

Two comparisons can be defined:

- **Inter-chain / interface CAD** — Boltz complex vs crystal complex,
  `[-inter-chain]` subselect. Captures "is the binding interface
  correctly predicted?" Requires a crystal reference; available on K81
  and V106 (which have crystal structures) but **not** on ProteinBase
  (designed binders, no crystal).
- **Single-chain / binder-fold CAD** — reference-binder vs
  Boltz-extracted-binder, `[-min-sep 1]` subselect. Captures "does the
  binder's predicted fold match its reference?" Reference = crystal
  chain B for K81/V106 or ESMFold monomer for PB. Computable uniformly.

### The honest result

With **interface CAD on K81/V106 and binder-fold CAD on PB** (the
historical non-uniform mix), an XGBoost residual model lifted R by
+0.059 over stock REFIT-CV on the unified 287-complex set under grouped
CV. The top SHAP features were `ic_pa × mean_pae_iface`,
`interface_residue_count`, `n_contacts`.

With **binder-fold CAD uniformly on all 287** (the honest apples-to-
apples setting), that lift collapses to +0.009 — within the noise
floor. Stock PRODIGY coefficients + an `ic_pa × ipTM` interaction
remain the best cross-domain linear model.

The +0.06 R that showed up in the non-uniform run was real but
**interface-CAD-specific**: it requires a crystal complex as reference,
which is not available for novel designed binders. If you're scoring
known-interface complexes you can use interface CAD and get the lift;
if you're scoring de-novo designs you can't.

### Why CAD's per-source univariate R = 0.5 doesn't translate to the model

On ProteinBase alone, `aac_cad_p50` has Pearson R = +0.50 vs log10 Kd.
But the same feature has R ≈ −0.17 on Kastritis 81 and R ≈ −0.05 on
Vreven BM5.5. Three sources, three different clouds in the CAD-vs-ΔG
scatter:

- K81/V106 complexes are *mature* natural interfaces; the binder is
  pre-folded in its bound conformation, so ESMFold / monomer predictions
  match Boltz-in-complex at uniformly high CAD (0.85–0.95). CAD carries
  no ΔG signal there.
- ProteinBase's designed binders often exhibit induced-fit
  rearrangements; ESMFold (which sees the binder alone) and Boltz (which
  sees it in complex) diverge when the binder is flexible. That
  flexibility correlates with tighter binding (negative ΔG, induced-fit
  energetic gain).

So the CAD signal lives in a specific population. A cross-source model
averaging over all three gets its per-population lifts diluted.

## What works in production, today

1. **For affinity on Boltz-predicted complexes where a crystal reference
   is available** (standard benchmarks, drug-target interactions with
   known structure): stock PRODIGY 2015 coefficients + an additive
   `ic_pa × mean_pae_iface` interaction term. +0.03 R over vanilla
   stock PRODIGY under grouped CV. Deployable as-is.
2. **For affinity on designed antibody-antigen binders with no crystal
   (ProteinBase-like)**: binder-fold CAD (ESMFold-vs-Boltz-binder) at
   Pearson R ≈ 0.50 univariate is the strongest signal we've found.
   A CAD-only model or a CAD + Boltz-confidence model beats stock
   PRODIGY in-sample on this regime, but the number of ΔG-annotated
   designed-binder complexes (N=100 here) is small for robust
   calibration.
3. **Cross-domain general PPI affinity**: stock PRODIGY stays best; no
   refit within our 287-complex union beats it under grouped CV.

## References

- Vangone A, Bonvin AMJJ. *Contacts-based prediction of binding affinity
  in protein–protein complexes.* eLife 4, e07454 (2015).
- Kastritis PL, Moal IH, Hwang H, Weng Z, Bates PA, Bonvin AMJJ, Janin
  J. *A structure-based benchmark for protein–protein binding affinity.*
  Protein Sci. 20(3), 482–491 (2011).
- Vreven T, Moal IH, Vangone A, Pierce BG, Kastritis PL, Torchala M,
  Chaleil R, Jimenez-Garcia B, Bates PA, Fernandez-Recio J, Bonvin AMJJ,
  Weng Z. *Updates to the integrated protein–protein interaction
  benchmarks.* J. Mol. Biol. 427(19), 3031–3041 (2015).
- Guest JD, Vreven T, Zhou J, Moal IH, Jeliazkov JR, Gray JJ, Weng Z,
  Pierce BG. *An expanded benchmark for antibody–antigen docking and
  affinity prediction.* Structure 29(6), 606–621 (2021).
- Olechnovic K, Venclovas C. *CAD-score: a new contact area difference-
  based function for evaluation of protein structural models.*
  Proteins 81(1), 149–162 (2013).
- Bludau I et al. *The structural context of posttranslational
  modifications at a proteome-wide scale.* PLoS Biol. 20(5), e3001636
  (2022). (structuremap PAE primitive.)
