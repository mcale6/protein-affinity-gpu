# AFDesign Notes

This note explains the stable differentiable helpers that now live in:

- `protein_affinity_gpu.af_design`
- `protein_affinity_gpu.contacts_soft`
- `protein_affinity_gpu.scoring_soft`
- `protein_affinity_gpu.sasa_soft`

The goal is to make the ColabDesign / AfDesign-style `ba_val` loss reusable
from normal Python imports instead of only from benchmark scripts.

## Stable Imports

```python
from protein_affinity_gpu.af_design import add_ba_val_loss
from protein_affinity_gpu.contacts_soft import calculate_residue_contacts_soft
from protein_affinity_gpu.scoring_soft import calculate_nis_percentages_soft
from protein_affinity_gpu.sasa_soft import calculate_sasa_batch_scan_soft
```

These are stable package modules. The old soft-SASA imports from
`protein_affinity_gpu.sasa_experimental` still work as a compatibility layer,
but `sasa_soft.py` is now the source of truth.

## `aux["seq"]["soft"]` vs `aux["seq"]["pseudo"]`

In ColabDesign, both tensors come from the optimized sequence logits, but they
do not mean the same thing:

- `aux["seq"]["soft"]` is the normalized residue-probability simplex.
- `aux["seq"]["pseudo"]` is the tensor ColabDesign feeds into AlphaFold during
  the current design stage.

For sequence-dependent auxiliary losses like `ba_val`, `soft` is usually the
better gradient carrier because it stays on a proper probability simplex and
behaves like an expectation over amino acids.

`pseudo` is still useful when you want the auxiliary loss to mirror the exact
sequence representation AlphaFold saw on the forward pass, even if that makes
the sequence branch less cleanly probabilistic.

The stable `add_ba_val_loss(...)` helper defaults to `binder_seq_mode="soft"`
for that reason.

## `design_logits()` vs `design_soft()`

The choice of design stage changes how the sequence is represented while the
optimizer is updating it:

- `design_logits()` optimizes the raw latent sequence parameters.
- `design_soft()` optimizes a softmaxed sequence distribution directly.

From a deep-learning perspective:

- `design_logits()` tends to commit faster and can produce sharper updates.
- `design_soft()` tends to keep more entropy in the sequence distribution,
  which makes the optimization landscape smoother and exploration longer.

This means a `soft` run often collapses more slowly to a single sequence, while
the logits run often makes harder sequence commitments earlier.

## Hard vs Soft Contacts

The hard contact kernel asks a binary question: is any valid atom pair within
the cutoff distance?

The soft contact kernel replaces that with a sigmoid around the cutoff and then
aggregates atom-pair probabilities into a residue-residue contact probability:

- hard contacts: exact inference-style decision
- soft contacts: differentiable probability in `[0, 1]`

As `beta -> infinity`, the soft contact kernel approaches the hard decision.

## Hard vs Soft NIS Thresholding

The hard NIS step uses a binary exposure gate:

- exposed if `relative_sasa >= threshold`
- buried otherwise

The soft NIS version replaces that step with a sigmoid around the threshold.
That keeps the NIS percentages differentiable with respect to SASA and, through
SASA, with respect to coordinates and sequence probabilities.

Again, increasing `beta` makes the soft gate behave more like the hard one.

## Hard vs Soft SASA

The hard Shrake–Rupley buried-point test is discrete:

- a sphere point is buried if it falls inside an occluding atom
- otherwise it is accessible

The soft SASA kernel replaces that buried-point decision with a sigmoid-smoothed
occlusion probability and accumulates accessibility in log-space for numerical
stability.

Why this matters:

- hard SASA is great for inference parity
- soft SASA gives non-zero gradients through the buried-point test

That makes the soft kernel suitable for backpropagation-based design losses.

## Why Soft Functions Backpropagate

The stable soft helpers are built out of standard JAX operations such as:

- `jax.nn.sigmoid`
- `jax.nn.softplus`
- matrix multiplication
- sums, products, concatenations, and exponentials

When those helpers are used inside ColabDesign's loss callback, their outputs
become part of the scalar loss that ColabDesign passes into `jax.value_and_grad`.
That is what gives you usable gradients.

The important contrast is:

- hard threshold / hard `any` / hard exposure gates produce sparse or zero
  gradient almost everywhere
- soft sigmoids and softplus produce smooth local gradients that the optimizer
  can actually follow

## What `ba_val` Actually Optimizes

The `ba_val` callback does not optimize `beta` directly. Instead, it computes a
scalar binding-energy proxy `dg` from the current AlphaFold structure and the
current binder sequence representation, and that scalar becomes one term in the
full AfDesign objective.

The path is:

```text
sequence logits
-> binder sequence probabilities / pseudo-sequence
-> target-binder contacts
-> complex SASA
-> relative SASA
-> NIS percentages
-> PRODIGY IC-NIS linear score (dg)
-> weighted total loss
```

More concretely:

- contacts are grouped into PRODIGY interaction classes
- SASA is computed on the whole complex and converted to relative SASA
- NIS percentages are derived from the relative SASA values
- those six features are fed into the PRODIGY linear model

The final `dg` term is:

```text
dg =
  -0.09459 * IC_CC
  -0.10007 * IC_CA
  +0.19577 * IC_PP
  -0.22671 * IC_PA
  +0.18681 * P_NIS_A
  +0.13810 * P_NIS_C
  -15.9433
```

where:

- `IC_CC` = charged-charged interface contacts
- `IC_CA` = charged-apolar interface contacts
- `IC_PP` = polar-polar interface contacts
- `IC_PA` = polar-apolar interface contacts
- `P_NIS_A` = percent apolar non-interacting surface
- `P_NIS_C` = percent charged non-interacting surface

In the Modal benchmark, this is then weighted as one part of the full design
loss:

```text
total_loss = other_afdesign_terms + ba_val_weight * dg
```

So with a positive `ba_val_weight`, the optimizer is pushed toward structures
and sequences that make the predicted `dg` more favorable under the PRODIGY
IC-NIS model.

## What `beta` Does and Does Not Do

The `beta` parameters in the soft helpers are fixed hyperparameters. They are
not learned model parameters in the current implementation.

What `beta` does:

- controls how sharp the soft contact switch is
- controls how sharp the soft SASA buried-point decision is
- controls how sharp the soft NIS exposure threshold is
- changes the size and locality of the gradient near each decision boundary

What `beta` does not do:

- it is not optimized by backpropagation in `add_ba_val_loss(...)`
- it is not part of the final PRODIGY linear score by itself
- it does not change what features the loss uses, only how smoothly those
  features are computed

So the clean mental model is:

- `dg` is the objective term being minimized
- `beta` shapes the differentiable path used to compute `dg`

## What Changes Theoretically During Optimization

If you compare a soft-design run against a more non-soft run, the main ML/DL
differences are:

- smoother gradient field: nearby sequences get related gradients instead of a
  mostly discrete signal
- slower collapse: the sequence distribution keeps entropy longer
- higher exploration: optimization can move by reweighting amino-acid
  probabilities before committing
- expectation-style scoring: the loss reflects mixtures over residues rather
  than only one hard sequence
- train / inference mismatch risk: once you discretize the final sequence, the
  hard sequence can behave a bit differently from the soft expectation that was
  optimized

In practice this usually means:

- soft losses are easier to optimize
- hard losses are closer to the final discretized design objective
- mixing the two is often useful, with soft terms early and harder terms later

## Practical Default

For AfDesign-style `ba_val` optimization, the recommended default is:

- `binder_seq_mode="soft"`
- `use_soft_contacts=True`
- `use_soft_nis=True`
- soft SASA enabled through `protein_affinity_gpu.sasa_soft`

That gives the cleanest differentiable path while keeping the underlying
PRODIGY-style score structure intact.

## TODO

### Isolate initialization from optimizer in "soft vs hard" comparisons

`benchmarks/modal_afdesign_ba_val.py` currently calls:

```python
af_model.restart(seed=seed, mode=["gumbel", "soft"], reset_opt=False)
```

independently of `design_mode`. So the "soft" and "hard-ish" runs share:

- the same seed
- the same initial logits (gumbel-sampled, passed through softmax)

and only differ in:

- the optimizer step (`design_soft` vs `design_logits`)
- `binder_seq_mode` (soft probabilities vs straight-through pseudo-sequence)
- `use_soft_contacts` / `use_soft_nis`

SASA stays soft in both because `add_ba_val_loss(...)` always uses
`calculate_sasa_batch_scan_soft(...)` internally.

This means the current "soft vs hard-ish" A/B is really measuring the effect of
the optimizer and the contact/NIS softness, not a clean hard/soft SASA split.
Two follow-ups worth doing:

1. Expose a `use_soft_sasa` toggle in `add_ba_val_loss(...)` and wire it up in
   the Modal entrypoint, so a true hard-SASA baseline is reachable.
2. Consider varying the `restart(mode=...)` initialization to match the design
   mode (e.g. a pure-gumbel or one-hot init for `design_logits` runs) if we
   want to isolate "optimizer shape" from "init shape".

Until (1) lands, runs from `modal_afdesign_ba_val.py` with
`use_soft_contacts=false` and `use_soft_nis=false` should be reported as
"hard-ish" rather than a true hard baseline.
