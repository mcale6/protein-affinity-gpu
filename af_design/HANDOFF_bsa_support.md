# Handoff: add Buried Surface Area (BSA) tracking to AFDesign

This note is for an agent adding BSA as a tracked quantity alongside
binder Cα RMSD. BSA is a more biologically meaningful interface metric
than RMSD-against-final-frame, and it is almost free to compute given
we already pay for soft SASA inside `add_ba_val_loss`.

## What BSA means here

Buried Surface Area is the delta between the SASA of the isolated
monomers and the SASA of the complex:

```text
BSA = SASA(target_alone) + SASA(binder_alone) − SASA(complex)
```

Units are Å². For a well-formed protein-protein interface, BSA grows
from ~0 (unbound / poorly docked) to several hundred Å² (typical
stable interface, ~600–1800 Å² for natural complexes).

Useful because:

- It's the physical observable the `ba_val` PRODIGY score is
  correlated with — contacts and NIS are proxies for BSA.
- Unlike Cα RMSD, BSA does not require picking a reference frame.
- Two runs with identical RMSD curves can have very different BSA
  curves, which tells you whether the binder is converging to the
  same pose or to different ones that happen to be rigid.

## Implementation options

There are three sensible places to put the BSA computation. Pick one,
do not implement all three.

### Option A — as a logged aux in `add_ba_val_loss` (cheapest code path, most expensive GPU)

Extend `src/protein_affinity_gpu/af_design.py` so that the callback
returns an extra logged value:

```python
return {"ba_val": jnp.squeeze(dg), "bsa": jnp.squeeze(bsa_total)}
```

This requires running `calculate_sasa_batch_scan_soft` three times per
forward pass (complex + target_alone + binder_alone). That roughly
triples SASA compute in the hot loop. The values that ColabDesign
keys on `bsa` automatically appear in `af_model._tmp["log"]` and
therefore land in `trajectory.json` without any further plumbing.

Trade-offs:

- **Pro:** one-line addition in the trajectory — nice for the plot.
- **Pro:** gradients of BSA are available if anyone later wants to
  use it as a loss term.
- **Con:** ~3× soft SASA scan memory + compute per step. With
  `checkpoint_body=True` the memory impact is bounded but not free.
- **Con:** makes the loss callback mutate two more large intermediate
  tensors during backprop even if BSA is never used in the loss.

### Option B — as a post-step callback in the Modal entrypoint (recommended default)

Extend the `_capture_binder_ca` callback in
`af_design/modal_afdesign_ba_val.py` into `_capture_binder_state`,
which also computes BSA from `aux["atom_positions"]` using the **hard**
SASA kernel (`calculate_sasa_batch_scan`, not `*_soft`) — since BSA is
purely a logged quantity, no gradient is needed.

The callback runs outside `jax.grad`, so there's no backprop cost.
Three cheap forward-only SASA calls per step instead of three
gradient-taped calls.

Sketch:

```python
from protein_affinity_gpu.sasa import calculate_sasa_batch_scan
from protein_affinity_gpu.sasa import generate_sphere_points
from protein_affinity_gpu.utils import residue_constants

_SPHERE = jnp.asarray(generate_sphere_points(sphere_points), dtype=jnp.float32)
bsa_history: list[float] = []

def _capture(model):
    aux = model.aux
    pos = jnp.asarray(aux["atom_positions"]).reshape(-1, 3)
    mask = jnp.asarray(aux["atom_mask"]).reshape(-1)
    # … build complex / target-only / binder-only masks …
    sasa_complex = calculate_sasa_batch_scan(pos, radii, mask_complex, 768, _SPHERE)
    sasa_target = calculate_sasa_batch_scan(pos, radii, mask_target_only, 768, _SPHERE)
    sasa_binder = calculate_sasa_batch_scan(pos, radii, mask_binder_only, 768, _SPHERE)
    bsa = float((sasa_target.sum() + sasa_binder.sum() - sasa_complex.sum()))
    bsa_history.append(bsa)
    # keep the existing binder Cα capture too
```

Serialise to `bsa_history.json` as a flat list of floats, and add it
to `summary["artifacts"]`.

Trade-offs:

- **Pro:** no backprop cost. No change to `af_design.py`.
- **Pro:** uses the fast hard kernel — one forward pass per monomer.
- **Con:** three JIT compiles the first time the callback fires (one
  per mask shape). Mitigate with a fixed complex layout so the shape
  is stable across iterations — it already is, since AF pads to a
  fixed length.
- **Con:** slightly more code in the entrypoint than Option A.

### Option C — post-hoc from the saved PDBs

After the design loop finishes, iterate over stored per-step
structures and rescore with the CPU pipeline
(`predict_binding_affinity`). This would give the full PRODIGY result
per step including BSA-like quantities.

The entrypoint currently only saves `best_design.pdb` and
`last_design.pdb`. To make this work we'd need to also save a PDB per
iteration (80 files @ ~180 KB each = ~15 MB — fine) or at least
per-step atom coords.

Trade-offs:

- **Pro:** no GPU SASA at all — CPU freesasa is already there.
- **Con:** either we eat disk + download cost for 80 PDBs, or we
  serialise per-step `[L, 37, 3]` coordinates ourselves.
- **Con:** CPU SASA over 80 structures will be slow relative to just
  computing it online.

## Recommendation

**Take Option B.** It keeps the loss function pure, it uses the hard
SASA kernel (correct answer, no β to tune, no checkpointing needed),
and it adds one readable JSON artifact. Option A only makes sense if
someone later wants BSA in the loss — leave that as a second PR.

If Option A is eventually wanted, it should probably be gated behind
`log_bsa: bool = False` in `add_ba_val_loss(...)` so the default hot
path stays minimal.

## What to change concretely

Regardless of option:

1. `af_design/modal_afdesign_ba_val.py`
   - Replace `_capture_binder_ca` with `_capture_design_state` that
     also appends BSA (float) to a `bsa_history` list.
   - Write `bsa_history.json` next to `binder_ca_history.json`.
   - Add `"bsa_history_json"` to `summary["artifacts"]`.

2. `af_design/plot_afdesign_rmsd.py` → rename to
   `af_design/plot_afdesign_traj.py` (BSA is not RMSD). Add a
   `--metric {rmsd, bsa, both}` flag:
   - `rmsd`: current behaviour.
   - `bsa`: one line per condition, y-axis "buried surface area (Å²)".
   - `both`: two-panel figure (RMSD top, BSA bottom).
   - Update the old command line in `af_design/HANDOFF_big_tests.md`.

3. `docs/AF_DESIGN.md`
   - Add a short subsection under "Practical Default" describing BSA
     as a tracked diagnostic.
   - Move the `use_soft_sasa` TODO entry to point out that BSA is now
     separately logged without needing that toggle.

4. `tests/` (optional but appreciated)
   - One smoke test that exercises the new callback on a tiny input
     and asserts BSA is monotonic-ish and positive across a converging
     trajectory. Use `benchmarks/downloads/kahraman_2013_t3/1IRA.pdb`
     chain X, binder_len=8, num_steps=3 — the same tiny shape the
     previous validation run used.

## Sanity checks before handing back

- `bsa_history.json` is an 80-length list of floats.
- All floats are ≥ 0 within ~10 Å² numerical tolerance.
- Typical final-iteration BSA on the 1A2K chain B target should land
  somewhere in [200, 1500] Å². If it comes back 0 or negative, the
  monomer masks are probably wrong.
- The existing `binder_ca_history.json` is still produced and
  unchanged.
- A 5-step sanity run on the reorganised entrypoint still completes in
  under ~90 s wall-clock (the tiny validation run's total).

## Reference pointers

- `src/protein_affinity_gpu/sasa.py` — hard SASA kernels
  (`calculate_sasa_batch_scan`, block kernel, sphere points).
- `src/protein_affinity_gpu/sasa_soft.py` — the soft analogue.
- `src/protein_affinity_gpu/af_design.py` — where `add_ba_val_loss`
  calls `calculate_sasa_batch_scan_soft`; this is the canonical
  example of "build a complex atom buffer + mask".
- `src/protein_affinity_gpu/utils/residue_constants.py` —
  `atom_type_num` (the 37) and the atom14/37 conversion tables;
  use `restype_atom14_mask` if you need to derive per-residue atom
  masks from sequence.

## Out of scope

- Do not implement all three options.
- Do not change the soft SASA kernel itself. BSA is logged with the
  hard kernel regardless of loss-mode configuration.
- Do not touch `benchmarks/` — this work stays under `af_design/` and
  `src/`.
