# Handoff: AFDesign soft vs hard-ish 80-step comparison on 1A2K

This note is for the next agent. The pipeline is already validated on a
tiny run; all that's left is to execute the two 80-step runs, produce
the RMSD plot, and write up the deliverables.

## What's already done

- `af_design/modal_afdesign_ba_val.py` — Modal entrypoint. Image, deps,
  and CLI flag parsing are all working. The entrypoint captures binder
  Cα coordinates every iteration via a ColabDesign design callback and
  serialises them to `binder_ca_history.json` alongside the usual
  ColabDesign artifacts.
- `af_design/plot_afdesign_traj.py` — reads per-run
  `binder_ca_history.json` + `bsa_history.json`, computes per-iteration
  backbone RMSD (Kabsch superposition) against the final frame and BSA,
  and writes a one- or two-panel PNG (`--metric rmsd|bsa|both`).
- `src/protein_affinity_gpu/af_design.py` — calls
  `calculate_sasa_batch_scan_soft(..., checkpoint_body=True)` so the
  SASA scan is wrapped in `jax.checkpoint`. That was needed to survive
  backprop on a 200+ residue complex; without it, the 80 GB A100
  OOM'd with a single 90 GB allocation.
- Validation runs already completed and downloaded:
  - `benchmarks/output/af-soft-test/` — 5 steps, soft mode, 1A2K chain B
  - `benchmarks/output/af-hardish-test/` — 5 steps, hard-ish, 1A2K chain B
  - `benchmarks/output/af-tiny/` — 3 steps, 1IRA chain X, 8-residue binder
  - `benchmarks/output/af_rmsd_test.png` — preview plot from the 5-step pair

These validation runs confirmed:

1. Image builds cleanly with `nvidia/cuda:12.4.1-runtime-ubuntu22.04`
   + `jax[cuda12]<0.5` + `tinygrad` + `ColabDesign@v1.1.1`.
2. The soft design path exercises `jax.sasa.scan` (logged during the run).
3. Per-step binder Cα capture works (3/5/5 frames captured as expected).
4. The Kabsch-RMSD plotting script produces a readable curve.

## What you need to do

Run **two** 80-step Modal jobs against `benchmarks/fixtures/1A2K.pdb`
chain B with a 20-residue binder, seed 0, then plot the RMSD curves.
Everything has been shaken down on 5-step dry runs — these are just
longer versions.

### 1. Soft run

```bash
MODAL_GPU=A100-80GB modal run af_design/modal_afdesign_ba_val.py \
  --pdb-path "$(pwd)/benchmarks/fixtures/1A2K.pdb" \
  --chain B \
  --binder-len 20 \
  --num-steps 80 \
  --seed 0 \
  --run-name af-soft \
  --design-mode soft \
  --design-temp 1.0 \
  --binder-seq-mode soft \
  --use-soft-contacts \
  --use-soft-nis \
  --local-output-dir benchmarks/output/af-soft
```

### 2. Hard-ish comparison run

```bash
MODAL_GPU=A100-80GB modal run af_design/modal_afdesign_ba_val.py \
  --pdb-path "$(pwd)/benchmarks/fixtures/1A2K.pdb" \
  --chain B \
  --binder-len 20 \
  --num-steps 80 \
  --seed 0 \
  --run-name af-hardish \
  --design-mode logits \
  --binder-seq-mode pseudo \
  --no-use-soft-contacts \
  --no-use-soft-nis \
  --local-output-dir benchmarks/output/af-hardish
```

Important flag caveats:

- Modal's typer wraps bool kwargs as `--flag`/`--no-flag`. **Do not**
  write `--use-soft-contacts true` — it will error with "Got 2 extra
  arguments." Use `--use-soft-contacts` / `--no-use-soft-contacts`.
- `--design-mode` accepts only `logits` or `soft`. The module-level
  constant `VALID_DESIGN_MODES` enforces this.
- `--binder-seq-mode` accepts only `soft` or `pseudo`.

### 3. Trajectory plot (RMSD + BSA)

```bash
uv run python af_design/plot_afdesign_traj.py \
  --soft-dir benchmarks/output/af-soft \
  --hardish-dir benchmarks/output/af-hardish \
  --output benchmarks/output/af_traj.png \
  --metric both
```

`--metric both` (default) writes a two-panel PNG with Cα RMSD on top
and BSA on the bottom. Drop to `--metric rmsd` or `--metric bsa` for a
single-panel figure.

For RMSD, the default reference is each run's own final frame. Add
`--reference soft-last` to compare both curves against the soft run's
final frame (more apples-to-apples between the two optimization
trajectories, at the cost of making the soft curve monotonically
decrease to zero by construction).

## Deliverables

Post back with:

1. **Success / fail** for both 80-step jobs (include the Modal app URLs
   that print at the end of each run).
2. **Exact commands** used (copy from above; note any deviations).
3. **Metric keys present in `trajectory.json`** — should be exactly
   `ba_val, con, exp_res, hard, i_con, i_pae, i_ptm, loss, models, pae,
   plddt, ptm, recycles, seq_ent, soft, temp`. No native RMSD key —
   that's why we built the Cα callback.
4. **Plot path** (`benchmarks/output/af_rmsd.png`).
5. **A short note** on whether the curves differ qualitatively —
   does the soft run converge more smoothly? Does hard-ish plateau
   earlier? Any interesting divergence point?
6. **Best metrics** from each `summary.json`: `ba_val`, `loss`,
   `plddt`, `i_ptm`, and the best sequence.
7. Do **not** edit `af_design/modal_afdesign_ba_val.py` unless a bug
   surfaces. If you do, say what you changed and why.

## Context you should read before starting

1. [`docs/AF_DESIGN.md`](../docs/AF_DESIGN.md) — soft vs hard
   rationale, what `ba_val` optimises, what `beta` does, the shared-
   initialization TODO.
2. The new "AFDesign Integration — soft-scan SASA" section in
   [`README.md`](../README.md) — one-screen summary of the soft+scan
   memory pattern.
3. The `add_ba_val_loss` callback in
   [`src/protein_affinity_gpu/af_design.py`](../src/protein_affinity_gpu/af_design.py)
   — one function, ~140 lines; read it top to bottom.

## Known wrinkles (already fixed, just so you recognise them if they re-appear)

- **Image build order.** `.add_local_dir("src", …)` must be the **last**
  step before `.add_local_dir` runs; `.env()` / `.workdir()` come
  before. Modal rejects the reversed order with `InvalidError`.
- **`tinygrad` is a required dependency** of the Modal image because
  `protein_affinity_gpu.__init__` → `results.py` → `utils/_array.py`
  imports it at module load. Removing it will fail the import of
  `add_ba_val_loss`.
- **JAX must be `<0.5`.** ColabDesign v1.1.1 calls
  `jax.lib.xla_bridge.get_backend()`, removed in later JAX.
- **SASA scan must be checkpointed in backprop.** `af_design.py`
  already passes `checkpoint_body=True`; don't disable it.
- **AlphaFold params tarball is ~3.5 GB** and is downloaded lazily
  into the Modal Volume on the first run; cached thereafter.

## Caveat on the comparison itself

Both runs share the same initialisation
(`restart(mode=["gumbel", "soft"], reset_opt=False, seed=0)`) and the
same soft SASA — SASA stays soft in both because `add_ba_val_loss`
always calls `calculate_sasa_batch_scan_soft`. So the "soft vs
hard-ish" A/B really only varies **(a)** the optimizer (`design_soft`
vs `design_logits`), **(b)** the binder sequence carrier (`soft`
probabilities vs `pseudo` straight-through), and **(c)** whether
contacts and NIS are sigmoid-smoothed. Write up results as "hard-ish"
or "less-soft baseline", not "hard". There is a TODO in
`docs/AF_DESIGN.md` to expose a `use_soft_sasa` toggle for a true hard
baseline — out of scope for this pass unless the user asks.
