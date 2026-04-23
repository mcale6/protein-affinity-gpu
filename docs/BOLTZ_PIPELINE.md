# Kastritis 81 — Boltz-2 Prediction Pipeline

> **Purpose.** Produce predicted structures + PAE for all 81 Kastritis complexes
> using Boltz-2 on Modal, then benchmark predicted-vs-crystal via MM-align
> (TM-score) alongside Boltz's own ipTM. Outputs feed Phase 2 of
> [docs/PAE.md](./PAE.md) — the PRODIGY-PAE calibration.
>
> **Status.** Design spec. Step 1 (prep) implemented & runnable; steps 2–6
> scaffolded below, to be built iteratively.

## Why Boltz-2 (not AF2-Multimer)

- **Open-source weights**, no parameter download from DeepMind bucket.
- **Native PAE + ipTM in the output JSON** (`confidence_{model}.json`).
- **Modal example already exists** (see `Setup` section below).
- **Template + MSA** both supported via one YAML — clean A/B ablation for our
  "what prior does Boltz need?" question.
- AF3 server would also work but is rate-limited (~20/day, ≈4 days for 81).

## Folder layout

```
benchmarks/
├── datasets/kastritis_81/
│   ├── dataset.json              # experimental ΔG + PRODIGY baseline  (existing)
│   ├── manifest.csv              # one row per complex, post-prep      (step 1)
│   └── README.md                 # (existing, updated with pipeline link)
├── downloads/kastritis_81/       # git-ignored
│   ├── {PDB}.pdb                 # raw crystal (existing, 81 files)
│   ├── cleaned/                  # step 1 output
│   │   └── {PDB}_AB.pdb          # rechained target=A, binder=B, renumbered
│   └── msa/                      # step 2 output
│       └── {hash16}.a3m          # keyed by sha256(sequence)[:16] to dedupe
├── downloads/kastritis_81_boltz_inputs/   # step 3 output, git-ignored
│   ├── msa_only/
│   │   └── {PDB}.yaml
│   └── template_msa/
│       └── {PDB}.yaml
├── output/kastritis_81_boltz/             # step 4 output, git-ignored
│   ├── msa_only/{PDB}/{predicted.cif, confidence.json, pae.npz}
│   └── template_msa/{PDB}/...
└── scripts/boltz_pipeline/
    ├── 01_prep_kastritis.py       # rechain + renumber + sequences + manifest
    ├── 02_fetch_msas.py           # MSAs for unique sequences
    ├── 03_build_boltz_yaml.py     # manifest → YAMLs (2 modes × 81)
    ├── 04_modal_boltz_predict.py  # Modal batch runner
    ├── 05_mmalign_tm.py           # predicted vs crystal → TM-score CSV
    ├── 06_plot_iptm_vs_tm.py      # scatter plot
    └── README.md                  # how to run, in order
```

## Manifest CSV schema (step 1 output)

One row per complex:

| column | type | notes |
|---|---|---|
| `pdb_id` | str | e.g. `1A2K` |
| `chains_target_orig` | str | e.g. `C` (original chain IDs, pre-rechain) |
| `chains_binder_orig` | str | e.g. `AB` |
| `cleaned_pdb` | str | relative path, always chain A = target, B = binder |
| `len_target` | int | residues in chain A after cleanup |
| `len_binder` | int | residues in chain B after cleanup |
| `seq_target` | str | one-letter AA |
| `seq_binder` | str | one-letter AA |
| `hash_target` | str | `sha256(seq)[:16]` — dedupe key for MSA cache |
| `hash_binder` | str | same |
| `msa_target` | str | filled by step 2: `downloads/kastritis_81/msa/{hash}.a3m` |
| `msa_binder` | str | same |
| `dg_exp` | float | from `dataset.json` (kcal/mol) |
| `ba_val_prodigy` | float | PRODIGY's own `ba_val`; sanity-check target |
| `functional_class` | str | `OG`, `EI`, … |
| `irmsd` | float | interface RMSD; proxy for flexibility |
| `bsa` | float | Å² |

**Format convention.** `:` is the target↔binder separator (as in dataset.json's
`"C:AB"`). Multi-chain partners are merged into a single chain in the cleaned
PDB — Boltz treats each side as one protein sequence.

## Pipeline

### Step 1 — Prep (`01_prep_kastritis.py`)

Port of `pdb_selchain` + `pdb_chain` + `pdb_reres` from
[pdb-tools](https://github.com/haddocking/pdb-tools) using Biopython:

- Parse `Interacting_chains` (e.g. `"C:AB"` → target=`C`, binder=`A,B`).
- Select only those chains from `downloads/kastritis_81/{PDB}.pdb`.
- Merge all target chains → new chain `A`; all binder chains → new chain `B`.
- Renumber residues contiguously per new chain (drop insertion codes).
- Drop ligands, HETATMs, non-standard residues.
- Extract one-letter sequences per new chain.
- Emit `downloads/kastritis_81/cleaned/{PDB}_AB.pdb` + manifest row.

**Run:** `python benchmarks/scripts/boltz_pipeline/01_prep_kastritis.py`

### Step 2 — MSA fetch (`02_fetch_msas.py`)

> **Open decision:** AlphaFold DB does not expose a raw-MSA API (as of 2026).
> Two routes:
>
> - **(a, recommended) Use Boltz's `--use_msa_server` flag in step 4.** Boltz
>   calls the ColabFold MMseqs2 server automatically; step 2 becomes a no-op,
>   `msa_target` / `msa_binder` stay empty in the manifest. Pro: one-line
>   change, no new code. Con: MSAs aren't cached across runs; network hits on
>   every prediction.
> - **(b) Pre-fetch from ColabFold MMseqs2 API**. Endpoint:
>   `https://api.colabfold.com/ticket/pair` (for complexes) or
>   `/ticket/msa` (per sequence). Full example in the ColabFold repo
>   ([sokrypton/ColabFold](https://github.com/sokrypton/ColabFold)). Dedupe
>   by `hash_target` / `hash_binder` to avoid re-querying identical chains.

Implementation sketch (route b):

```python
for seq_hash, seq in unique_sequences(manifest):
    out = MSA_DIR / f"{seq_hash}.a3m"
    if out.exists():
        continue
    a3m = fetch_colabfold_msa(seq)   # POST to api.colabfold.com
    out.write_text(a3m)
update_manifest_with_msa_paths(manifest_csv)
```

### Step 3 — Boltz YAML build (`03_build_boltz_yaml.py`)

Two modes per complex. Both Boltz-2 schema, written per
[boltz-community/boltz docs](https://github.com/jwohlwend/boltz).

**`msa_only/{PDB}.yaml`** — no structural prior:

```yaml
version: 1
sequences:
  - protein:
      id: A
      sequence: "{seq_target}"
      msa: "{msa_target}"          # or omit + use --use_msa_server at run-time
  - protein:
      id: B
      sequence: "{seq_binder}"
      msa: "{msa_binder}"
# no templates block → MSA-only prediction
```

**`template_msa/{PDB}.yaml`** — MSA + crystal as structural prior:

> **Open decision:** "template" semantics for this ablation. Three options,
> ranked by info-leakage from high to low:
>
> 1. **Self-template**: feed `cleaned/{PDB}_AB.pdb` as template. Upper-bound
>    sanity check ("can Boltz stay put given the right answer?"). Leaks
>    ground truth — don't over-interpret TM-score.
> 2. **AFDB monomer templates**: one template per chain from AFDB. Realistic
>    "AF-enriched" prior, no complex-level leakage. Requires UniProt mapping.
> 3. **Foldseek homolog**: a related PDB entry. Matches real deployment but
>    adds Foldseek as a dep.
>
> Recommend starting with **(1)** to confirm Boltz round-trips, then **(2)**
> for the meaningful condition.

```yaml
version: 1
sequences:
  - protein: {id: A, sequence: "{seq_target}", msa: "{msa_target}"}
  - protein: {id: B, sequence: "{seq_binder}", msa: "{msa_binder}"}
templates:
  - cif: "{template_path}"   # self-template OR AFDB monomer — see decision above
```

### Step 4 — Modal batch predict (`04_modal_boltz_predict.py`)

Based on Modal's [Boltz-2 example](https://modal.com/docs/examples/boltz_predict).
One function per complex × mode (162 calls), using Modal's concurrency to
parallelise over H100s. Store model weights + CCD once in a
`modal.Volume("boltz-models")`.

```python
@app.function(image=image, volumes={models_dir: boltz_vol}, gpu="H100",
              timeout=10*MINUTES)
def boltz_inference(yaml_text: str, pdb_id: str, mode: str) -> bytes:
    input_path = Path("input.yaml"); input_path.write_text(yaml_text)
    subprocess.run(
        ["boltz", "predict", input_path,
         "--use_msa_server",                 # only if step 2 = route (a)
         "--cache", str(models_dir),
         "--out_dir", f"out/{mode}/{pdb_id}"],
        check=True,
    )
    return tar_dir(f"out/{mode}/{pdb_id}")

@app.local_entrypoint()
def main():
    rows = read_manifest()
    futures = []
    for row in rows:
        for mode in ("msa_only", "template_msa"):
            yaml_text = (YAML_DIR / mode / f"{row['pdb_id']}.yaml").read_text()
            futures.append(boltz_inference.spawn(yaml_text, row['pdb_id'], mode))
    for f in futures:
        unpack(f.get())
```

**Per-complex outputs (Boltz default names):**
- `{PDB}_model_0.cif` — predicted structure
- `confidence_{PDB}_model_0.json` — has `complex_iptm`, `iptm`, `ptm`, `plddt`, `pair_chains_iptm`
- `pae_{PDB}_model_0.npz` — the PAE matrix we actually want for Phase 2

### Step 5 — MM-align TM-score (`05_mmalign_tm.py`)

For each `(pdb_id, mode)`: `MMalign predicted.cif cleaned/{pdb_id}_AB.pdb` → parse TM-score.

- [MM-align binary](https://zhanggroup.org/MM-align/) — multi-chain capable.
- Parse `TM-score=0.8547 (if normalized by length of Chain_1=...)` line.
- Output: `benchmarks/output/kastritis_81_boltz/tm_scores.csv` with columns
  `pdb_id, mode, tm_score, rmsd, aligned_len`.

### Step 6 — Plot (`06_plot_iptm_vs_tm.py`)

Scatter, x = TM-score vs. crystal ("ground truth"), y = ipTM (Boltz self-confidence).

- Two series: `msa_only` vs `template_msa`, different colors.
- Optionally colour by `iRMSD` bucket (rigid / medium / flexible) for stratified insight.
- Diagonal y=x reference line (perfect self-assessment).
- Per-series Spearman ρ(ipTM, TM) reported in the legend.
- Write `benchmarks/output/kastritis_81_boltz/iptm_vs_tm.png` + `.csv`.

## Output wired into Phase 2 of PAE calibration

Once step 4 completes, every complex has a `pae_{PDB}_model_0.npz`. `contacts_pae.py`
already has `load_pae_json` — we'll add a thin `load_pae_npz` variant, and the
Phase 2 calibration driver consumes:

```
(crystal_pdb, predicted_cif, pae_npz, dg_exp)
```

per complex, for the three-way ΔG comparison (crystal PRODIGY / AF-predicted
PRODIGY / AF-predicted PRODIGY-PAE).

## Open decisions (to resolve before step 2–4 land)

1. **MSA source:** `--use_msa_server` (simpler) or pre-fetched ColabFold API
   (reproducible). Affects whether step 2 exists.
2. **Template semantics:** self-template / AFDB monomer / Foldseek homolog
   (see step 3 sidebar).
3. **Boltz version:** `boltz==2.1.1` in the Modal example; pin or track latest?
4. **Modal GPU budget:** 162 inferences × ~3 min/H100 ≈ 8 h wall at low
   concurrency. OK?

## References

- Boltz-2: [MIT Jameel Clinic tech report](https://jclinic.mit.edu/boltz-2/) · [jwohlwend/boltz](https://github.com/jwohlwend/boltz)
- Modal Boltz-2 example: https://modal.com/docs/examples/boltz_predict
- MM-align: https://zhanggroup.org/MM-align/
- pdb-tools (source for rechain/reres logic): https://github.com/haddocking/pdb-tools
- Kastritis 81 dataset: `benchmarks/datasets/kastritis_81/README.md`
- Phase 2 consumer: `docs/PAE.md`
