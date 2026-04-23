# Kastritis 81 — PRODIGY calibration set

81 protein–protein complexes with experimental ΔG (kcal/mol), the exact set
PRODIGY-IC was calibrated and evaluated on (Kastritis 2011 + Vangone & Bonvin
2015). Source: [haddocking/prodigy/tests/test_data/dataset.json + dataset.tgz](https://github.com/haddocking/prodigy/tree/main/tests/test_data).

## Layout

```
benchmarks/
├── datasets/kastritis_81/
│   ├── dataset.json        # 26 KB — metadata, committed
│   └── README.md           # this file
└── downloads/kastritis_81/ # 81 × .pdb — git-ignored; re-fetch from PRODIGY repo
```

## `dataset.json` schema

```json
{
  "1A2K": {
    "Interacting_chains": "C:AB",    // target_chains : binder_chains
    "Functional_class": "OG",
    "DG": -9.3,                       // experimental ΔG in kcal/mol — GROUND TRUTH
    "ba_val": -9.0,                   // PRODIGY's own prediction (reference baseline)
    "Experimental_method": "ITC",
    "iRMSD": 1.11,                    // interface RMSD (rigid ≲ 1 Å, flexible ≳ 2 Å)
    "BSA": 1603,                      // buried surface area, Å²
    "CC": 5, "CP": 4, "AC": 20,       // PRODIGY contact counts
    "PP": 2, "AP": 11, "AA": 25,
    "nis_p": 31.65, "nis_a": 41.77, "nis_c": 26.58  // PRODIGY NIS percentages
  },
  ...
}
```

Note: the `ba_val` field in the upstream dataset is the **PRODIGY-IC
prediction**, which maps to the identically-named `ba_val` loss term in this
repo (`src/protein_affinity_gpu/af_design.py:add_ba_val_loss`).

## Dataset statistics

| | |
|---|---|
| Complexes | 81 |
| ΔG range | −18.60 to −4.30 kcal/mol |
| BSA range | 808–3347 Å² |
| iRMSD range | 0.31–3.28 Å |
| Experimental methods | SPR (39), ITC (20), spectroscopy (14), stopped-flow (8) |
| Unique chain specs | 36 (mix of single- and multi-chain partners, e.g. `A:B`, `A:BC`, `AB:C`) |
| Functional classes | OX (19), EI (12), OG (11), A (9), ES (7), NC (8), OR (8), ER (4), AB (3) |

## Re-fetch commands

```bash
# Metadata (committed; only needed if missing):
curl -sL https://raw.githubusercontent.com/haddocking/prodigy/main/tests/test_data/dataset.json \
  -o benchmarks/datasets/kastritis_81/dataset.json

# Structures (git-ignored):
mkdir -p benchmarks/downloads/kastritis_81
curl -sL https://raw.githubusercontent.com/haddocking/prodigy/main/tests/test_data/dataset.tgz \
  | tar -xz -C /tmp \
  && mv /tmp/PRODIGYdataset/*.pdb benchmarks/downloads/kastritis_81/ \
  && rmdir /tmp/PRODIGYdataset
```

## Status

- [x] Experimental ΔG + crystal PDBs in place (this dir + `benchmarks/downloads/kastritis_81/`).
- [ ] AF2-Multimer-predicted structures + PAE JSON per complex — **not yet generated**.
  - Sequences can be extracted from the bound crystal PDBs using `Interacting_chains` to define target vs. binder.
  - Chain-spec parser needs to handle multi-chain partners (e.g. `AB:C`); the existing `predict.predict_binding_affinity(selection="A,B")` assumes single-chain each side.
- [ ] Calibration driver (crystal PRODIGY sanity check + PAE-gated PRODIGY) — deferred, see `docs/PAE.md` Phase 2.

## References

- Kastritis PL et al., 2011, *J. Mol. Biol.* — the original 81-complex affinity set.
- Vangone A & Bonvin AMJJ, 2015, *eLife* — PRODIGY-IC calibration on this set.
- Upstream dataset: https://github.com/haddocking/prodigy/tree/main/tests/test_data
