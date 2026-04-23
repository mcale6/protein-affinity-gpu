# Vreven BM5.5 — processed metadata

Clean CSVs derived from `benchmarks/downloads/vreven_bm55/Table_BM5.5.xlsx`
(+ merged ΔG data from two external sources). Citations for raw sources: see
`benchmarks/downloads/vreven_bm55/README.md`.

## Files

| file | content |
|---|---|
| `manifest.csv` | Base BM5.5 manifest (257 rows, no ΔG) |
| `manifest_with_dg.csv` | Same 257 rows + `dg_exp`, `kd_nm`, `dg_source` columns (ΔG populated for 106 complexes, NaN otherwise) |
| `manifest_affinity_only.csv` | Subset restricted to the **106 ΔG-annotated** complexes — ready to drop into pipelines that need affinity targets |

## ΔG coverage (how `manifest_affinity_only.csv` is built)

The BM5.5 docking table has no affinity column. ΔG values are sourced by
joining two disjoint external tables keyed on the 4-char complex PDB code:

| Source | N in BM5.5 | Notes |
|---|---:|---|
| `Kastritis81` — `benchmarks/datasets/kastritis_81/dataset.json` `.DG` | 64 | 17 of the original 81 Kastritis complexes are no longer in BM5.5 (renamed / replaced) |
| `PierceAbAg` — `benchmarks/downloads/vreven_bm55/pierce_ab/antibody_antigen_affinities.txt` (42 with ΔG out of 67 antibody-antigen cases) | 42 | All 42 Pierce-with-ΔG complexes are in BM5.5; **zero overlap with Kastritis 81** |
| **Union** | **106** | The two tables are complementary |

The 151 BM5.5 complexes still without ΔG could be recovered from the
**Affinity Benchmark v2** table (Moal & Bates 2012 + Vreven 2015 additions),
hosted at <https://bmm.crick.ac.uk/~bmmadmin/Affinity/>. Merging that is a
future task — the current 106 is already a clean, tractable calibration
target.

## Affinity subset at a glance

- **106 complexes** = 64 Kastritis + 42 Pierce
- Strata (PRODIGY cutoffs iRMSD 1.5 / 2.2 Å): **80 rigid / 17 medium / 9 difficult**
- Categories: AA 41, OX 17, OG 11, EI 10, AS 9, ES 7, OR 7, ER 4
- Better flexibility coverage than Kastritis 81 alone (K81 had only 6 complexes at iRMSD ≥ 2.2 Å; here we have 9 difficult + 17 medium = 26 non-rigid)
- Antibody-antigen is now the largest single class (50 AA+AS out of 106)

## manifest.csv / manifest_with_dg.csv columns

| column | meaning |
|---|---|
| `pdb_id` | 4-char complex PDB code (pseudo-codes `9QFW`, `BAAD`, `BOYV`, `BP57`, `CP57` disambiguated per upstream README) |
| `Complex` | upstream format `<PDB>_<receptor_chains>:<ligand_chains>` e.g. `1AHW_AB:C` |
| `chains_spec` | the `<rec>:<lig>` portion (receptor left, ligand right) |
| `Cat.` | functional category — AA (antibody-antigen), AS (antibody-small antigen), OX (other; enzyme-substrate-like), EI (enzyme-inhibitor), ER (enzyme-regulator), OR (other regulator), OG (other G-protein-containing), ES (enzyme-substrate) |
| `PDB ID 1` | source PDB of the receptor chains |
| `Protein 1` | receptor protein name |
| `PDB ID 2` | source PDB of the ligand chains |
| `Protein 2` | ligand protein name |
| `I-RMSD (Å)` | interface RMSD between bound and unbound states (flexibility indicator) |
| `ΔASA(Å2)` | buried surface area on binding (proxy for interface size) |
| `BM version introduced` | version at which this complex entered the benchmark (2.0, 3.0, 4.0, 5.0, 5.5) |
| `stratum_iRMSD15_22` | `rigid` (iRMSD < 1.5), `medium` (1.5 ≤ iRMSD < 2.2), `difficult` (≥ 2.2) — PRODIGY-paper cutoff convention |
| `dg_exp` | experimental ΔG (kcal/mol), NaN if not available — present in `manifest_with_dg.csv` only |
| `kd_nm` | dissociation constant in nM, present only for Pierce Ab-Ag-sourced rows |
| `dg_source` | `Kastritis81` / `PierceAbAg` / empty — identifies which external table contributed the ΔG |

## Strata overview

- **177 rigid / 44 medium / 36 difficult** under our cutoffs
- 162 rigid-body / 60 medium / 35 difficult per the paper's section headers in the xlsx
  (paper uses a slightly different iRMSD boundary between medium and difficult)

