#!/usr/bin/env python3
"""Prep Vreven BM5.5 affinity subset for Boltz-2: rechain, renumber,
extract sequences, emit manifest.

Vreven BM5.5 ships two separate PDB files per complex:
    {pdb_id}_r_b.pdb   — bound receptor (the "target" in our convention)
    {pdb_id}_l_b.pdb   — bound ligand   (the "binder")

We merge receptor chains into ``A``, ligand chains into ``B``, renumber
residues contiguously from 1, drop HETATMs and non-standard residues,
and save a single cleaned PDB per complex — identical schema to the
Kastritis 81 prep output so downstream Boltz-pipeline scripts
(02_generate_msa → 06_plot_boltz_eval) work unchanged after a manifest-path
switch.

Inputs
    benchmarks/datasets/vreven_bm55/manifest_affinity_only.csv
    benchmarks/downloads/vreven_bm55/benchmark5.5/structures/{PDB}_{r,l}_b.pdb

Outputs
    benchmarks/downloads/vreven_bm55/cleaned/{PDB}_AB.pdb
    benchmarks/datasets/vreven_bm55/manifest_boltz.csv

See docs/BOLTZ_PIPELINE.md for the pipeline design; see docs/PAE.md
§ "Vreven v5.5 — staged for Phase 2 v2 validation" for motivation.
"""
from __future__ import annotations

import argparse
import csv
from hashlib import sha256
from pathlib import Path

from Bio.PDB import Chain, Model, PDBIO, PDBParser, Residue, Structure

ROOT = Path(__file__).resolve().parents[3]
AFFINITY_CSV = ROOT / "benchmarks/datasets/vreven_bm55/manifest_affinity_only.csv"
SRC_DIR = ROOT / "benchmarks/downloads/vreven_bm55/benchmark5.5/structures"
OUT_PDB_DIR = ROOT / "benchmarks/downloads/vreven_bm55/cleaned"
MANIFEST_CSV = ROOT / "benchmarks/datasets/vreven_bm55/manifest_boltz.csv"

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def is_std_aa(res) -> bool:
    return res.get_resname() in THREE_TO_ONE and res.id[0] == " "


def merge_all_chains(model, new_id: str) -> Chain.Chain:
    """Merge every std-AA residue from ``model`` into a single renumbered chain.

    Order: iterate the model's chains in their stored order, keep residue
    ordering within each chain. The Vreven ``_r_b.pdb`` files already hold
    only receptor chains; ``_l_b.pdb`` only ligand chains — so "all chains"
    cleanly maps to the target/binder distinction.
    """
    new_chain = Chain.Chain(new_id)
    idx = 1
    for chain in model:
        for res in chain:
            if not is_std_aa(res):
                continue
            new_res = Residue.Residue((" ", idx, " "), res.resname, "")
            for atom in res:
                new_res.add(atom.copy())
            new_chain.add(new_res)
            idx += 1
    return new_chain


def chain_sequence(chain: Chain.Chain) -> str:
    return "".join(THREE_TO_ONE[r.resname] for r in chain)


def first_model(path: Path, parser: PDBParser, key: str):
    structure = parser.get_structure(key, path)
    return next(iter(structure))


def prep(pdb_id: str, row: dict, parser: PDBParser) -> dict:
    rb_path = SRC_DIR / f"{pdb_id}_r_b.pdb"
    lb_path = SRC_DIR / f"{pdb_id}_l_b.pdb"
    if not rb_path.exists():
        raise FileNotFoundError(f"missing {rb_path.relative_to(ROOT)}")
    if not lb_path.exists():
        raise FileNotFoundError(f"missing {lb_path.relative_to(ROOT)}")

    model_r = first_model(rb_path, parser, f"{pdb_id}_r")
    model_l = first_model(lb_path, parser, f"{pdb_id}_l")

    chain_a = merge_all_chains(model_r, "A")   # receptor → target
    chain_b = merge_all_chains(model_l, "B")   # ligand   → binder
    na, nb = len(list(chain_a)), len(list(chain_b))
    if na == 0 or nb == 0:
        raise RuntimeError(
            f"empty chain after merge: |A|={na}, |B|={nb} "
            f"(rb={rb_path.name}, lb={lb_path.name})"
        )

    new_struct = Structure.Structure(pdb_id)
    new_model = Model.Model(0)
    new_model.add(chain_a)
    new_model.add(chain_b)
    new_struct.add(new_model)

    OUT_PDB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PDB_DIR / f"{pdb_id}_AB.pdb"
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(out_path))

    seq_t = chain_sequence(chain_a)
    seq_b = chain_sequence(chain_b)

    # Vreven manifest has chain spec as "<rec>:<lig>" — preserve raw labels
    # for traceability even though we drop them in the cleaned PDB.
    spec = row.get("chains_spec", "")
    rec_chains, lig_chains = spec.split(":") if ":" in spec else ("", "")

    return {
        "pdb_id": pdb_id,
        "chains_target_orig": rec_chains,
        "chains_binder_orig": lig_chains,
        "cleaned_pdb": str(out_path.relative_to(ROOT)),
        "len_target": len(seq_t),
        "len_binder": len(seq_b),
        "seq_target": seq_t,
        "seq_binder": seq_b,
        "hash_target": sha256(seq_t.encode()).hexdigest()[:16],
        "hash_binder": sha256(seq_b.encode()).hexdigest()[:16],
        "msa_target": "",   # filled by step 2
        "msa_binder": "",
        # ΔG + stratification + Vreven metadata, pass-through
        "dg_exp": row["dg_exp"],
        "kd_nm": row.get("kd_nm", ""),
        "dg_source": row["dg_source"],
        "ba_val_prodigy": "",   # Vreven table has no PRODIGY baseline
        "functional_class": row["Cat."],
        "irmsd": row["I-RMSD (Å)"],
        "stratum_iRMSD15_22": row["stratum_iRMSD15_22"],
        "bsa": row["ΔASA(Å2)"],
        "bm_version_introduced": row["BM version introduced"],
    }


def _load_affinity_rows(limit: int, pdb_ids: str) -> list[dict]:
    rows = []
    with AFFINITY_CSV.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    if pdb_ids.strip():
        keep = {p.strip() for p in pdb_ids.split(",") if p.strip()}
        missing = keep - {r["pdb_id"] for r in rows}
        if missing:
            print(f"[warn] requested pdb_ids not in manifest: {sorted(missing)}")
        rows = [r for r in rows if r["pdb_id"] in keep]
    elif limit > 0:
        rows = rows[:limit]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0,
                    help="process only first N rows (0 = all 106)")
    ap.add_argument("--pdb-ids", default="",
                    help="comma-separated subset of pdb_ids")
    args = ap.parse_args()

    if not AFFINITY_CSV.exists():
        raise SystemExit(f"missing affinity manifest: "
                          f"{AFFINITY_CSV.relative_to(ROOT)}")
    if not SRC_DIR.exists():
        raise SystemExit(f"missing BM5.5 structures: "
                          f"{SRC_DIR.relative_to(ROOT)}")

    rows_in = _load_affinity_rows(args.limit, args.pdb_ids)
    if not rows_in:
        raise SystemExit("no rows to process")
    print(f"[cfg] {len(rows_in)} complexes to prep "
          f"(from {AFFINITY_CSV.relative_to(ROOT)})")
    print(f"[cfg] out PDBs: {OUT_PDB_DIR.relative_to(ROOT)}/")
    print(f"[cfg] out manifest: {MANIFEST_CSV.relative_to(ROOT)}")

    parser = PDBParser(QUIET=True)
    rows_out: list[dict] = []
    fails: list[tuple[str, str]] = []
    for row in rows_in:
        pdb_id = row["pdb_id"]
        try:
            out = prep(pdb_id, row, parser)
        except Exception as exc:  # noqa: BLE001
            fails.append((pdb_id, str(exc)[:160]))
            print(f"[FAIL] {pdb_id}: {exc}")
            continue
        rows_out.append(out)
        print(f"[ ok ] {pdb_id}  "
              f"T={out['len_target']:>4}  B={out['len_binder']:>4}  "
              f"dG={out['dg_exp']}  {out['stratum_iRMSD15_22']}")

    if not rows_out:
        raise SystemExit("no rows produced; nothing to write")

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows_out[0].keys())
    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows_out)

    # Summary
    n_ok = len(rows_out); n_fail = len(fails)
    strata = {}
    for r in rows_out:
        strata[r["stratum_iRMSD15_22"]] = strata.get(r["stratum_iRMSD15_22"], 0) + 1
    print()
    print(f"Wrote {n_ok}/{len(rows_in)} rows to "
          f"{MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Cleaned PDBs in {OUT_PDB_DIR.relative_to(ROOT)}")
    print(f"Strata: {strata}")
    if fails:
        print(f"Failures ({n_fail}):")
        for pid, err in fails:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    main()
