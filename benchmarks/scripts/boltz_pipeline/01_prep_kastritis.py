#!/usr/bin/env python3
"""Prep Kastritis 81 for Boltz-2: rechain, renumber, extract sequences, emit manifest.

Quick-and-dirty port of pdb-tools (pdb_selchain + pdb_chain + pdb_reres)
via Biopython:

  * select only the chains listed in ``Interacting_chains`` (e.g. "C:AB"),
  * merge all target chains into chain ``A`` and all binder chains into ``B``,
  * renumber residues per chain contiguously from 1 (drop insertion codes),
  * drop HETATMs and non-standard residues,
  * extract one-letter sequences.

Inputs
    benchmarks/datasets/kastritis_81/dataset.json
    benchmarks/downloads/kastritis_81/{PDB}.pdb

Outputs
    benchmarks/downloads/kastritis_81/cleaned/{PDB}_AB.pdb
    benchmarks/datasets/kastritis_81/manifest.csv

See docs/BOLTZ_PIPELINE.md for the full design.
"""
from __future__ import annotations

import csv
import json
from hashlib import sha256
from pathlib import Path

from Bio.PDB import Chain, Model, PDBIO, PDBParser, Residue, Structure

ROOT = Path(__file__).resolve().parents[3]
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"
SRC_DIR = ROOT / "benchmarks/downloads/kastritis_81"
OUT_PDB_DIR = SRC_DIR / "cleaned"
MANIFEST_CSV = ROOT / "benchmarks/datasets/kastritis_81/manifest.csv"

THREE_TO_ONE = {
    "ALA": "A", "CYS": "C", "ASP": "D", "GLU": "E", "PHE": "F",
    "GLY": "G", "HIS": "H", "ILE": "I", "LYS": "K", "LEU": "L",
    "MET": "M", "ASN": "N", "PRO": "P", "GLN": "Q", "ARG": "R",
    "SER": "S", "THR": "T", "VAL": "V", "TRP": "W", "TYR": "Y",
}


def parse_chains_spec(spec: str) -> tuple[list[str], list[str]]:
    """``'C:AB'`` -> (``['C']``, ``['A', 'B']``)."""
    target_str, binder_str = spec.split(":")
    return list(target_str), list(binder_str)


def is_std_aa(res) -> bool:
    """Standard amino acid residue with hetflag == ' '."""
    return res.get_resname() in THREE_TO_ONE and res.id[0] == " "


def merge_chains(model, chain_ids: list[str], new_id: str) -> Chain.Chain:
    """Merge residues from ``chain_ids`` into a single renumbered chain."""
    new_chain = Chain.Chain(new_id)
    idx = 1
    for cid in chain_ids:
        if cid not in model:
            continue
        for res in model[cid]:
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


def prep(pdb_id: str, spec: str, parser: PDBParser) -> dict:
    """Renumber + extract sequences from a PRODIGY-shipped PDB.

    Convention confirmed: every PDB in ``PRODIGYdataset.tgz`` is already
    pre-split into chain ``A`` (target) + chain ``B`` (binder), regardless
    of the original RCSB chain labels recorded in ``Interacting_chains``.
    We keep the original spec only as manifest metadata.
    """
    target_ids_orig, binder_ids_orig = parse_chains_spec(spec)
    src_path = SRC_DIR / f"{pdb_id}.pdb"
    structure = parser.get_structure(pdb_id, src_path)
    model = next(iter(structure))  # use first model (NMR -> first conformer)

    new_struct = Structure.Structure(pdb_id)
    new_model = Model.Model(0)
    new_struct.add(new_model)
    chain_a = merge_chains(model, ["A"], "A")
    chain_b = merge_chains(model, ["B"], "B")

    na, nb = len(list(chain_a)), len(list(chain_b))
    if na == 0 or nb == 0:
        raise RuntimeError(
            f"expected PRODIGY PDB to have chains A and B; got "
            f"source_chains={[c.id for c in model]}, |A|={na}, |B|={nb}"
        )
    new_model.add(chain_a)
    new_model.add(chain_b)

    OUT_PDB_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_PDB_DIR / f"{pdb_id}_AB.pdb"
    io = PDBIO()
    io.set_structure(new_struct)
    io.save(str(out_path))

    seq_t = chain_sequence(chain_a)
    seq_b = chain_sequence(chain_b)
    return {
        "pdb_id": pdb_id,
        "chains_target_orig": "".join(target_ids_orig),  # RCSB labels, for traceability
        "chains_binder_orig": "".join(binder_ids_orig),
        "cleaned_pdb": str(out_path.relative_to(ROOT)),
        "len_target": len(seq_t),
        "len_binder": len(seq_b),
        "seq_target": seq_t,
        "seq_binder": seq_b,
        "hash_target": sha256(seq_t.encode()).hexdigest()[:16],
        "hash_binder": sha256(seq_b.encode()).hexdigest()[:16],
        "msa_target": "",    # filled by step 2
        "msa_binder": "",
    }


def main() -> None:
    parser = PDBParser(QUIET=True)
    dataset = json.loads(DATASET_JSON.read_text())

    rows: list[dict] = []
    fails: list[tuple[str, str]] = []
    for pdb_id, meta in sorted(dataset.items()):
        try:
            row = prep(pdb_id, meta["Interacting_chains"], parser)
        except Exception as exc:  # noqa: BLE001
            fails.append((pdb_id, str(exc)))
            print(f"[FAIL] {pdb_id}: {exc}")
            continue
        row.update({
            "dg_exp": meta["DG"],
            "ba_val_prodigy": meta["ba_val"],
            "functional_class": meta["Functional_class"],
            "irmsd": meta["iRMSD"],
            "bsa": meta["BSA"],
        })
        rows.append(row)
        print(f"[ ok ] {pdb_id}  T={row['len_target']:>4}  B={row['len_binder']:>4}")

    if not rows:
        raise SystemExit("no rows produced; nothing to write")

    MANIFEST_CSV.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with MANIFEST_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print(f"Wrote {len(rows)}/{len(dataset)} rows to {MANIFEST_CSV.relative_to(ROOT)}")
    print(f"Cleaned PDBs in {OUT_PDB_DIR.relative_to(ROOT)}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, err in fails:
            print(f"  {pid}: {err}")


if __name__ == "__main__":
    main()
