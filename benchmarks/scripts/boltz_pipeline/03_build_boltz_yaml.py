#!/usr/bin/env python3
"""Build Boltz-2 YAML inputs from the Kastritis 81 manifest.

Two YAMLs per complex:
  - ``msa_only/{PDB}.yaml``       -- no templates; pure MSA prediction.
  - ``template_msa/{PDB}.yaml``   -- cleaned crystal CIF as self-template.

Self-template notes:
  We use the cleaned crystal (chain A = target, B = binder) as the template.
  This is an upper-bound sanity check ("does Boltz round-trip when given the
  answer?"), not a realistic-deployment signal. See docs/BOLTZ_PIPELINE.md.

MSAs: not pre-fetched. YAMLs omit the ``msa`` field -- the Modal runner
(step 4) passes ``--use_msa_server`` to Boltz so ColabFold's MMseqs2 fetches
paired + unpaired MSAs on demand.

Template path convention:
  The YAML references ``cif: template.cif`` by basename only. The Modal
  runner writes ``template.cif`` next to the YAML in a per-job temp dir;
  Boltz resolves relative template paths from the YAML's directory.

Inputs
    benchmarks/datasets/kastritis_81/manifest.csv
    benchmarks/downloads/kastritis_81/cleaned/{PDB}_AB.pdb

Outputs
    benchmarks/downloads/kastritis_81/cleaned/{PDB}_AB.cif          (template, per complex)
    benchmarks/downloads/kastritis_81_boltz_inputs/msa_only/{PDB}.yaml
    benchmarks/downloads/kastritis_81_boltz_inputs/template_msa/{PDB}.yaml
"""
from __future__ import annotations

import csv
from pathlib import Path

import gemmi

ROOT = Path(__file__).resolve().parents[3]
MANIFEST_CSV = ROOT / "benchmarks/datasets/kastritis_81/manifest.csv"
CLEANED_DIR = ROOT / "benchmarks/downloads/kastritis_81/cleaned"
YAML_ROOT = ROOT / "benchmarks/downloads/kastritis_81_boltz_inputs"


def pdb_to_cif(pdb_path: Path, cif_path: Path) -> None:
    """Convert PDB -> mmCIF using gemmi with proper ``_entity_poly_seq`` rows.

    Boltz's template parser (``parse_mmcif`` -> ``parse_polymer``) indexes
    into the per-entity sequence list. ``gemmi.Structure.setup_entities()``
    creates entities but leaves ``full_sequence`` empty, which yields
    ``IndexError: list index out of range``. We populate it explicitly
    from the chain residues.
    """
    if cif_path.exists() and cif_path.stat().st_mtime >= pdb_path.stat().st_mtime:
        return
    structure = gemmi.read_structure(str(pdb_path))
    structure.setup_entities()
    for model in structure:
        for chain in model:
            entity = structure.get_entity(chain.name) or next(
                (e for e in structure.entities if chain.name in e.subchains), None
            )
            if entity is None:
                continue
            entity.full_sequence = [r.name for r in chain if r.het_flag != "H"]
    structure.make_mmcif_document().write_file(str(cif_path))


def yaml_msa_only(seq_target: str, seq_binder: str) -> str:
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq_target}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {seq_binder}\n"
    )


def yaml_template_msa(seq_target: str, seq_binder: str) -> str:
    # Template is ``template.cif`` -- the Modal runner places the cleaned
    # crystal CIF at that relative path before invoking boltz.
    return (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {seq_target}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {seq_binder}\n"
        "templates:\n"
        "  - cif: template.cif\n"
    )


def main() -> None:
    rows = list(csv.DictReader(MANIFEST_CSV.open()))
    if not rows:
        raise SystemExit(f"manifest empty: {MANIFEST_CSV}")

    (YAML_ROOT / "msa_only").mkdir(parents=True, exist_ok=True)
    (YAML_ROOT / "template_msa").mkdir(parents=True, exist_ok=True)

    for row in rows:
        pdb_id = row["pdb_id"]
        seq_target = row["seq_target"]
        seq_binder = row["seq_binder"]

        pdb_path = CLEANED_DIR / f"{pdb_id}_AB.pdb"
        cif_path = CLEANED_DIR / f"{pdb_id}_AB.cif"
        pdb_to_cif(pdb_path, cif_path)

        (YAML_ROOT / "msa_only" / f"{pdb_id}.yaml").write_text(
            yaml_msa_only(seq_target, seq_binder)
        )
        (YAML_ROOT / "template_msa" / f"{pdb_id}.yaml").write_text(
            yaml_template_msa(seq_target, seq_binder)
        )
        print(f"[ ok ] {pdb_id}")

    print()
    print(f"Wrote {len(rows)} x 2 YAMLs to {YAML_ROOT.relative_to(ROOT)}")
    print(f"Wrote {len(rows)} CIFs to {CLEANED_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
