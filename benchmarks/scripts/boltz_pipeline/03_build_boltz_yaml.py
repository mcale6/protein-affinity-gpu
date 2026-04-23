#!/usr/bin/env python3
"""Build Boltz-2 YAML inputs from a calibration-dataset manifest.

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

Usage:
    python 03_build_boltz_yaml.py --dataset kastritis
    python 03_build_boltz_yaml.py --dataset vreven

Inputs / outputs are resolved via ``dataset_registry.get_paths(<name>)``.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import gemmi

sys.path.insert(0, str(Path(__file__).parent))
from dataset_registry import AVAILABLE, get_paths  # noqa: E402


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
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=AVAILABLE,
                    help="calibration benchmark to build YAMLs for")
    args = ap.parse_args()
    paths = get_paths(args.dataset)

    rows = list(csv.DictReader(paths.manifest.open()))
    if not rows:
        raise SystemExit(f"manifest empty: {paths.manifest}")

    (paths.yaml_root / "msa_only").mkdir(parents=True, exist_ok=True)
    (paths.yaml_root / "template_msa").mkdir(parents=True, exist_ok=True)

    for row in rows:
        pdb_id = row["pdb_id"]
        seq_target = row["seq_target"]
        seq_binder = row["seq_binder"]

        pdb_path = paths.cleaned_dir / f"{pdb_id}_AB.pdb"
        cif_path = paths.cleaned_dir / f"{pdb_id}_AB.cif"
        pdb_to_cif(pdb_path, cif_path)

        (paths.yaml_root / "msa_only" / f"{pdb_id}.yaml").write_text(
            yaml_msa_only(seq_target, seq_binder)
        )
        (paths.yaml_root / "template_msa" / f"{pdb_id}.yaml").write_text(
            yaml_template_msa(seq_target, seq_binder)
        )
        print(f"[ ok ] {pdb_id}")

    print()
    print(f"[{paths.display}]")
    print(f"Wrote {len(rows)} x 2 YAMLs to {paths.yaml_root}")
    print(f"Wrote {len(rows)} CIFs to {paths.cleaned_dir}")


if __name__ == "__main__":
    main()
