#!/usr/bin/env python3
"""Extract crystallographic interface residues from a multi-chain PDB.

Prints a ``--hotspot`` string (``"A42,A45,A89"``) for every target-chain
residue with at least one heavy atom within ``--cutoff`` Å of any
partner-chain heavy atom.

Example — EGFR/HER2 interface on 8HGO (target=EGFR chain A, partner=HER2 B)::

    python af_design/extract_interface_hotspots.py \
        --pdb af_design/input/8hgo_AB.pdb \
        --target A --partner B --cutoff 5.0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


def _chain_heavy_atoms(chain) -> tuple[np.ndarray, list[tuple[str, int]]]:
    coords: list[list[float]] = []
    per_atom_res: list[tuple[str, int]] = []
    for residue in chain:
        if residue.id[0] != " ":
            continue
        resnum = int(residue.id[1])
        for atom in residue:
            if atom.element == "H":
                continue
            coords.append(list(atom.coord))
            per_atom_res.append((chain.id, resnum))
    return np.asarray(coords, dtype=np.float32), per_atom_res


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdb", type=Path, required=True)
    parser.add_argument("--target", required=True, help="Target chain ID")
    parser.add_argument("--partner", required=True, help="Partner chain ID")
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Interface cutoff in Å (default 5.0)")
    parser.add_argument("--top-k", type=int, default=0,
                        help="Keep only the N residues closest to the partner "
                             "chain (0 = keep all; use e.g. 4 for AFDesign hotspots)")
    args = parser.parse_args()

    structure = PDBParser(QUIET=True).get_structure("s", str(args.pdb))
    model = next(iter(structure))

    target_chain = model[args.target]
    partner_chain = model[args.partner]

    tgt_xyz, tgt_res = _chain_heavy_atoms(target_chain)
    prt_xyz, _prt_res = _chain_heavy_atoms(partner_chain)

    # min distance from each target atom to any partner atom
    # O(Nt * Np) on the distance matrix — fine for <100k atoms.
    diffs = tgt_xyz[:, None, :] - prt_xyz[None, :, :]
    d2 = (diffs * diffs).sum(axis=-1)
    d_min_per_atom = np.sqrt(d2.min(axis=1))

    close_mask = d_min_per_atom < args.cutoff
    hotspot_residues: dict[int, float] = {}
    for hit, (_, resnum), d_min in zip(
        close_mask, tgt_res, d_min_per_atom
    ):
        if not hit:
            continue
        if resnum not in hotspot_residues or d_min < hotspot_residues[resnum]:
            hotspot_residues[resnum] = float(d_min)

    if args.top_k > 0:
        # Rank by proximity (smallest d_min first), keep the N closest, then
        # re-sort by residue number for readable output.
        ranked = sorted(hotspot_residues.items(), key=lambda kv: kv[1])
        kept = sorted(r for r, _ in ranked[: args.top_k])
    else:
        kept = sorted(hotspot_residues)

    hotspot_str = ",".join(f"{args.target}{r}" for r in kept)
    print(hotspot_str)
    suffix = f" (top-{args.top_k} by proximity)" if args.top_k > 0 else ""
    print(
        f"# {len(kept)}/{len(hotspot_residues)} residues on chain {args.target} "
        f"within {args.cutoff} Å of chain {args.partner}{suffix}"
    )


if __name__ == "__main__":
    main()
