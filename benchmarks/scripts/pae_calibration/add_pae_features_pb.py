#!/usr/bin/env python3
"""Compute mean_pae_contacts + mean_pae_interface for ProteinBase.

Uses the same formulation as ``augmented_refit.extract_features`` on K81/V106:

  pae_ab                 = slice_pae_inter(pae_full, n_target, n_binder)
  mean_pae_interface     = mean of pae_ab over all inter-chain pairs
  mean_pae_contacts      = mean of pae_ab at pairs where min_heavy_dist ≤ 5.5 Å

Target chain = A, binder chain = B in the Boltz complex. PAE JSONs live
in ``benchmarks/downloads/proteinbase/pae/{proteinbase_id}_{target}.json``
(AFDB v3+ format: ``predicted_aligned_error`` nested list).

Output: ``benchmarks/output/proteinbase/pae_calibration/pae_features.csv``
with columns: proteinbase_id, mode, mean_pae_contacts, mean_pae_interface,
n_contacts_5p5A.

Usage:
    python add_pae_features_pb.py
"""
from __future__ import annotations

import csv
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from protein_affinity_gpu.contacts_pae import slice_pae_inter  # noqa: E402

PB_ROWS = ROOT / "benchmarks/output/proteinbase/proteinbase_kd_boltz_pae_rows.csv"
PB_STRUCT_DIR = ROOT / "benchmarks/downloads/proteinbase/structures"
PB_PAE_DIR = ROOT / "benchmarks/downloads/proteinbase/pae"
OUT_DIR = ROOT / "benchmarks/output/proteinbase/pae_calibration"


def _safe_name(v: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", v).strip("_")


def _structure_path(row: dict) -> Path:
    stem = _safe_name(f"{row['proteinbase_id']}_{row['target']}")
    return PB_STRUCT_DIR / f"{stem}.cif"


def _pae_path(row: dict) -> Path:
    stem = _safe_name(f"{row['proteinbase_id']}_{row['target']}")
    return PB_PAE_DIR / f"{stem}.json"


def load_pae_json(path: Path) -> np.ndarray:
    data = json.loads(path.read_text())
    if isinstance(data, list): data = data[0]
    if "predicted_aligned_error" in data:
        return np.asarray(data["predicted_aligned_error"], dtype=np.float32)
    if "pae" in data:
        return np.asarray(data["pae"], dtype=np.float32)
    raise ValueError(f"no PAE key in {path}")


def extract_ab_positions(cif_path: Path):
    """Return per-chain atom37-like heavy-atom positions + masks for A, B."""
    parser = MMCIFParser(QUIET=True)
    s = parser.get_structure("c", str(cif_path))
    model = next(iter(s))
    out = {}
    for cid in ("A", "B"):
        if cid not in model:
            raise ValueError(f"chain {cid} missing in {cif_path}")
        residues = []
        for r in model[cid]:
            if r.id[0].strip(): continue     # skip hetatm/waters
            heavy = [(a.element or "").strip() != "H" for a in r]
            atoms = [a for a, h in zip(r, heavy) if h]
            if not atoms: continue
            pos = np.zeros((14, 3), dtype=np.float32)
            mask = np.zeros(14, dtype=bool)
            for i, a in enumerate(atoms[:14]):
                pos[i] = a.get_coord()
                mask[i] = True
            residues.append((pos, mask))
        if not residues:
            raise ValueError(f"empty chain {cid} in {cif_path}")
        pos = np.stack([r[0] for r in residues])
        mask = np.stack([r[1] for r in residues])
        out[cid] = (pos, mask)
    return out


def min_heavy_dist(pos_a, pos_b, mask_a, mask_b):
    diff = pos_a[:, None, :, None, :] - pos_b[None, :, None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    valid = mask_a[:, None, :, None] & mask_b[None, :, None, :]
    dist = np.sqrt(np.maximum(dist2, 0.0))
    dist = np.where(valid, dist, np.inf)
    return np.min(dist, axis=(2, 3))


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = OUT_DIR / "pae_features.csv"

    with PB_ROWS.open() as f:
        rows_in = list(csv.DictReader(f))

    results: list[dict] = []
    fails: list[tuple[str, str]] = []
    t0 = time.time()
    for i, row in enumerate(rows_in, 1):
        pid = row["proteinbase_id"]
        cif = _structure_path(row)
        pae = _pae_path(row)
        if not cif.exists():
            fails.append((pid, f"no CIF {cif.name}")); continue
        if not pae.exists():
            fails.append((pid, f"no PAE {pae.name}")); continue
        try:
            chains = extract_ab_positions(cif)
            pae_full = load_pae_json(pae)
            n_t = chains["A"][0].shape[0]
            n_b = chains["B"][0].shape[0]
            if pae_full.shape[0] < n_t + n_b:
                raise ValueError(f"PAE {pae_full.shape} < {n_t+n_b}")
            pae_ab = slice_pae_inter(pae_full, n_t, n_b, symmetrize=True).astype(np.float32)
            md = min_heavy_dist(
                chains["A"][0], chains["B"][0],
                chains["A"][1], chains["B"][1],
            )
            contacts = md <= 5.5
            n_contacts = int(contacts.sum())
            if n_contacts > 0:
                mean_pae_contacts = float(pae_ab[contacts].mean())
            else:
                mean_pae_contacts = float("nan")
            mean_pae_interface = float(pae_ab.mean())
            results.append({
                "proteinbase_id": pid, "mode": "msa_only",
                "mean_pae_contacts": mean_pae_contacts,
                "mean_pae_interface": mean_pae_interface,
                "n_contacts_5p5A": n_contacts,
            })
            if i % 20 == 0 or i == 1 or i == len(rows_in):
                print(f"[{i:>3d}/{len(rows_in)}] {pid:<30s}  "
                      f"<PAE@contacts>={mean_pae_contacts:.2f}  "
                      f"<PAE@iface>={mean_pae_interface:.2f}  "
                      f"n_contacts={n_contacts}")
        except Exception as exc:  # noqa: BLE001
            fails.append((pid, str(exc)[:120]))
            print(f"[FAIL] {pid}: {exc}")

    if not results:
        print("[fatal] no rows"); return 1

    fields = list(results[0].keys())
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        w.writerows(results)
    print(f"\nWrote {len(results)} rows in {time.time() - t0:.1f}s to {out_csv}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, err in fails[:10]: print(f"  {pid}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
