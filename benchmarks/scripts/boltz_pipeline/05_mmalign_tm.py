#!/usr/bin/env python3
"""Compare Boltz predictions to crystals via US-align + parse ipTM from Boltz.

For each ``<output_root>/{mode}/{PDB}_{mode}/`` dir:
  * locate ``*_model_0.cif`` (predicted structure; step-04 diffusion rollout 0).
  * locate ``confidence_*_model_0.json`` (Boltz self-confidence metrics).
  * run US-align predicted vs. ``<cleaned_dir>/{PDB}_AB.pdb``.
  * parse TM-scores (normalized by chain-1 and chain-2 lengths).
  * write ``<output_root>/tm_scores.csv``.

Default US-align binary: ``benchmarks/tools/USalign`` (compiled from source;
single C++ file; multimer-capable, modern successor to MM-align).
Override with ``--usalign-bin``.

Use as:

    python 05_mmalign_tm.py --dataset kastritis
    python 05_mmalign_tm.py --dataset vreven --pdb-ids 2OOB --modes msa_only
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset_registry import AVAILABLE, get_paths  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_USALIGN = ROOT / "benchmarks/tools/USalign"

# US-align prints two TM lines:
# TM-score= 0.8547 (normalized by length of Structure_1: L=X, d0=Y)
# TM-score= 0.8812 (normalized by length of Structure_2: L=Z, d0=W)
TM_RE = re.compile(
    r"TM-score=\s*([\d.]+)\s*\(normalized by length of Structure_(\d):\s*L=(\d+),\s*d0=([\d.]+)"
)
RMSD_RE = re.compile(r"RMSD=\s*([\d.]+)")
ALEN_RE = re.compile(r"Aligned length=\s*(\d+)")


def find_prediction_artifacts(pred_dir: Path) -> tuple[Path, Path] | None:
    """Return (predicted_cif, confidence_json). None if not found."""
    cif = next(pred_dir.rglob("*_model_0.cif"), None)
    conf = next(pred_dir.rglob("confidence_*_model_0.json"), None)
    return (cif, conf) if cif and conf else None


def run_usalign(usalign_bin: Path, pred_cif: Path, ref_pdb: Path) -> dict:
    cmd = [str(usalign_bin), str(pred_cif), str(ref_pdb), "-mm", "1", "-ter", "1"]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = proc.stdout
    tm_by_ref = {}
    for m in TM_RE.finditer(out):
        tm_by_ref[int(m.group(2))] = {
            "tm": float(m.group(1)),
            "length": int(m.group(3)),
            "d0": float(m.group(4)),
        }
    rmsd_match = RMSD_RE.search(out)
    alen_match = ALEN_RE.search(out)
    return {
        "tm_ref_pred": tm_by_ref.get(1, {}).get("tm"),
        "tm_ref_crystal": tm_by_ref.get(2, {}).get("tm"),
        "len_pred": tm_by_ref.get(1, {}).get("length"),
        "len_crystal": tm_by_ref.get(2, {}).get("length"),
        "rmsd": float(rmsd_match.group(1)) if rmsd_match else None,
        "aligned_len": int(alen_match.group(1)) if alen_match else None,
        "stdout": out,
    }


def parse_confidence(conf_path: Path) -> dict:
    d = json.loads(conf_path.read_text())
    return {
        "iptm": d.get("iptm") or d.get("complex_iptm"),
        "ptm": d.get("ptm") or d.get("complex_ptm"),
        "plddt": d.get("complex_plddt"),
        "confidence_score": d.get("confidence_score"),
    }


def iter_predictions(out_root: Path, pdb_ids: set[str] | None,
                     modes: set[str] | None):
    for mode_dir in sorted(out_root.iterdir()):
        if not mode_dir.is_dir():
            continue
        mode = mode_dir.name
        if modes and mode not in modes:
            continue
        for pred_dir in sorted(mode_dir.iterdir()):
            if not pred_dir.is_dir():
                continue
            # {PDB}_{mode}/
            pdb_id = pred_dir.name.removesuffix(f"_{mode}")
            if pdb_ids and pdb_id not in pdb_ids:
                continue
            yield pdb_id, mode, pred_dir


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    ap.add_argument("--usalign-bin", type=Path, default=DEFAULT_USALIGN)
    ap.add_argument("--pdb-ids", default="", help="comma-separated; default = all")
    ap.add_argument("--modes", default="", help="comma-separated; default = all")
    args = ap.parse_args()

    paths = get_paths(args.dataset)
    out_root = paths.output_root
    crystal_dir = paths.cleaned_dir
    tm_csv = out_root / "tm_scores.csv"

    if not args.usalign_bin.exists():
        print(f"[fatal] USalign not found: {args.usalign_bin}", file=sys.stderr)
        print("         compile: g++ -O3 -ffast-math -o USalign USalign.cpp", file=sys.stderr)
        return 2
    if not out_root.exists():
        print(f"[fatal] output root missing: {out_root} — run step 04 first", file=sys.stderr)
        return 2

    pdb_ids = {s.strip() for s in args.pdb_ids.split(",") if s.strip()} or None
    modes = {s.strip() for s in args.modes.split(",") if s.strip()} or None

    rows: list[dict] = []
    for pdb_id, mode, pred_dir in iter_predictions(out_root, pdb_ids, modes):
        arts = find_prediction_artifacts(pred_dir)
        if arts is None:
            print(f"[skip] {pdb_id}/{mode}: no *_model_0.cif / confidence JSON under {pred_dir}")
            continue
        pred_cif, conf_json = arts
        ref_pdb = crystal_dir / f"{pdb_id}_AB.pdb"
        if not ref_pdb.exists():
            print(f"[skip] {pdb_id}/{mode}: reference crystal missing at {ref_pdb}")
            continue
        try:
            tm = run_usalign(args.usalign_bin, pred_cif, ref_pdb)
        except subprocess.CalledProcessError as e:
            print(f"[FAIL] {pdb_id}/{mode}: USalign failed: {e.stderr[:300]}")
            continue
        conf = parse_confidence(conf_json)
        row = {
            "pdb_id": pdb_id,
            "mode": mode,
            "tm_ref_crystal": tm["tm_ref_crystal"],
            "tm_ref_pred": tm["tm_ref_pred"],
            "rmsd": tm["rmsd"],
            "aligned_len": tm["aligned_len"],
            "len_pred": tm["len_pred"],
            "len_crystal": tm["len_crystal"],
            "iptm": conf["iptm"],
            "ptm": conf["ptm"],
            "plddt": conf["plddt"],
            "confidence_score": conf["confidence_score"],
        }
        rows.append(row)
        print(
            f"[ ok ] {pdb_id:>6}/{mode:<13} "
            f"TM(crystal)={row['tm_ref_crystal']:.3f} "
            f"TM(pred)={row['tm_ref_pred']:.3f} "
            f"RMSD={row['rmsd']:.2f}A "
            f"ipTM={row['iptm']:.3f} "
            f"pTM={row['ptm']:.3f} "
            f"pLDDT={row['plddt']:.3f}"
        )

    if not rows:
        print("\n[warn] no predictions found. Run step 04 first.")
        return 1

    tm_csv.parent.mkdir(parents=True, exist_ok=True)
    with tm_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n[{paths.display}]")
    print(f"Wrote {len(rows)} rows to {tm_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
