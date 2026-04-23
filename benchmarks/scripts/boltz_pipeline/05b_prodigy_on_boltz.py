#!/usr/bin/env python3
"""Run standard PRODIGY IC-NIS on each Boltz-predicted structure.

Consumes Boltz outputs from step 4, emits ``prodigy_scores.csv`` with the
standard (non-PAE) predicted dG. Feeds step 6 panel 2.

Columns:
  pdb_id, mode,
  dg_pred_boltz,        # PRODIGY on Boltz-predicted CIF
  dg_exp,               # experimental ground truth (from manifest)
  dg_prodigy_baseline,  # PRODIGY ba_val on crystal (if dataset provides it)
  ic_cc, ic_ca, ic_pp, ic_pa, nis_a, nis_c  # contact + NIS features

Uses ``backend="cpu"`` for freesasa speed — 162 single-shot calls; JAX
compile-per-shape overhead would dominate.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from dataset_registry import AVAILABLE, get_paths  # noqa: E402
from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad  # noqa: E402


def find_predicted_cif(out_root: Path, mode: str, pdb_id: str) -> Path | None:
    pred_dir = out_root / mode / f"{pdb_id}_{mode}"
    return next(pred_dir.rglob("*_model_0.cif"), None)


def score_one(cif: Path, selection: str = "A,B"):
    """Run PRODIGY via tinygrad. ``mode="bucketed"`` pads N to a multiple of
    ``bucket_step`` so the TinyJit cache is keyed on a handful of shapes --
    one compile amortises across many structures in the sweep.
    """
    res = predict_binding_affinity_tinygrad(
        cif,
        selection=selection,
        quiet=True,
        mode="bucketed",
        bucket_step=2048,
    )
    dg = float(res.binding_affinity)
    contacts = list(res.contact_types.values)  # [AA, CC, PP, AC, AP, CP]
    return dg, {
        "ic_cc": float(contacts[1]),  # CC
        "ic_ca": float(contacts[3]),  # AC
        "ic_pp": float(contacts[2]),  # PP
        "ic_pa": float(contacts[4]),  # AP
        "nis_a": float(res.nis_aliphatic),
        "nis_c": float(res.nis_charged),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    args = ap.parse_args()

    paths = get_paths(args.dataset)
    out_root = paths.output_root
    manifest_csv = paths.manifest
    tm_csv = out_root / "tm_scores.csv"
    prodigy_csv = out_root / "prodigy_scores.csv"

    manifest = {r["pdb_id"]: r for r in csv.DictReader(manifest_csv.open())}
    if not tm_csv.exists():
        print(f"[fatal] {tm_csv} not found — run 05_mmalign_tm.py first")
        sys.exit(2)
    tm_rows = list(csv.DictReader(tm_csv.open()))
    if not tm_rows:
        print(f"[fatal] empty TM CSV: {tm_csv}")
        sys.exit(2)

    rows: list[dict] = []
    fails: list[tuple[str, str, str]] = []
    for tm_row in tm_rows:
        pdb_id, mode = tm_row["pdb_id"], tm_row["mode"]
        cif = find_predicted_cif(out_root, mode, pdb_id)
        if cif is None:
            fails.append((pdb_id, mode, "no predicted CIF"))
            continue
        try:
            dg, feats = score_one(cif)
        except Exception as exc:  # noqa: BLE001
            fails.append((pdb_id, mode, str(exc)[:160]))
            print(f"[FAIL] {pdb_id}/{mode}: {exc}")
            continue
        manifest_row = manifest[pdb_id]
        baseline = ""
        if paths.has_prodigy_baseline and manifest_row.get("ba_val_prodigy"):
            baseline = float(manifest_row["ba_val_prodigy"])
        row = {
            "pdb_id": pdb_id,
            "mode": mode,
            "dg_pred_boltz": dg,
            "dg_exp": float(manifest_row["dg_exp"]),
            "dg_prodigy_baseline": baseline,
            **feats,
        }
        rows.append(row)
        bline = (f"baseline={baseline:+7.2f}" if isinstance(baseline, float)
                 else "baseline=    n/a")
        print(
            f"[ ok ] {pdb_id:>6}/{mode:<13}"
            f"  dG_pred={dg:+7.2f}"
            f"  dG_exp={row['dg_exp']:+7.2f}"
            f"  {bline}"
        )

    if not rows:
        print("[fatal] no rows produced")
        sys.exit(1)

    with prodigy_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{paths.display}]")
    print(f"Wrote {len(rows)} rows to {prodigy_csv}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, m, err in fails:
            print(f"  {pid}/{m}: {err}")


if __name__ == "__main__":
    main()
