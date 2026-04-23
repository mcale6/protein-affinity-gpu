#!/usr/bin/env python3
"""Run standard PRODIGY IC-NIS on each Boltz-predicted structure.

Consumes Boltz outputs from step 4, emits ``prodigy_scores.csv`` with the
standard (non-PAE) predicted dG. Feeds step 6 panel 2.

Columns:
  pdb_id, mode,
  dg_pred_boltz,        # PRODIGY on Boltz-predicted CIF
  dg_exp,               # experimental ground truth (from manifest)
  dg_prodigy_baseline,  # PRODIGY's own published ba_val (from dataset.json; on crystal)
  ic_cc, ic_ca, ic_pp, ic_pa, nis_a, nis_c  # contact + NIS features

Uses ``backend="cpu"`` for freesasa speed — 162 single-shot calls; JAX
compile-per-shape overhead would dominate.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from protein_affinity_gpu.experimental import predict_binding_affinity_tinygrad  # noqa: E402

OUT_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
MANIFEST_CSV = ROOT / "benchmarks/datasets/kastritis_81/manifest.csv"
TM_CSV = OUT_ROOT / "tm_scores.csv"
PRODIGY_CSV = OUT_ROOT / "prodigy_scores.csv"


def find_predicted_cif(mode: str, pdb_id: str) -> Path | None:
    pred_dir = OUT_ROOT / mode / f"{pdb_id}_{mode}"
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
    manifest = {r["pdb_id"]: r for r in csv.DictReader(MANIFEST_CSV.open())}
    tm_rows = list(csv.DictReader(TM_CSV.open()))
    if not tm_rows:
        print(f"[fatal] empty TM CSV: {TM_CSV}")
        sys.exit(2)

    rows: list[dict] = []
    fails: list[tuple[str, str, str]] = []
    for tm_row in tm_rows:
        pdb_id, mode = tm_row["pdb_id"], tm_row["mode"]
        cif = find_predicted_cif(mode, pdb_id)
        if cif is None:
            fails.append((pdb_id, mode, "no predicted CIF"))
            continue
        try:
            dg, feats = score_one(cif)
        except Exception as exc:  # noqa: BLE001
            fails.append((pdb_id, mode, str(exc)[:160]))
            print(f"[FAIL] {pdb_id}/{mode}: {exc}")
            continue
        row = {
            "pdb_id": pdb_id,
            "mode": mode,
            "dg_pred_boltz": dg,
            "dg_exp": float(manifest[pdb_id]["dg_exp"]),
            "dg_prodigy_baseline": float(manifest[pdb_id]["ba_val_prodigy"]),
            **feats,
        }
        rows.append(row)
        print(
            f"[ ok ] {pdb_id:>6}/{mode:<13}"
            f"  dG_pred={dg:+7.2f}"
            f"  dG_exp={row['dg_exp']:+7.2f}"
            f"  baseline={row['dg_prodigy_baseline']:+7.2f}"
        )

    if not rows:
        print("[fatal] no rows produced")
        sys.exit(1)

    with PRODIGY_CSV.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {PRODIGY_CSV.relative_to(ROOT)}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, m, err in fails:
            print(f"  {pid}/{m}: {err}")


if __name__ == "__main__":
    main()
