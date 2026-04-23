#!/usr/bin/env python3
"""Compute interface CAD-score-LT + local distribution features.

For each complex, compares the Boltz msa_only model_0 prediction to the
cleaned crystal AB.pdb with ``subselect_contacts="[-inter-chain]"`` (interface
only) and records local scores at residue, atom, and per-contact granularity.

Output columns per complex:

  Global:
    cad_rr, cad_rr_f1, cad_aa, cad_aa_f1,
    cad_rr_target_area, cad_rr_model_area,
    cad_rr_tp, cad_rr_fp, cad_rr_fn

  Per-residue distribution (at interface residues; A=target, B=binder):
    resi_cad_{mean,std,min,p10,p25,p50,p75,p90,max},
    resi_cad_{A,B}_mean,
    resi_cad_frac_{below_0_3,below_0_5,above_0_7,above_0_9},
    resi_n_total, resi_n_false_positive

  Per-atom distribution (backbone = N CA C O, sidechain = other):
    atom_cad_{mean,std,min,p10,p25,p50,p75,p90,max},
    atom_cad_{A,B}_mean, atom_cad_{bb,sc}_mean,
    atom_cad_frac_{below_0_3,below_0_5,above_0_7,above_0_9},
    atom_n_total, atom_n_false_positive

  Per-contact residue-residue:
    rrc_cad_{mean,std,p10,p50,p90},
    rrc_n_{model_only,target_only,shared,total},
    rrc_cad_frac_{below_0_3,above_0_7}

  Per-contact atom-atom:
    aac_cad_{mean,std,p10,p50,p90},
    aac_n_{model_only,target_only,shared,total},
    aac_cad_frac_{below_0_3,above_0_7}

In addition, a JSONL file ``cadscore_arrays_{mode}.jsonl`` ships the full
per-residue and per-atom arrays (one record per complex) for downstream
models that want to consume raw distributions.

Usage:
    python score_cadscorelt_complex.py --dataset vreven
    python score_cadscorelt_complex.py --dataset kastritis --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

try:
    import cadscorelt
except ImportError as exc:
    raise SystemExit(
        "cadscorelt is required. Install: "
        "`.venv/bin/python -m pip install cadscorelt`"
    ) from exc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "benchmarks/scripts/boltz_pipeline"))

import quick_pae_calib as qpc  # noqa: E402
from dataset_registry import AVAILABLE, get_paths  # noqa: E402

BACKBONE_ATOMS = {"N", "CA", "C", "O"}


def find_predicted_cif(out_root: Path, mode: str, pdb_id: str) -> Path | None:
    pred = out_root / mode / f"{pdb_id}_{mode}"
    return next(pred.rglob("*_model_0.cif"), None)


def _dist_stats(values: np.ndarray, prefix: str,
                drop_false_positives: bool = True) -> dict[str, float]:
    """Compute {mean, std, min, p10, p25, p50, p75, p90, max, fraction} stats."""
    out: dict[str, float] = {}
    if drop_false_positives:
        mask = values != -1.0
        v = values[mask]
    else:
        v = values
    total = len(values)
    n_fp = int((values == -1.0).sum())
    if len(v) == 0:
        for k in ("mean", "std", "min", "p10", "p25", "p50", "p75", "p90", "max"):
            out[f"{prefix}_{k}"] = float("nan")
        for k in ("frac_below_0_3", "frac_below_0_5",
                   "frac_above_0_7", "frac_above_0_9"):
            out[f"{prefix}_{k}"] = float("nan")
    else:
        out[f"{prefix}_mean"] = float(v.mean())
        out[f"{prefix}_std"] = float(v.std()) if len(v) > 1 else 0.0
        out[f"{prefix}_min"] = float(v.min())
        out[f"{prefix}_max"] = float(v.max())
        for p, name in ((10, "p10"), (25, "p25"), (50, "p50"),
                          (75, "p75"), (90, "p90")):
            out[f"{prefix}_{name}"] = float(np.percentile(v, p))
        out[f"{prefix}_frac_below_0_3"] = float((v < 0.3).mean())
        out[f"{prefix}_frac_below_0_5"] = float((v < 0.5).mean())
        out[f"{prefix}_frac_above_0_7"] = float((v > 0.7).mean())
        out[f"{prefix}_frac_above_0_9"] = float((v > 0.9).mean())
    out[f"{prefix.split('_cad')[0]}_n_total"] = total
    out[f"{prefix.split('_cad')[0]}_n_false_positive"] = n_fp
    return out


def _first_row_dict(table) -> dict:
    df = table.to_pandas()
    return df.iloc[0].to_dict() if len(df) else {}


def score_one(reference: Path, prediction: Path) -> tuple[dict, dict]:
    """Run CAD-score-LT with local recording. Returns (flat_features, arrays)."""
    csc = cadscorelt.CADScoreComputer.init(
        remap_chains=True,
        subselect_contacts="[-inter-chain]",
        score_atom_atom_contacts=True,
        record_local_scores=True,
    )
    csc.add_target_structure_from_file(str(reference), "crystal")
    csc.add_model_structure_from_file(str(prediction), "boltz")

    rr_global = _first_row_dict(csc.get_all_cadscores_residue_residue_summarized_globally())
    aa_global = _first_row_dict(csc.get_all_cadscores_atom_atom_summarized_globally())

    df_resi = csc.get_local_cadscores_residue_residue_summarized_per_residue(
        "crystal", "boltz").to_pandas()
    df_atom = csc.get_local_cadscores_atom_atom_summarized_per_atom(
        "crystal", "boltz").to_pandas()
    df_rrc = csc.get_local_cadscores_residue_residue("crystal", "boltz").to_pandas()
    df_aac = csc.get_local_cadscores_atom_atom("crystal", "boltz").to_pandas()

    feats: dict = {}

    # Global
    def _f(d: dict, k: str) -> float:
        v = d.get(k)
        try:
            return float(v) if v is not None else float("nan")
        except (TypeError, ValueError):
            return float("nan")
    feats["cad_rr"] = _f(rr_global, "CAD_score")
    feats["cad_rr_f1"] = _f(rr_global, "F1_of_areas")
    feats["cad_rr_target_area"] = _f(rr_global, "target_area")
    feats["cad_rr_model_area"] = _f(rr_global, "model_area")
    feats["cad_rr_tp"] = _f(rr_global, "TP_area")
    feats["cad_rr_fp"] = _f(rr_global, "FP_area")
    feats["cad_rr_fn"] = _f(rr_global, "FN_area")
    feats["cad_aa"] = _f(aa_global, "CAD_score")
    feats["cad_aa_f1"] = _f(aa_global, "F1_of_areas")

    # Per-residue distribution
    resi_cad = df_resi["CAD_score"].to_numpy() if len(df_resi) else np.array([])
    feats.update(_dist_stats(resi_cad, "resi_cad"))
    if len(df_resi):
        chA = df_resi[df_resi["ID_chain"] == "A"]["CAD_score"].to_numpy()
        chB = df_resi[df_resi["ID_chain"] == "B"]["CAD_score"].to_numpy()
        chA = chA[chA != -1.0]
        chB = chB[chB != -1.0]
        feats["resi_cad_A_mean"] = float(chA.mean()) if len(chA) else float("nan")
        feats["resi_cad_B_mean"] = float(chB.mean()) if len(chB) else float("nan")
    else:
        feats["resi_cad_A_mean"] = float("nan")
        feats["resi_cad_B_mean"] = float("nan")

    # Per-atom distribution
    atom_cad = df_atom["CAD_score"].to_numpy() if len(df_atom) else np.array([])
    feats.update(_dist_stats(atom_cad, "atom_cad"))
    if len(df_atom):
        dfa = df_atom[df_atom["CAD_score"] != -1.0]
        chA = dfa[dfa["ID_chain"] == "A"]["CAD_score"].to_numpy()
        chB = dfa[dfa["ID_chain"] == "B"]["CAD_score"].to_numpy()
        bb = dfa[dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        sc = dfa[~dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        feats["atom_cad_A_mean"] = float(chA.mean()) if len(chA) else float("nan")
        feats["atom_cad_B_mean"] = float(chB.mean()) if len(chB) else float("nan")
        feats["atom_cad_bb_mean"] = float(bb.mean()) if len(bb) else float("nan")
        feats["atom_cad_sc_mean"] = float(sc.mean()) if len(sc) else float("nan")
    else:
        for k in ("A_mean", "B_mean", "bb_mean", "sc_mean"):
            feats[f"atom_cad_{k}"] = float("nan")

    # Per-contact residue-residue
    rrc_cad = df_rrc["CAD_score"].to_numpy() if len(df_rrc) else np.array([])
    rrc_stats = _dist_stats(rrc_cad, "rrc_cad",
                             drop_false_positives=False)  # keep -1 for diagnostics
    # rrc_n_total and rrc_n_false_positive were added by _dist_stats — rename
    feats["rrc_cad_mean"] = rrc_stats["rrc_cad_mean"]
    feats["rrc_cad_std"] = rrc_stats["rrc_cad_std"]
    feats["rrc_cad_p10"] = rrc_stats["rrc_cad_p10"]
    feats["rrc_cad_p50"] = rrc_stats["rrc_cad_p50"]
    feats["rrc_cad_p90"] = rrc_stats["rrc_cad_p90"]
    feats["rrc_cad_frac_below_0_3"] = rrc_stats["rrc_cad_frac_below_0_3"]
    feats["rrc_cad_frac_above_0_7"] = rrc_stats["rrc_cad_frac_above_0_7"]
    if len(df_rrc):
        # model_only contacts: CAD == -1 (false positive); shared: > 0;
        # target_only is implied by missing rows — we can't count directly
        # from this table, but the global FN area measures the same quantity.
        feats["rrc_n_total"] = int(len(df_rrc))
        feats["rrc_n_model_only"] = int((df_rrc["CAD_score"] == -1.0).sum())
        feats["rrc_n_shared"] = int((df_rrc["CAD_score"] >= 0).sum())
    else:
        feats["rrc_n_total"] = 0
        feats["rrc_n_model_only"] = 0
        feats["rrc_n_shared"] = 0

    # Per-contact atom-atom
    aac_cad = df_aac["CAD_score"].to_numpy() if len(df_aac) else np.array([])
    aac_stats = _dist_stats(aac_cad, "aac_cad", drop_false_positives=False)
    feats["aac_cad_mean"] = aac_stats["aac_cad_mean"]
    feats["aac_cad_std"] = aac_stats["aac_cad_std"]
    feats["aac_cad_p10"] = aac_stats["aac_cad_p10"]
    feats["aac_cad_p50"] = aac_stats["aac_cad_p50"]
    feats["aac_cad_p90"] = aac_stats["aac_cad_p90"]
    feats["aac_cad_frac_below_0_3"] = aac_stats["aac_cad_frac_below_0_3"]
    feats["aac_cad_frac_above_0_7"] = aac_stats["aac_cad_frac_above_0_7"]
    if len(df_aac):
        feats["aac_n_total"] = int(len(df_aac))
        feats["aac_n_model_only"] = int((df_aac["CAD_score"] == -1.0).sum())
        feats["aac_n_shared"] = int((df_aac["CAD_score"] >= 0).sum())
    else:
        feats["aac_n_total"] = 0
        feats["aac_n_model_only"] = 0
        feats["aac_n_shared"] = 0

    # Raw arrays for JSONL
    arrays = {
        "per_residue_cad": df_resi[["ID_chain", "ID_rnum", "CAD_score",
                                      "F1_of_areas"]].to_dict("records"),
        "per_atom_cad": df_atom[["ID_chain", "ID_rnum", "ID_atom_name",
                                  "CAD_score", "F1_of_areas"]].to_dict("records"),
        "per_contact_rr": df_rrc[["ID1_chain", "ID1_rnum", "ID2_chain",
                                    "ID2_rnum", "CAD_score"]].to_dict("records"),
        "per_contact_aa": df_aac[["ID1_chain", "ID1_rnum", "ID1_atom_name",
                                    "ID2_chain", "ID2_rnum", "ID2_atom_name",
                                    "CAD_score"]].to_dict("records"),
    }
    return feats, arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    paths = get_paths(args.dataset)
    qpc.set_dataset(args.dataset)
    out_dir = (Path(args.out_dir) if args.out_dir
               else paths.output_root / "pae_calibration" / "cadscorelt")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cadscore_features_{args.mode}.csv"
    out_jsonl = out_dir / f"cadscore_arrays_{args.mode}.jsonl"

    manifest = paths.manifest
    pdb_ids: list[str] = []
    with manifest.open() as f:
        for row in csv.DictReader(f):
            pdb_ids.append(row["pdb_id"])
    if args.limit > 0:
        pdb_ids = pdb_ids[:args.limit]

    cadscorelt.enable_considering_residue_names()

    rows: list[dict] = []
    fails: list[tuple[str, str]] = []
    t0 = time.time()
    jsonl_fh = out_jsonl.open("w")
    for i, pid in enumerate(pdb_ids, 1):
        ref = paths.cleaned_dir / f"{pid}_AB.pdb"
        pred = find_predicted_cif(paths.output_root, args.mode, pid)
        if not ref.exists():
            fails.append((pid, f"missing ref {ref}")); continue
        if pred is None:
            fails.append((pid, "no predicted CIF")); continue
        try:
            feats, arrays = score_one(ref, pred)
        except Exception as exc:  # noqa: BLE001
            fails.append((pid, str(exc)[:120]))
            print(f"[FAIL] {pid}: {exc}")
            continue
        row = {"pdb_id": pid, "mode": args.mode, **feats}
        rows.append(row)
        jsonl_fh.write(json.dumps({
            "pdb_id": pid, "mode": args.mode, **arrays,
        }, default=float) + "\n")
        if i % 10 == 0 or i == len(pdb_ids) or i == 1:
            print(f"[{i:>3d}/{len(pdb_ids)}] {pid:<6s}  "
                  f"cad_rr={feats['cad_rr']:.3f}  "
                  f"atom_cad_mean={feats['atom_cad_mean']:.3f}  "
                  f"resi_n_total={feats['resi_n_total']}  "
                  f"atom_n_total={feats['atom_n_total']}")
    jsonl_fh.close()

    if not rows:
        print("[fatal] no rows produced"); return 1

    all_fields = sorted({k for r in rows for k in r})
    lead = ["pdb_id", "mode"]
    fieldnames = lead + [f for f in all_fields if f not in lead]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{paths.display}]  wrote {len(rows)} rows × {len(fieldnames)} cols "
          f"in {time.time() - t0:.1f}s")
    print(f"  CSV:   {out_csv}")
    print(f"  JSONL: {out_jsonl}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, err in fails[:10]: print(f"  {pid}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
