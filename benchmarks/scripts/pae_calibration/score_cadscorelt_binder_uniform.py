#!/usr/bin/env python3
"""Uniform binder-fold CAD-score-LT on all K81/V106/PB complexes.

The original ``score_cadscorelt_complex.py`` scores inter-chain contacts
(Boltz complex vs crystal complex) for K81/V106. PB uses single-chain
binder-fold CAD (ESMFold vs Boltz binder chain) via
``score_cadscorelt_proteinbase.py``. Column names are shared but semantics
differ.

This script produces **uniform binder-fold CAD** for **K81 and V106** by:
- target = crystal binder chain B (extracted from ``{pdb}_AB.pdb``)
- model  = Boltz binder chain B  (extracted from the predicted CIF)
- subselect_contacts = "[-min-sep 1]"  (single-chain, matches PB)
- record_local_scores = True

The 65-feature schema matches both ``score_cadscorelt_complex`` and
``score_cadscorelt_proteinbase``. Combined with PB's existing output,
all 287 complexes share identically-computed CAD columns.

Outputs (per dataset):
    {boltz_output}/pae_calibration/cadscorelt_binder/
        cadscore_binder_features_msa_only.csv
        cadscore_binder_arrays_msa_only.jsonl

Usage:
    python score_cadscorelt_binder_uniform.py --dataset vreven
    python score_cadscorelt_binder_uniform.py --dataset kastritis
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

try:
    import cadscorelt
except ImportError as exc:
    raise SystemExit("cadscorelt required") from exc

try:
    from Bio.PDB import MMCIFParser, PDBIO, PDBParser, Select
except ImportError as exc:
    raise SystemExit("Biopython required") from exc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "benchmarks/scripts/boltz_pipeline"))

import quick_pae_calib as qpc  # noqa: E402
from dataset_registry import AVAILABLE, get_paths  # noqa: E402
from score_cadscorelt_complex import (  # noqa: E402
    BACKBONE_ATOMS, _dist_stats, _first_row_dict,
)


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)
    def accept_chain(self, chain) -> bool:  # noqa: ANN001
        return chain.id in self.chain_ids


def _extract_chain_to_pdb(src_path: Path, chain_id: str, out_path: Path) -> None:
    """Write a single chain from a PDB/CIF source into an output PDB."""
    if src_path.suffix.lower() == ".cif":
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure(src_path.stem, str(src_path))
    chain_ids = {c.id for c in structure.get_chains()}
    if chain_id not in chain_ids:
        raise ValueError(
            f"chain {chain_id!r} not in {src_path}; found {sorted(chain_ids)}"
        )
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect([chain_id]))


def find_predicted_cif(out_root: Path, mode: str, pdb_id: str) -> Path | None:
    pred = out_root / mode / f"{pdb_id}_{mode}"
    return next(pred.rglob("*_model_0.cif"), None)


def score_one(reference_complex: Path, prediction: Path,
              chain: str = "B") -> tuple[dict, dict]:
    """Single-chain binder-fold CAD.

    Extracts chain ``chain`` from both sides, runs CAD with [-min-sep 1]
    and records local scores. Returns (flat_features, arrays).
    """
    with tempfile.TemporaryDirectory(prefix="k81v106-binder-cad-") as tmp:
        tmp_p = Path(tmp)
        ref_binder = tmp_p / "ref_binder.pdb"
        mod_binder = tmp_p / "mod_binder.pdb"
        _extract_chain_to_pdb(reference_complex, chain, ref_binder)
        _extract_chain_to_pdb(prediction, chain, mod_binder)

        csc = cadscorelt.CADScoreComputer.init(
            remap_chains=True,
            subselect_contacts="[-min-sep 1]",
            score_atom_atom_contacts=True,
            record_local_scores=True,
        )
        csc.add_target_structure_from_file(str(ref_binder), "ref_binder")
        csc.add_model_structure_from_file(str(mod_binder), "mod_binder")

        rr_global = _first_row_dict(
            csc.get_all_cadscores_residue_residue_summarized_globally())
        aa_global = _first_row_dict(
            csc.get_all_cadscores_atom_atom_summarized_globally())

        df_resi = csc.get_local_cadscores_residue_residue_summarized_per_residue(
            "ref_binder", "mod_binder").to_pandas()
        df_atom = csc.get_local_cadscores_atom_atom_summarized_per_atom(
            "ref_binder", "mod_binder").to_pandas()
        df_rrc = csc.get_local_cadscores_residue_residue(
            "ref_binder", "mod_binder").to_pandas()
        df_aac = csc.get_local_cadscores_atom_atom(
            "ref_binder", "mod_binder").to_pandas()

    def _f(d, k):
        v = d.get(k)
        try: return float(v) if v is not None else float("nan")
        except (TypeError, ValueError): return float("nan")

    feats = {
        "cad_rr": _f(rr_global, "CAD_score"),
        "cad_rr_f1": _f(rr_global, "F1_of_areas"),
        "cad_rr_target_area": _f(rr_global, "target_area"),
        "cad_rr_model_area": _f(rr_global, "model_area"),
        "cad_rr_tp": _f(rr_global, "TP_area"),
        "cad_rr_fp": _f(rr_global, "FP_area"),
        "cad_rr_fn": _f(rr_global, "FN_area"),
        "cad_aa": _f(aa_global, "CAD_score"),
        "cad_aa_f1": _f(aa_global, "F1_of_areas"),
    }

    resi_cad = df_resi["CAD_score"].to_numpy() if len(df_resi) else np.array([])
    feats.update(_dist_stats(resi_cad, "resi_cad"))
    v = resi_cad[resi_cad != -1.0] if len(resi_cad) else resi_cad
    feats["resi_cad_A_mean"] = float(v.mean()) if len(v) else float("nan")
    feats["resi_cad_B_mean"] = float("nan")      # single chain

    atom_cad = df_atom["CAD_score"].to_numpy() if len(df_atom) else np.array([])
    feats.update(_dist_stats(atom_cad, "atom_cad"))
    if len(df_atom):
        dfa = df_atom[df_atom["CAD_score"] != -1.0]
        bb = dfa[dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        sc = dfa[~dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        feats["atom_cad_A_mean"] = (
            float(dfa["CAD_score"].mean()) if len(dfa) else float("nan"))
        feats["atom_cad_B_mean"] = float("nan")
        feats["atom_cad_bb_mean"] = float(bb.mean()) if len(bb) else float("nan")
        feats["atom_cad_sc_mean"] = float(sc.mean()) if len(sc) else float("nan")
    else:
        for k in ("A_mean", "B_mean", "bb_mean", "sc_mean"):
            feats[f"atom_cad_{k}"] = float("nan")

    rrc_cad = df_rrc["CAD_score"].to_numpy() if len(df_rrc) else np.array([])
    rrc_stats = _dist_stats(rrc_cad, "rrc_cad", drop_false_positives=False)
    for k in ("mean", "std", "p10", "p50", "p90",
              "frac_below_0_3", "frac_above_0_7"):
        feats[f"rrc_cad_{k}"] = rrc_stats[f"rrc_cad_{k}"]
    if len(df_rrc):
        feats["rrc_n_total"] = int(len(df_rrc))
        feats["rrc_n_model_only"] = int((df_rrc["CAD_score"] == -1.0).sum())
        feats["rrc_n_shared"] = int((df_rrc["CAD_score"] >= 0).sum())
    else:
        feats["rrc_n_total"] = 0; feats["rrc_n_model_only"] = 0
        feats["rrc_n_shared"] = 0

    aac_cad = df_aac["CAD_score"].to_numpy() if len(df_aac) else np.array([])
    aac_stats = _dist_stats(aac_cad, "aac_cad", drop_false_positives=False)
    for k in ("mean", "std", "p10", "p50", "p90",
              "frac_below_0_3", "frac_above_0_7"):
        feats[f"aac_cad_{k}"] = aac_stats[f"aac_cad_{k}"]
    if len(df_aac):
        feats["aac_n_total"] = int(len(df_aac))
        feats["aac_n_model_only"] = int((df_aac["CAD_score"] == -1.0).sum())
        feats["aac_n_shared"] = int((df_aac["CAD_score"] >= 0).sum())
    else:
        feats["aac_n_total"] = 0; feats["aac_n_model_only"] = 0
        feats["aac_n_shared"] = 0

    arrays = {
        "per_residue_cad": df_resi[
            ["ID_chain", "ID_rnum", "CAD_score", "F1_of_areas"]
        ].to_dict("records"),
        "per_atom_cad": df_atom[
            ["ID_chain", "ID_rnum", "ID_atom_name", "CAD_score", "F1_of_areas"]
        ].to_dict("records"),
        "per_contact_rr": df_rrc[
            ["ID1_chain", "ID1_rnum", "ID2_chain", "ID2_rnum", "CAD_score"]
        ].to_dict("records"),
        "per_contact_aa": df_aac[
            ["ID1_chain", "ID1_rnum", "ID1_atom_name",
             "ID2_chain", "ID2_rnum", "ID2_atom_name", "CAD_score"]
        ].to_dict("records"),
    }
    return feats, arrays


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--chain", default="B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    paths = get_paths(args.dataset)
    qpc.set_dataset(args.dataset)
    out_dir = paths.output_root / "pae_calibration" / "cadscorelt_binder"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"cadscore_binder_features_{args.mode}.csv"
    out_jsonl = out_dir / f"cadscore_binder_arrays_{args.mode}.jsonl"

    with paths.manifest.open() as f:
        pdb_ids = [row["pdb_id"] for row in csv.DictReader(f)]
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
            feats, arrays = score_one(ref, pred, chain=args.chain)
        except Exception as exc:  # noqa: BLE001
            fails.append((pid, str(exc)[:120]))
            print(f"[FAIL] {pid}: {exc}")
            continue
        row = {"pdb_id": pid, "mode": args.mode, **feats}
        rows.append(row)
        jsonl_fh.write(json.dumps({
            "pdb_id": pid, "mode": args.mode, **arrays,
        }, default=float) + "\n")
        if i % 20 == 0 or i == len(pdb_ids) or i == 1:
            print(f"[{i:>3d}/{len(pdb_ids)}] {pid:<6s}  "
                  f"cad_rr={feats['cad_rr']:.3f}  "
                  f"atom_cad_mean={feats['atom_cad_mean']:.3f}")
    jsonl_fh.close()

    if not rows:
        print("[fatal] no rows produced"); return 1

    fieldnames = ["pdb_id", "mode"] + sorted(set(rows[0].keys()) - {"pdb_id", "mode"})
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, restval="")
        w.writeheader()
        w.writerows(rows)
    print(f"\n[{paths.display}]  wrote {len(rows)} rows × {len(fieldnames)} cols "
          f"in {time.time() - t0:.1f}s")
    print(f"  CSV:   {out_csv}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, err in fails[:10]: print(f"  {pid}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
