#!/usr/bin/env python3
"""Compute local CAD-score-LT stats for ProteinBase (binder-fold geometry).

PB's existing CAD script (``benchmarks/scripts/proteinbase_pipeline/
04_score_cadscorelt.py``) compares the ESMFold binder-only structure to
the Boltz binder chain extracted from the complex. It stores only the 9
global CAD columns.

This script runs the same single-chain comparison but with
``record_local_scores=True`` to extract the 65 per-residue / per-atom /
per-contact distribution features that K81 / V106 already have under
``score_cadscorelt_complex.py``. The resulting CSV matches the K81/V106
schema column-for-column — filling the atom-CAD NaN gap that blocks PB
from generalising in the unified Phase 3 models.

**Semantic note.** For PB the CAD stats describe *binder-fold quality*
(ESMFold vs Boltz binder-in-complex); for K81/V106 they describe
*interface quality* (crystal complex vs Boltz complex). Both are
"how trustworthy is the region that matters for affinity" but on
different scopes. Consumers of the unified dataset can consult
``cad_arrays_jsonl`` for the raw arrays and the ``source`` column for
semantic context.

Usage:
    python score_cadscorelt_proteinbase.py
    python score_cadscorelt_proteinbase.py --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
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

try:
    from Bio.PDB import MMCIFParser, PDBIO, Select
except ImportError as exc:
    raise SystemExit(
        "Biopython required for chain extraction: "
        "`.venv/bin/python -m pip install biopython`"
    ) from exc

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from score_cadscorelt_complex import (  # noqa: E402 — reuse stat extractors
    BACKBONE_ATOMS, _dist_stats, _first_row_dict,
)

DEFAULT_ROWS = ROOT / "benchmarks/output/proteinbase/proteinbase_kd_boltz_pae_rows.csv"
DEFAULT_STRUCT_DIR = ROOT / "benchmarks/downloads/proteinbase/structures"
DEFAULT_ESMFOLD_DIR = ROOT / "benchmarks/downloads/proteinbase/esmfold"
OUT_DIR = ROOT / "benchmarks/output/proteinbase/pae_calibration/cadscorelt"


class ChainSelect(Select):
    def __init__(self, chain_ids):
        self.chain_ids = set(chain_ids)

    def accept_chain(self, chain) -> bool:  # noqa: ANN001
        return chain.id in self.chain_ids


def _safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _structure_path(row, structures_dir: Path) -> Path:
    stem = _safe_name(f"{row['proteinbase_id']}_{row['target']}")
    return structures_dir / f"{stem}.cif"


def _esmfold_path(row, esmfold_dir: Path) -> Path:
    stem = _safe_name(f"{row['proteinbase_id']}_esmfold")
    return esmfold_dir / f"{stem}.cif"


def _write_chain_pdb(structure_path: Path, chain_id: str, out_path: Path) -> None:
    structure = MMCIFParser(QUIET=True).get_structure(
        structure_path.stem, str(structure_path)
    )
    chain_ids = {chain.id for chain in structure.get_chains()}
    if chain_id not in chain_ids:
        raise ValueError(
            f"Chain {chain_id!r} not in {structure_path}; found {sorted(chain_ids)!r}"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    io = PDBIO()
    io.set_structure(structure)
    io.save(str(out_path), ChainSelect([chain_id]))


def score_one(target_path: Path, model_complex_path: Path,
              model_chain: str = "B") -> tuple[dict, dict]:
    """Single-chain CAD with local scores.

    Target: ESMFold binder (full CIF, single chain).
    Model: Boltz binder chain extracted from the complex CIF.
    """
    with tempfile.TemporaryDirectory(prefix="pb-cad-local-") as tmp:
        model_pdb = Path(tmp) / "boltz_binder.pdb"
        _write_chain_pdb(model_complex_path, model_chain, model_pdb)

        csc = cadscorelt.CADScoreComputer.init(
            remap_chains=True,
            subselect_contacts="[-min-sep 1]",
            score_atom_atom_contacts=True,
            record_local_scores=True,
        )
        csc.add_target_structure_from_file(str(target_path), "esmfold_binder")
        csc.add_model_structure_from_file(str(model_pdb), "boltz_binder")

        rr_global = _first_row_dict(
            csc.get_all_cadscores_residue_residue_summarized_globally())
        aa_global = _first_row_dict(
            csc.get_all_cadscores_atom_atom_summarized_globally())

        df_resi = csc.get_local_cadscores_residue_residue_summarized_per_residue(
            "esmfold_binder", "boltz_binder").to_pandas()
        df_atom = csc.get_local_cadscores_atom_atom_summarized_per_atom(
            "esmfold_binder", "boltz_binder").to_pandas()
        df_rrc = csc.get_local_cadscores_residue_residue(
            "esmfold_binder", "boltz_binder").to_pandas()
        df_aac = csc.get_local_cadscores_atom_atom(
            "esmfold_binder", "boltz_binder").to_pandas()

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

    # Per-residue
    resi_cad = df_resi["CAD_score"].to_numpy() if len(df_resi) else np.array([])
    feats.update(_dist_stats(resi_cad, "resi_cad"))
    # Single-chain: no A/B split meaningful; fill A mean with overall, B mean NaN.
    v = resi_cad[resi_cad != -1.0] if len(resi_cad) else resi_cad
    feats["resi_cad_A_mean"] = float(v.mean()) if len(v) else float("nan")
    feats["resi_cad_B_mean"] = float("nan")

    # Per-atom
    atom_cad = df_atom["CAD_score"].to_numpy() if len(df_atom) else np.array([])
    feats.update(_dist_stats(atom_cad, "atom_cad"))
    if len(df_atom):
        dfa = df_atom[df_atom["CAD_score"] != -1.0]
        bb = dfa[dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        sc = dfa[~dfa["ID_atom_name"].isin(BACKBONE_ATOMS)]["CAD_score"].to_numpy()
        feats["atom_cad_A_mean"] = (
            float(dfa["CAD_score"].mean()) if len(dfa) else float("nan")
        )
        feats["atom_cad_B_mean"] = float("nan")
        feats["atom_cad_bb_mean"] = float(bb.mean()) if len(bb) else float("nan")
        feats["atom_cad_sc_mean"] = float(sc.mean()) if len(sc) else float("nan")
    else:
        for k in ("A_mean", "B_mean", "bb_mean", "sc_mean"):
            feats[f"atom_cad_{k}"] = float("nan")

    # Per-contact rr
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
        feats["rrc_n_total"] = 0
        feats["rrc_n_model_only"] = 0
        feats["rrc_n_shared"] = 0

    # Per-contact aa
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
        feats["aac_n_total"] = 0
        feats["aac_n_model_only"] = 0
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
    ap.add_argument("--rows", type=Path, default=DEFAULT_ROWS)
    ap.add_argument("--structures-dir", type=Path, default=DEFAULT_STRUCT_DIR)
    ap.add_argument("--esmfold-dir", type=Path, default=DEFAULT_ESMFOLD_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--model-chain", default="B")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = args.out_dir / "cadscore_features_msa_only.csv"
    out_jsonl = args.out_dir / "cadscore_arrays_msa_only.jsonl"

    if not args.rows.exists():
        raise SystemExit(f"Missing rows CSV: {args.rows}")

    with args.rows.open() as f:
        rows_in = list(csv.DictReader(f))
    if args.limit > 0:
        rows_in = rows_in[:args.limit]

    cadscorelt.enable_considering_residue_names()

    rows: list[dict] = []
    fails: list[tuple[str, str]] = []
    t0 = time.time()
    jsonl_fh = out_jsonl.open("w")
    for i, r in enumerate(rows_in, 1):
        pid = r["proteinbase_id"]
        target = _esmfold_path(r, args.esmfold_dir)
        model = _structure_path(r, args.structures_dir)
        if not target.exists():
            fails.append((pid, f"missing ESMFold {target.name}")); continue
        if not model.exists():
            fails.append((pid, f"missing Boltz {model.name}")); continue
        try:
            feats, arrays = score_one(target, model, model_chain=args.model_chain)
        except Exception as exc:  # noqa: BLE001
            fails.append((pid, str(exc)[:120]))
            print(f"[FAIL] {pid}: {exc}")
            continue
        row = {"pdb_id": pid, "mode": "msa_only", **feats}
        rows.append(row)
        jsonl_fh.write(json.dumps({
            "pdb_id": pid, "mode": "msa_only", **arrays,
        }, default=float) + "\n")
        if i % 20 == 0 or i == len(rows_in) or i == 1:
            print(f"[{i:>3d}/{len(rows_in)}] {pid:<30s}  "
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
    print(f"\nWrote {len(rows)} rows × {len(fieldnames)} cols in "
          f"{time.time() - t0:.1f}s")
    print(f"  CSV:   {out_csv}")
    print(f"  JSONL: {out_jsonl}")
    if fails:
        print(f"Failures ({len(fails)}):")
        for pid, err in fails[:10]: print(f"  {pid}: {err}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
