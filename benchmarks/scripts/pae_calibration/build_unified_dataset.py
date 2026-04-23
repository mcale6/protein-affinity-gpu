#!/usr/bin/env python3
"""Build a unified features table across K81 + V106 + ProteinBase.

Each complex contributes one row with a ``source`` column identifying the
origin dataset. Columns common to all three are named consistently;
dataset-specific columns are preserved where available (NaN elsewhere).

Outputs:
    benchmarks/output/unified/unified_features.csv
    benchmarks/output/unified/unified_features.jsonl    (richer per-row)

CSV columns (common schema):
    pdb_id, source,
    dg_exp_kcal_mol, kd_m, log10_kd,
    irmsd, stratum,                                      # K81/V106 only
    # PRODIGY IC counts (stock 4 + 2 extended)
    ic_cc, ic_ca, ic_pp, ic_pa, ic_aa, ic_cp,
    nis_a, nis_c, nis_p,                                 # nis_p only where available
    dg_prodigy_boltz,
    # Boltz confidence
    boltz_iptm, boltz_ptm, boltz_plddt, boltz_confidence_score,
    boltz_complex_plddt, boltz_complex_iplddt, boltz_complex_pde,
    # PAE summaries (K81/V106 only for now)
    mean_pae_contacts, mean_pae_interface, n_contacts,
    # Interface geometry (CAD-score-LT)
    cad_rr, cad_rr_f1, cad_aa, cad_aa_f1,
    # PB-specific extras
    pdockq, pdockq2, lis, ipsae, min_ipsae,
    shape_complementarity, interface_residue_count,
    # derived
    functional_class

JSONL adds: sequences, structure paths, URLs where available.

Usage:
    python build_unified_dataset.py
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(ROOT / "benchmarks/scripts/boltz_pipeline"))

from dataset_registry import get_paths  # noqa: E402

OUT_DIR = ROOT / "benchmarks/output/unified"

# RT ln(10) at 298 K in kcal/mol — converts log10(Kd_M) to ΔG.
RT_LN10_298K = 1.3637        # ΔG(kcal/mol) = 1.3637 × log10(Kd_M)

# Column maps. ProteinBase uses ``prodigy_contacts_<pair>``; K81/V106 use
# ``ic_<pair>`` with reversed charge/polar/aliphatic letter ordering for
# the asymmetric pairs.
PB_IC_MAP = {
    "prodigy_contacts_cc": "ic_cc",
    "prodigy_contacts_ac": "ic_ca",        # apolar-charged == charged-apolar
    "prodigy_contacts_pp": "ic_pp",
    "prodigy_contacts_ap": "ic_pa",        # apolar-polar == polar-apolar
    "prodigy_contacts_aa": "ic_aa",
    "prodigy_contacts_cp": "ic_cp",
}
PB_NIS_MAP = {
    "prodigy_nis_aliphatic": "nis_a",
    "prodigy_nis_charged": "nis_c",
    "prodigy_nis_polar": "nis_p",
}
PB_BOLTZ_MAP = {
    "boltz2_iptm": "boltz_iptm",
    "boltz2_ptm": "boltz_ptm",
    "boltz2_plddt": "boltz_plddt",
    "boltz2_complex_plddt": "boltz_complex_plddt",
    "boltz2_complex_iplddt": "boltz_complex_iplddt",
    "boltz2_complex_pde": "boltz_complex_pde",
    "boltz2_pdockq": "pdockq",
    "boltz2_pdockq2": "pdockq2",
    "boltz2_lis": "lis",
    "boltz2_ipsae": "ipsae",
    "boltz2_min_ipsae": "min_ipsae",
    "shape_complimentarity_boltz2_binder_ss": "shape_complementarity",
    "interface_residue_count": "interface_residue_count",
}
PB_CAD_MAP = {
    "cadscorelt_rr_CAD_score": "cad_rr",
    "cadscorelt_rr_F1_of_areas": "cad_rr_f1",
    "cadscorelt_aa_CAD_score": "cad_aa",
    "cadscorelt_aa_F1_of_areas": "cad_aa_f1",
}


def _num(x: Any) -> float:
    if x is None or x == "":
        return float("nan")
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _kd_to_dg_kcal(kd_m: float) -> float:
    if not math.isfinite(kd_m) or kd_m <= 0:
        return float("nan")
    return RT_LN10_298K * math.log10(kd_m)


def _dg_kcal_to_kd_m(dg: float) -> float:
    if not math.isfinite(dg):
        return float("nan")
    return 10.0 ** (dg / RT_LN10_298K)


def load_k81_v106(dataset: str) -> pd.DataFrame:
    """Load K81 or V106 with uniform (binder-fold) CAD features.

    Switch from the original inter-chain CAD (cadscorelt/) to the uniform
    binder-fold CAD (cadscorelt_binder/) so the schema matches PB. The
    interface-specific CAD file is still on disk but not used here.
    """
    paths = get_paths(dataset)
    feats_csv = (paths.output_root / "pae_calibration/augmented_refit"
                 / "features_msa_only.csv")
    cad_csv = (paths.output_root / "pae_calibration/cadscorelt_binder"
               / "cadscore_binder_features_msa_only.csv")
    cad_arrays = (paths.output_root / "pae_calibration/cadscorelt_binder"
                  / "cadscore_binder_arrays_msa_only.jsonl")
    if not feats_csv.exists():
        raise SystemExit(f"Missing {feats_csv} — run augmented_refit first")
    if not cad_csv.exists():
        raise SystemExit(
            f"Missing {cad_csv} — run score_cadscorelt_binder_uniform first"
        )

    feats = pd.read_csv(feats_csv)
    cad = pd.read_csv(cad_csv)
    # feats has the OLD inter-chain CAD columns too; drop them so the
    # uniform binder-fold CAD from cadscorelt_binder/ is what lands.
    cad_cols_in_feats = [c for c in feats.columns
                          if c.startswith(("cad_", "resi_", "atom_",
                                            "rrc_", "aac_"))]
    feats = feats.drop(columns=cad_cols_in_feats)

    df = feats.merge(cad, on=["pdb_id", "mode"], how="left")

    df["dg_exp_kcal_mol"] = df["dg_exp"]
    df["kd_m"] = df["dg_exp"].apply(_dg_kcal_to_kd_m)
    df["log10_kd"] = df["dg_exp"] / RT_LN10_298K
    df["source"] = "Kastritis81" if dataset == "kastritis" else "VrevenBM5.5"
    df["cad_arrays_jsonl"] = (
        str(cad_arrays.relative_to(ROOT)) if cad_arrays.exists() else ""
    )

    df = df.rename(columns={
        "iptm": "boltz_iptm", "ptm": "boltz_ptm", "plddt": "boltz_plddt",
        "confidence_score": "boltz_confidence_score",
    })
    return df


def load_proteinbase() -> pd.DataFrame:
    prod_csv = ROOT / "benchmarks/output/proteinbase/prodigy_tinygrad_scores.csv"
    cad_csv = ROOT / "benchmarks/output/proteinbase/proteinbase_kd_boltz_pae_cadscorelt.csv"
    local_cad_csv = (ROOT / "benchmarks/output/proteinbase/pae_calibration"
                     / "cadscorelt/cadscore_features_msa_only.csv")
    local_cad_arrays = (ROOT / "benchmarks/output/proteinbase/pae_calibration"
                        / "cadscorelt/cadscore_arrays_msa_only.jsonl")
    if not prod_csv.exists():
        raise SystemExit(f"Missing {prod_csv}")
    if not cad_csv.exists():
        raise SystemExit(f"Missing {cad_csv}")

    prod = pd.read_csv(prod_csv)
    cad = pd.read_csv(cad_csv)
    df = cad.merge(prod, on="proteinbase_id", how="left", suffixes=("", "_prod"))

    df = df.rename(columns={**PB_IC_MAP, **PB_NIS_MAP, **PB_BOLTZ_MAP, **PB_CAD_MAP})
    df["pdb_id"] = df["proteinbase_id"]
    df["source"] = "ProteinBase"
    df["mode"] = "msa_only"      # needed for PAE merge key
    df["dg_prodigy_boltz"] = pd.to_numeric(df.get("prodigy_ba_val"),
                                            errors="coerce")

    # Merge in local CAD features (binder-fold geometry, single-chain).
    # With uniform binder-fold CAD (score_cadscorelt_binder_uniform.py for
    # K81/V106, score_cadscorelt_proteinbase.py for PB) all 287 complexes
    # now have identically-computed CAD columns: single-chain comparison
    # of the reference binder vs Boltz-extracted binder chain, [-min-sep 1]
    # subselect.
    if local_cad_csv.exists():
        local_cad = pd.read_csv(local_cad_csv)
        # Drop any CAD overlap cols from the base df before merging.
        cad_cols_in_df = [c for c in df.columns
                          if c.startswith(("cad_", "resi_", "atom_",
                                            "rrc_", "aac_"))]
        df = df.drop(columns=cad_cols_in_df)
        # PB CAD CSV uses pdb_id = proteinbase_id.
        df = df.merge(local_cad, left_on="pdb_id", right_on="pdb_id", how="left",
                       suffixes=("", "_pb_cad"))
        df["cad_arrays_jsonl"] = (str(local_cad_arrays.relative_to(ROOT))
                                   if local_cad_arrays.exists() else "")
    else:
        df["cad_arrays_jsonl"] = ""

    # Merge in PAE features (matching K81/V106 semantics).
    pae_csv = (ROOT / "benchmarks/output/proteinbase/pae_calibration"
               / "pae_features.csv")
    if pae_csv.exists():
        pae = pd.read_csv(pae_csv)
        pae = pae.rename(columns={"proteinbase_id": "pdb_id",
                                    "n_contacts_5p5A": "n_contacts_pae"})
        df = df.merge(pae[["pdb_id", "mode", "mean_pae_contacts",
                            "mean_pae_interface"]],
                       on=["pdb_id", "mode"], how="left",
                       suffixes=("_old", ""))
        # If there was a mean_pae_* already (shouldn't be; PB didn't have it),
        # drop the _old variant.
        for c in ("mean_pae_contacts_old", "mean_pae_interface_old"):
            if c in df.columns:
                df = df.drop(columns=c)

    # Affinity: PB ships log10_kd_median (in molar). Derive dg_exp_kcal_mol.
    df["log10_kd"] = pd.to_numeric(df.get("log10_kd_median"), errors="coerce")
    df["kd_m"] = df["log10_kd"].apply(
        lambda x: 10.0 ** x if pd.notna(x) else float("nan")
    )
    df["dg_exp_kcal_mol"] = df["log10_kd"] * RT_LN10_298K

    # PAE summaries are populated by the merge above; only set defaults if
    # the merge didn't run (e.g. PAE CSV missing).
    if "mean_pae_contacts" not in df.columns:
        df["mean_pae_contacts"] = float("nan")
    if "mean_pae_interface" not in df.columns:
        df["mean_pae_interface"] = float("nan")
    df["n_contacts"] = (
        df[["ic_cc", "ic_ca", "ic_pp", "ic_pa", "ic_aa", "ic_cp"]].sum(axis=1)
    )
    df["irmsd"] = float("nan")
    df["stratum"] = ""
    df["functional_class"] = ""   # PB categories are different (design_class)
    return df


# Common columns emitted in the unified CSV — UNIFORM across all 287 complexes.
# Every column below is computed identically on K81, V106, and PB.
# Non-uniform features (pdockq, ipsae, shape_complementarity, irmsd, stratum,
# etc.) are kept in the raw data but dropped from the regression feature pool.
UNIFIED_COLS = [
    "pdb_id", "source",
    "dg_exp_kcal_mol", "kd_m", "log10_kd",
    # Metadata only (NOT features — may be NaN on PB)
    "irmsd", "stratum", "functional_class",
    # PRODIGY IC+NIS — uniform (predict_binding_affinity_tinygrad on Boltz CIF)
    "ic_cc", "ic_ca", "ic_pp", "ic_pa", "ic_aa", "ic_cp",
    "nis_a", "nis_c",
    # Boltz confidence — uniform across all 3 datasets
    "boltz_iptm", "boltz_ptm", "boltz_plddt",
    # PAE summaries — uniform (inter-chain PAE block; PB filled by
    # add_pae_features_pb.py)
    "mean_pae_contacts", "mean_pae_interface", "n_contacts",
    # CAD-score-LT — uniform (single-chain binder-fold: reference_binder vs
    # Boltz_binder_extracted, [-min-sep 1])
    "cad_rr", "cad_rr_f1", "cad_aa", "cad_aa_f1",
    "cad_rr_target_area", "cad_rr_model_area",
    "cad_rr_tp", "cad_rr_fp", "cad_rr_fn",
    "resi_cad_mean", "resi_cad_std", "resi_cad_min", "resi_cad_max",
    "resi_cad_p10", "resi_cad_p25", "resi_cad_p50", "resi_cad_p75", "resi_cad_p90",
    "resi_cad_A_mean",
    "resi_cad_frac_below_0_3", "resi_cad_frac_below_0_5",
    "resi_cad_frac_above_0_7", "resi_cad_frac_above_0_9",
    "resi_n_total", "resi_n_false_positive",
    "atom_cad_mean", "atom_cad_std", "atom_cad_min", "atom_cad_max",
    "atom_cad_p10", "atom_cad_p25", "atom_cad_p50", "atom_cad_p75", "atom_cad_p90",
    "atom_cad_A_mean", "atom_cad_bb_mean", "atom_cad_sc_mean",
    "atom_cad_frac_below_0_3", "atom_cad_frac_below_0_5",
    "atom_cad_frac_above_0_7", "atom_cad_frac_above_0_9",
    "atom_n_total", "atom_n_false_positive",
    "rrc_cad_mean", "rrc_cad_std", "rrc_cad_p10", "rrc_cad_p50", "rrc_cad_p90",
    "rrc_cad_frac_below_0_3", "rrc_cad_frac_above_0_7",
    "rrc_n_total", "rrc_n_model_only", "rrc_n_shared",
    "aac_cad_mean", "aac_cad_std", "aac_cad_p10", "aac_cad_p50", "aac_cad_p90",
    "aac_cad_frac_below_0_3", "aac_cad_frac_above_0_7",
    "aac_n_total", "aac_n_model_only", "aac_n_shared",
    # Pointer to raw per-atom / per-residue CAD arrays (JSONL)
    "cad_arrays_jsonl",
]


def _unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in UNIFIED_COLS:
        if c not in out.columns:
            out[c] = float("nan") if c not in ("stratum", "functional_class") else ""
    return out[UNIFIED_COLS].copy()


def build_jsonl_row(row: pd.Series, raw: pd.Series) -> dict:
    """Richer per-complex record with sequences/paths where available."""
    base = {k: (None if (isinstance(row[k], float) and math.isnan(row[k])) else row[k])
            for k in UNIFIED_COLS}
    extras: dict[str, Any] = {}
    for k in ("seq_target", "seq_binder", "binder_sequence", "target",
              "name", "author", "design_method", "classification",
              "binding", "binding_strength",
              "boltz2_structure_url", "esmfold_structure_url", "pae_url",
              "cleaned_pdb"):
        if k in raw.index and pd.notna(raw[k]):
            extras[k] = raw[k]
    base["extras"] = extras
    return base


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    k81 = load_k81_v106("kastritis")
    v106 = load_k81_v106("vreven")
    pb = load_proteinbase()

    frames = [k81, v106, pb]
    # Keep raw columns for JSONL extras
    raw = pd.concat(frames, ignore_index=True, sort=False)
    unified = _unify_columns(raw)

    # Deduplicate on (pdb_id, source) — K81 ∩ V106 intentionally duplicated
    # to preserve two observations.
    n_before = len(unified)
    unified = unified.drop_duplicates(subset=["pdb_id", "source"])
    print(f"[unify] total rows = {len(unified)}  "
          f"({(unified.source == 'Kastritis81').sum()} K81, "
          f"{(unified.source == 'VrevenBM5.5').sum()} V106, "
          f"{(unified.source == 'ProteinBase').sum()} PB)")
    if n_before != len(unified):
        print(f"  (dedupe removed {n_before - len(unified)} within-source "
              f"duplicates)")

    csv_path = OUT_DIR / "unified_features.csv"
    unified.to_csv(csv_path, index=False)
    print(f"[write] {csv_path}")

    # JSONL
    jsonl_path = OUT_DIR / "unified_features.jsonl"
    with jsonl_path.open("w") as f:
        for idx in unified.index:
            row = unified.loc[idx]
            raw_row = raw.loc[idx]
            rec = build_jsonl_row(row, raw_row)
            f.write(json.dumps(rec, default=float) + "\n")
    print(f"[write] {jsonl_path}")

    # Quick stats
    print()
    print("Per-source summary:")
    for src, sub in unified.groupby("source"):
        n_dg = sub["dg_exp_kcal_mol"].notna().sum()
        n_cad = sub["cad_rr"].notna().sum()
        n_iptm = sub["boltz_iptm"].notna().sum()
        print(f"  {src:<15s} N={len(sub):>3d}  "
              f"dg_exp={n_dg}  cad_rr={n_cad}  iptm={n_iptm}")


if __name__ == "__main__":
    main()
