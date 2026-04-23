#!/usr/bin/env python3
"""Experiment B — pLDDT-gated NIS reparametrisation.

Phase 2 v1 (see ``docs/PAE.md``) found that four independent PAE-gating
strategies all saturated at R ≈ 0.62 (msa_only) / 0.67 (template_msa) vs the
crystal R = 0.74 target. Every strategy touched only IC-side terms. This
script tests the complementary hypothesis: on Boltz predictions, surface
residues in uncertain regions (low pLDDT) have miscalled SASA, so the NIS
terms themselves leak noise. Three pLDDT-based reparametrisations are
evaluated:

Variant 1 — global-pLDDT scaling
    %NIS_a_v1 = %NIS_a * mean_plddt
    %NIS_c_v1 = %NIS_c * mean_plddt
    (mean_plddt already in [0, 1] from tm_scores.csv)

Variant 2 — per-residue high-pLDDT gate
    f_high = count(plddt_residue >= 0.7) / N_total
    %NIS_a_v2 = %NIS_a * f_high
    %NIS_c_v2 = %NIS_c * f_high
    (approximates "fraction of the structure we trust")

Variant 3 — interface/bulk pLDDT contrast
    mean_plddt_interface = mean over residues with any inter-chain contact at 5.5 A
    mean_plddt_bulk      = mean over all other residues
    mod = mean_plddt_interface / mean_plddt_bulk
    %NIS_a_v3 = %NIS_a * mod
    %NIS_c_v3 = %NIS_c * mod

For every variant the 6 PRODIGY coefficients + intercept are refit via
4-fold CV x 10 repeats (paper convention). In-sample R and FIXED-stock-coef
R (applied to the ORIGINAL ungated NIS) are reported as baselines.

Usage:
    python plddt_nis.py                            # both modes
    python plddt_nis.py --mode msa_only
    python plddt_nis.py --out-dir /tmp/foo
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from diagnostic_refit import (  # noqa: E402
    COEFFS_STOCK, INTERCEPT_STOCK, IRMSD_CUTOFF,
    FEATURE_ORDER, fit_ols, kfold_cv_R, R_RMSE,
)

BOLTZ_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"
PRODIGY_CSV = BOLTZ_ROOT / "prodigy_scores.csv"
TM_CSV = BOLTZ_ROOT / "tm_scores.csv"

# --------------------------------------------------------------------------
# pLDDT-gated NIS variant enum
# --------------------------------------------------------------------------

VARIANTS = ("v1_global", "v2_highplddt_frac", "v3_iface_bulk_ratio")

# Per-residue threshold for variant 2.  Boltz pLDDT files are in [0, 1],
# so "plddt >= 70%" → raw >= 0.70.
V2_PLDDT_THRESHOLD = 0.70

# Distance cutoff for interface definition in variant 3 (PRODIGY's 5.5 A).
D_CUT_INTERFACE = 5.5


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_stock_prodigy_rows(mode: str) -> dict[str, dict]:
    """{pdb_id: {ic_cc, ic_ca, ic_pp, ic_pa, nis_a, nis_c, dg_exp, ...}}."""
    out = {}
    with PRODIGY_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != mode:
                continue
            out[row["pdb_id"]] = {
                "ic_cc": float(row["ic_cc"]),
                "ic_ca": float(row["ic_ca"]),
                "ic_pp": float(row["ic_pp"]),
                "ic_pa": float(row["ic_pa"]),
                "nis_a": float(row["nis_a"]),
                "nis_c": float(row["nis_c"]),
                "dg_exp": float(row["dg_exp"]),
                "dg_prodigy_baseline": float(row["dg_prodigy_baseline"]),
            }
    return out


def load_tm_mean_plddt(mode: str) -> dict[str, float]:
    out = {}
    with TM_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != mode:
                continue
            out[row["pdb_id"]] = float(row["plddt"])
    return out


def load_irmsd_map() -> dict[str, float]:
    d = json.loads(DATASET_JSON.read_text())
    return {k: float(v["iRMSD"]) for k, v in d.items()}


# --------------------------------------------------------------------------
# pLDDT npz + interface residue identification
# --------------------------------------------------------------------------

def _plddt_npz_path(mode: str, pdb_id: str) -> Path:
    return (BOLTZ_ROOT / mode / f"{pdb_id}_{mode}"
            / "boltz_results_input" / "predictions" / "input"
            / "plddt_input_model_0.npz")


def _cif_path(mode: str, pdb_id: str) -> Path:
    return (BOLTZ_ROOT / mode / f"{pdb_id}_{mode}"
            / "boltz_results_input" / "predictions" / "input"
            / "input_model_0.cif")


def load_plddt_residues(mode: str, pdb_id: str) -> np.ndarray | None:
    """Return per-residue pLDDT in [0, 1].  Boltz already provides per-res."""
    p = _plddt_npz_path(mode, pdb_id)
    if not p.exists():
        return None
    d = np.load(p)
    if "plddt" not in d:
        return None
    arr = np.asarray(d["plddt"]).astype(np.float32).ravel()
    # If any value > 1.5, assume the distribution is 0-100 and normalise.
    if arr.max() > 1.5:
        arr = arr / 100.0
    return arr


def load_interface_residue_mask(
    mode: str, pdb_id: str, d_cut: float = D_CUT_INTERFACE,
) -> tuple[np.ndarray, int, int] | None:
    """Boolean mask [N_total] — True for residues with any inter-chain
    heavy-atom contact at ``d_cut``.  Uses the same CIF parser as
    quick_pae_calib.  Returns (mask, n_t, n_b), residue order = chain A then
    chain B (matches the pLDDT array layout).
    """
    cif = _cif_path(mode, pdb_id)
    if not cif.exists():
        return None
    # Lazy import to avoid pulling Bio.PDB at module top.
    sys.path.insert(0, str(Path(__file__).parent))
    from quick_pae_calib import parse_boltz_cif, min_heavy_dist  # noqa: E402

    try:
        (pos_t, mask_t, _), (pos_b, mask_b, _) = parse_boltz_cif(cif)
    except (KeyError, ValueError) as exc:
        print(f"[skip] {pdb_id}/{mode}: CIF parse failed — {exc}")
        return None
    md = min_heavy_dist(pos_t, pos_b, mask_t, mask_b)
    contacts = md <= d_cut  # [N_t, N_b]
    iface_t = contacts.any(axis=1)
    iface_b = contacts.any(axis=0)
    iface_full = np.concatenate([iface_t, iface_b])
    return iface_full, pos_t.shape[0], pos_b.shape[0]


# --------------------------------------------------------------------------
# Feature builder
# --------------------------------------------------------------------------

def build_feature_df(mode: str) -> pd.DataFrame:
    """Build a DataFrame with stock + all three variant features per pdb."""
    stock = load_stock_prodigy_rows(mode)
    tm_plddt = load_tm_mean_plddt(mode)
    irmsd_map = load_irmsd_map()

    rows = []
    for pdb_id, feats in sorted(stock.items()):
        plddt_res = load_plddt_residues(mode, pdb_id)
        if plddt_res is None:
            print(f"[skip] {pdb_id}/{mode}: missing per-residue pLDDT")
            continue

        # Variant 1: global mean pLDDT (prefer per-residue mean over
        # tm_scores scalar — they agree to <1e-3 in spot checks).
        mean_plddt_res = float(plddt_res.mean())
        mean_plddt_tm = float(tm_plddt.get(pdb_id, np.nan))
        v1_scale = mean_plddt_res

        # Variant 2: fraction of residues with plddt >= 0.70.
        v2_scale = float((plddt_res >= V2_PLDDT_THRESHOLD).mean())

        # Variant 3: mean pLDDT at interface / mean pLDDT bulk.
        iface_res = load_interface_residue_mask(mode, pdb_id)
        if iface_res is None:
            print(f"[skip] {pdb_id}/{mode}: interface mask failed")
            continue
        iface_mask, _, _ = iface_res
        if len(iface_mask) != len(plddt_res):
            print(f"[skip] {pdb_id}/{mode}: iface_mask length "
                  f"{len(iface_mask)} != plddt length {len(plddt_res)}")
            continue
        if iface_mask.any() and (~iface_mask).any():
            mean_iface = float(plddt_res[iface_mask].mean())
            mean_bulk = float(plddt_res[~iface_mask].mean())
            v3_scale = mean_iface / (mean_bulk + 1e-8)
        else:
            # Degenerate — all residues at interface, or none.  Fall back
            # to mean pLDDT (equivalent to variant 1).
            mean_iface = float(plddt_res.mean())
            mean_bulk = mean_iface
            v3_scale = 1.0

        nis_a = float(np.clip(feats["nis_a"], 0, 100))
        nis_c = float(np.clip(feats["nis_c"], 0, 100))

        rows.append({
            "pdb_id": pdb_id, "mode": mode,
            "dg_exp": feats["dg_exp"],
            "irmsd": irmsd_map.get(pdb_id, np.nan),
            # stock IC (unchanged across variants)
            "ic_cc": feats["ic_cc"], "ic_ca": feats["ic_ca"],
            "ic_pp": feats["ic_pp"], "ic_pa": feats["ic_pa"],
            # original NIS
            "nis_a": nis_a, "nis_c": nis_c,
            # variant scalars
            "mean_plddt_residues": mean_plddt_res,
            "mean_plddt_tm": mean_plddt_tm,
            "f_highplddt": v2_scale,
            "mean_plddt_iface": mean_iface,
            "mean_plddt_bulk": mean_bulk,
            "iface_bulk_ratio": v3_scale,
            "n_iface_res": int(iface_mask.sum()),
            "n_total_res": int(len(plddt_res)),
            # gated NIS per variant
            "nis_a_v1_global": nis_a * v1_scale,
            "nis_c_v1_global": nis_c * v1_scale,
            "nis_a_v2_highplddt_frac": nis_a * v2_scale,
            "nis_c_v2_highplddt_frac": nis_c * v2_scale,
            "nis_a_v3_iface_bulk_ratio": nis_a * v3_scale,
            "nis_c_v3_iface_bulk_ratio": nis_c * v3_scale,
        })
    df = pd.DataFrame(rows)
    print(f"[features] {mode}: built {len(df)} rows")
    return df


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def _X_stock(df: pd.DataFrame) -> np.ndarray:
    return df[list(FEATURE_ORDER)].to_numpy(dtype=np.float64)


def _X_variant(df: pd.DataFrame, variant: str) -> np.ndarray:
    cols = ["ic_cc", "ic_ca", "ic_pp", "ic_pa",
            f"nis_a_{variant}", f"nis_c_{variant}"]
    return df[cols].to_numpy(dtype=np.float64)


def evaluate_mode(df: pd.DataFrame) -> dict:
    y = df["dg_exp"].to_numpy(dtype=np.float64)
    is_flex = (df["irmsd"] > IRMSD_CUTOFF).to_numpy()
    n_rigid = int((~is_flex).sum()); n_flex = int(is_flex.sum())

    # Stock FIXED baseline (ORIGINAL NIS + stock coefs).
    X_stock = _X_stock(df)
    pred_fixed = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed, rmse_fixed = R_RMSE(pred_fixed, y)
    r_fixed_rigid, _ = R_RMSE(pred_fixed[~is_flex], y[~is_flex])
    r_fixed_flex, _ = R_RMSE(pred_fixed[is_flex], y[is_flex])

    # Stock REFIT CV baseline (ORIGINAL NIS + refit coefs).
    cv_stock = kfold_cv_R(X_stock, y, k=4, n_repeats=10, seed=0)
    stock_refit_coefs, stock_refit_icept = fit_ols(X_stock, y)

    variants_out = {}
    for v in VARIANTS:
        X_v = _X_variant(df, v)
        # In-sample
        coefs_in, icept_in = fit_ols(X_v, y)
        pred_in = X_v @ coefs_in + icept_in
        r_in, rmse_in = R_RMSE(pred_in, y)
        # 4-fold CV x 10 repeats
        cv = kfold_cv_R(X_v, y, k=4, n_repeats=10, seed=0)
        preds_cv = cv["preds_mean"]
        r_cv_rigid, _ = R_RMSE(preds_cv[~is_flex], y[~is_flex])
        r_cv_flex, _ = R_RMSE(preds_cv[is_flex], y[is_flex])
        # FIXED: apply STOCK coefs to gated NIS (not very meaningful — the
        # stock intercept assumes stock NIS scale — report for completeness).
        pred_fixed_on_v = X_v @ COEFFS_STOCK + INTERCEPT_STOCK
        r_fixed_on_v, rmse_fixed_on_v = R_RMSE(pred_fixed_on_v, y)
        variants_out[v] = {
            "in": {"R": r_in, "RMSE": rmse_in,
                   "coefs": coefs_in.tolist(), "intercept": float(icept_in)},
            "cv": {"R_mean": cv["R_mean"], "R_std": cv["R_std"],
                   "RMSE_mean": cv["RMSE_mean"], "RMSE_std": cv["RMSE_std"],
                   "R_rigid": r_cv_rigid, "R_flex": r_cv_flex,
                   "preds_mean": preds_cv},
            "fixed_on_gated": {"R": r_fixed_on_v, "RMSE": rmse_fixed_on_v},
        }

    return {
        "mode": df["mode"].iloc[0], "N": len(y),
        "n_rigid": n_rigid, "n_flex": n_flex,
        "y": y, "is_flex": is_flex,
        "stock_fixed": {
            "R": r_fixed, "RMSE": rmse_fixed,
            "R_rigid": r_fixed_rigid, "R_flex": r_fixed_flex,
            "preds": pred_fixed,
        },
        "stock_refit_cv": {
            "R_mean": cv_stock["R_mean"], "R_std": cv_stock["R_std"],
            "RMSE_mean": cv_stock["RMSE_mean"],
            "RMSE_std": cv_stock["RMSE_std"],
            "preds_mean": cv_stock["preds_mean"],
            "coefs_in_sample": stock_refit_coefs.tolist(),
            "intercept_in_sample": float(stock_refit_icept),
        },
        "variants": variants_out,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def _fmt(val: float) -> str:
    return f"{val:+.3f}" if not np.isnan(val) else "  nan "


def pick_best_variant(results: list[dict]) -> tuple[str, str, float, float]:
    """Return (mode, variant, dR_vs_fixed, dR_vs_refit_cv).

    "Best" = largest CV R across all (mode, variant).  Returned deltas are
    relative to (a) stock FIXED and (b) stock REFIT CV in the same mode.
    """
    best_key = ("", "", -9.99, -9.99)
    best_cv_R = -9.99
    for res in results:
        r_fixed = res["stock_fixed"]["R"]
        r_refit_cv = res["stock_refit_cv"]["R_mean"]
        for v in VARIANTS:
            r_cv = res["variants"][v]["cv"]["R_mean"]
            if r_cv > best_cv_R:
                best_cv_R = r_cv
                best_key = (res["mode"], v,
                            r_cv - r_fixed, r_cv - r_refit_cv)
    return best_key


def verdict_line(best_dR: float) -> str:
    if best_dR >= 0.03:
        return "**HELPS**"
    if best_dR >= 0.0:
        return "**MARGINAL**"
    return "**NO-HELP**"


def write_report(path: Path, results: list[dict],
                 crystal_R: float, crystal_RMSE: float):
    best_mode, best_v, best_dR_fixed, best_dR_refit = pick_best_variant(results)
    # Success-criterion verdict follows the spec (ΔR vs stock FIXED).
    lines = [
        f"# pLDDT-gated NIS on K81: {verdict_line(best_dR_fixed)}",
        "",
        f"Best: mode={best_mode}  variant={best_v}  "
        f"ΔR = {best_dR_fixed:+.3f} (CV mean vs stock FIXED; "
        f"success criterion).",
        f"Apples-to-apples: ΔR = {best_dR_refit:+.3f} vs stock REFIT CV "
        f"(same 4-fold × 10 protocol, different features).",
        "",
        f"Crystal reference (ba_val vs DG): R = {crystal_R:.3f},  "
        f"RMSE = {crystal_RMSE:.2f} kcal/mol.",
        "Paper published (4-fold CV x 10 on crystal): R ≈ 0.73, RMSE ≈ 1.89.",
        "",
        "Note: the headline success criterion compares CV R against stock "
        "FIXED, which does not share the same CV penalty.  A fair "
        "apples-to-apples comparison is **ΔR vs stock REFIT CV** — same "
        "6-feature 4-fold × 10 protocol, just un-gated NIS.",
        "",
        "## Variants",
        "",
        f"- **v1_global**: %NIS_χ · mean_plddt  (mean_plddt ∈ [0, 1]).",
        f"- **v2_highplddt_frac**: %NIS_χ · (count(plddt≥{V2_PLDDT_THRESHOLD}) / N_total).",
        f"- **v3_iface_bulk_ratio**: %NIS_χ · (⟨plddt⟩_iface / ⟨plddt⟩_bulk); "
        f"interface = any inter-chain heavy-atom contact ≤ {D_CUT_INTERFACE} Å.",
        "",
        f"Flexibility cutoff: iRMSD > {IRMSD_CUTOFF} Å (paper convention).",
        "",
    ]

    for res in results:
        mode = res["mode"]
        r_fixed = res["stock_fixed"]["R"]
        rmse_fixed = res["stock_fixed"]["RMSE"]
        r_refit_cv = res["stock_refit_cv"]["R_mean"]
        lines += [
            f"## mode = {mode}  (N={res['N']}, rigid={res['n_rigid']}, "
            f"flex={res['n_flex']})",
            "",
            "### Headline table",
            "",
            "| Model | R (CV mean ± std) | R (in-sample) | RMSE (CV) | "
            "ΔR vs stock FIXED | ΔR vs stock REFIT-CV |",
            "|---|---:|---:|---:|---:|---:|",
            f"| stock FIXED             | — | {_fmt(r_fixed)} | "
            f"{rmse_fixed:.2f} | 0.000 | — |",
            f"| stock REFIT (6 coefs)   | "
            f"{_fmt(r_refit_cv)} ± "
            f"{res['stock_refit_cv']['R_std']:.3f} | — | "
            f"{res['stock_refit_cv']['RMSE_mean']:.2f} | "
            f"{r_refit_cv - r_fixed:+.3f} | 0.000 |",
        ]
        for v in VARIANTS:
            cv = res["variants"][v]["cv"]
            ins = res["variants"][v]["in"]
            dR_fixed = cv["R_mean"] - r_fixed
            dR_refit = cv["R_mean"] - r_refit_cv
            lines.append(
                f"| {v:<22s} | {_fmt(cv['R_mean'])} ± {cv['R_std']:.3f} | "
                f"{_fmt(ins['R'])} | {cv['RMSE_mean']:.2f} | "
                f"{dR_fixed:+.3f} | {dR_refit:+.3f} |"
            )
        lines += [
            "",
            "### Stratified R (CV preds_mean)",
            "",
            "| variant | R (rigid) | R (flex) |",
            "|---|---:|---:|",
            f"| stock FIXED | {_fmt(res['stock_fixed']['R_rigid'])} | "
            f"{_fmt(res['stock_fixed']['R_flex'])} |",
        ]
        for v in VARIANTS:
            cv = res["variants"][v]["cv"]
            lines.append(
                f"| {v:<22s} | {_fmt(cv['R_rigid'])} | "
                f"{_fmt(cv['R_flex'])} |"
            )
        lines.append("")

        # Refit coef table for the best variant in this mode.
        local_best_v = max(
            VARIANTS, key=lambda v: res["variants"][v]["cv"]["R_mean"]
        )
        local_best_cv = res["variants"][local_best_v]["cv"]
        local_best_in = res["variants"][local_best_v]["in"]
        lines += [
            f"### Refit coefficients for best-in-mode variant "
            f"({local_best_v}, CV R = {local_best_cv['R_mean']:+.3f}, "
            f"in-sample R = {local_best_in['R']:+.3f})",
            "",
            "| feature          | stock      | refit      | Δ          |",
            "|---|---:|---:|---:|",
        ]
        coefs = local_best_in["coefs"]
        icept = local_best_in["intercept"]
        for name, stock_c, new_c in zip(FEATURE_ORDER, COEFFS_STOCK, coefs):
            lines.append(
                f"| {name:<16s} | {stock_c:+.5f} | {new_c:+.5f} | "
                f"{new_c - stock_c:+.5f} |"
            )
        lines.append(
            f"| intercept        | {INTERCEPT_STOCK:+.3f}   | "
            f"{icept:+.3f}   | {icept - INTERCEPT_STOCK:+.3f}   |"
        )
        lines.append("")

    lines += [
        "## Verdict rules",
        "",
        "- ΔR ≥ +0.03 at CV (vs stock FIXED) → **HELPS**.",
        "- 0 ≤ ΔR < +0.03 → **MARGINAL** (within CV noise).",
        "- ΔR < 0 → **NO-HELP** (pLDDT-gated NIS destroys signal or adds noise).",
        "",
        "If all variants fail, that confirms NIS is not the bottleneck for "
        "the Boltz→crystal R gap on Kastritis 81 — consistent with the v1 "
        "result that no PAE/confidence feature survived AIC selection.",
    ]

    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Plot
# --------------------------------------------------------------------------

def plot_best(res: dict, out_dir: Path):
    mode = res["mode"]
    y = res["y"]
    is_flex = res["is_flex"]
    pred_fixed = res["stock_fixed"]["preds"]
    # Best variant = largest CV R_mean in this mode.
    best_v = max(
        VARIANTS, key=lambda v: res["variants"][v]["cv"]["R_mean"]
    )
    pred_best = res["variants"][best_v]["cv"]["preds_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6), sharex=True, sharey=True)
    for ax, pred, ttl in zip(
        axes,
        (pred_fixed, pred_best),
        (f"stock FIXED (N={res['N']})",
         f"best variant [{best_v}] 4-fold CV"),
    ):
        r, rmse = R_RMSE(pred, y)
        ax.scatter(y[~is_flex], pred[~is_flex], alpha=0.6, s=22, color="C0",
                   label=f"rigid (n={(~is_flex).sum()})")
        ax.scatter(y[is_flex], pred[is_flex], alpha=0.7, s=28, color="C3",
                   marker="^", label=f"flex (n={is_flex.sum()})")
        lo = float(min(y.min(), pred.min())) - 1
        hi = float(max(y.max(), pred.max())) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{ttl}\nR={r:.2f}  RMSE={rmse:.2f}  ({mode})")
        ax.set_xlabel("ΔG_exp (kcal/mol)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("ΔG_pred (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out_dir / f"plddt_nis_scatter_{mode}.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Features CSV
# --------------------------------------------------------------------------

FEATURES_CSV_COLS = [
    "pdb_id", "mode", "dg_exp", "irmsd",
    "ic_cc", "ic_ca", "ic_pp", "ic_pa",
    "nis_a", "nis_c",
    "mean_plddt_residues", "mean_plddt_tm", "f_highplddt",
    "mean_plddt_iface", "mean_plddt_bulk", "iface_bulk_ratio",
    "n_iface_res", "n_total_res",
    "nis_a_v1_global", "nis_c_v1_global",
    "nis_a_v2_highplddt_frac", "nis_c_v2_highplddt_frac",
    "nis_a_v3_iface_bulk_ratio", "nis_c_v3_iface_bulk_ratio",
]


def write_features_csv(path: Path, dfs: list[pd.DataFrame]):
    full = pd.concat(dfs, axis=0, ignore_index=True)
    full = full[FEATURES_CSV_COLS]
    full.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", default="both",
                    choices=["msa_only", "template_msa", "both"])
    ap.add_argument("--out-dir", default=None,
                    help="Default: benchmarks/output/kastritis_81_boltz/"
                         "pae_calibration/plddt_nis/")
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else BOLTZ_ROOT / "pae_calibration" / "plddt_nis")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = (["msa_only", "template_msa"] if args.mode == "both"
             else [args.mode])

    # Crystal reference from dataset.json (ba_val vs DG).
    truth = json.loads(DATASET_JSON.read_text())
    pdbs = sorted(truth)
    dg_exp_all = np.array([float(truth[p]["DG"]) for p in pdbs])
    ba_val_all = np.array([float(truth[p]["ba_val"]) for p in pdbs])
    crystal_R, crystal_RMSE = R_RMSE(ba_val_all, dg_exp_all)
    print(f"[ref] crystal (ba_val vs DG)  R={crystal_R:+.3f}  "
          f"RMSE={crystal_RMSE:.2f}")

    results = []
    feat_dfs = []
    for mode in modes:
        print(f"\n=== mode = {mode} ===")
        df = build_feature_df(mode)
        feat_dfs.append(df)

        res = evaluate_mode(df)
        results.append(res)

        print(f"  N={res['N']}  rigid={res['n_rigid']}  flex={res['n_flex']}")
        f = res["stock_fixed"]
        print(f"  STOCK FIXED            R={f['R']:+.3f}  "
              f"RMSE={f['RMSE']:.2f}  "
              f"(rigid {f['R_rigid']:+.3f} | flex {f['R_flex']:+.3f})")
        s = res["stock_refit_cv"]
        print(f"  STOCK REFIT CV         R={s['R_mean']:+.3f}±{s['R_std']:.3f}"
              f"  RMSE={s['RMSE_mean']:.2f}±{s['RMSE_std']:.2f}")
        for v in VARIANTS:
            cv = res["variants"][v]["cv"]
            ins = res["variants"][v]["in"]
            dR = cv["R_mean"] - f["R"]
            print(f"  {v:<22s} CV R={cv['R_mean']:+.3f}±{cv['R_std']:.3f}  "
                  f"in-sample R={ins['R']:+.3f}  "
                  f"ΔR_vs_fixed={dR:+.3f}  "
                  f"(rigid {cv['R_rigid']:+.3f} | flex {cv['R_flex']:+.3f})")

        plot_best(res, out_dir)
        print(f"[plot] {(out_dir / f'plddt_nis_scatter_{mode}.png').relative_to(ROOT)}")

    # Report
    write_report(out_dir / "report.md", results, crystal_R, crystal_RMSE)
    print(f"[report] {(out_dir / 'report.md').relative_to(ROOT)}")

    # Features CSV (both modes)
    feat_csv = out_dir / "plddt_nis_features.csv"
    write_features_csv(feat_csv, feat_dfs)
    print(f"[features] {feat_csv.relative_to(ROOT)}")

    best_mode, best_v, best_dR_fixed, best_dR_refit = pick_best_variant(results)
    print(f"\n[verdict] best: mode={best_mode} variant={best_v}  "
          f"ΔR_fixed={best_dR_fixed:+.3f}  "
          f"ΔR_refitCV={best_dR_refit:+.3f}  — "
          f"{verdict_line(best_dR_fixed)}")


if __name__ == "__main__":
    main()
