#!/usr/bin/env python3
"""Experiment B — augmented feature pool + forward-stepwise AIC selection.

Experiment A showed stock IC+NIS features on Boltz have R ≈ 0.63 (msa_only)
and 0.67 (template_msa) with either stock or refit coefficients. The ceiling
is *feature-limited*, not coefficient-limited.

This script augments the 6-feature PRODIGY pool with candidates derived from
Boltz confidence output:

    + IC_aa, IC_cp       — contact types the 2015 paper's AIC dropped on
                            crystals; may become useful on Boltz
    + ipTM, pTM, pLDDT   — global Boltz confidence scalars
                            (from tm_scores.csv)
    + confidence_score   — Boltz's composite scalar
    + mean_pae_contacts  — mean inter-chain PAE at 5.5 Å contacts
                            (computed from pae_input_model_0.npz)
    + mean_pae_interface — mean inter-chain PAE over *all* inter-residue
                            pairs (global uncertainty at the interface
                            region)
    + n_contacts         — total inter-residue contacts at 5.5 Å

→ 13 candidate features. Forward-stepwise AIC picks a subset; reported
alongside 4-fold CV × 10 repeats R on the selected model.

Usage:
    python augmented_refit.py              # both modes
    python augmented_refit.py --mode msa_only
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from quick_pae_calib import (  # noqa: E402
    BOLTZ_ROOT, classify_ic, classify_stratum, load_complex,
    load_dataset_truth, load_stock_prodigy,
)
from diagnostic_refit import (  # noqa: E402
    COEFFS_STOCK, INTERCEPT_STOCK, IRMSD_CUTOFF, R_RMSE, fit_ols,
    kfold_cv_R,
)

TM_CSV = BOLTZ_ROOT / "tm_scores.csv"
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"

# All 6 pair types (needed for IC_aa, IC_cp as well as the 4 stock).
# Indices: 0 = Aliphatic, 1 = Charged, 2 = Polar.


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def classify_ic_full(contacts: np.ndarray, char_t: np.ndarray,
                     char_b: np.ndarray) -> dict:
    """Return all 6 IC-pair counts plus total n_contacts."""
    cbool = contacts.astype(np.int32)

    def _sum(ti: int, bj: int) -> int:
        sel = (char_t[:, None] == ti) & (char_b[None, :] == bj)
        return int((cbool * sel).sum())

    ic_aa = _sum(0, 0)
    ic_cc = _sum(1, 1)
    ic_pp = _sum(2, 2)
    ic_ca = _sum(0, 1) + _sum(1, 0)
    ic_pa = _sum(0, 2) + _sum(2, 0)
    ic_cp = _sum(1, 2) + _sum(2, 1)
    return {
        "ic_cc": ic_cc, "ic_ca": ic_ca, "ic_pp": ic_pp, "ic_pa": ic_pa,
        "ic_aa": ic_aa, "ic_cp": ic_cp,
        "n_contacts": ic_aa + ic_cc + ic_pp + ic_ca + ic_pa + ic_cp,
    }


def load_tm_scalars(mode: str) -> dict[str, dict]:
    """{pdb_id: {iptm, ptm, plddt, confidence_score}}."""
    out = {}
    with TM_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != mode:
                continue
            out[row["pdb_id"]] = {
                "iptm": float(row["iptm"]),
                "ptm": float(row["ptm"]),
                "plddt": float(row["plddt"]),
                "confidence_score": float(row["confidence_score"]),
            }
    return out


def extract_features(mode: str) -> pd.DataFrame:
    """Build feature DataFrame for all 81 complexes."""
    truth = load_dataset_truth()
    stock = load_stock_prodigy()
    tm_scalars = load_tm_scalars(mode)
    pdb_ids = sorted(truth.keys())

    rows = []
    t0 = time.time()
    for pid in pdb_ids:
        c = load_complex(pid, mode, truth, stock)
        if c is None:
            print(f"[skip] {pid}/{mode}")
            continue
        contacts = c.min_dist <= 5.5       # [N_t, N_b]
        ic_feats = classify_ic_full(contacts, c.char_t, c.char_b)

        if contacts.any():
            mean_pae_contacts = float(c.pae_ab[contacts].mean())
        else:
            mean_pae_contacts = float(np.nan)
        mean_pae_interface = float(c.pae_ab.mean())

        tm = tm_scalars.get(pid, {})
        rows.append({
            "pdb_id": pid, "mode": mode,
            "dg_exp": c.dg_exp, "ba_val": c.ba_val, "irmsd": c.irmsd,
            "stratum": classify_stratum(c.irmsd),
            "nis_a": float(np.clip(c.nis_a, 0, 100)),
            "nis_c": float(np.clip(c.nis_c, 0, 100)),
            **ic_feats,
            "mean_pae_contacts": mean_pae_contacts,
            "mean_pae_interface": mean_pae_interface,
            "iptm": tm.get("iptm", np.nan),
            "ptm": tm.get("ptm", np.nan),
            "plddt": tm.get("plddt", np.nan),
            "confidence_score": tm.get("confidence_score", np.nan),
        })
    print(f"[features] built {len(rows)} rows for {mode} in "
          f"{time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Forward-stepwise AIC selection
# --------------------------------------------------------------------------

CANDIDATE_FEATURES = [
    "ic_cc", "ic_ca", "ic_pp", "ic_pa",         # stock 4
    "nis_a", "nis_c",                            # stock 2
    "ic_aa", "ic_cp",                            # IC types paper's AIC dropped
    "iptm", "ptm", "plddt", "confidence_score",  # global Boltz confidence
    "mean_pae_contacts", "mean_pae_interface",   # PAE summaries
    "n_contacts",                                # total contact count
]


def aic_ols(X: np.ndarray, y: np.ndarray) -> tuple[float, np.ndarray]:
    """AIC under Gaussian error assumption: AIC = n·ln(RSS/n) + 2k."""
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    resid = y - X_aug @ beta
    rss = float(np.sum(resid ** 2))
    n, k = X.shape[0], X_aug.shape[1]
    aic = n * np.log(rss / n) + 2 * k
    return aic, beta


def forward_stepwise_aic(df: pd.DataFrame,
                         candidates: list[str] = CANDIDATE_FEATURES,
                         y_col: str = "dg_exp") -> tuple[list[str], dict]:
    """Greedy forward selection: add the feature that most reduces AIC.

    Stops when no remaining feature lowers AIC.
    """
    y = df[y_col].to_numpy()
    # Drop candidates with NaN rows (rare — e.g., zero-contact complexes).
    candidates = [c for c in candidates if not df[c].isna().any()]
    selected: list[str] = []
    remaining = list(candidates)
    # Null model: just intercept.
    _, beta0 = aic_ols(np.zeros((len(y), 0)), y)
    best_aic = float(len(y) * np.log(np.var(y, ddof=0) + 1e-12) + 2)
    trace = [{"step": 0, "selected": [], "aic": best_aic}]

    while remaining:
        candidate_aics = []
        for c in remaining:
            cols = selected + [c]
            X = df[cols].to_numpy()
            a, _ = aic_ols(X, y)
            candidate_aics.append((a, c))
        candidate_aics.sort(key=lambda t: t[0])
        best_new_aic, best_new = candidate_aics[0]
        if best_new_aic < best_aic - 1e-6:
            selected.append(best_new)
            remaining.remove(best_new)
            best_aic = best_new_aic
            trace.append({
                "step": len(selected),
                "added": best_new,
                "selected": list(selected),
                "aic": best_aic,
            })
        else:
            break
    X_final = df[selected].to_numpy()
    aic_final, beta_final = aic_ols(X_final, y)
    coefs = beta_final[:-1]; icept = float(beta_final[-1])
    return selected, {
        "aic": aic_final, "coefs": coefs, "intercept": icept,
        "trace": trace,
    }


# --------------------------------------------------------------------------
# Per-mode evaluation
# --------------------------------------------------------------------------

def evaluate_mode(df: pd.DataFrame) -> dict:
    y = df["dg_exp"].to_numpy()
    is_flex = (df["irmsd"] > IRMSD_CUTOFF).to_numpy()
    n_rigid = int((~is_flex).sum()); n_flex = int(is_flex.sum())

    # Baseline: stock 6 features + stock coefficients.
    stock_feats = ["ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c"]
    X_stock = df[stock_feats].to_numpy()
    pred_fixed = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed, rmse_fixed = R_RMSE(pred_fixed, y)

    # Baseline: stock 6 features, refit on all 81, in-sample.
    coefs_stock_refit, icept_stock_refit = fit_ols(X_stock, y)
    pred_stock_refit = X_stock @ coefs_stock_refit + icept_stock_refit
    r_stock_refit_in, rmse_stock_refit_in = R_RMSE(pred_stock_refit, y)
    cv_stock = kfold_cv_R(X_stock, y, k=4, n_repeats=10, seed=0)

    # Augmented: stepwise AIC over all candidates.
    selected, aic_out = forward_stepwise_aic(df)
    X_aug = df[selected].to_numpy()
    pred_aug_in = X_aug @ aic_out["coefs"] + aic_out["intercept"]
    r_aug_in, rmse_aug_in = R_RMSE(pred_aug_in, y)
    cv_aug = kfold_cv_R(X_aug, y, k=4, n_repeats=10, seed=0)
    r_cv_rigid, _ = R_RMSE(cv_aug["preds_mean"][~is_flex], y[~is_flex])
    r_cv_flex, _  = R_RMSE(cv_aug["preds_mean"][is_flex], y[is_flex])

    return {
        "mode": df["mode"].iloc[0], "N": len(y),
        "n_rigid": n_rigid, "n_flex": n_flex,
        "stock_fixed": {"R": r_fixed, "RMSE": rmse_fixed},
        "stock_refit_in": {"R": r_stock_refit_in,
                            "RMSE": rmse_stock_refit_in},
        "stock_refit_cv": cv_stock,
        "aug_selected": selected,
        "aug_trace": aic_out["trace"],
        "aug_coefs": aic_out["coefs"],
        "aug_intercept": aic_out["intercept"],
        "aug_aic": aic_out["aic"],
        "aug_in": {"R": r_aug_in, "RMSE": rmse_aug_in},
        "aug_cv": cv_aug,
        "aug_cv_rigid": r_cv_rigid,
        "aug_cv_flex": r_cv_flex,
        "y": y, "pred_aug_cv": cv_aug["preds_mean"],
        "pred_stock_fixed": pred_fixed,
        "is_flex": is_flex,
    }


# --------------------------------------------------------------------------
# Report + plot
# --------------------------------------------------------------------------

def write_report(path: Path, results: list[dict], crystal_R: float,
                 crystal_RMSE: float):
    lines = [
        "# Experiment B — augmented features + stepwise AIC",
        "",
        f"Crystal reference (`ba_val` vs `DG`): R = {crystal_R:.3f}, "
        f"RMSE = {crystal_RMSE:.2f} kcal/mol",
        "",
        f"Candidate feature pool ({len(CANDIDATE_FEATURES)}):",
        "  " + ", ".join(CANDIDATE_FEATURES),
        "",
        "Forward stepwise AIC — starts empty, adds the feature that most "
        "reduces AIC, stops when no addition lowers AIC.",
        "",
    ]
    for res in results:
        lines += [
            f"## mode = {res['mode']}   "
            f"(N={res['N']}, rigid={res['n_rigid']}, flex={res['n_flex']})",
            "",
            "### R/RMSE table",
            "",
            "| Model | R (all) | RMSE (all) | Notes |",
            "|---|---:|---:|---|",
            f"| stock 6-feat FIXED coefs | {res['stock_fixed']['R']:+.3f} | "
            f"{res['stock_fixed']['RMSE']:.2f} | reproduces Boltz-stock |",
            f"| stock 6-feat REFIT in-sample | {res['stock_refit_in']['R']:+.3f} | "
            f"{res['stock_refit_in']['RMSE']:.2f} | 6 coefs + 1 intercept |",
            f"| stock 6-feat REFIT 4-fold CV | "
            f"{res['stock_refit_cv']['R_mean']:+.3f} ± "
            f"{res['stock_refit_cv']['R_std']:.3f} | "
            f"{res['stock_refit_cv']['RMSE_mean']:.2f} ± "
            f"{res['stock_refit_cv']['RMSE_std']:.2f} | |",
            f"| **augmented AIC REFIT in-sample** | "
            f"**{res['aug_in']['R']:+.3f}** | "
            f"**{res['aug_in']['RMSE']:.2f}** | "
            f"{len(res['aug_selected'])} features selected |",
            f"| **augmented AIC REFIT 4-fold CV** | "
            f"**{res['aug_cv']['R_mean']:+.3f} ± "
            f"{res['aug_cv']['R_std']:.3f}** | "
            f"**{res['aug_cv']['RMSE_mean']:.2f} ± "
            f"{res['aug_cv']['RMSE_std']:.2f}** | |",
            "",
            "### Stratified (augmented CV, mean predictions)",
            "",
            "| stratum | N | R |",
            "|---|---:|---:|",
            f"| rigid  | {res['n_rigid']} | {res['aug_cv_rigid']:+.3f} |",
            f"| flex   | {res['n_flex']}  | {res['aug_cv_flex']:+.3f} |",
            "",
            "### Selected features & coefficients",
            "",
            "| step | added feature | AIC |",
            "|---:|---|---:|",
        ]
        for t in res["aug_trace"]:
            added = t.get("added", "(null / intercept only)")
            lines.append(f"| {t['step']} | {added} | {t['aic']:.2f} |")
        lines += [
            "",
            "| feature | coef |",
            "|---|---:|",
        ]
        for f, c in zip(res["aug_selected"], res["aug_coefs"]):
            lines.append(f"| {f} | {c:+.5f} |")
        lines.append(f"| intercept | {res['aug_intercept']:+.3f} |")
        lines.append("")
    lines += [
        "## Interpretation key",
        "",
        "- **If augmented CV R > stock CV R**: new features carry orthogonal "
        "signal → genuine improvement.",
        "- **If augmented in-sample R ≫ augmented CV R**: overfit — AIC "
        "selection picked noise.",
        "- **If augmented CV R ≈ stock CV R**: no new signal; the Boltz→"
        "crystal gap is not recoverable from available confidence features.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_scatter(res: dict, out_dir: Path):
    y = res["y"]
    pred_fixed = res["pred_stock_fixed"]
    pred_aug_cv = res["pred_aug_cv"]
    is_flex = res["is_flex"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, pred, ttl in zip(
        axes,
        (pred_fixed, pred_aug_cv),
        ("stock 6-feat FIXED coefs",
         f"augmented AIC CV ({len(res['aug_selected'])} feats)"),
    ):
        r, rmse = R_RMSE(pred, y)
        ax.scatter(y[~is_flex], pred[~is_flex], alpha=0.6, s=22, color="C0",
                    label=f"rigid (n={(~is_flex).sum()})")
        ax.scatter(y[is_flex], pred[is_flex], alpha=0.7, s=28, color="C3",
                    marker="^", label=f"flex (n={is_flex.sum()})")
        lo = min(y.min(), pred.min()) - 1
        hi = max(y.max(), pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{ttl}\nR={r:.2f}  RMSE={rmse:.2f}  ({res['mode']})")
        ax.set_xlabel("ΔG_exp (kcal/mol)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("ΔG_pred (kcal/mol)")
    fig.tight_layout()
    fig.savefig(out_dir / f"augmented_scatter_{res['mode']}.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="both",
                    choices=["msa_only", "template_msa", "both"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = (Path(args.out_dir) if args.out_dir
               else BOLTZ_ROOT / "pae_calibration" / "augmented_refit")
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = (["msa_only", "template_msa"] if args.mode == "both"
             else [args.mode])

    truth = json.loads(DATASET_JSON.read_text())
    pdbs = sorted(truth)
    dg_exp_all = np.array([float(truth[p]["DG"]) for p in pdbs])
    ba_val_all = np.array([float(truth[p]["ba_val"]) for p in pdbs])
    crystal_R, crystal_RMSE = R_RMSE(ba_val_all, dg_exp_all)
    print(f"[ref] crystal  R={crystal_R:+.3f}  RMSE={crystal_RMSE:.2f}")

    results = []
    for mode in modes:
        print(f"\n=== mode = {mode} ===")
        df = extract_features(mode)
        feat_csv = out_dir / f"features_{mode}.csv"
        df.to_csv(feat_csv, index=False)
        print(f"[write] {feat_csv.relative_to(ROOT)}")
        res = evaluate_mode(df)
        results.append(res)

        print(f"  stock 6-feat FIXED        R={res['stock_fixed']['R']:+.3f} "
              f"RMSE={res['stock_fixed']['RMSE']:.2f}")
        print(f"  stock 6-feat REFIT in-sample R={res['stock_refit_in']['R']:+.3f} "
              f"RMSE={res['stock_refit_in']['RMSE']:.2f}")
        print(f"  stock 6-feat REFIT 4-fold CV R="
              f"{res['stock_refit_cv']['R_mean']:+.3f} ± "
              f"{res['stock_refit_cv']['R_std']:.3f}")
        print(f"  AIC selected ({len(res['aug_selected'])}): "
              f"{res['aug_selected']}")
        print(f"  augmented in-sample       R={res['aug_in']['R']:+.3f} "
              f"RMSE={res['aug_in']['RMSE']:.2f}")
        print(f"  augmented 4-fold CV       R="
              f"{res['aug_cv']['R_mean']:+.3f} ± "
              f"{res['aug_cv']['R_std']:.3f}  "
              f"(rigid R={res['aug_cv_rigid']:+.3f}, "
              f"flex R={res['aug_cv_flex']:+.3f})")

        plot_scatter(res, out_dir)

    write_report(out_dir / "report.md", results, crystal_R, crystal_RMSE)
    print(f"\n[done] {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
