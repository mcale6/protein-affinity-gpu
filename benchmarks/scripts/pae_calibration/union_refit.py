#!/usr/bin/env python3
"""Union K81 + V106 refit — combine both feature sets for the final coef set.

N_union = 17 Kastritis-only + 106 Vreven = **123 complexes**:
  - 64 complexes are in both K81 and V106 → use V106 features (more recent
    Boltz run, diffusion_samples=2 best-ipTM selection)
  - 17 K81-only complexes → use K81 features
  - 42 Pierce Ab-Ag additions only in V106 → V106 features

The union is the calibration set for the candidate ``NIS_COEFFICIENTS_PAE``
that would replace the 2015 coefficients once the PAE-aware formula is
adopted.

Outputs (under ``benchmarks/output/union_k81_v106/pae_calibration/``):
  features_union.csv
  report.md
  scatter_union.png
  union_coefficients.json

Usage:
    python union_refit.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from diagnostic_refit import (  # noqa: E402
    COEFFS_STOCK, INTERCEPT_STOCK, IRMSD_CUTOFF, R_RMSE, fit_ols,
    kfold_cv_R,
)
from augmented_refit import forward_stepwise_aic  # noqa: E402

K81_FEATS_CSV = (ROOT / "benchmarks/output/kastritis_81_boltz/"
                 "pae_calibration/augmented_refit/features_msa_only.csv")
V106_FEATS_CSV = (ROOT / "benchmarks/output/vreven_bm55_boltz/"
                  "pae_calibration/augmented_refit/features_msa_only.csv")
OUT_DIR = ROOT / "benchmarks/output/union_k81_v106/pae_calibration"

STOCK_FEATS = ["ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c"]
IC_FEATS = ["ic_cc", "ic_ca", "ic_pp", "ic_pa"]


def build_union() -> pd.DataFrame:
    if not K81_FEATS_CSV.exists():
        raise SystemExit(
            f"{K81_FEATS_CSV} missing — run "
            "`augmented_refit.py --dataset kastritis --mode msa_only` first"
        )
    if not V106_FEATS_CSV.exists():
        raise SystemExit(
            f"{V106_FEATS_CSV} missing — run "
            "`augmented_refit.py --dataset vreven --mode msa_only` first"
        )
    k81 = pd.read_csv(K81_FEATS_CSV)
    v106 = pd.read_csv(V106_FEATS_CSV)
    # Keep V106 version of overlap (newer, best-ipTM sample). K81-only goes in.
    k81_only = k81[~k81.pdb_id.isin(v106.pdb_id)].copy()
    k81_only["source"] = "K81"
    v106 = v106.copy()
    v106["source"] = "V106"
    union = pd.concat([v106, k81_only], ignore_index=True)
    # Ensure NIS in [0,100] (some K81 rows occasionally have tiny overflow)
    for col in ("nis_a", "nis_c"):
        union[col] = union[col].clip(lower=0, upper=100)
    return union


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for ic in IC_FEATS:
        out[f"{ic}_x_iptm"] = out[ic].to_numpy() * out["iptm"].to_numpy()
        out[f"{ic}_x_paec"] = (out[ic].to_numpy()
                                * out["mean_pae_contacts"].to_numpy())
    return out


def eval_cv(X: np.ndarray, y: np.ndarray, is_flex: np.ndarray,
            n_repeats: int = 10) -> dict:
    cv = kfold_cv_R(X, y, k=4, n_repeats=n_repeats, seed=0)
    preds_cv = cv["preds_mean"]
    coefs, icept = fit_ols(X, y)
    preds_in = X @ coefs + icept
    r_in, rmse_in = R_RMSE(preds_in, y)
    r_cv_rigid, _ = R_RMSE(preds_cv[~is_flex], y[~is_flex])
    r_cv_flex, _ = R_RMSE(preds_cv[is_flex], y[is_flex])
    return {
        "R_cv": cv["R_mean"], "R_cv_std": cv["R_std"],
        "RMSE_cv": cv["RMSE_mean"], "RMSE_cv_std": cv["RMSE_std"],
        "R_in": r_in, "RMSE_in": rmse_in,
        "R_cv_rigid": r_cv_rigid, "R_cv_flex": r_cv_flex,
        "coefs": coefs, "intercept": float(icept),
        "preds_cv": preds_cv, "preds_in": preds_in,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    union = build_union()
    print(f"[union] {len(union)} complexes "
          f"(K81-only: {(union.source=='K81').sum()}, "
          f"V106: {(union.source=='V106').sum()})")

    y = union["dg_exp"].to_numpy()
    is_flex = (union["irmsd"] > IRMSD_CUTOFF).to_numpy()
    n_rigid = int((~is_flex).sum()); n_flex = int(is_flex.sum())
    print(f"[strata] rigid={n_rigid}, flex={n_flex}")

    union.to_csv(OUT_DIR / "features_union.csv", index=False)

    # Baseline 1: stock FIXED coefs on 6 stock feats
    X_stock = union[STOCK_FEATS].to_numpy()
    pred_fixed = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed, rmse_fixed = R_RMSE(pred_fixed, y)

    # Baseline 2: stock REFIT 4-fold CV on 6 stock feats
    stock_refit = eval_cv(X_stock, y, is_flex)

    # Augmented AIC (main effects, 13 candidates)
    aug_candidates = STOCK_FEATS + [
        "ic_aa", "ic_cp", "iptm", "ptm", "plddt", "confidence_score",
        "mean_pae_contacts", "mean_pae_interface", "n_contacts",
    ]
    aug_candidates = [c for c in aug_candidates if not union[c].isna().any()]
    aug_selected, aug_out = forward_stepwise_aic(
        union, candidates=aug_candidates, y_col="dg_exp",
    )
    X_aug = union[aug_selected].to_numpy()
    aug_res = eval_cv(X_aug, y, is_flex)
    aug_res["features"] = aug_selected
    aug_res["aic_trace"] = aug_out["trace"]

    # Interaction AIC (14 candidates = 6 stock + 4 IC×ipTM + 4 IC×⟨PAE⟩)
    union_x = build_interactions(union)
    int_candidates = STOCK_FEATS + [
        f"{ic}_x_iptm" for ic in IC_FEATS
    ] + [
        f"{ic}_x_paec" for ic in IC_FEATS
    ]
    int_candidates = [c for c in int_candidates if not union_x[c].isna().any()]
    int_selected, int_out = forward_stepwise_aic(
        union_x, candidates=int_candidates, y_col="dg_exp",
    )
    X_int = union_x[int_selected].to_numpy()
    int_res = eval_cv(X_int, y, is_flex)
    int_res["features"] = int_selected
    int_res["aic_trace"] = int_out["trace"]

    # Single-interaction ablation: stock 6 + one interaction at a time
    ablation: list[dict] = []
    for ic in IC_FEATS:
        for mod in ("iptm", "paec"):
            feat = f"{ic}_x_{mod}"
            feats = STOCK_FEATS + [feat]
            X = union_x[feats].to_numpy()
            r = eval_cv(X, y, is_flex)
            ablation.append({
                "interaction": feat,
                "R_cv": r["R_cv"], "R_cv_std": r["R_cv_std"],
                "RMSE_cv": r["RMSE_cv"], "R_in": r["R_in"],
                "dR_vs_fixed": r["R_cv"] - r_fixed,
                "dR_vs_refit": r["R_cv"] - stock_refit["R_cv"],
            })
    ablation.sort(key=lambda d: -d["R_cv"])

    # PAE-aware candidate: stock 6 + ic_pa×iptm (the top cross-dataset signal)
    cand_feats = STOCK_FEATS + ["ic_pa_x_iptm"]
    X_cand = union_x[cand_feats].to_numpy()
    cand_res = eval_cv(X_cand, y, is_flex)
    cand_res["features"] = cand_feats

    # Report
    print()
    print(f"  stock FIXED           R={r_fixed:+.3f}  RMSE={rmse_fixed:.2f}")
    print(f"  stock REFIT CV        R={stock_refit['R_cv']:+.3f} ± "
          f"{stock_refit['R_cv_std']:.3f}  RMSE={stock_refit['RMSE_cv']:.2f}")
    print(f"  augmented AIC CV      R={aug_res['R_cv']:+.3f} ± "
          f"{aug_res['R_cv_std']:.3f}  selected={aug_selected}")
    print(f"  interaction AIC CV    R={int_res['R_cv']:+.3f} ± "
          f"{int_res['R_cv_std']:.3f}  selected={int_selected}")
    print(f"  PAE candidate         R={cand_res['R_cv']:+.3f} ± "
          f"{cand_res['R_cv_std']:.3f}  "
          f"(stock 6 + ic_pa×iptm)")
    print()
    print("  [ablation — single-interaction addition to stock REFIT]")
    print(f"  {'interaction':<28s} {'R_cv':>8s} {'dR vs FIXED':>12s} "
          f"{'dR vs REFIT':>12s}")
    for row in ablation:
        print(f"  {row['interaction']:<28s} {row['R_cv']:>+8.3f} "
              f"{row['dR_vs_fixed']:>+12.3f} {row['dR_vs_refit']:>+12.3f}")

    # Save coefficients
    coef_json = {
        "description": "PAE-aware PRODIGY candidate coefficients (union K81+V106, N=123)",
        "dataset": "union K81 + V106",
        "N": int(len(union)),
        "features": cand_res["features"],
        "coefficients": {
            name: float(val) for name, val in zip(
                cand_res["features"], cand_res["coefs"]
            )
        },
        "intercept": cand_res["intercept"],
        "metrics": {
            "R_in_sample": float(cand_res["R_in"]),
            "RMSE_in_sample": float(cand_res["RMSE_in"]),
            "R_cv_mean": float(cand_res["R_cv"]),
            "R_cv_std": float(cand_res["R_cv_std"]),
            "RMSE_cv_mean": float(cand_res["RMSE_cv"]),
            "R_cv_rigid": float(cand_res["R_cv_rigid"]),
            "R_cv_flex": float(cand_res["R_cv_flex"]),
        },
        "baselines": {
            "stock_FIXED_R": float(r_fixed),
            "stock_FIXED_RMSE": float(rmse_fixed),
            "stock_REFIT_CV_R": float(stock_refit["R_cv"]),
            "stock_REFIT_CV_RMSE": float(stock_refit["RMSE_cv"]),
        },
    }
    (OUT_DIR / "union_coefficients.json").write_text(
        json.dumps(coef_json, indent=2)
    )
    print(f"\n[write] {(OUT_DIR / 'union_coefficients.json').relative_to(ROOT)}")

    # Scatter: stock FIXED vs PAE candidate CV predictions
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True, sharey=True)
    for ax, pred, title in zip(
        axes,
        (pred_fixed, cand_res["preds_cv"]),
        ("stock FIXED (2015 coefs)",
         "stock 6 + ic_pa × ipTM (4-fold CV × 10)"),
    ):
        r, rmse = R_RMSE(pred, y)
        ax.scatter(y[~is_flex], pred[~is_flex], alpha=0.6, s=22,
                    color="C0", label=f"rigid (n={(~is_flex).sum()})")
        ax.scatter(y[is_flex], pred[is_flex], alpha=0.7, s=28, color="C3",
                    marker="^", label=f"flex (n={is_flex.sum()})")
        lo = min(y.min(), pred.min()) - 1
        hi = max(y.max(), pred.max()) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{title}\nR={r:.2f}  RMSE={rmse:.2f}  N={len(y)}")
        ax.set_xlabel("ΔG_exp (kcal/mol)")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)
    axes[0].set_ylabel("ΔG_pred (kcal/mol)")
    fig.suptitle("Union K81 + V106 (N=123) — PAE candidate coefficients")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(OUT_DIR / "scatter_union.png", dpi=140)
    plt.close(fig)
    print(f"[write] {(OUT_DIR / 'scatter_union.png').relative_to(ROOT)}")

    # Report md
    lines = [
        "# Union K81 + V106 refit — Phase 2 v2 final calibration",
        "",
        f"N_union = **{len(union)}** complexes "
        f"({(union.source=='K81').sum()} K81-only, "
        f"{(union.source=='V106').sum()} V106 — includes 64 in both, resolved to V106)",
        f"Strata (iRMSD > {IRMSD_CUTOFF} Å): rigid = {n_rigid}, flex = {n_flex}",
        "",
        "## Model comparison (4-fold CV × 10 repeats)",
        "",
        "| Model | R (CV) ± std | R (in-sample) | RMSE CV | ΔR vs FIXED | Features |",
        "|---|---:|---:|---:|---:|---|",
        (f"| stock FIXED (2015) | {r_fixed:+.3f} | {r_fixed:+.3f} | "
         f"{rmse_fixed:.2f} | +0.000 | 6 stock |"),
        (f"| stock REFIT CV | {stock_refit['R_cv']:+.3f} ± "
         f"{stock_refit['R_cv_std']:.3f} | {stock_refit['R_in']:+.3f} | "
         f"{stock_refit['RMSE_cv']:.2f} | "
         f"{stock_refit['R_cv'] - r_fixed:+.3f} | 6 stock |"),
        (f"| augmented AIC | {aug_res['R_cv']:+.3f} ± "
         f"{aug_res['R_cv_std']:.3f} | {aug_res['R_in']:+.3f} | "
         f"{aug_res['RMSE_cv']:.2f} | "
         f"{aug_res['R_cv'] - r_fixed:+.3f} | {', '.join(aug_selected)} |"),
        (f"| interaction AIC | {int_res['R_cv']:+.3f} ± "
         f"{int_res['R_cv_std']:.3f} | {int_res['R_in']:+.3f} | "
         f"{int_res['RMSE_cv']:.2f} | "
         f"{int_res['R_cv'] - r_fixed:+.3f} | {', '.join(int_selected)} |"),
        (f"| **PAE candidate** | **{cand_res['R_cv']:+.3f} ± "
         f"{cand_res['R_cv_std']:.3f}** | {cand_res['R_in']:+.3f} | "
         f"{cand_res['RMSE_cv']:.2f} | "
         f"**{cand_res['R_cv'] - r_fixed:+.3f}** | **6 stock + ic_pa × ipTM** |"),
        "",
        "## Single-interaction ablation (stock REFIT + 1)",
        "",
        "| Interaction | R_CV | ΔR vs FIXED | ΔR vs REFIT |",
        "|---|---:|---:|---:|",
    ]
    for row in ablation:
        lines.append(
            f"| `{row['interaction']}` | {row['R_cv']:+.3f} ± "
            f"{row['R_cv_std']:.3f} | {row['dR_vs_fixed']:+.3f} | "
            f"{row['dR_vs_refit']:+.3f} |"
        )
    lines += [
        "",
        "## PAE-candidate coefficients",
        "",
        "Stock 6 features + ic_pa × ipTM, fitted by OLS on the union N=123.",
        "These are candidate values for ``NIS_COEFFICIENTS_PAE`` in "
        "``src/protein_affinity_gpu/scoring.py`` once Phase 3 is unblocked.",
        "",
        "| Feature | Coefficient |",
        "|---|---:|",
    ]
    for name, val in zip(cand_res["features"], cand_res["coefs"]):
        lines.append(f"| `{name}` | {val:+.5f} |")
    lines.append(f"| intercept | {cand_res['intercept']:+.4f} |")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n")
    print(f"[write] {(OUT_DIR / 'report.md').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
