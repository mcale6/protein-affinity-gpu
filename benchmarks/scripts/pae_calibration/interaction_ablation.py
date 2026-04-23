#!/usr/bin/env python3
"""Ablation: which IC × ipTM (and IC × ⟨PAE⟩) interaction carries the signal?

Experiment C (``interaction_refit.py``) establishes that AIC-sparse
interaction terms lift R above stock REFIT-CV. But AIC picks jointly —
is the lift driven by one dominant interaction, or distributed across
several?

This script adds each candidate interaction individually to the 6-feat
stock REFIT, runs the same 4-fold CV × 10 repeats protocol, and reports
ΔR vs stock REFIT-CV. Four IC × ipTM and four IC × mean_PAE_contacts
interactions = 8 single-addition models.

Usage:
    python interaction_ablation.py --dataset vreven
    python interaction_ablation.py --dataset kastritis --mode msa_only
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

import quick_pae_calib as qpc  # noqa: E402
from diagnostic_refit import (  # noqa: E402
    COEFFS_STOCK, INTERCEPT_STOCK, IRMSD_CUTOFF, R_RMSE, fit_ols,
    kfold_cv_R,
)

STOCK_FEATS = ["ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c"]
IC_FEATS = ["ic_cc", "ic_ca", "ic_pp", "ic_pa"]
CONFIDENCE_MODS = ("iptm", "mean_pae_contacts")


def load_features(mode: str) -> pd.DataFrame:
    aug_dir = qpc.BOLTZ_ROOT / "pae_calibration" / "augmented_refit"
    p = aug_dir / f"features_{mode}.csv"
    if not p.exists():
        raise SystemExit(
            f"Missing {p} — run augmented_refit.py --dataset "
            f"{qpc.DATASET_NAME} --mode {mode} first."
        )
    df = pd.read_csv(p)
    df["nis_a"] = df["nis_a"].clip(lower=0, upper=100)
    df["nis_c"] = df["nis_c"].clip(lower=0, upper=100)
    return df


def build_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for ic in IC_FEATS:
        for mod in CONFIDENCE_MODS:
            out[f"{ic}_x_{mod}"] = out[ic].to_numpy() * out[mod].to_numpy()
    return out


def eval_model(df: pd.DataFrame, feats: list[str], y: np.ndarray,
               is_flex: np.ndarray) -> dict:
    X = df[feats].to_numpy()
    coefs, icept = fit_ols(X, y)
    pred_in = X @ coefs + icept
    r_in, rmse_in = R_RMSE(pred_in, y)
    cv = kfold_cv_R(X, y, k=4, n_repeats=10, seed=0)
    preds_cv = cv["preds_mean"]
    r_cv_rigid, _ = R_RMSE(preds_cv[~is_flex], y[~is_flex])
    r_cv_flex, _ = R_RMSE(preds_cv[is_flex], y[is_flex])
    return {
        "R_in": r_in, "RMSE_in": rmse_in,
        "R_cv": cv["R_mean"], "R_cv_std": cv["R_std"],
        "RMSE_cv": cv["RMSE_mean"], "RMSE_cv_std": cv["RMSE_std"],
        "R_cv_rigid": r_cv_rigid, "R_cv_flex": r_cv_flex,
        "coefs": coefs, "intercept": icept,
    }


def run(dataset: str, mode: str) -> dict:
    qpc.set_dataset(dataset)
    df = build_interactions(load_features(mode))
    y = df["dg_exp"].to_numpy()
    is_flex = (df["irmsd"] > IRMSD_CUTOFF).to_numpy()

    # Baselines
    X_stock = df[STOCK_FEATS].to_numpy()
    pred_fixed = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed, rmse_fixed = R_RMSE(pred_fixed, y)
    stock_fixed = {
        "R_cv": r_fixed, "R_cv_std": 0.0,
        "RMSE_cv": rmse_fixed, "RMSE_cv_std": 0.0,
        "R_in": r_fixed, "R_cv_rigid": float("nan"), "R_cv_flex": float("nan"),
    }
    stock_refit = eval_model(df, STOCK_FEATS, y, is_flex)

    # Each single-interaction addition
    singles = {}
    for mod in CONFIDENCE_MODS:
        for ic in IC_FEATS:
            feat = f"{ic}_x_{mod}"
            feats = STOCK_FEATS + [feat]
            singles[feat] = eval_model(df, feats, y, is_flex)

    return {
        "mode": mode, "N": len(y),
        "n_rigid": int((~is_flex).sum()), "n_flex": int(is_flex.sum()),
        "stock_fixed": stock_fixed,
        "stock_refit": stock_refit,
        "singles": singles,
    }


def print_report(res: dict, dataset: str) -> None:
    print(f"[{dataset} / {res['mode']}]  N={res['N']}  "
          f"rigid={res['n_rigid']}  flex={res['n_flex']}")
    print()
    print(f"{'model':<30s} {'R (CV)':>10s} {'±std':>6s} {'ΔR vs FIXED':>12s} "
          f"{'ΔR vs REFIT':>12s} {'RMSE CV':>9s}")
    print("-" * 90)
    fixed_R = res["stock_fixed"]["R_cv"]
    refit_R = res["stock_refit"]["R_cv"]
    print(f"{'stock FIXED':<30s} {fixed_R:>+10.3f} {'':>6s} "
          f"{0.0:>+12.3f} {'':>12s} {res['stock_fixed']['RMSE_cv']:>9.2f}")
    print(f"{'stock REFIT CV':<30s} {refit_R:>+10.3f} "
          f"{res['stock_refit']['R_cv_std']:>6.3f} "
          f"{refit_R - fixed_R:>+12.3f} {'':>12s} "
          f"{res['stock_refit']['RMSE_cv']:>9.2f}")
    print()
    # Rank singles by ΔR vs stock REFIT (matched comparator)
    items = sorted(
        res["singles"].items(),
        key=lambda kv: -kv[1]["R_cv"],
    )
    for feat, v in items:
        print(f"{'+ ' + feat:<30s} {v['R_cv']:>+10.3f} "
              f"{v['R_cv_std']:>6.3f} "
              f"{v['R_cv'] - fixed_R:>+12.3f} "
              f"{v['R_cv'] - refit_R:>+12.3f} "
              f"{v['RMSE_cv']:>9.2f}")


def write_csv(res: dict, dataset: str, out_path: Path) -> None:
    rows = [
        {
            "model": "stock_FIXED", "R_cv": res["stock_fixed"]["R_cv"],
            "R_cv_std": 0.0, "RMSE_cv": res["stock_fixed"]["RMSE_cv"],
            "R_cv_rigid": res["stock_fixed"]["R_cv_rigid"],
            "R_cv_flex": res["stock_fixed"]["R_cv_flex"],
            "R_in": res["stock_fixed"]["R_in"],
        },
        {
            "model": "stock_REFIT_CV", "R_cv": res["stock_refit"]["R_cv"],
            "R_cv_std": res["stock_refit"]["R_cv_std"],
            "RMSE_cv": res["stock_refit"]["RMSE_cv"],
            "R_cv_rigid": res["stock_refit"]["R_cv_rigid"],
            "R_cv_flex": res["stock_refit"]["R_cv_flex"],
            "R_in": res["stock_refit"]["R_in"],
        },
    ]
    for feat, v in res["singles"].items():
        rows.append({
            "model": f"stock + {feat}", "R_cv": v["R_cv"],
            "R_cv_std": v["R_cv_std"], "RMSE_cv": v["RMSE_cv"],
            "R_cv_rigid": v["R_cv_rigid"], "R_cv_flex": v["R_cv_flex"],
            "R_in": v["R_in"],
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="vreven",
                    choices=["kastritis", "vreven"])
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    qpc.set_dataset(args.dataset)
    out_dir = (Path(args.out_dir) if args.out_dir
                else qpc.BOLTZ_ROOT / "pae_calibration" / "interaction_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = run(args.dataset, args.mode)
    print_report(res, args.dataset)
    write_csv(res, args.dataset, out_dir / f"ablation_{args.mode}.csv")
    print(f"\nWrote {out_dir / f'ablation_{args.mode}.csv'}")


if __name__ == "__main__":
    main()
