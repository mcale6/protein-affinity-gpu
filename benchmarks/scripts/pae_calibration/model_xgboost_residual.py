#!/usr/bin/env python3
"""Two-stage residual model: stock PRODIGY + XGBoost on the residual.

Central research question: can a non-linear model (XGBoost) discover
interactions beyond the cross-dataset-stable ``ic_pa × ipTM`` seen in
``diagnostic_refit.py`` on the 287-complex unified dataset?

Pipeline (Stage 1 is deterministic, Stage 2 is learned):

    Stage 1 : dg_pred_stock  = f(6 stock IC/NIS, COEFFS_STOCK)
    Stage 2 : residual_hat   = XGBoost(non-stock features)
              dg_pred_final  = dg_pred_stock + residual_hat

The 287 rows span K81 (81, ITC), V106 (106, mixed), PB (100, SPR/BLI). K81 ∩
V106 share 64 pdb_ids, so all CV is **grouped on pdb_id**. PB has NaN
atom-level CAD / PAE / Boltz-confidence-score columns; XGBoost handles NaN
natively (``missing=nan``).

Evaluation protocol:

    - grouped 4-fold × 10 repeats   (primary)
    - leave-one-source-out (K81+V106 → PB, K81+PB → V106, V106+PB → K81)
    - fit on K81+V106, predict PB       (extrapolation check)

Baselines reported for context:
    - FIXED  (stock coefs, no refit)
    - REFIT-CV (grouped OLS on 6 stock features)

SHAP analysis on the best-seed full-data fit:
    - mean |SHAP| bar + beeswarm (top 20)
    - per-feature mean |SHAP| CSV
    - top-10 SHAP interactions (check whether ``ic_pa × ipTM`` or
      ``ic_cc × mean_pae_contacts`` appear)
    - partial-dependence scatters for top 3 features

Outputs (worktree-local):
    benchmarks/output/unified/xgboost_residual/
        report.md
        calibration_grouped_cv.png
        calibration_heldout_pb.png
        shap_summary.png
        pdp_top3.png
        feature_importance.csv
        shap_interactions.csv
        predictions_grouped_cv.csv

Usage:
    python model_xgboost_residual.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402
import xgboost as xgb  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
CSV_PATH = ROOT / "benchmarks/output/unified/unified_features.csv"
OUT_DIR = ROOT / "benchmarks/output/unified/xgboost_residual"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Stock PRODIGY (2015) coefficients — Stage 1
# --------------------------------------------------------------------------
STOCK_FEATURES = ("ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c")
COEFFS_STOCK = np.array(
    [-0.09459, -0.10007, 0.19577, -0.22671, 0.18681, 0.13810], dtype=np.float64,
)
INTERCEPT_STOCK = -15.9433

# --------------------------------------------------------------------------
# XGBoost hyperparameters (N=287, small, shallow tree ensemble)
# --------------------------------------------------------------------------
XGB_PARAMS = dict(
    n_estimators=200,
    max_depth=3,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:squarederror",
    tree_method="hist",
    enable_categorical=False,
    missing=np.nan,
    random_state=0,
    verbosity=0,
)

N_REPEATS = 10
N_FOLDS = 4
N_BOOT = 500


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------
def load_data() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH)
    # Clip NIS% to [0, 100] to match PRODIGY convention; NaN rows (PB nis_p)
    # are left untouched so XGBoost treats them as missing.
    for col in ("nis_a", "nis_c", "nis_p"):
        if col in df.columns:
            df[col] = df[col].where(df[col].isna(), df[col].clip(0, 100))
    return df


def stock_prediction(df: pd.DataFrame) -> np.ndarray:
    X = df[list(STOCK_FEATURES)].to_numpy(dtype=np.float64)
    # Clip NIS columns 4 and 5 for stock formula (safety; already clipped).
    X[:, 4] = np.clip(X[:, 4], 0, 100)
    X[:, 5] = np.clip(X[:, 5], 0, 100)
    return X @ COEFFS_STOCK + INTERCEPT_STOCK


def build_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return the design matrix for Stage-2 XGBoost and the feature names."""
    non_feat = {
        # identifiers & target
        "pdb_id", "source", "dg_exp_kcal_mol", "kd_m", "log10_kd",
        "irmsd", "stratum", "functional_class",
        "dg_prodigy_boltz",                # redundant with STOCK features
        "cad_arrays_jsonl",                # path, not a feature
    }
    # Drop the 6 stock features from Stage 2: residual must be non-trivial.
    non_feat |= set(STOCK_FEATURES)

    feats = [c for c in df.columns if c not in non_feat]

    X = df[feats].copy()
    # Engineered interactions — seed XGBoost with a handful of known/plausible
    # ones; tree splits can still discover others. Guard against NaN: if any
    # factor is NaN, the product is NaN → XGBoost treats as missing.
    X["int__ic_pa_x_iptm"]           = df["ic_pa"] * df["boltz_iptm"]
    X["int__ic_cc_x_mean_pae_ct"]    = df["ic_cc"] * df["mean_pae_contacts"]
    X["int__cad_rr_x_iptm"]          = df["cad_rr"] * df["boltz_iptm"]
    X["int__ic_pa_x_mean_pae_iface"] = df["ic_pa"] * df["mean_pae_interface"]
    X["int__atom_cad_mean_x_iptm"]   = df["atom_cad_mean"] * df["boltz_iptm"]

    # Ensure numeric (everything should be; safety net).
    X = X.apply(pd.to_numeric, errors="coerce")
    feat_names = list(X.columns)
    return X, feat_names


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    mask = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[mask], y[mask]
    if len(y) < 3 or np.std(pred) == 0:
        return {"R": float("nan"), "rho": float("nan"),
                "RMSE": float("nan"), "MAE": float("nan"), "N": int(len(y))}
    r = float(pearsonr(pred, y)[0])
    rho = float(spearmanr(pred, y).correlation)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    return {"R": r, "rho": rho, "RMSE": rmse, "MAE": mae, "N": int(len(y))}


def bootstrap_R_ci(pred: np.ndarray, y: np.ndarray, n_boot: int = N_BOOT,
                   seed: int = 0) -> tuple[float, float, float]:
    mask = np.isfinite(pred) & np.isfinite(y)
    pred, y = pred[mask], y[mask]
    if len(y) < 3:
        return (float("nan"),) * 3
    rng = np.random.default_rng(seed)
    N = len(y)
    rs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, N, size=N)
        if np.std(pred[idx]) == 0:
            rs[b] = np.nan
        else:
            rs[b] = pearsonr(pred[idx], y[idx])[0]
    rs = rs[np.isfinite(rs)]
    r_point = float(pearsonr(pred, y)[0])
    lo, hi = (float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5)))
    return r_point, lo, hi


# --------------------------------------------------------------------------
# Fold assignment — grouped CV with repeat-level group shuffling
# --------------------------------------------------------------------------
def grouped_kfold_indices(groups: np.ndarray, n_splits: int,
                          seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Grouped K-fold where the *group* order is shuffled by seed.

    sklearn's ``GroupKFold`` is deterministic in group order; to emulate 10
    different repeats we permute group identifiers per seed, then re-hand to
    ``GroupKFold``.
    """
    rng = np.random.default_rng(seed)
    uniq = np.unique(groups)
    perm = rng.permutation(uniq)
    remap = {g: i for i, g in enumerate(perm)}
    new_groups = np.array([remap[g] for g in groups])
    gkf = GroupKFold(n_splits=n_splits)
    return list(gkf.split(np.zeros(len(groups)), groups=new_groups))


# --------------------------------------------------------------------------
# Training loops
# --------------------------------------------------------------------------
def fit_xgb_with_val(X_tr: np.ndarray, y_tr: np.ndarray,
                     seed: int) -> xgb.XGBRegressor:
    """Fit XGBoost on residuals with a 20 % internal validation split for
    early stopping (validation chosen per seed, disjoint of CV test)."""
    rng = np.random.default_rng(seed)
    N = len(y_tr)
    n_val = max(5, int(0.2 * N))
    idx = rng.permutation(N)
    val_idx = idx[:n_val]
    tr_idx = idx[n_val:]

    params = dict(XGB_PARAMS, random_state=seed, early_stopping_rounds=20)
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr[tr_idx], y_tr[tr_idx],
        eval_set=[(X_tr[val_idx], y_tr[val_idx])],
        verbose=False,
    )
    return model


def ols_fit(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    Xa = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return beta[:-1], float(beta[-1])


# --------------------------------------------------------------------------
# Experiment: grouped 4-fold × 10 repeats (residual XGBoost + two OLS baselines)
# --------------------------------------------------------------------------
def grouped_cv_experiment(df: pd.DataFrame, X_full: pd.DataFrame, y: np.ndarray,
                          dg_stock: np.ndarray, pdb_groups: np.ndarray):
    """Primary experiment. Returns dict of aggregate metrics + per-row means."""
    N = len(y)
    X_np = X_full.to_numpy(dtype=np.float64)
    stock_X = df[list(STOCK_FEATURES)].to_numpy(dtype=np.float64)
    # Clip NIS defensively.
    stock_X[:, 4] = np.clip(stock_X[:, 4], 0, 100)
    stock_X[:, 5] = np.clip(stock_X[:, 5], 0, 100)

    # Stash per-repeat out-of-fold predictions to average later.
    preds_xgb_final = np.full((N_REPEATS, N), np.nan)
    preds_refit    = np.full((N_REPEATS, N), np.nan)
    residual_preds = np.full((N_REPEATS, N), np.nan)

    per_repeat_R_xgb = []
    per_repeat_R_refit = []
    per_repeat_R_fixed = []  # deterministic but reported each repeat for uniform mean/std

    fixed_pred = dg_stock  # doesn't depend on fold
    per_repeat_R_fixed_val = metrics(fixed_pred, y)["R"]

    for rep in range(N_REPEATS):
        splits = grouped_kfold_indices(pdb_groups, N_FOLDS, seed=rep)
        rep_pred_xgb = np.full(N, np.nan)
        rep_resid   = np.full(N, np.nan)
        rep_refit   = np.full(N, np.nan)

        for train_idx, test_idx in splits:
            # Stage 1 is fixed: dg_stock already computed.
            # Stage 2: fit XGBoost on residual within train.
            r_train = y[train_idx] - dg_stock[train_idx]
            model = fit_xgb_with_val(X_np[train_idx], r_train, seed=rep)
            resid_hat = model.predict(X_np[test_idx])
            rep_resid[test_idx]  = resid_hat
            rep_pred_xgb[test_idx] = dg_stock[test_idx] + resid_hat

            # Baseline: OLS refit on 6 stock features (grouped CV).
            coefs, icept = ols_fit(stock_X[train_idx], y[train_idx])
            rep_refit[test_idx] = stock_X[test_idx] @ coefs + icept

        preds_xgb_final[rep] = rep_pred_xgb
        residual_preds[rep]  = rep_resid
        preds_refit[rep]     = rep_refit
        per_repeat_R_xgb.append(metrics(rep_pred_xgb, y)["R"])
        per_repeat_R_refit.append(metrics(rep_refit, y)["R"])
        per_repeat_R_fixed.append(per_repeat_R_fixed_val)
        print(f"  rep {rep}: R(xgb)={per_repeat_R_xgb[-1]:+.3f}   "
              f"R(refit)={per_repeat_R_refit[-1]:+.3f}")

    mean_pred_xgb = np.nanmean(preds_xgb_final, axis=0)
    mean_pred_refit = np.nanmean(preds_refit, axis=0)
    mean_residual  = np.nanmean(residual_preds, axis=0)

    # Bootstrap CIs on the mean prediction.
    xgb_m = metrics(mean_pred_xgb, y)
    _, xgb_lo, xgb_hi = bootstrap_R_ci(mean_pred_xgb, y, seed=0)
    refit_m = metrics(mean_pred_refit, y)
    _, refit_lo, refit_hi = bootstrap_R_ci(mean_pred_refit, y, seed=1)
    fixed_m = metrics(fixed_pred, y)
    _, fixed_lo, fixed_hi = bootstrap_R_ci(fixed_pred, y, seed=2)

    return {
        "mean_pred_xgb_final": mean_pred_xgb,
        "mean_pred_refit":     mean_pred_refit,
        "mean_residual":       mean_residual,
        "fixed_pred":          fixed_pred,
        "xgb":   {**xgb_m, "R_ci": (xgb_lo, xgb_hi),
                  "R_rep_mean": float(np.mean(per_repeat_R_xgb)),
                  "R_rep_std":  float(np.std(per_repeat_R_xgb))},
        "refit": {**refit_m, "R_ci": (refit_lo, refit_hi),
                  "R_rep_mean": float(np.mean(per_repeat_R_refit)),
                  "R_rep_std":  float(np.std(per_repeat_R_refit))},
        "fixed": {**fixed_m, "R_ci": (fixed_lo, fixed_hi)},
        "per_repeat_R_xgb": per_repeat_R_xgb,
    }


# --------------------------------------------------------------------------
# Experiment: leave-one-source-out
# --------------------------------------------------------------------------
def loso_experiment(df: pd.DataFrame, X_full: pd.DataFrame, y: np.ndarray,
                    dg_stock: np.ndarray) -> dict:
    """Leave-one-source-out (train on 2, test on 1). Runs one model per fold."""
    X_np = X_full.to_numpy(dtype=np.float64)
    sources = df["source"].to_numpy()
    out = {}
    for held in ("ProteinBase", "VrevenBM5.5", "Kastritis81"):
        tr = sources != held
        te = sources == held
        r_train = y[tr] - dg_stock[tr]
        model = fit_xgb_with_val(X_np[tr], r_train, seed=0)
        resid_hat = model.predict(X_np[te])
        dg_final = dg_stock[te] + resid_hat
        m = metrics(dg_final, y[te])
        _, lo, hi = bootstrap_R_ci(dg_final, y[te], seed=0)
        out[held] = {**m, "R_ci": (lo, hi), "pred": dg_final,
                     "y": y[te], "pdb_ids": df.loc[te, "pdb_id"].tolist()}
    return out


# --------------------------------------------------------------------------
# SHAP analysis
# --------------------------------------------------------------------------
def run_shap(X_full: pd.DataFrame, y: np.ndarray, dg_stock: np.ndarray,
             out_dir: Path, feat_names: list[str]) -> dict:
    """Fit a final XGBoost on all 287 residuals (seed=0) and run SHAP."""
    X_np = X_full.to_numpy(dtype=np.float64)
    residual = y - dg_stock
    # No early stopping on full-data fit; use the same n_estimators.
    params = dict(XGB_PARAMS, random_state=0)
    params.pop("early_stopping_rounds", None)
    model = xgb.XGBRegressor(**params)
    model.fit(X_np, residual, verbose=False)

    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(X_np)              # (N, F)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    ranked = [(feat_names[i], float(mean_abs[i])) for i in order]

    # CSV
    pd.DataFrame(ranked, columns=["feature", "mean_abs_shap"]).to_csv(
        out_dir / "feature_importance.csv", index=False
    )

    # Summary plot (top-20 beeswarm + bar).
    top20 = order[:20]
    fig = plt.figure(figsize=(8, 9))
    shap.summary_plot(shap_vals[:, top20], X_full.iloc[:, top20],
                      feature_names=[feat_names[i] for i in top20],
                      show=False, plot_size=None)
    fig = plt.gcf()
    fig.set_size_inches(8, 8)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # SHAP interaction values on top 15 features (reduce cost; full is O(F^2 N)).
    top15 = order[:15]
    X_top = X_np[:, top15]
    # TreeExplainer wants the model to see same dims as training — use all X.
    inter_vals = explainer.shap_interaction_values(X_np)  # (N, F, F)
    # Restrict to top-15 × top-15 slab, zero diagonal, keep only upper tri.
    inter_slab = inter_vals[:, top15][:, :, top15]
    # Off-diagonal interaction strength = 2|SHAP_ij| (symmetric with i<j).
    F = len(top15)
    rows = []
    for i in range(F):
        for j in range(i + 1, F):
            strength = float(np.abs(inter_slab[:, i, j] + inter_slab[:, j, i]).mean())
            rows.append((feat_names[top15[i]], feat_names[top15[j]], strength))
    rows.sort(key=lambda r: -r[2])
    inter_df = pd.DataFrame(rows, columns=["feature_a", "feature_b",
                                            "mean_abs_interaction"])
    inter_df.head(30).to_csv(out_dir / "shap_interactions.csv", index=False)

    # Partial-dependence (SHAP scatter) for top 3 features.
    top3 = order[:3]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for ax, idx in zip(axes, top3):
        feat = feat_names[idx]
        x_vals = X_np[:, idx]
        s_vals = shap_vals[:, idx]
        mask = np.isfinite(x_vals)
        ax.scatter(x_vals[mask], s_vals[mask], s=14, alpha=0.6, c="C0")
        ax.axhline(0, color="k", lw=0.7, alpha=0.4)
        ax.set_xlabel(feat); ax.set_ylabel(f"SHAP({feat})")
        ax.set_title(f"mean|SHAP|={mean_abs[idx]:.3f}")
        ax.grid(alpha=0.3)
    fig.suptitle("Top-3 partial dependence (XGBoost residual model)")
    fig.tight_layout()
    fig.savefig(out_dir / "pdp_top3.png", dpi=140)
    plt.close(fig)

    return {"ranked": ranked, "top5": ranked[:5], "top10": ranked[:10],
            "interactions": rows[:10], "model": model, "shap_values": shap_vals,
            "top3_names": [feat_names[i] for i in top3]}


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
_SRC_COLOR = {
    "Kastritis81": "C0",
    "VrevenBM5.5": "C3",
    "ProteinBase": "C2",
}


def calibration_plot(pred: np.ndarray, y: np.ndarray, sources: np.ndarray,
                     out_path: Path, title: str):
    m = metrics(pred, y)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for src, color in _SRC_COLOR.items():
        mask = sources == src
        if mask.sum() == 0:
            continue
        ax.scatter(y[mask], pred[mask], s=24, alpha=0.65, color=color,
                   label=f"{src} (n={mask.sum()})", edgecolor="none")
    lo = float(np.nanmin([np.nanmin(y), np.nanmin(pred)])) - 1.0
    hi = float(np.nanmax([np.nanmax(y), np.nanmax(pred)])) + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\Delta G_\mathrm{exp}$ (kcal/mol)")
    ax.set_ylabel(r"$\Delta G_\mathrm{pred}$ (kcal/mol)")
    ax.set_title(
        f"{title}\nR={m['R']:+.3f}  RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}  "
        f"N={m['N']}"
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report writer
# --------------------------------------------------------------------------
def write_report(out_dir: Path, df: pd.DataFrame, cv: dict, loso: dict,
                 shap_summary: dict, extrap: dict):
    xgb_R = cv["xgb"]["R"]; xgb_lo, xgb_hi = cv["xgb"]["R_ci"]
    refit_R = cv["refit"]["R"]; fixed_R = cv["fixed"]["R"]
    delta_vs_refit = xgb_R - refit_R
    delta_vs_fixed = xgb_R - fixed_R

    if delta_vs_refit >= 0.05:
        verdict = "HELPS  (XGBoost residual delivers ΔR ≥ +0.05 over stock REFIT-CV)"
    elif delta_vs_refit >= 0.03:
        verdict = "MARGINAL  (ΔR ≈ +0.03 over stock REFIT-CV)"
    elif delta_vs_refit >= 0:
        verdict = "NEUTRAL  (no meaningful gain over stock REFIT-CV)"
    else:
        verdict = "HURTS  (XGBoost underperforms stock REFIT-CV)"

    # Qualitative SHAP check: does the cross-dataset-stable ic_pa × iptm appear
    # either as a top SHAP pair interaction, or as a top-ranked engineered
    # feature (seeded product) with meaningful mean|SHAP|?
    top_pair_names = [(a.lower(), b.lower()) for a, b, _ in shap_summary["interactions"]]

    def _pair_contains(tokens_a: list[str], tokens_b: list[str]) -> bool:
        for a, b in top_pair_names:
            if (any(t in a for t in tokens_a) and any(t in b for t in tokens_b)) or \
               (any(t in b for t in tokens_a) and any(t in a for t in tokens_b)):
                return True
        return False

    top20_feats = {n.lower() for n, _ in shap_summary["ranked"][:20]}
    # Either the engineered ic_pa*iptm is in top-20 features ...
    engineered_ic_pa_iptm = any("ic_pa" in f and "iptm" in f for f in top20_feats)
    # ... or ic_pa and iptm-like features appear together as a SHAP interaction.
    shap_pair_ic_pa_iptm = _pair_contains(["ic_pa"], ["iptm"])
    has_ic_pa_iptm = engineered_ic_pa_iptm or shap_pair_ic_pa_iptm

    engineered_ic_cc_pae = any("ic_cc" in f and "pae" in f for f in top20_feats)
    shap_pair_ic_cc_pae = _pair_contains(["ic_cc"], ["mean_pae_contacts", "pae"])
    has_ic_cc_pae = engineered_ic_cc_pae or shap_pair_ic_cc_pae

    lines = [
        "# XGBoost residual on unified 287-complex set",
        "",
        f"**Verdict: {verdict}**",
        "",
        "Stage 1 = deterministic stock PRODIGY (2015 coefs on 6 IC/NIS features).",
        "Stage 2 = XGBoost(n_estimators=200, max_depth=3, lr=0.05) on the residual,",
        "given every non-stock column (78 raw + 5 engineered interactions).",
        "",
        "## 1. Primary grouped CV (4-fold × 10 repeats, pdb_id groups)",
        "",
        "| Policy | R | 95% CI | ρ (Spearman) | RMSE | MAE | ΔR vs stock REFIT-CV |",
        "|---|---:|:---:|---:|---:|---:|---:|",
        (f"| FIXED (stock coefs)   | {fixed_R:+.3f} | "
         f"[{cv['fixed']['R_ci'][0]:+.3f}, {cv['fixed']['R_ci'][1]:+.3f}] | "
         f"{cv['fixed']['rho']:+.3f} | {cv['fixed']['RMSE']:.2f} | "
         f"{cv['fixed']['MAE']:.2f} | {fixed_R - refit_R:+.3f} |"),
        (f"| REFIT-CV (6 stock OLS)| {refit_R:+.3f} | "
         f"[{cv['refit']['R_ci'][0]:+.3f}, {cv['refit']['R_ci'][1]:+.3f}] | "
         f"{cv['refit']['rho']:+.3f} | {cv['refit']['RMSE']:.2f} | "
         f"{cv['refit']['MAE']:.2f} | ref |"),
        (f"| **XGBoost residual**  | **{xgb_R:+.3f}** | "
         f"[{xgb_lo:+.3f}, {xgb_hi:+.3f}] | "
         f"{cv['xgb']['rho']:+.3f} | {cv['xgb']['RMSE']:.2f} | "
         f"{cv['xgb']['MAE']:.2f} | **{delta_vs_refit:+.3f}** |"),
        "",
        f"Per-repeat R (XGBoost residual): "
        f"{cv['xgb']['R_rep_mean']:+.3f} ± {cv['xgb']['R_rep_std']:.3f} "
        f"(mean ± std over {N_REPEATS} repeats).",
        "",
        "## 2. Leave-one-source-out",
        "",
        "| Held-out source | N | R | 95% CI | ρ | RMSE | MAE |",
        "|---|---:|---:|:---:|---:|---:|---:|",
    ]
    for src in ("ProteinBase", "VrevenBM5.5", "Kastritis81"):
        m = loso[src]
        lines.append(
            f"| {src} | {m['N']} | {m['R']:+.3f} | "
            f"[{m['R_ci'][0]:+.3f}, {m['R_ci'][1]:+.3f}] | "
            f"{m['rho']:+.3f} | {m['RMSE']:.2f} | {m['MAE']:.2f} |"
        )
    lines += [
        "",
        "## 3. Extrapolation: fit on K81+V106, predict PB (one-shot)",
        "",
        (f"- R = {extrap['R']:+.3f}   95% CI [{extrap['R_ci'][0]:+.3f}, "
         f"{extrap['R_ci'][1]:+.3f}]"),
        f"- ρ = {extrap['rho']:+.3f}",
        f"- RMSE = {extrap['RMSE']:.2f} kcal/mol",
        f"- MAE  = {extrap['MAE']:.2f} kcal/mol",
        f"- N = {extrap['N']}",
        "",
        "## 4. SHAP — top-20 feature importances (mean |SHAP|)",
        "",
        "| rank | feature | mean |SHAP| |",
        "|---:|---|---:|",
    ]
    for rk, (name, val) in enumerate(shap_summary["ranked"][:20], start=1):
        lines.append(f"| {rk} | `{name}` | {val:.4f} |")
    lines += [
        "",
        "### Top-5 SHAP features",
        "",
    ] + [f"{rk}. `{n}`   mean|SHAP|={v:.4f}"
         for rk, (n, v) in enumerate(shap_summary["top5"], start=1)] + [
        "",
        "## 5. SHAP interaction values — top 10 pairs",
        "",
        "| rank | feature A | feature B | mean |interaction| |",
        "|---:|---|---|---:|",
    ]
    for rk, (a, b, val) in enumerate(shap_summary["interactions"][:10], start=1):
        lines.append(f"| {rk} | `{a}` | `{b}` | {val:.4f} |")
    lines += [
        "",
        "### Qualitative checks",
        "",
        (f"- `ic_pa × boltz_iptm`: **{'YES' if has_ic_pa_iptm else 'NO'}** "
         f"(engineered feature in top-20: "
         f"{'yes' if engineered_ic_pa_iptm else 'no'}; "
         f"pair in top-10 SHAP interactions: "
         f"{'yes' if shap_pair_ic_pa_iptm else 'no'})"),
        (f"- `ic_cc × mean_pae_contacts`: **{'YES' if has_ic_cc_pae else 'NO'}** "
         f"(engineered feature in top-20: "
         f"{'yes' if engineered_ic_cc_pae else 'no'}; "
         f"pair in top-10 SHAP interactions: "
         f"{'yes' if shap_pair_ic_cc_pae else 'no'})"),
        "",
        "## 6. Plots",
        "",
        "- `calibration_grouped_cv.png` — OOF predictions vs ΔG_exp over "
        "the 4-fold × 10 repeat grouped CV, all 287 points.",
        "- `calibration_heldout_pb.png` — fit on K81+V106, predict PB (100 pts).",
        "- `shap_summary.png` — beeswarm of top-20 features.",
        "- `pdp_top3.png` — SHAP partial-dependence for the top 3 features "
        f"({', '.join('`%s`' % n for n in shap_summary['top3_names'])}).",
        "",
        "## 7. Files",
        "",
        "- `predictions_grouped_cv.csv` — `pdb_id, source, dg_exp, dg_pred_stock, "
        "residual_pred, dg_pred_final` (means over 10 repeats).",
        "- `feature_importance.csv` — all features ranked by mean|SHAP|.",
        "- `shap_interactions.csv` — top-30 pair interactions.",
        "",
        "## Success criterion",
        "",
        f"- ΔR vs REFIT-CV = {delta_vs_refit:+.3f}  "
        f"({'≥ +0.05 → HELPS' if delta_vs_refit >= 0.05 else ('≥ +0.03 → MARGINAL' if delta_vs_refit >= 0.03 else '< +0.03 → not HELPS')})",
        f"- ΔR vs FIXED    = {delta_vs_fixed:+.3f}",
        "",
    ]
    (out_dir / "report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    df = load_data()
    dg_stock = stock_prediction(df)
    y = df["dg_exp_kcal_mol"].to_numpy(dtype=np.float64)

    X_full, feat_names = build_features(df)
    print(f"[load] {len(df)} rows  |  {len(feat_names)} features for XGBoost")
    print(f"[sources] {dict(df.source.value_counts())}")

    # -------------------------------------------------------------- primary CV
    print("\n=== grouped 4-fold × 10 repeats ===")
    pdb_groups = df["pdb_id"].to_numpy()
    cv = grouped_cv_experiment(df, X_full, y, dg_stock, pdb_groups)
    print(f"[cv]    XGBoost R = {cv['xgb']['R']:+.3f}   RMSE = {cv['xgb']['RMSE']:.2f}")
    print(f"        REFIT  R = {cv['refit']['R']:+.3f}")
    print(f"        FIXED  R = {cv['fixed']['R']:+.3f}")

    # -------------------------------------------------------------- plots
    calibration_plot(
        cv["mean_pred_xgb_final"], y, df["source"].to_numpy(),
        OUT_DIR / "calibration_grouped_cv.png",
        "Grouped CV (mean OOF pred over 10 repeats)",
    )

    # ---------------------------------------------------- leave-one-source-out
    print("\n=== leave-one-source-out ===")
    loso = loso_experiment(df, X_full, y, dg_stock)
    for src, m in loso.items():
        print(f"  {src:<14s}  N={m['N']:3d}  R={m['R']:+.3f}   "
              f"RMSE={m['RMSE']:.2f}   MAE={m['MAE']:.2f}")

    # ---------------------------------------------------- extrapolation to PB
    # This equals loso["ProteinBase"], but we re-plot separately.
    pb = loso["ProteinBase"]
    calibration_plot(
        pb["pred"], pb["y"],
        np.array(["ProteinBase"] * len(pb["y"])),
        OUT_DIR / "calibration_heldout_pb.png",
        "Held-out ProteinBase (train = K81 + V106)",
    )

    # ----------------------------------------------------------------- SHAP
    print("\n=== SHAP on full-data fit ===")
    shap_summary = run_shap(X_full, y, dg_stock, OUT_DIR, feat_names)
    for rk, (name, val) in enumerate(shap_summary["top10"], start=1):
        print(f"  {rk:2d}. {name:36s}  mean|SHAP|={val:.4f}")
    print("  top interactions:")
    for a, b, val in shap_summary["interactions"][:10]:
        print(f"    {a:32s}  ×  {b:32s}   |I|={val:.4f}")

    # -------------------------------------------------- predictions_grouped_cv.csv
    out_df = pd.DataFrame({
        "pdb_id": df["pdb_id"].values,
        "source": df["source"].values,
        "dg_exp": y,
        "dg_pred_stock": dg_stock,
        "residual_pred": cv["mean_residual"],
        "dg_pred_final": cv["mean_pred_xgb_final"],
    })
    out_df.to_csv(OUT_DIR / "predictions_grouped_cv.csv", index=False)

    # ---------------------------------------------------------------- report
    write_report(OUT_DIR, df, cv, loso, shap_summary, pb)

    # Save a compact JSON summary for downstream agents.
    summary = {
        "N": int(len(df)),
        "n_features": len(feat_names),
        "primary": {
            "xgb":   {"R": cv["xgb"]["R"],   "R_ci": cv["xgb"]["R_ci"],
                      "RMSE": cv["xgb"]["RMSE"], "MAE": cv["xgb"]["MAE"],
                      "rho": cv["xgb"]["rho"],
                      "R_rep_mean": cv["xgb"]["R_rep_mean"],
                      "R_rep_std":  cv["xgb"]["R_rep_std"]},
            "refit": {"R": cv["refit"]["R"], "R_ci": cv["refit"]["R_ci"],
                      "RMSE": cv["refit"]["RMSE"], "MAE": cv["refit"]["MAE"],
                      "rho": cv["refit"]["rho"]},
            "fixed": {"R": cv["fixed"]["R"], "R_ci": cv["fixed"]["R_ci"],
                      "RMSE": cv["fixed"]["RMSE"], "MAE": cv["fixed"]["MAE"],
                      "rho": cv["fixed"]["rho"]},
            "delta_R_vs_refit": cv["xgb"]["R"] - cv["refit"]["R"],
            "delta_R_vs_fixed": cv["xgb"]["R"] - cv["fixed"]["R"],
        },
        "loso": {k: {kk: v[kk] for kk in ("R", "R_ci", "RMSE", "MAE", "rho", "N")}
                 for k, v in loso.items()},
        "extrapolation_to_PB": {k: pb[k] for k in ("R", "R_ci", "RMSE", "MAE", "rho", "N")},
        "top5_shap": shap_summary["top5"],
        "top10_interactions": shap_summary["interactions"][:10],
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(f"\n[done] {OUT_DIR}")


if __name__ == "__main__":
    main()
