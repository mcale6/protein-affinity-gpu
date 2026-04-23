#!/usr/bin/env python3
"""PRODIGY 2015 + top-K CAD hybrid — conservative shippable model.

The stock PRODIGY 2015 linear combination of (ic_cc, ic_ca, ic_pp, ic_pa,
nis_a, nis_c) is held **fixed** (no refit).  A small additive correction is
learned from a tiny subset of CAD local-geometry features — specifically the
top-K that best explain the residual `dg_exp - dg_pred_stock` on the unified
K81+V106 training block.

Three hybrid sizes are compared (H1 / H3 / H5) against:
    * baseline-F  — stock FIXED on 6 features, no fitting
    * baseline-R  — stock REFIT CV on 6 features (7 params)
    * Full        — all top-15 CAD features stacked on stock (overfit bound)

Training / CV protocol:
    * feature ranking on K81+V106 only (PB has NaN atom/local CAD)
    * primary eval:  4-fold grouped-by-`pdb_id` × 10 repeats
    * secondary:     leave-one-source-out (K81, V106, PB)
    * secondary:     fit-on-K81+V106 → predict-PB extrapolation

Outputs (relative to repo root):
    benchmarks/output/unified/prodigy_cad_hybrid/
      report.md
      ranking_univariate.png
      calibration_H3_grouped_cv.png
      calibration_H3_heldout_pb.png
      model_comparison.png
      feature_ranking.csv
      hybrid_H3_coefs.csv
      predictions_H3_grouped_cv.csv

Usage:
    python benchmarks/scripts/pae_calibration/model_prodigy_cad_hybrid.py
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402

# The script lives at <repo>/benchmarks/scripts/pae_calibration/…; when run
# from a git worktree the unified CSV lives in the main repo, so we walk up
# looking for it and fall back to parents[3] if nothing is found.
def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    # Primary candidate: go 3 up from the file (scripts / pae_calibration / repo).
    primary = here.parents[3]
    candidates = [primary]
    # If we're inside a `.claude/worktrees/<name>` worktree, also try the main repo.
    parts = primary.parts
    if ".claude" in parts:
        idx = parts.index(".claude")
        candidates.append(Path(*parts[:idx]))
    for cand in candidates:
        if (cand / "benchmarks/output/unified/unified_features.csv").exists():
            return cand
    return primary  # last resort — will surface a clear error at load


ROOT = _find_repo_root()
CSV_PATH = ROOT / "benchmarks/output/unified/unified_features.csv"
# Output goes to the worktree if we're inside one; otherwise the main repo.
WORKTREE_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = WORKTREE_ROOT / "benchmarks/output/unified/prodigy_cad_hybrid"

# ---------------------------------------------------------------------------
# Constants — stock 2015 PRODIGY coefficients
# ---------------------------------------------------------------------------
STOCK_FEATURES = ("ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c")
COEFFS_STOCK = np.array(
    [-0.09459, -0.10007, 0.19577, -0.22671, 0.18681, 0.13810],
    dtype=np.float64,
)
INTERCEPT_STOCK = -15.9433

# Columns that are IDs, labels, duplicate target representations, or blobs —
# never candidates for CAD-correction features.
EXCLUDE_FROM_CANDIDATES = {
    "pdb_id", "source", "dg_exp_kcal_mol", "kd_m", "log10_kd",
    "stratum", "functional_class", "cad_arrays_jsonl",
    "dg_prodigy_boltz",  # all-null in unified CSV
    "nis_p",             # all-null in unified CSV (K81 + V106 coverage)
    *STOCK_FEATURES,
}

MIN_COVERAGE = 187  # need full coverage across K81+V106 for fair ranking

# Paper headline (stock FIXED on crystal, 4-fold CV × 10 repeats)
PAPER_REFERENCE_R = 0.73
PAPER_REFERENCE_RMSE = 1.89

# Hybrid sizes to fit
HYBRID_SIZES = (1, 3, 5)
FULL_SIZE = 15

N_BOOTSTRAP = 500
K_FOLDS = 4
N_REPEATS = 10


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_unified() -> pd.DataFrame:
    return pd.read_csv(CSV_PATH)


def stock_prediction(df: pd.DataFrame) -> np.ndarray:
    """Deterministic stock PRODIGY 2015 prediction, no refit."""
    X = df[list(STOCK_FEATURES)].to_numpy(dtype=np.float64, copy=True)
    # clip NIS% to valid [0, 100] — matches diagnostic_refit convention
    X[:, 4] = np.clip(X[:, 4], 0, 100)
    X[:, 5] = np.clip(X[:, 5], 0, 100)
    return X @ COEFFS_STOCK + INTERCEPT_STOCK


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def metrics(pred: np.ndarray, y: np.ndarray) -> dict:
    """Return Pearson R, Spearman rho, RMSE, MAE."""
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if np.std(pred) == 0 or np.std(y) == 0:
        return {"R": float("nan"), "rho": float("nan"),
                "RMSE": float("nan"), "MAE": float("nan")}
    r = float(pearsonr(pred, y)[0])
    rho = float(spearmanr(pred, y).correlation)
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    mae = float(np.mean(np.abs(pred - y)))
    return {"R": r, "rho": rho, "RMSE": rmse, "MAE": mae}


def bootstrap_R_CI(pred: np.ndarray, y: np.ndarray,
                   groups: np.ndarray | None = None,
                   n: int = N_BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    """Bootstrap 95 % CI for Pearson R.

    If ``groups`` is supplied we resample **complexes** (unique ``pdb_id``),
    otherwise we resample rows.
    """
    rng = np.random.default_rng(seed)
    pred = np.asarray(pred); y = np.asarray(y)
    if groups is None:
        N = len(y); rows = np.arange(N)
    else:
        groups = np.asarray(groups)
    vals = []
    if groups is None:
        for _ in range(n):
            idx = rng.choice(rows, size=rows.size, replace=True)
            if np.std(pred[idx]) == 0 or np.std(y[idx]) == 0:
                continue
            r = pearsonr(pred[idx], y[idx])[0]
            if np.isfinite(r):
                vals.append(r)
    else:
        unique_g = np.unique(groups)
        g_to_rows = {g: np.where(groups == g)[0] for g in unique_g}
        for _ in range(n):
            sampled = rng.choice(unique_g, size=unique_g.size, replace=True)
            idx = np.concatenate([g_to_rows[g] for g in sampled])
            if np.std(pred[idx]) == 0 or np.std(y[idx]) == 0:
                continue
            r = pearsonr(pred[idx], y[idx])[0]
            if np.isfinite(r):
                vals.append(r)
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


# ---------------------------------------------------------------------------
# Model fitting
# ---------------------------------------------------------------------------
def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Simple OLS with an explicit intercept column."""
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta[:-1], float(beta[-1])


def predict_hybrid(df_rows: pd.DataFrame,
                   coefs: np.ndarray, intercept: float,
                   cad_features: list[str]) -> np.ndarray:
    """stock + (CAD · coefs + intercept)."""
    dg_stock = stock_prediction(df_rows)
    if not cad_features:
        return dg_stock
    X_cad = df_rows[cad_features].to_numpy(dtype=np.float64, copy=True)
    return dg_stock + X_cad @ coefs + intercept


def standardise_train_apply(X_train: np.ndarray, X_apply: np.ndarray
                            ) -> tuple[np.ndarray, np.ndarray,
                                       np.ndarray, np.ndarray]:
    """Z-score each column using train mean/std, applied to both sets."""
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0, ddof=0)
    sd = np.where(sd > 0, sd, 1.0)
    return (X_train - mu) / sd, (X_apply - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# Feature ranking — univariate |R(feature, residual)| on K81 ∪ V106
# ---------------------------------------------------------------------------
def rank_features(df_train: pd.DataFrame) -> pd.DataFrame:
    dg_stock = stock_prediction(df_train)
    residual = df_train["dg_exp_kcal_mol"].to_numpy() - dg_stock

    candidates = [c for c in df_train.columns
                  if c not in EXCLUDE_FROM_CANDIDATES]
    records = []
    for col in candidates:
        s = df_train[col]
        try:
            s = pd.to_numeric(s, errors="raise")
        except Exception:
            continue
        mask = s.notna().to_numpy() & np.isfinite(s.to_numpy())
        n_nonnan = int(mask.sum())
        if n_nonnan < MIN_COVERAGE:
            continue
        x = s.to_numpy()[mask]
        r_res = residual[mask]
        if np.std(x) == 0:
            continue
        R = pearsonr(x, r_res)[0]
        # also report R vs raw target for context
        y = df_train["dg_exp_kcal_mol"].to_numpy()[mask]
        R_y = pearsonr(x, y)[0]
        records.append({
            "feature": col,
            "R_vs_residual": float(R),
            "abs_R_vs_residual": float(abs(R)),
            "R_vs_dg_exp": float(R_y),
            "n_non_nan": n_nonnan,
        })
    out = pd.DataFrame.from_records(records)
    out = out.sort_values("abs_R_vs_residual", ascending=False,
                          ignore_index=True)
    return out


# ---------------------------------------------------------------------------
# Grouped CV — fit CAD coefs on residual
# ---------------------------------------------------------------------------
def grouped_kfold_indices(groups: np.ndarray, k: int, seed: int
                          ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Yield (train_idx, test_idx) pairs for k-fold grouped on `groups`."""
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(set(groups.tolist())))
    rng.shuffle(unique)
    folds = np.array_split(unique, k)
    out = []
    for fi in range(k):
        test_groups = set(folds[fi].tolist())
        test_mask = np.array([g in test_groups for g in groups])
        train_idx = np.where(~test_mask)[0]
        test_idx = np.where(test_mask)[0]
        out.append((train_idx, test_idx))
    return out


def grouped_cv_predictions(df_train: pd.DataFrame,
                           cad_features: list[str],
                           k: int = K_FOLDS,
                           n_repeats: int = N_REPEATS,
                           seed: int = 0) -> dict:
    """4-fold × 10 repeats grouped-by-pdb_id CV for the additive CAD model.

    Returns mean predictions across repeats plus the per-repeat R.
    """
    N = len(df_train)
    y = df_train["dg_exp_kcal_mol"].to_numpy()
    groups = df_train["pdb_id"].to_numpy()
    dg_stock = stock_prediction(df_train)
    residual = y - dg_stock

    rng = np.random.default_rng(seed)
    repeat_seeds = rng.integers(0, 2 ** 31 - 1, size=n_repeats)

    # preds_sum / counts averaged across repeats -> one prediction per row
    preds_sum = np.zeros(N)
    preds_count = np.zeros(N, dtype=int)
    R_by_repeat = []
    coef_matrix = []  # (repeat*fold, len(cad_features))
    intercept_list = []

    if cad_features:
        X_cad_full = df_train[cad_features].to_numpy(dtype=np.float64,
                                                     copy=True)
    else:
        X_cad_full = np.zeros((N, 0))

    for rep_i, s in enumerate(repeat_seeds):
        folds = grouped_kfold_indices(groups, k, seed=int(s))
        preds_rep = np.full(N, np.nan)
        for train_idx, test_idx in folds:
            if cad_features:
                X_tr = X_cad_full[train_idx]
                X_te = X_cad_full[test_idx]
                X_tr_z, X_te_z, _, _ = standardise_train_apply(X_tr, X_te)
                coefs, icept = fit_ols(X_tr_z, residual[train_idx])
                preds_rep[test_idx] = (
                    dg_stock[test_idx] + X_te_z @ coefs + icept
                )
                coef_matrix.append(coefs)
                intercept_list.append(icept)
            else:
                preds_rep[test_idx] = dg_stock[test_idx]
        mask_ok = np.isfinite(preds_rep)
        if mask_ok.sum() >= 2 and np.std(preds_rep[mask_ok]) > 0:
            R_by_repeat.append(float(pearsonr(preds_rep[mask_ok],
                                              y[mask_ok])[0]))
        preds_sum += np.where(mask_ok, preds_rep, 0.0)
        preds_count += mask_ok.astype(int)

    preds_mean = preds_sum / np.where(preds_count > 0, preds_count, 1)
    ok = preds_count > 0
    R_by_repeat = np.asarray(R_by_repeat)
    return {
        "preds_mean": preds_mean,
        "ok_mask": ok,
        "R_by_repeat_mean": float(R_by_repeat.mean()) if len(R_by_repeat) else float("nan"),
        "R_by_repeat_std": float(R_by_repeat.std()) if len(R_by_repeat) else float("nan"),
        "coef_matrix": np.asarray(coef_matrix) if coef_matrix else np.zeros((0, len(cad_features))),
        "intercepts": np.asarray(intercept_list) if intercept_list else np.zeros((0,)),
    }


def baseline_F_grouped_cv(df_train: pd.DataFrame) -> dict:
    """Stock FIXED — no fitting, so no CV variance. Return mean preds."""
    y = df_train["dg_exp_kcal_mol"].to_numpy()
    dg_stock = stock_prediction(df_train)
    return {"preds_mean": dg_stock, "ok_mask": np.ones(len(y), dtype=bool),
            "R_by_repeat_mean": float("nan"),
            "R_by_repeat_std": float("nan")}


def baseline_R_grouped_cv(df_train: pd.DataFrame,
                          k: int = K_FOLDS,
                          n_repeats: int = N_REPEATS,
                          seed: int = 0) -> dict:
    """Full 7-parameter PRODIGY-style refit, grouped by pdb_id CV.

    Fits (6 IC/NIS coefs + intercept) on train split, predicts test split.
    """
    N = len(df_train)
    y = df_train["dg_exp_kcal_mol"].to_numpy()
    groups = df_train["pdb_id"].to_numpy()
    X = df_train[list(STOCK_FEATURES)].to_numpy(dtype=np.float64, copy=True)
    X[:, 4] = np.clip(X[:, 4], 0, 100)
    X[:, 5] = np.clip(X[:, 5], 0, 100)

    rng = np.random.default_rng(seed)
    repeat_seeds = rng.integers(0, 2 ** 31 - 1, size=n_repeats)
    preds_sum = np.zeros(N); preds_count = np.zeros(N, dtype=int)
    R_by_repeat = []
    for s in repeat_seeds:
        folds = grouped_kfold_indices(groups, k, seed=int(s))
        preds_rep = np.full(N, np.nan)
        for train_idx, test_idx in folds:
            coefs, icept = fit_ols(X[train_idx], y[train_idx])
            preds_rep[test_idx] = X[test_idx] @ coefs + icept
        mask_ok = np.isfinite(preds_rep)
        if mask_ok.sum() >= 2 and np.std(preds_rep[mask_ok]) > 0:
            R_by_repeat.append(float(pearsonr(preds_rep[mask_ok],
                                              y[mask_ok])[0]))
        preds_sum += np.where(mask_ok, preds_rep, 0.0)
        preds_count += mask_ok.astype(int)
    preds_mean = preds_sum / np.where(preds_count > 0, preds_count, 1)
    R_by_repeat = np.asarray(R_by_repeat)
    return {"preds_mean": preds_mean, "ok_mask": preds_count > 0,
            "R_by_repeat_mean": float(R_by_repeat.mean()) if len(R_by_repeat) else float("nan"),
            "R_by_repeat_std": float(R_by_repeat.std()) if len(R_by_repeat) else float("nan")}


# ---------------------------------------------------------------------------
# Leave-one-source-out
# ---------------------------------------------------------------------------
def loso_hybrid(df_all: pd.DataFrame, cad_features: list[str]) -> dict:
    """Leave-one-source-out for hybrid (CAD correction on stock).

    Returns per-source metrics.  A source is skipped if any selected CAD
    feature is missing for every row in that source.
    """
    y_all = df_all["dg_exp_kcal_mol"].to_numpy()
    dg_stock_all = stock_prediction(df_all)
    sources = df_all["source"].to_numpy()
    unique_sources = sorted(set(sources.tolist()))
    per = {}
    for held in unique_sources:
        test_mask = sources == held
        train_mask = ~test_mask
        # only train on rows with all CAD features available
        if cad_features:
            cad_ok_train = df_all.loc[train_mask, cad_features].notna().all(axis=1)
            cad_ok_test = df_all.loc[test_mask, cad_features].notna().all(axis=1)
        else:
            cad_ok_train = pd.Series(True, index=df_all.index[train_mask])
            cad_ok_test = pd.Series(True, index=df_all.index[test_mask])
        train_idx = df_all.index[train_mask][cad_ok_train.values]
        test_idx = df_all.index[test_mask][cad_ok_test.values]
        n_train = len(train_idx); n_test = len(test_idx)
        if n_test < 3 and cad_features:
            # CAD features aren't available on this block -> fall back to stock
            # so we still report a number instead of leaving a blank row.
            test_idx_raw = df_all.index[test_mask]
            y_test_raw = y_all[df_all.index.get_indexer(test_idx_raw)]
            pred_stock = dg_stock_all[df_all.index.get_indexer(test_idx_raw)]
            m = metrics(pred_stock, y_test_raw)
            lo, hi = bootstrap_R_CI(
                pred_stock, y_test_raw,
                groups=df_all.loc[test_idx_raw, "pdb_id"].to_numpy(),
            )
            per[held] = {
                "n_train": int(n_train),
                "n_test": int(len(test_idx_raw)),
                **m, "R_CI_lo": lo, "R_CI_hi": hi,
                "note": ("CAD features NaN on held-out block — hybrid "
                         "reduces to stock PRODIGY."),
            }
            continue
        if n_train < max(10, len(cad_features) + 2) or n_test < 3:
            per[held] = {"note": f"n_train={n_train}, n_test={n_test} — skipped",
                         "n_train": n_train, "n_test": n_test}
            continue
        y_train = y_all[df_all.index.get_indexer(train_idx)]
        y_test = y_all[df_all.index.get_indexer(test_idx)]
        dg_s_train = dg_stock_all[df_all.index.get_indexer(train_idx)]
        dg_s_test = dg_stock_all[df_all.index.get_indexer(test_idx)]
        resid_train = y_train - dg_s_train
        if cad_features:
            X_tr = df_all.loc[train_idx, cad_features].to_numpy(dtype=np.float64)
            X_te = df_all.loc[test_idx, cad_features].to_numpy(dtype=np.float64)
            X_tr_z, X_te_z, _, _ = standardise_train_apply(X_tr, X_te)
            coefs, icept = fit_ols(X_tr_z, resid_train)
            pred = dg_s_test + X_te_z @ coefs + icept
        else:
            pred = dg_s_test
        m = metrics(pred, y_test)
        lo, hi = bootstrap_R_CI(pred, y_test,
                                 groups=df_all.loc[test_idx, "pdb_id"].to_numpy())
        per[held] = {"n_train": int(n_train), "n_test": int(n_test),
                     **m, "R_CI_lo": lo, "R_CI_hi": hi}
    return per


# ---------------------------------------------------------------------------
# Extrapolation: fit K81+V106, predict PB
# ---------------------------------------------------------------------------
def extrapolate_to_pb(df_all: pd.DataFrame, cad_features: list[str]) -> dict:
    """Fit the additive CAD model on K81+V106, evaluate on PB.

    If any selected CAD feature is entirely NaN on PB the hybrid prediction
    reduces to the plain stock PRODIGY baseline on that subset (we report
    both "stock on PB" and "hybrid on PB" and flag the missing features).
    """
    train = df_all[df_all["source"].isin(["Kastritis81", "VrevenBM5.5"])]
    test = df_all[df_all["source"] == "ProteinBase"].copy()

    # availability per feature on PB
    feature_availability = {c: int(test[c].notna().sum()) if c in test.columns else 0
                             for c in cad_features}

    # stock prediction on PB — always available
    dg_stock_pb = stock_prediction(test)
    stock_m = metrics(dg_stock_pb, test["dg_exp_kcal_mol"].to_numpy())
    stock_ci = bootstrap_R_CI(dg_stock_pb,
                              test["dg_exp_kcal_mol"].to_numpy(),
                              groups=test["pdb_id"].to_numpy())

    missing = [c for c in cad_features if feature_availability[c] == 0]
    if cad_features and missing:
        # cannot apply CAD correction -> hybrid = stock on PB
        return {
            "n_train": len(train),
            "n_test": len(test),
            "n_test_after_cad_mask": 0,
            "feature_availability": feature_availability,
            "missing_on_pb": missing,
            "stock_only": {**stock_m,
                           "R_CI_lo": stock_ci[0], "R_CI_hi": stock_ci[1]},
            "hybrid": None,
            "note": ("selected CAD features absent from ProteinBase "
                     "(local/atom-CAD is NaN on single-chain PB entries); "
                     "hybrid reduces to stock PRODIGY on this block"),
        }

    # Full hybrid path — available
    # drop rows with any NaN in selected CAD features
    ok = test[cad_features].notna().all(axis=1) if cad_features else pd.Series(True, index=test.index)
    test_ok = test[ok]
    y_te = test_ok["dg_exp_kcal_mol"].to_numpy()
    dg_s_te = stock_prediction(test_ok)

    y_tr = train["dg_exp_kcal_mol"].to_numpy()
    dg_s_tr = stock_prediction(train)
    resid_tr = y_tr - dg_s_tr
    X_tr = train[cad_features].to_numpy(dtype=np.float64, copy=True) if cad_features else np.zeros((len(train), 0))
    X_te = test_ok[cad_features].to_numpy(dtype=np.float64, copy=True) if cad_features else np.zeros((len(test_ok), 0))
    if cad_features:
        X_tr_z, X_te_z, _, _ = standardise_train_apply(X_tr, X_te)
        coefs, icept = fit_ols(X_tr_z, resid_tr)
        pred = dg_s_te + X_te_z @ coefs + icept
    else:
        pred = dg_s_te
    m = metrics(pred, y_te)
    ci_lo, ci_hi = bootstrap_R_CI(pred, y_te,
                                  groups=test_ok["pdb_id"].to_numpy())
    return {
        "n_train": len(train),
        "n_test": len(test),
        "n_test_after_cad_mask": len(test_ok),
        "feature_availability": feature_availability,
        "missing_on_pb": missing,
        "stock_only": {**stock_m,
                       "R_CI_lo": stock_ci[0], "R_CI_hi": stock_ci[1]},
        "hybrid": {**m, "R_CI_lo": ci_lo, "R_CI_hi": ci_hi},
        "note": "",
    }


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_model(df_train: pd.DataFrame, df_all: pd.DataFrame,
              label: str, cad_features: list[str],
              is_refit_baseline: bool = False) -> dict:
    """Primary grouped-CV evaluation + metrics.

    When ``is_refit_baseline`` is True we run the PRODIGY-6-feature refit
    instead of the stock+CAD hybrid.
    """
    if is_refit_baseline:
        cv = baseline_R_grouped_cv(df_train)
    elif label == "baseline-F":
        cv = baseline_F_grouped_cv(df_train)
    else:
        cv = grouped_cv_predictions(df_train, cad_features)
    preds = cv["preds_mean"]
    y = df_train["dg_exp_kcal_mol"].to_numpy()
    groups = df_train["pdb_id"].to_numpy()
    mask = cv["ok_mask"]
    m = metrics(preds[mask], y[mask])
    lo, hi = bootstrap_R_CI(preds[mask], y[mask], groups=groups[mask])
    return {
        "label": label,
        "cad_features": cad_features,
        "R": m["R"], "rho": m["rho"], "RMSE": m["RMSE"], "MAE": m["MAE"],
        "R_CI_lo": lo, "R_CI_hi": hi,
        "R_by_repeat_mean": cv.get("R_by_repeat_mean"),
        "R_by_repeat_std": cv.get("R_by_repeat_std"),
        "preds_mean": preds,
        "ok_mask": mask,
        "cv_coef_matrix": cv.get("coef_matrix"),
        "cv_intercepts": cv.get("intercepts"),
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
SOURCE_COLORS = {
    "Kastritis81": "#1f77b4",
    "VrevenBM5.5": "#d62728",
    "ProteinBase": "#2ca02c",
}


def plot_ranking(ranking: pd.DataFrame, out_path: Path, top_n: int = 15):
    top = ranking.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(top["feature"], top["abs_R_vs_residual"],
                   color="#4c72b0")
    for bar, r in zip(bars, top["R_vs_residual"]):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{r:+.2f}", va="center", fontsize=8)
    ax.set_xlabel("|Pearson R(feature, residual)|")
    ax.set_title(f"Top {top_n} CAD features — univariate association with\n"
                 f"(dg_exp - dg_pred_stock) on K81+V106  (n={MIN_COVERAGE})")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_calibration(pred: np.ndarray, y: np.ndarray,
                     sources: np.ndarray, title: str,
                     out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    for src, color in SOURCE_COLORS.items():
        m = sources == src
        if not m.any():
            continue
        ax.scatter(y[m], pred[m], alpha=0.7, s=24, color=color,
                   label=f"{src} (n={m.sum()})")
    lo = min(y.min(), pred.min()) - 1
    hi = max(y.max(), pred.max()) + 1
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    r_tot, rho_tot, rmse_tot, mae_tot = (
        metrics(pred, y)["R"], metrics(pred, y)["rho"],
        metrics(pred, y)["RMSE"], metrics(pred, y)["MAE"],
    )
    ax.set_title(f"{title}\nR={r_tot:.3f}  ρ={rho_tot:.3f}  "
                 f"RMSE={rmse_tot:.2f}  MAE={mae_tot:.2f}  n={len(y)}")
    ax.set_xlabel("ΔG_exp  (kcal/mol)")
    ax.set_ylabel("ΔG_pred  (kcal/mol)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_model_comparison(rows: list[dict], out_path: Path):
    """Bars for R (with CI) and RMSE vs model size (in # fitted params)."""
    labels = [r["label"] for r in rows]
    R = np.array([r["R"] for r in rows])
    R_lo = np.array([r["R_CI_lo"] for r in rows])
    R_hi = np.array([r["R_CI_hi"] for r in rows])
    RMSE = np.array([r["RMSE"] for r in rows])
    x = np.arange(len(labels))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    # R
    axes[0].bar(x, R, color="#4c72b0")
    yerr_lo = np.maximum(R - R_lo, 0); yerr_hi = np.maximum(R_hi - R, 0)
    axes[0].errorbar(x, R, yerr=[yerr_lo, yerr_hi], fmt="none",
                     ecolor="k", lw=1)
    for xi, val in zip(x, R):
        axes[0].text(xi, val + 0.01, f"{val:.3f}", ha="center", fontsize=9)
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, rotation=20)
    axes[0].set_ylabel("Pearson R  (grouped 4-fold × 10)")
    axes[0].set_title("Correlation with ΔG_exp (bootstrap 95 % CI, K81+V106 N=187)")
    axes[0].grid(axis="y", alpha=0.3)
    # RMSE
    axes[1].bar(x, RMSE, color="#c44e52")
    for xi, val in zip(x, RMSE):
        axes[1].text(xi, val + 0.02, f"{val:.2f}", ha="center", fontsize=9)
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, rotation=20)
    axes[1].set_ylabel("RMSE  (kcal/mol)")
    axes[1].set_title("RMSE")
    axes[1].grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def verdict_label(delta_R_vs_refit: float) -> str:
    if delta_R_vs_refit >= 0.05:
        return "**HELPS**"
    if delta_R_vs_refit <= -0.02:
        return "**NO-HELP**"
    return "**MARGINAL**"


def format_ci(v: float, lo: float, hi: float) -> str:
    if not np.isfinite(lo) or not np.isfinite(hi):
        return f"{v:+.3f}"
    return f"{v:+.3f} [{lo:+.3f}, {hi:+.3f}]"


def build_report(ranking: pd.DataFrame,
                 model_rows: list[dict],
                 h3_loso: dict,
                 h3_extrapolate: dict,
                 baseline_F: dict, baseline_R: dict,
                 out_path: Path):
    # verdict logic — H3 compared to stock REFIT CV
    R_baseline_R = baseline_R["R"]
    # find H3 row
    h3 = next(r for r in model_rows if r["label"] == "H3")
    delta = h3["R"] - R_baseline_R
    verdict = verdict_label(delta)
    h3_feats = ", ".join(f"`{f}`" for f in h3["cad_features"])

    lines = []
    lines += [
        f"# PRODIGY 2015 + top-K CAD hybrid — unified N=287",
        "",
        f"**Verdict**: PRODIGY + top-K CAD hybrid on unified N=287: {verdict}",
        "",
        f"- H3 grouped-CV R = {format_ci(h3['R'], h3['R_CI_lo'], h3['R_CI_hi'])}  "
        f"(ΔR vs stock-REFIT-CV = {delta:+.3f}, threshold for HELPS = +0.05)",
        f"- H3 selected CAD features: {h3_feats}",
        "",
        "## Protocol",
        "",
        "- Feature ranking on K81 ∪ V106 only (N = 187) — ProteinBase has NaN "
        "on atom/local CAD.",
        "- Each CAD feature must have ≥ 187 non-NaN values (full K81+V106 "
        "coverage) to be eligible.",
        "- Stock PRODIGY 2015 linear combination is held **fixed** — "
        "intercept = –15.9433 and coefficients for (ic_cc, ic_ca, ic_pp, "
        "ic_pa, nis_a, nis_c) are the published crystal values.",
        "- Hybrid models add an OLS correction on z-scored CAD features "
        "against the residual `dg_exp - dg_pred_stock`.",
        f"- Primary evaluation: {K_FOLDS}-fold grouped-by-`pdb_id` × "
        f"{N_REPEATS} repeats (mean-over-repeats prediction, per-complex "
        "bootstrap 95 % CI over {N_BOOTSTRAP} resamples of pdb_id).".format(
            N_BOOTSTRAP=N_BOOTSTRAP),
        "- Group CV is mandatory — K81 and V106 share 64 pdb_ids so random "
        "folds leak.",
        "",
    ]

    # Top-15 ranking table
    lines += [
        "## Top-15 CAD features by |R(feature, dg_exp - dg_pred_stock)|  "
        "(K81+V106, N=187)",
        "",
        "| rank | feature | R(feat, residual) | R(feat, ΔG_exp) | n non-NaN |",
        "|---:|---|---:|---:|---:|",
    ]
    for i, row in ranking.head(15).iterrows():
        lines.append(f"| {i + 1} | `{row['feature']}` | "
                     f"{row['R_vs_residual']:+.3f} | "
                     f"{row['R_vs_dg_exp']:+.3f} | "
                     f"{row['n_non_nan']} |")
    lines.append("")

    # Model-comparison table
    lines += [
        "## Model comparison — grouped-CV on K81+V106  (N = 187)",
        "",
        "| model | fitted params | CAD features | R (95 % CI) | ρ | RMSE | "
        "MAE | ΔR vs baseline-F |",
        "|---|---:|---|---:|---:|---:|---:|---:|",
    ]
    F_R = baseline_F["R"]
    for r in model_rows:
        label = r["label"]
        nfeat = len(r["cad_features"])
        feats_show = (", ".join(f"`{f}`" for f in r["cad_features"])
                      if r["cad_features"] else "—")
        # fitted params count
        if label == "baseline-F":
            n_params = 0
        elif label == "baseline-R":
            n_params = 7  # 6 coefs + intercept
        else:
            # hybrid: len(cad) coefs + 1 intercept on residual
            n_params = nfeat + 1
        lines.append(
            f"| {label} | {n_params} | {feats_show} | "
            f"{format_ci(r['R'], r['R_CI_lo'], r['R_CI_hi'])} | "
            f"{r['rho']:+.3f} | {r['RMSE']:.2f} | {r['MAE']:.2f} | "
            f"{r['R'] - F_R:+.3f} |"
        )
    lines.append("")

    # H3 pick
    lines += [
        "## H3 selected CAD features (the shippable recommendation)",
        "",
        f"`{', '.join(h3['cad_features'])}`",
        "",
        "H3 coefficients (after z-scoring CAD features) from grouped-CV, "
        "mean over all 40 fold-fits:",
        "",
        "| feature | mean coef | std coef | sign stability |",
        "|---|---:|---:|---:|",
    ]
    # stability: fraction of folds sharing sign of mean
    if h3.get("cv_coef_matrix") is not None and len(h3["cv_coef_matrix"]):
        C = h3["cv_coef_matrix"]
        for j, feat in enumerate(h3["cad_features"]):
            col = C[:, j]
            mean_c = float(col.mean())
            std_c = float(col.std())
            sign_frac = float((np.sign(col) == np.sign(mean_c)).mean())
            lines.append(f"| `{feat}` | {mean_c:+.3f} | {std_c:.3f} | "
                         f"{sign_frac:.2%} |")
    lines.append("")

    # Leave-one-source-out for H3
    lines += [
        "## Leave-one-source-out — H3",
        "",
        "| held-out source | n_train | n_test | R (95 % CI) | ρ | RMSE | MAE | note |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for src, info in h3_loso.items():
        if "R" not in info:
            lines.append(
                f"| {src} | {info['n_train']} | {info['n_test']} | "
                f"{info.get('note', '—')} | — | — | — | — |"
            )
            continue
        note = info.get("note", "")
        lines.append(
            f"| {src} | {info['n_train']} | {info['n_test']} | "
            f"{format_ci(info['R'], info['R_CI_lo'], info['R_CI_hi'])} | "
            f"{info['rho']:+.3f} | {info['RMSE']:.2f} | {info['MAE']:.2f} | "
            f"{note} |"
        )
    lines.append("")

    # Extrapolation to PB
    lines += [
        "## Extrapolation to ProteinBase — fit on K81+V106, predict PB",
        "",
    ]
    ex = h3_extrapolate
    if ex["missing_on_pb"]:
        lines += [
            f"- Selected CAD features unavailable on PB: "
            f"`{', '.join(ex['missing_on_pb'])}`",
            f"- Note: {ex['note']}",
            "",
            "Stock PRODIGY on PB (no CAD correction possible):",
            "",
            f"  R = {format_ci(ex['stock_only']['R'], ex['stock_only']['R_CI_lo'], ex['stock_only']['R_CI_hi'])} "
            f"| ρ = {ex['stock_only']['rho']:+.3f} | RMSE = "
            f"{ex['stock_only']['RMSE']:.2f} | MAE = "
            f"{ex['stock_only']['MAE']:.2f} | N = {ex['n_test']}",
            "",
        ]
    else:
        h = ex["hybrid"]; s = ex["stock_only"]
        lines += [
            f"- PB N = {ex['n_test']}  (usable after CAD-mask: {ex['n_test_after_cad_mask']})",
            "",
            "| model | R (95 % CI) | ρ | RMSE | MAE |",
            "|---|---:|---:|---:|---:|",
            f"| stock (no CAD) | {format_ci(s['R'], s['R_CI_lo'], s['R_CI_hi'])} "
            f"| {s['rho']:+.3f} | {s['RMSE']:.2f} | {s['MAE']:.2f} |",
            f"| **H3 hybrid** | {format_ci(h['R'], h['R_CI_lo'], h['R_CI_hi'])} "
            f"| {h['rho']:+.3f} | {h['RMSE']:.2f} | {h['MAE']:.2f} |",
            "",
        ]

    # PRODIGY paper reference
    lines += [
        "## Comparison to PRODIGY 2015 paper",
        "",
        f"- Paper: 4-fold CV × 10 repeats on **crystal** structures — R = "
        f"{PAPER_REFERENCE_R:.2f}, RMSE = {PAPER_REFERENCE_RMSE:.2f} kcal/mol.",
        f"- Our stock FIXED on Boltz predictions, grouped-CV K81+V106: R = "
        f"{format_ci(baseline_F['R'], baseline_F['R_CI_lo'], baseline_F['R_CI_hi'])}, "
        f"RMSE = {baseline_F['RMSE']:.2f}.",
        f"- Our stock REFIT-CV 7-param, grouped-CV K81+V106: R = "
        f"{format_ci(baseline_R['R'], baseline_R['R_CI_lo'], baseline_R['R_CI_hi'])}, "
        f"RMSE = {baseline_R['RMSE']:.2f}.",
        "",
        "## Sparsity check: does H3 beat H5?",
        "",
    ]
    h5 = next(r for r in model_rows if r["label"] == "H5")
    h1 = next(r for r in model_rows if r["label"] == "H1")
    d_h3_vs_h5 = h3["R"] - h5["R"]
    if h3["R"] >= h5["R"]:
        sparse_msg = ("**YES** — H3 meets or exceeds H5 grouped-CV R "
                      "→ CAD residual signal is sparse (3 features suffice; "
                      "more overfits).")
    else:
        sparse_msg = ("**NO** — H5 beats H3 by ΔR={:+.3f} → CAD signal "
                      "benefits from additional features; H3 may be too "
                      "conservative.".format(-d_h3_vs_h5))
    lines += [
        f"- H1 R = {h1['R']:+.3f},  H3 R = {h3['R']:+.3f},  H5 R = {h5['R']:+.3f}",
        f"- ΔR(H3 − H5) = {d_h3_vs_h5:+.3f}",
        f"- {sparse_msg}",
        "",
    ]

    out_path.write_text("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_unified()
    print(f"[data] unified N = {len(df)}  ({dict(df['source'].value_counts())})")

    train_mask = df["source"].isin(["Kastritis81", "VrevenBM5.5"])
    df_train = df[train_mask].reset_index(drop=True)
    df_all = df.reset_index(drop=True)
    print(f"[data] train (K81+V106) N = {len(df_train)}")

    # ---------- Step 1: ranking ----------
    print("[rank] ranking features by |R(feat, residual)| ...")
    ranking = rank_features(df_train)
    ranking.to_csv(out_dir / "feature_ranking.csv", index=False)
    top15 = ranking.head(15)["feature"].tolist()
    print(f"[rank] top-15 features: {top15}")
    plot_ranking(ranking, out_dir / "ranking_univariate.png")

    # ---------- Step 2: hybrid models ----------
    print("[fit] grouped-CV baselines and hybrids (this is quick) ...")
    baseline_F = run_model(df_train, df_all,
                            label="baseline-F", cad_features=[])
    baseline_R = run_model(df_train, df_all,
                            label="baseline-R", cad_features=[],
                            is_refit_baseline=True)
    hybrids = []
    for k in HYBRID_SIZES:
        feats = top15[:k]
        hybrids.append(run_model(df_train, df_all,
                                 label=f"H{k}", cad_features=feats))
    full = run_model(df_train, df_all,
                      label="Full", cad_features=top15[:FULL_SIZE])

    model_rows = [baseline_F, baseline_R] + hybrids + [full]

    for r in model_rows:
        print(f"  {r['label']:12s}  "
              f"R={r['R']:+.3f} [{r['R_CI_lo']:+.3f}, {r['R_CI_hi']:+.3f}]  "
              f"ρ={r['rho']:+.3f}  "
              f"RMSE={r['RMSE']:.2f}  MAE={r['MAE']:.2f}  "
              f"nfeat={len(r['cad_features'])}")

    # ---------- Step 3: LOSO for H3 ----------
    h3 = next(r for r in model_rows if r["label"] == "H3")
    print("[loso] leave-one-source-out for H3 ...")
    h3_loso = loso_hybrid(df_all, h3["cad_features"])
    for src, info in h3_loso.items():
        if "R" in info:
            print(f"  {src:15s} n_train={info['n_train']} n_test={info['n_test']} "
                  f"R={info['R']:+.3f}  RMSE={info['RMSE']:.2f}")
        else:
            print(f"  {src:15s} {info}")

    # ---------- Step 4: extrapolate to PB ----------
    print("[extrapolate] fit K81+V106, predict PB ...")
    h3_extrapolate = extrapolate_to_pb(df_all, h3["cad_features"])
    if h3_extrapolate["missing_on_pb"]:
        print(f"  missing on PB: {h3_extrapolate['missing_on_pb']}")
        s = h3_extrapolate["stock_only"]
        print(f"  stock on PB:  R={s['R']:+.3f}  RMSE={s['RMSE']:.2f}  "
              f"N={h3_extrapolate['n_test']}")
    else:
        h = h3_extrapolate["hybrid"]
        print(f"  H3 on PB:  R={h['R']:+.3f}  RMSE={h['RMSE']:.2f}  "
              f"N={h3_extrapolate['n_test_after_cad_mask']}")

    # ---------- Plots ----------
    sources_train = df_train["source"].to_numpy()
    plot_calibration(h3["preds_mean"][h3["ok_mask"]],
                     df_train["dg_exp_kcal_mol"].to_numpy()[h3["ok_mask"]],
                     sources_train[h3["ok_mask"]],
                     "H3 hybrid — grouped-CV (K81+V106)",
                     out_dir / "calibration_H3_grouped_cv.png")

    df_pb = df[df["source"] == "ProteinBase"].reset_index(drop=True)
    if h3_extrapolate["missing_on_pb"]:
        # Fallback: plot stock only for PB
        pred_pb = stock_prediction(df_pb)
        y_pb = df_pb["dg_exp_kcal_mol"].to_numpy()
        plot_calibration(pred_pb, y_pb,
                          df_pb["source"].to_numpy(),
                          "PB extrapolation — stock-only "
                          "(H3 CAD features NaN on PB)",
                          out_dir / "calibration_H3_heldout_pb.png")
    else:
        # Fit on train, apply to PB
        train = df_all[df_all["source"].isin(["Kastritis81", "VrevenBM5.5"])]
        dg_s_tr = stock_prediction(train)
        y_tr = train["dg_exp_kcal_mol"].to_numpy()
        resid_tr = y_tr - dg_s_tr
        X_tr = train[h3["cad_features"]].to_numpy(dtype=np.float64)
        ok_pb = df_pb[h3["cad_features"]].notna().all(axis=1)
        X_te = df_pb.loc[ok_pb, h3["cad_features"]].to_numpy(dtype=np.float64)
        X_tr_z, X_te_z, _, _ = standardise_train_apply(X_tr, X_te)
        coefs, icept = fit_ols(X_tr_z, resid_tr)
        dg_s_te = stock_prediction(df_pb.loc[ok_pb])
        pred_pb = dg_s_te + X_te_z @ coefs + icept
        y_pb = df_pb.loc[ok_pb, "dg_exp_kcal_mol"].to_numpy()
        plot_calibration(pred_pb, y_pb,
                          df_pb.loc[ok_pb, "source"].to_numpy(),
                          "H3 hybrid — held-out PB (extrapolation)",
                          out_dir / "calibration_H3_heldout_pb.png")

    plot_model_comparison(model_rows, out_dir / "model_comparison.png")

    # ---------- CSV outputs ----------
    # hybrid_H3_coefs.csv — mean coef over 40 CV fits, z-scored feature space
    C = h3.get("cv_coef_matrix")
    icpts = h3.get("cv_intercepts")
    rows_h3 = []
    if C is not None and len(C):
        for j, feat in enumerate(h3["cad_features"]):
            col = C[:, j]
            rows_h3.append({
                "feature": feat,
                "coef_mean_z": float(col.mean()),
                "coef_std_z": float(col.std()),
                "sign_stability": float(
                    (np.sign(col) == np.sign(col.mean())).mean()),
            })
        rows_h3.append({
            "feature": "(intercept_on_residual)",
            "coef_mean_z": float(icpts.mean()),
            "coef_std_z": float(icpts.std()),
            "sign_stability": float(
                (np.sign(icpts) == np.sign(icpts.mean())).mean()),
        })
    pd.DataFrame(rows_h3).to_csv(out_dir / "hybrid_H3_coefs.csv", index=False)

    # predictions_H3_grouped_cv.csv — per-complex
    pred_df = pd.DataFrame({
        "pdb_id": df_train["pdb_id"],
        "source": df_train["source"],
        "dg_exp_kcal_mol": df_train["dg_exp_kcal_mol"],
        "dg_pred_stock": stock_prediction(df_train),
        "dg_pred_H3": h3["preds_mean"],
        "ok_H3": h3["ok_mask"],
    })
    pred_df.to_csv(out_dir / "predictions_H3_grouped_cv.csv", index=False)

    # ---------- Report ----------
    build_report(ranking, model_rows, h3_loso, h3_extrapolate,
                 baseline_F, baseline_R,
                 out_dir / "report.md")

    print(f"[done] wrote {out_dir}")


if __name__ == "__main__":
    main()
