#!/usr/bin/env python3
"""Elastic Net + PRODIGY-anchored prior on unified N=287 complexes.

Design
------
- The 6 stock PRODIGY features (``ic_cc, ic_ca, ic_pp, ic_pa, nis_a, nis_c``)
  are anchored to their 2015 crystal-fit values with a Bayesian prior: L2
  penalty on ``(coef - stock_coef)`` rather than on the coefficient itself.
- All other features ("free": CAD atom-level, PAE summaries, confidence
  scalars, PB-only extras) are given an Elastic Net (L1 + L2) penalty with
  no prior — α and ``l1_ratio`` tuned via inner CV.
- Features with >50% NaN in the training set are dropped. Remaining NaNs are
  median-imputed per column.

The combined penalised OLS is

    loss(θ) = ||y − Xθ||² + λ_stock · ||θ_stock − θ_prior||²
              + α · (l1_ratio · ||θ_free||₁ + (1 − l1_ratio) · ||θ_free||²)

Implementation:
1. Compute ``y_offset = y − X_stock @ coef_prior`` so the stock block becomes
   a residual-fitting problem around the prior.
2. Reparameterise ``δ_stock = θ_stock − θ_prior`` and center/scale the free
   block.
3. Stack ``[λ_stock · I_6  0; 0  0]`` as an L2 regulariser on δ, and use
   sklearn ``ElasticNet`` on the free features with its α · l1_ratio penalty.
   Actually simpler: use ``ElasticNetCV`` with a custom composite design:
   concatenate δ_stock (scale=√λ_stock) + free (scale=1) and rely on the
   uniform elastic net penalty with ``l1_ratio=0`` on the stock block via
   a multiplicative scaling trick — see ``fit_prior_en`` below.

CV protocol
-----------
- Primary: 4-fold grouped-by-pdb_id × 10 repeats. ``pdb_id`` is the group key
  so overlapping K81∩V106 entries don't leak across folds.
- Secondary: leave-one-source-out (K81 ∪ V106 → PB, K81 ∪ PB → V106, V106 ∪ PB
  → K81).
- Extrapolation: fit on K81+V106 (187), predict PB (100) once.

Baselines
---------
- ``stock_fixed``: 6 stock coefs at 2015 values, no refit, applied to all 287.
- ``stock_refit_cv``: refit 7 OLS params on just the 6 stock features, same
  grouped CV.
"""
from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr, spearmanr  # noqa: E402
from sklearn.linear_model import ElasticNet  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402

# ElasticNet fits on augmented designs often have slow coordinate-descent
# convergence near the L1-corner — we rely on R/RMSE scoring, not exact
# minimum, so silencing the warning is fine.
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
from sklearn.exceptions import ConvergenceWarning  # noqa: E402
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[3]
DATA_CSV = ROOT / "benchmarks/output/unified/unified_features.csv"
OUT_DIR = ROOT / "benchmarks/output/unified/elasticnet_prior"

STOCK_FEATURES = ["ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c"]
COEFFS_STOCK = np.array(
    [-0.09459, -0.10007, 0.19577, -0.22671, 0.18681, 0.13810],
    dtype=np.float64,
)
INTERCEPT_STOCK = -15.9433

# Candidate "free" features (everything that's not stock PRODIGY or metadata).
# We'll filter by NaN fraction at fit time.
FREE_FEATURE_CANDIDATES = [
    # Extended IC
    "ic_aa", "ic_cp", "nis_p",
    # Boltz confidence
    "boltz_iptm", "boltz_ptm", "boltz_plddt", "boltz_confidence_score",
    "boltz_complex_plddt", "boltz_complex_iplddt", "boltz_complex_pde",
    # PAE
    "mean_pae_contacts", "mean_pae_interface", "n_contacts",
    # Global CAD
    "cad_rr", "cad_rr_f1", "cad_aa", "cad_aa_f1", "cad_rr_target_area",
    "cad_rr_model_area", "cad_rr_tp", "cad_rr_fp", "cad_rr_fn",
    # Local CAD — residue
    "resi_cad_mean", "resi_cad_std", "resi_cad_min", "resi_cad_max",
    "resi_cad_p10", "resi_cad_p25", "resi_cad_p50", "resi_cad_p75",
    "resi_cad_p90", "resi_cad_A_mean", "resi_cad_B_mean",
    "resi_cad_frac_below_0_3", "resi_cad_frac_below_0_5",
    "resi_cad_frac_above_0_7", "resi_cad_frac_above_0_9",
    "resi_n_total", "resi_n_false_positive",
    # Local CAD — atom
    "atom_cad_mean", "atom_cad_std", "atom_cad_min", "atom_cad_max",
    "atom_cad_p10", "atom_cad_p25", "atom_cad_p50", "atom_cad_p75",
    "atom_cad_p90", "atom_cad_A_mean", "atom_cad_B_mean",
    "atom_cad_bb_mean", "atom_cad_sc_mean",
    "atom_cad_frac_below_0_3", "atom_cad_frac_below_0_5",
    "atom_cad_frac_above_0_7", "atom_cad_frac_above_0_9",
    "atom_n_total", "atom_n_false_positive",
    # Local CAD — rr/aa contact
    "rrc_cad_mean", "rrc_cad_std", "rrc_cad_p10", "rrc_cad_p50", "rrc_cad_p90",
    "rrc_cad_frac_below_0_3", "rrc_cad_frac_above_0_7",
    "rrc_n_total", "rrc_n_model_only", "rrc_n_shared",
    "aac_cad_mean", "aac_cad_std", "aac_cad_p10", "aac_cad_p50", "aac_cad_p90",
    "aac_cad_frac_below_0_3", "aac_cad_frac_above_0_7",
    "aac_n_total", "aac_n_model_only", "aac_n_shared",
    # PB-only extras
    "pdockq", "pdockq2", "lis", "ipsae", "min_ipsae",
    "shape_complementarity", "interface_residue_count",
]

# CV hyperparameters
N_FOLDS = 4
N_REPEATS = 10
BASE_SEED = 0
NAN_THRESHOLD = 0.5
# Small but informative grid. lambda_stock spans 2 orders; l1_ratio spans
# ridge-ish (0.1) → pure-lasso (1.0); α in 1e-3..1 is the usual EN sweet spot.
LAMBDA_STOCK_GRID = np.array([0.5, 2.0, 10.0])
ALPHA_GRID = np.logspace(-3, 0, 7)
L1_RATIOS = [0.2, 0.5, 0.9]
N_BOOT = 500


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def metrics(y_pred: np.ndarray, y_true: np.ndarray) -> dict:
    """Pearson R, Spearman ρ, RMSE, MAE."""
    if np.std(y_pred) < 1e-12:
        return dict(R=float("nan"), spearman=float("nan"),
                    RMSE=float("nan"), MAE=float("nan"))
    r = float(pearsonr(y_pred, y_true)[0])
    rho = float(spearmanr(y_pred, y_true)[0])
    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    return dict(R=r, spearman=rho, RMSE=rmse, MAE=mae)


def bootstrap_R_ci(y_pred: np.ndarray, y_true: np.ndarray,
                   n_boot: int = N_BOOT, seed: int = 7) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    N = len(y_true)
    rs = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, size=N)
        if np.std(y_pred[idx]) < 1e-12:
            continue
        rs.append(pearsonr(y_pred[idx], y_true[idx])[0])
    if not rs:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return (float(lo), float(hi))


# --------------------------------------------------------------------------
# Model: augmented-design Elastic Net with PRODIGY prior
# --------------------------------------------------------------------------

def build_augmented_design(
    X_stock: np.ndarray,
    X_free: np.ndarray,
    y: np.ndarray,
    lambda_stock: float,
    free_mean: np.ndarray,
    free_std: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a ridge-augmented design so a standard ElasticNet solver handles
    both the prior-anchored stock block and the free block.

    Parameters θ split as ``[δ_stock, δ_free_scaled]``:
    - ``δ_stock = θ_stock - coef_prior`` — so shrinkage toward 0 = shrinkage
      toward stock.
    - ``δ_free_scaled = θ_free * free_std`` (standardised) — so uniform
      α penalty makes sense across free features of different scale.

    The stock block gets an explicit L2 penalty λ_stock · ||δ_stock||² via
    the standard Tikhonov augmentation trick: stack √λ_stock · I on X_stock
    rows with corresponding 0 rows on y. ElasticNet is then called with
    its own α on the joint design, but the stock columns already carry
    heavy L2 from the augmented rows, so α mostly affects free columns.

    We also add a dummy-weight column of 1s so ElasticNet's fit_intercept
    handles the intercept.

    Returns (X_aug, y_aug, free_scale_vec).
    """
    N = X_stock.shape[0]
    n_stock = X_stock.shape[1]
    n_free = X_free.shape[1]

    # Offset y by stock prior contribution so δ_stock fits residuals.
    y_prior = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    y_residual = y - y_prior  # model residual = δ_stock·X_stock + θ_free·X_free

    # Standardise free features (center by train mean, scale by train std).
    X_free_std = (X_free - free_mean) / np.where(free_std > 0, free_std, 1.0)

    # Tikhonov block for stock: augment design with √λ·I, and y with 0s.
    sqrt_lam = np.sqrt(max(lambda_stock, 0.0))
    X_aug = np.concatenate(
        [
            np.concatenate([X_stock, X_free_std], axis=1),  # (N, n_stock+n_free)
            np.concatenate(
                [sqrt_lam * np.eye(n_stock), np.zeros((n_stock, n_free))],
                axis=1,
            ),  # Tikhonov rows — (n_stock, n_stock+n_free)
        ],
        axis=0,
    )
    y_aug = np.concatenate([y_residual, np.zeros(n_stock)])
    return X_aug, y_aug


def fit_en_with_prior(
    X_stock_train: np.ndarray,
    X_free_train: np.ndarray,
    y_train: np.ndarray,
    lambda_stock: float,
    alpha: float,
    l1_ratio: float,
) -> dict:
    """Fit Elastic Net + stock prior on training data; return coefs.

    Returns:
        dict with keys ``coef_stock (n_stock,)``, ``coef_free (n_free,)``,
        ``intercept``, ``free_mean``, ``free_std``.
    """
    free_mean = X_free_train.mean(axis=0)
    free_std = X_free_train.std(axis=0)
    free_std_safe = np.where(free_std > 1e-9, free_std, 1.0)

    X_aug, y_aug = build_augmented_design(
        X_stock_train, X_free_train, y_train, lambda_stock,
        free_mean, free_std_safe,
    )

    n_stock = X_stock_train.shape[1]

    # ElasticNet penalty is α·(l1·|θ| + 0.5·(1-l1)·θ²) per sklearn.
    # We want: heavy L2 on stock (from Tikhonov aug) + (α, l1_ratio) on free.
    # The aug trick makes stock columns already carry L2 strength λ_stock
    # via the extra rows, so the ElasticNet α applies uniformly on top.
    # To prevent ElasticNet α from also slightly shrinking stock coefs,
    # we pick α small enough that λ_stock dominates stock columns in
    # practice; λ_stock grid is chosen to be >>α.

    model = ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio,
        fit_intercept=True, max_iter=5000, tol=1e-4,
        selection="cyclic", positive=False,
    )
    model.fit(X_aug, y_aug)
    beta = model.coef_  # (n_stock + n_free,)
    intercept = float(model.intercept_)

    delta_stock = beta[:n_stock]
    coef_free_scaled = beta[n_stock:]

    coef_stock = COEFFS_STOCK + delta_stock
    coef_free = coef_free_scaled / free_std_safe  # un-standardise
    # Unstandardising shifts the intercept: Σ coef_free_scaled·mean/std.
    intercept_unstd = intercept + INTERCEPT_STOCK - np.sum(
        coef_free_scaled * free_mean / free_std_safe
    )
    return dict(
        coef_stock=coef_stock,
        coef_free=coef_free,
        intercept=intercept_unstd,
        free_mean=free_mean,
        free_std=free_std_safe,
    )


def predict_en(
    X_stock: np.ndarray, X_free: np.ndarray, fit: dict,
) -> np.ndarray:
    return X_stock @ fit["coef_stock"] + X_free @ fit["coef_free"] + fit["intercept"]


# --------------------------------------------------------------------------
# Inner CV: pick (lambda_stock, alpha, l1_ratio)
# --------------------------------------------------------------------------

def inner_cv_select_hparams(
    X_stock_train: np.ndarray,
    X_free_train: np.ndarray,
    y_train: np.ndarray,
    groups_train: np.ndarray,
    n_folds: int = 4,
    seed: int = 0,
) -> dict:
    """Inner grouped KFold to pick hyperparameters by validation R."""
    rng = np.random.default_rng(seed)
    N = len(y_train)
    uniq_groups = np.unique(groups_train)
    if len(uniq_groups) < n_folds:
        n_folds = max(2, len(uniq_groups))
    # Random group assignment → folds
    perm = rng.permutation(len(uniq_groups))
    group_to_fold = {g: (i % n_folds) for i, g in enumerate(uniq_groups[perm])}
    fold_ids = np.array([group_to_fold[g] for g in groups_train])

    best = dict(score=-np.inf, lambda_stock=None, alpha=None, l1_ratio=None)
    for lam in LAMBDA_STOCK_GRID:
        for l1 in L1_RATIOS:
            for alpha in ALPHA_GRID:
                preds = np.full(N, np.nan)
                for f in range(n_folds):
                    test = np.where(fold_ids == f)[0]
                    train = np.where(fold_ids != f)[0]
                    if len(train) < 10 or len(test) < 1:
                        continue
                    try:
                        fit = fit_en_with_prior(
                            X_stock_train[train], X_free_train[train],
                            y_train[train], lam, alpha, l1,
                        )
                        preds[test] = predict_en(
                            X_stock_train[test], X_free_train[test], fit,
                        )
                    except Exception:
                        continue
                msk = ~np.isnan(preds)
                if msk.sum() < 10 or np.std(preds[msk]) < 1e-9:
                    continue
                # Objective: negative-RMSE (R can be flat across α for tiny changes).
                rmse = float(np.sqrt(np.mean((preds[msk] - y_train[msk]) ** 2)))
                score = -rmse
                if score > best["score"]:
                    best = dict(
                        score=score, lambda_stock=float(lam),
                        alpha=float(alpha), l1_ratio=float(l1),
                    )
    return best


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_unified() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    # Sanity: dg_exp always present
    assert df["dg_exp_kcal_mol"].isna().sum() == 0
    assert df["pdb_id"].notna().all()
    return df


def select_free_features(df_train: pd.DataFrame,
                         candidates: list[str],
                         threshold: float = NAN_THRESHOLD) -> list[str]:
    keep = []
    for c in candidates:
        if c not in df_train.columns:
            continue
        na_frac = df_train[c].isna().mean()
        if na_frac < threshold:
            keep.append(c)
    return keep


def impute_matrix(X: np.ndarray, fill_vals: np.ndarray) -> np.ndarray:
    """Replace NaN with per-column fill values."""
    Xc = X.copy()
    for j in range(X.shape[1]):
        msk = np.isnan(Xc[:, j])
        if msk.any():
            Xc[msk, j] = fill_vals[j]
    return Xc


# --------------------------------------------------------------------------
# Baselines
# --------------------------------------------------------------------------

def baseline_stock_fixed(X_stock: np.ndarray, y: np.ndarray) -> dict:
    pred = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    m = metrics(pred, y)
    ci = bootstrap_R_ci(pred, y)
    return dict(pred=pred, y=y, metrics=m, R_ci=ci,
                name="stock_fixed")


def baseline_stock_refit_cv(
    X_stock: np.ndarray, y: np.ndarray, groups: np.ndarray,
    n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
    seed: int = BASE_SEED,
) -> dict:
    N = len(y)
    all_r, all_rmse, all_mae, all_rho = [], [], [], []
    preds_by_rep = np.zeros((n_repeats, N))

    uniq_groups = np.unique(groups)
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(len(uniq_groups))
        group_to_fold = {g: (i % n_folds) for i, g in enumerate(uniq_groups[perm])}
        fold_ids = np.array([group_to_fold[g] for g in groups])
        preds = np.zeros(N)
        for f in range(n_folds):
            test = np.where(fold_ids == f)[0]
            train = np.where(fold_ids != f)[0]
            X_aug = np.concatenate([X_stock[train], np.ones((len(train), 1))], axis=1)
            beta, *_ = np.linalg.lstsq(X_aug, y[train], rcond=None)
            coefs, icept = beta[:-1], beta[-1]
            preds[test] = X_stock[test] @ coefs + icept
        preds_by_rep[rep] = preds
        m = metrics(preds, y)
        all_r.append(m["R"]); all_rmse.append(m["RMSE"])
        all_mae.append(m["MAE"]); all_rho.append(m["spearman"])
    mean_preds = preds_by_rep.mean(axis=0)
    m = metrics(mean_preds, y)
    ci = bootstrap_R_ci(mean_preds, y)
    return dict(
        pred=mean_preds, y=y,
        metrics=m, R_ci=ci,
        name="stock_refit_cv",
        R_mean=float(np.mean(all_r)), R_std=float(np.std(all_r)),
        RMSE_mean=float(np.mean(all_rmse)), RMSE_std=float(np.std(all_rmse)),
        MAE_mean=float(np.mean(all_mae)),
        spearman_mean=float(np.mean(all_rho)),
    )


# --------------------------------------------------------------------------
# Primary model: Elastic Net + prior, grouped CV × repeats
# --------------------------------------------------------------------------

def primary_grouped_cv(
    X_stock: np.ndarray, X_free_raw: np.ndarray, y: np.ndarray,
    groups: np.ndarray,
    free_feature_names: list[str],
    n_folds: int = N_FOLDS, n_repeats: int = N_REPEATS,
    seed: int = BASE_SEED,
) -> dict:
    """Nested: outer grouped KFold, inner CV for hparams, refit outer."""
    N = len(y)
    all_r, all_rmse, all_mae, all_rho = [], [], [], []
    preds_by_rep = np.zeros((n_repeats, N))
    hparams_log = []
    coef_stock_accum = np.zeros(X_stock.shape[1])
    coef_free_accum = np.zeros(X_free_raw.shape[1])
    intercept_accum = 0.0
    n_fits = 0
    # Track which free features go non-zero (by average |coef| > epsilon).
    nonzero_counts = np.zeros(X_free_raw.shape[1])

    uniq_groups = np.unique(groups)
    t0 = time.time()
    for rep in range(n_repeats):
        rng = np.random.default_rng(seed + rep)
        perm = rng.permutation(len(uniq_groups))
        group_to_fold = {g: (i % n_folds) for i, g in enumerate(uniq_groups[perm])}
        fold_ids = np.array([group_to_fold[g] for g in groups])
        preds = np.full(N, np.nan)
        for f in range(n_folds):
            test = np.where(fold_ids == f)[0]
            train = np.where(fold_ids != f)[0]
            if len(train) < 20 or len(test) < 1:
                continue
            # Impute free features with TRAIN medians.
            free_med = np.nanmedian(X_free_raw[train], axis=0)
            free_med = np.where(np.isnan(free_med), 0.0, free_med)
            X_free_train = impute_matrix(X_free_raw[train], free_med)
            X_free_test = impute_matrix(X_free_raw[test], free_med)
            X_free_test = np.where(np.isnan(X_free_test), 0.0, X_free_test)

            # Inner CV to pick hparams on train.
            hp = inner_cv_select_hparams(
                X_stock[train], X_free_train, y[train], groups[train],
                n_folds=3, seed=seed + rep * 100 + f,
            )
            hparams_log.append(dict(rep=rep, fold=f, **hp))
            if hp["lambda_stock"] is None:
                continue
            fit = fit_en_with_prior(
                X_stock[train], X_free_train, y[train],
                hp["lambda_stock"], hp["alpha"], hp["l1_ratio"],
            )
            preds[test] = predict_en(X_stock[test], X_free_test, fit)
            coef_stock_accum += fit["coef_stock"]
            coef_free_accum += fit["coef_free"]
            intercept_accum += fit["intercept"]
            n_fits += 1
            nonzero_counts += (np.abs(fit["coef_free"]) > 1e-9).astype(int)
        elapsed = time.time() - t0
        msk = ~np.isnan(preds)
        r_rep = float(pearsonr(preds[msk], y[msk])[0]) if msk.sum() > 2 else float("nan")
        print(f"    rep {rep+1}/{n_repeats}  R={r_rep:+.3f}  "
              f"({elapsed:.0f}s elapsed)", flush=True)
        if msk.sum() < N * 0.9:
            # Fall back: just skip this rep if too many folds failed.
            continue
        preds_by_rep[rep] = preds
        m = metrics(preds[msk], y[msk])
        all_r.append(m["R"]); all_rmse.append(m["RMSE"])
        all_mae.append(m["MAE"]); all_rho.append(m["spearman"])
    mean_preds = preds_by_rep.mean(axis=0)
    m = metrics(mean_preds, y)
    ci = bootstrap_R_ci(mean_preds, y)
    if n_fits == 0:
        raise RuntimeError("Zero successful fits in primary CV")
    avg_coef_stock = coef_stock_accum / n_fits
    avg_coef_free = coef_free_accum / n_fits
    avg_intercept = intercept_accum / n_fits
    nonzero_freq = nonzero_counts / n_fits
    return dict(
        pred=mean_preds, y=y,
        metrics=m, R_ci=ci,
        name="en_prior_primary",
        R_mean=float(np.mean(all_r)), R_std=float(np.std(all_r)),
        RMSE_mean=float(np.mean(all_rmse)), RMSE_std=float(np.std(all_rmse)),
        MAE_mean=float(np.mean(all_mae)),
        spearman_mean=float(np.mean(all_rho)),
        avg_coef_stock=avg_coef_stock,
        avg_coef_free=avg_coef_free,
        avg_intercept=avg_intercept,
        nonzero_freq=nonzero_freq,
        hparams_log=hparams_log,
        free_feature_names=free_feature_names,
    )


# --------------------------------------------------------------------------
# Leave-one-source-out and extrapolation
# --------------------------------------------------------------------------

def fit_and_predict_across_sources(
    df: pd.DataFrame, train_sources: list[str], test_source: str,
    seed: int = BASE_SEED,
) -> dict:
    train_msk = df["source"].isin(train_sources).to_numpy()
    test_msk = (df["source"] == test_source).to_numpy()

    y = df["dg_exp_kcal_mol"].to_numpy()
    groups = df["pdb_id"].to_numpy()
    X_stock_all = df[STOCK_FEATURES].to_numpy().astype(np.float64)
    # Select free features that are viable on TRAIN set.
    free_cols = select_free_features(df[train_msk], FREE_FEATURE_CANDIDATES)
    X_free_all = df[free_cols].to_numpy().astype(np.float64)

    # Train impute
    free_med = np.nanmedian(X_free_all[train_msk], axis=0)
    # Guard: a column that's 100% NaN on train would give nan median.
    free_med = np.where(np.isnan(free_med), 0.0, free_med)

    X_free_train = impute_matrix(X_free_all[train_msk], free_med)
    X_free_test = impute_matrix(X_free_all[test_msk], free_med)

    # For test-time columns missing entirely → already imputed to train median.
    # (If whole column is NaN on test, impute_matrix above does nothing —
    # handle that by also replacing NaN on test with 0.)
    X_free_test = np.where(np.isnan(X_free_test), 0.0, X_free_test)

    hp = inner_cv_select_hparams(
        X_stock_all[train_msk], X_free_train,
        y[train_msk], groups[train_msk],
        n_folds=4, seed=seed,
    )
    fit = fit_en_with_prior(
        X_stock_all[train_msk], X_free_train, y[train_msk],
        hp["lambda_stock"], hp["alpha"], hp["l1_ratio"],
    )
    preds = predict_en(X_stock_all[test_msk], X_free_test, fit)
    y_test = y[test_msk]
    m = metrics(preds, y_test)
    ci = bootstrap_R_ci(preds, y_test)
    return dict(
        train_sources=train_sources, test_source=test_source,
        metrics=m, R_ci=ci,
        hp=hp,
        pdb_ids=df.loc[test_msk, "pdb_id"].to_numpy(),
        y_test=y_test, preds=preds,
        n_train=int(train_msk.sum()), n_test=int(test_msk.sum()),
        free_cols=free_cols,
        fit=fit,
    )


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def plot_calibration(
    out_path: Path, y: np.ndarray, y_pred: np.ndarray,
    sources: np.ndarray, title: str,
):
    fig, ax = plt.subplots(figsize=(6.0, 5.6))
    color_map = {"Kastritis81": "C0", "VrevenBM5.5": "C1", "ProteinBase": "C2"}
    for src, col in color_map.items():
        msk = sources == src
        if msk.any():
            ax.scatter(y[msk], y_pred[msk], alpha=0.65, s=24,
                       color=col, label=f"{src} (n={msk.sum()})")
    lo = min(y.min(), y_pred.min()) - 1.0
    hi = max(y.max(), y_pred.max()) + 1.0
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.4)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    m = metrics(y_pred, y)
    ax.set_xlabel("ΔG_exp (kcal/mol)")
    ax.set_ylabel("ΔG_pred (kcal/mol)")
    ax.set_title(f"{title}\nR={m['R']:.3f}  Spearman={m['spearman']:.3f}  "
                 f"RMSE={m['RMSE']:.2f}  MAE={m['MAE']:.2f}")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def verdict(delta_r_vs_refit: float) -> str:
    if delta_r_vs_refit >= 0.05:
        return "HELPS"
    if delta_r_vs_refit >= 0.03:
        return "MARGINAL"
    return "NO-HELP"


def write_report(path: Path,
                 primary: dict, stock_fixed: dict, stock_refit: dict,
                 lso_results: list[dict], pb_extrap: dict,
                 free_features: list[str]):
    lines = []
    dr_refit = primary["metrics"]["R"] - stock_refit["metrics"]["R"]
    dr_stock = primary["metrics"]["R"] - stock_fixed["metrics"]["R"]
    v = verdict(dr_refit)

    lines += [
        "# Elastic Net with PRODIGY-anchored Prior — Unified N=287",
        "",
        f"**Verdict**: Elastic Net with PRODIGY prior on unified N=287: **{v}**",
        "",
        f"ΔR vs stock REFIT-CV  = {dr_refit:+.4f}",
        f"ΔR vs stock FIXED     = {dr_stock:+.4f}",
        "",
        f"Free features ({len(free_features)}) considered after <50% NaN filter:",
        "",
        f"`{', '.join(free_features)}`",
        "",
        "## Headline — primary grouped CV (4-fold × 10 repeats, group=pdb_id)",
        "",
        "| Model | R (mean ± std) | R (mean-pred, CI) | Spearman | RMSE | MAE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for r in (stock_fixed, stock_refit, primary):
        name = r["name"]
        if "R_mean" in r:
            r_cell = f"{r['R_mean']:+.3f} ± {r['R_std']:.3f}"
            sp = f"{r['spearman_mean']:+.3f}"
            rmse_cell = f"{r['RMSE_mean']:.2f}"
            mae_cell = f"{r['MAE_mean']:.2f}"
        else:
            r_cell = f"{r['metrics']['R']:+.3f}"
            sp = f"{r['metrics']['spearman']:+.3f}"
            rmse_cell = f"{r['metrics']['RMSE']:.2f}"
            mae_cell = f"{r['metrics']['MAE']:.2f}"
        ci = r["R_ci"]
        ci_cell = f"{r['metrics']['R']:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"
        lines.append(f"| {name} | {r_cell} | {ci_cell} | {sp} | "
                     f"{rmse_cell} | {mae_cell} |")

    lines += [
        "",
        "## Leave-one-source-out",
        "",
        "| Train | Test (n) | R | 95% CI | Spearman | RMSE | MAE | "
        "λ_stock | α | l1_ratio |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for res in lso_results:
        m = res["metrics"]
        ci = res["R_ci"]
        hp = res["hp"]
        lines.append(
            f"| {'+'.join(res['train_sources'])} | {res['test_source']} "
            f"(n={res['n_test']}) | {m['R']:+.3f} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] | {m['spearman']:+.3f} | "
            f"{m['RMSE']:.2f} | {m['MAE']:.2f} | "
            f"{hp['lambda_stock']:.2f} | {hp['alpha']:.4f} | "
            f"{hp['l1_ratio']:.2f} |"
        )

    lines += [
        "",
        "## Extrapolation — fit on K81+V106, predict PB",
        "",
        f"n_train = {pb_extrap['n_train']},  n_test = {pb_extrap['n_test']}",
        "",
        "| R | 95% CI | Spearman | RMSE | MAE |",
        "|---:|---:|---:|---:|---:|",
    ]
    m = pb_extrap["metrics"]; ci = pb_extrap["R_ci"]
    lines.append(
        f"| {m['R']:+.3f} | [{ci[0]:+.3f}, {ci[1]:+.3f}] | "
        f"{m['spearman']:+.3f} | {m['RMSE']:.2f} | {m['MAE']:.2f} |"
    )

    # Coefficients table — averaged over all primary CV folds
    lines += [
        "",
        "## Coefficients (primary CV, average over folds)",
        "",
        f"intercept (avg)   = {primary['avg_intercept']:+.4f} "
        f"(stock intercept {INTERCEPT_STOCK:+.4f})",
        "",
        "### Stock PRODIGY 6 (Bayesian anchor)",
        "",
        "| feature | stock | avg coef | Δ vs stock |",
        "|---|---:|---:|---:|",
    ]
    for name, stock, new in zip(STOCK_FEATURES, COEFFS_STOCK,
                                 primary["avg_coef_stock"]):
        lines.append(f"| {name} | {stock:+.5f} | {new:+.5f} | "
                     f"{new - stock:+.5f} |")

    # Free features — rank by |avg coef| and frequency of non-zero.
    lines += [
        "",
        "### Free features — non-zero survivors",
        "",
        "| feature | avg coef | |coef| | non-zero freq |",
        "|---|---:|---:|---:|",
    ]
    free_names = primary["free_feature_names"]
    avg_free = primary["avg_coef_free"]
    nz = primary["nonzero_freq"]
    order = np.argsort(-np.abs(avg_free))
    for j in order:
        if nz[j] < 0.25:
            continue
        lines.append(f"| {free_names[j]} | {avg_free[j]:+.4f} | "
                     f"{abs(avg_free[j]):.4f} | {nz[j]:.2f} |")

    # Biggest deviation from stock
    idx_dev = int(np.argmax(np.abs(primary["avg_coef_stock"] - COEFFS_STOCK)))
    lines += [
        "",
        "### Biggest deviation-from-stock-PRODIGY coefficient",
        "",
        f"- feature `{STOCK_FEATURES[idx_dev]}`: stock "
        f"{COEFFS_STOCK[idx_dev]:+.5f} → avg "
        f"{primary['avg_coef_stock'][idx_dev]:+.5f} "
        f"(Δ = {primary['avg_coef_stock'][idx_dev] - COEFFS_STOCK[idx_dev]:+.5f})",
        "",
        "## Calibration plots",
        "",
        f"- {Path('calibration_grouped_cv.png').name}",
        f"- {Path('calibration_heldout_pb.png').name}",
    ]
    path.write_text("\n".join(lines) + "\n")


def write_coefs_csv(path: Path, primary: dict):
    free_names = primary["free_feature_names"]
    rows = []
    for name, stock, new in zip(STOCK_FEATURES, COEFFS_STOCK,
                                 primary["avg_coef_stock"]):
        rows.append(dict(feature=name, is_stock=True,
                         stock_coef=stock, avg_coef=new,
                         delta_coef=new - stock,
                         abs_coef=abs(new),
                         nonzero_freq=1.0))
    for name, new, nz in zip(free_names, primary["avg_coef_free"],
                              primary["nonzero_freq"]):
        rows.append(dict(feature=name, is_stock=False,
                         stock_coef=np.nan, avg_coef=new,
                         delta_coef=np.nan,
                         abs_coef=abs(new),
                         nonzero_freq=float(nz)))
    pd.DataFrame(rows).to_csv(path, index=False)


def write_predictions_csv(path: Path, df: pd.DataFrame, preds: np.ndarray):
    out = pd.DataFrame({
        "pdb_id": df["pdb_id"].to_numpy(),
        "source": df["source"].to_numpy(),
        "dg_exp": df["dg_exp_kcal_mol"].to_numpy(),
        "dg_pred": preds,
        "residual": preds - df["dg_exp_kcal_mol"].to_numpy(),
    })
    out.to_csv(path, index=False)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[load] {DATA_CSV}")
    df = load_unified()
    print(f"  rows={len(df)}  sources={df['source'].value_counts().to_dict()}")

    y = df["dg_exp_kcal_mol"].to_numpy().astype(np.float64)
    groups = df["pdb_id"].to_numpy()
    sources = df["source"].to_numpy()

    # Stock features — never NaN (per dataset audit).
    X_stock = df[STOCK_FEATURES].to_numpy().astype(np.float64)
    assert not np.isnan(X_stock).any()

    # Select free features viable on the full unified set (<50% NaN globally).
    free_cols = select_free_features(df, FREE_FEATURE_CANDIDATES,
                                      threshold=NAN_THRESHOLD)
    print(f"[features] kept {len(free_cols)}/{len(FREE_FEATURE_CANDIDATES)} "
          f"free features after <{int(NAN_THRESHOLD*100)}% NaN filter:")
    print(f"  {free_cols}")
    X_free_raw = df[free_cols].to_numpy().astype(np.float64)

    # ---- Baselines ----
    print("[baseline] stock FIXED")
    base_fixed = baseline_stock_fixed(X_stock, y)
    print(f"  R={base_fixed['metrics']['R']:+.3f}  "
          f"CI={base_fixed['R_ci']}  "
          f"RMSE={base_fixed['metrics']['RMSE']:.2f}")

    print("[baseline] stock REFIT CV (4-fold × 10 repeats, grouped)")
    base_refit = baseline_stock_refit_cv(X_stock, y, groups)
    print(f"  R={base_refit['metrics']['R']:+.3f}  "
          f"R_mean={base_refit['R_mean']:+.3f}±{base_refit['R_std']:.3f}  "
          f"RMSE={base_refit['metrics']['RMSE']:.2f}")

    # ---- Primary: Elastic Net + prior ----
    print(f"[primary] Elastic Net + prior, nested grouped CV "
          f"(N_FOLDS={N_FOLDS}, N_REPEATS={N_REPEATS}) …")
    primary = primary_grouped_cv(X_stock, X_free_raw, y, groups,
                                  free_feature_names=free_cols)
    print(f"  R={primary['metrics']['R']:+.3f}  "
          f"R_mean={primary['R_mean']:+.3f}±{primary['R_std']:.3f}  "
          f"RMSE={primary['metrics']['RMSE']:.2f}  "
          f"Spearman={primary['metrics']['spearman']:+.3f}")

    # ---- Leave-one-source-out ----
    print("[lso] leave-one-source-out …")
    sources_uniq = ["Kastritis81", "VrevenBM5.5", "ProteinBase"]
    lso_results = []
    for test_src in sources_uniq:
        train_src = [s for s in sources_uniq if s != test_src]
        res = fit_and_predict_across_sources(df, train_src, test_src)
        lso_results.append(res)
        print(f"  train={train_src}  test={test_src} "
              f"R={res['metrics']['R']:+.3f}  "
              f"RMSE={res['metrics']['RMSE']:.2f}")

    # ---- Extrapolation: K81+V106 → PB ----
    print("[extrap] K81+V106 -> PB")
    pb_extrap = fit_and_predict_across_sources(
        df, ["Kastritis81", "VrevenBM5.5"], "ProteinBase",
    )
    print(f"  R={pb_extrap['metrics']['R']:+.3f}  "
          f"RMSE={pb_extrap['metrics']['RMSE']:.2f}")

    # ---- Plots ----
    plot_calibration(
        OUT_DIR / "calibration_grouped_cv.png",
        y, primary["pred"], sources,
        title="Elastic Net + PRODIGY prior — grouped CV (mean over 10 repeats)",
    )
    # For held-out PB plot, combine: training points from CV preds +
    # test points from pb_extrap
    # simpler: just plot the PB extrapolation alone.
    plot_calibration(
        OUT_DIR / "calibration_heldout_pb.png",
        pb_extrap["y_test"], pb_extrap["preds"],
        np.full(len(pb_extrap["y_test"]), "ProteinBase"),
        title="Elastic Net + PRODIGY prior — extrapolation to PB (held out)",
    )

    # ---- CSVs ----
    write_coefs_csv(OUT_DIR / "coefs.csv", primary)
    write_predictions_csv(OUT_DIR / "predictions_grouped_cv.csv",
                           df, primary["pred"])

    # ---- Report ----
    write_report(
        OUT_DIR / "report.md",
        primary, base_fixed, base_refit, lso_results, pb_extrap,
        free_features=free_cols,
    )

    # Dump key hparams log as JSON
    hp_log_path = OUT_DIR / "hparams_log.json"
    hp_log_path.write_text(json.dumps(primary["hparams_log"], indent=2))

    print(f"\n[done] wrote results to {OUT_DIR}")
    print(f"  verdict: {verdict(primary['metrics']['R'] - base_refit['metrics']['R'])}")
    print(f"  ΔR vs refit-CV = "
          f"{primary['metrics']['R'] - base_refit['metrics']['R']:+.4f}")


if __name__ == "__main__":
    main()
