#!/usr/bin/env python3
"""Experiment A (entropy surrogate) — augment PRODIGY with a PAE-derived
conformational-entropy feature on Kastritis 81 Boltz predictions.

v1 AIC had treated `mean_PAE_contacts` as an unconstrained main effect among
13 candidates. Here we test a *sign-constrained* augmentation driven by a
thermodynamic prior:

        ΔG_pred = PRODIGY_6(IC, NIS)  +  c · S(PAE)  +  Q   with  c ≥ 0.

High interface PAE ⇒ floppy binding interface ⇒ conformational-entropy cost
⇒ weaker (less-negative) ΔG.  Since more negative ΔG means stronger binding,
the PAE coefficient should be **non-negative**: adding PAE must *increase*
(weaken) ΔG_pred.

Four S-variants, for each complex with contact mask
``contact_ij = 1(min_heavy_dist_ij ≤ 5.5)`` and inter-chain PAE block
``pae_ab``:

    S_mean              = mean(pae_ab[contact_mask])
    S_max               = max(pae_ab[contact_mask])
    S_sum               = sum(pae_ab[contact_mask])
    S_contact_weighted  = sum(pae_ab[contact_mask]) / sqrt(n_contacts)

Two coefficient-fitting policies:

    (a) Unconstrained OLS      (np.linalg.lstsq)
    (b) Sign-constrained       (scipy.optimize.lsq_linear, c ≥ 0)

Evaluation: in-sample R + 4-fold CV × 10 repeats + RMSE, stratified by
iRMSD > 1.0 Å (paper convention).

Usage:
    python entropy_surrogate.py                            # both modes
    python entropy_surrogate.py --mode msa_only
    python entropy_surrogate.py --mode template_msa --out-dir .../custom
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.optimize import lsq_linear  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from quick_pae_calib import (  # noqa: E402
    BOLTZ_ROOT, Complex, load_complex, load_dataset_truth,
    load_stock_prodigy,
)
from diagnostic_refit import (  # noqa: E402
    COEFFS_STOCK, INTERCEPT_STOCK, IRMSD_CUTOFF, R_RMSE, fit_ols,
    kfold_cv_R,
)

DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"

# Stock 6-feature PRODIGY order (matches COEFFS_STOCK indexing).
STOCK_FEATURES = ("ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c")

# S-variants we test.
S_VARIANTS = ("S_mean", "S_max", "S_sum", "S_contact_weighted")

# Policies: (label, name-for-bound-sign). None = unconstrained.
POLICIES = (
    ("OLS",       None),       # unconstrained OLS on augmented features
    ("NNsign",    "nonneg"),   # c ≥ 0 on S term only
)

# Improvement threshold for success (documented in the spec).
DELTA_R_FLOOR = 0.03


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------

def compute_s_variants(comp: Complex, d_cut: float = 5.5
                       ) -> dict:
    """Return the 4 S-variants and n_contacts for a single complex.

    Contact mask uses only min-heavy-distance (no PAE gating) — we want S(PAE)
    as an independent feature, not a gated contact count.
    """
    contact_mask = comp.min_dist <= d_cut      # [N_t, N_b] bool
    n_contacts = int(contact_mask.sum())
    if n_contacts == 0:
        # Should not happen on Kastritis 81, but guard anyway.
        return {
            "n_contacts": 0,
            "S_mean": np.nan, "S_max": np.nan,
            "S_sum": np.nan, "S_contact_weighted": np.nan,
        }
    pae_at_contacts = comp.pae_ab[contact_mask]
    s_mean = float(pae_at_contacts.mean())
    s_max = float(pae_at_contacts.max())
    s_sum = float(pae_at_contacts.sum())
    s_cw = s_sum / float(np.sqrt(n_contacts))
    return {
        "n_contacts": n_contacts,
        "S_mean": s_mean, "S_max": s_max,
        "S_sum": s_sum, "S_contact_weighted": s_cw,
    }


def build_features(mode: str) -> pd.DataFrame:
    """Build the feature DataFrame for all 81 complexes in a given mode."""
    truth = load_dataset_truth()
    stock = load_stock_prodigy()
    pdb_ids = sorted(truth.keys())

    rows = []
    t0 = time.time()
    for pid in pdb_ids:
        c = load_complex(pid, mode, truth, stock)
        if c is None:
            print(f"[skip] {pid}/{mode}")
            continue
        s_feats = compute_s_variants(c, d_cut=5.5)

        # Pull stock IC+NIS from prodigy_scores.csv (matches diagnostic_refit).
        ic_stock = c.ic_stock
        if not ic_stock:
            print(f"[skip] {pid}/{mode}: missing stock IC/NIS row in "
                  f"prodigy_scores.csv")
            continue
        rows.append({
            "pdb_id": pid, "mode": mode,
            "dg_exp": c.dg_exp, "ba_val": c.ba_val, "irmsd": c.irmsd,
            # stock 6 (from prodigy_scores.csv to match the diagnostic script)
            "ic_cc": float(ic_stock["ic_cc"]),
            "ic_ca": float(ic_stock["ic_ca"]),
            "ic_pp": float(ic_stock["ic_pp"]),
            "ic_pa": float(ic_stock["ic_pa"]),
            "nis_a": float(np.clip(ic_stock["nis_a"], 0, 100)),
            "nis_c": float(np.clip(ic_stock["nis_c"], 0, 100)),
            # entropy surrogates
            **s_feats,
        })
    print(f"[features] built {len(rows)} rows for {mode} in "
          f"{time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Fitters
# --------------------------------------------------------------------------

def fit_aug_ols(X_aug: np.ndarray, y: np.ndarray
                ) -> tuple[np.ndarray, float]:
    """Unconstrained OLS on the augmented (6 stock + 1 S) design matrix.

    Returns (coefs[7], intercept).
    """
    X_bias = np.concatenate([X_aug, np.ones((X_aug.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_bias, y, rcond=None)
    return beta[:-1], float(beta[-1])


def fit_aug_nonneg_s(X_aug: np.ndarray, y: np.ndarray
                     ) -> tuple[np.ndarray, float]:
    """Constrained fit: 6 stock coefs free, S coef (column 6) ≥ 0, intercept free.

    Uses ``scipy.optimize.lsq_linear`` with box bounds on the augmented
    design matrix.
    """
    X_bias = np.concatenate([X_aug, np.ones((X_aug.shape[0], 1))], axis=1)
    n_cols = X_bias.shape[1]    # 6 stock + 1 S + 1 intercept = 8
    lo = np.full(n_cols, -np.inf)
    hi = np.full(n_cols, np.inf)
    lo[6] = 0.0                 # S coef ≥ 0 (physical prior)
    res = lsq_linear(X_bias, y, bounds=(lo, hi), method="bvls")
    beta = res.x
    return beta[:-1], float(beta[-1])


def fit_from_policy(policy: tuple) -> callable:
    _, kind = policy
    if kind is None:
        return fit_aug_ols
    if kind == "nonneg":
        return fit_aug_nonneg_s
    raise ValueError(f"unknown policy kind: {kind}")


# --------------------------------------------------------------------------
# Generic CV using a user-supplied fitter
# --------------------------------------------------------------------------

def kfold_cv_generic(X: np.ndarray, y: np.ndarray,
                     fitter: callable, k: int = 4, n_repeats: int = 10,
                     seed: int = 0) -> dict:
    """4-fold CV × n_repeats on an arbitrary fitter.

    ``fitter(X_train, y_train)`` must return ``(coefs[n_feat], intercept)``.
    """
    rng = np.random.default_rng(seed)
    N = len(y)
    R_vals: list[float] = []
    RMSE_vals: list[float] = []
    preds_all = np.zeros((n_repeats, N))
    for rep in range(n_repeats):
        perm = rng.permutation(N)
        folds = np.array_split(perm, k)
        preds = np.zeros(N)
        for fi in range(k):
            test = folds[fi]
            train = np.setdiff1d(np.arange(N), test, assume_unique=False)
            coefs, icept = fitter(X[train], y[train])
            preds[test] = X[test] @ coefs + icept
        preds_all[rep] = preds
        r, rmse = R_RMSE(preds, y)
        R_vals.append(r); RMSE_vals.append(rmse)
    return {
        "R_mean": float(np.mean(R_vals)),
        "R_std": float(np.std(R_vals)),
        "RMSE_mean": float(np.mean(RMSE_vals)),
        "RMSE_std": float(np.std(RMSE_vals)),
        "preds_mean": preds_all.mean(axis=0),
    }


# --------------------------------------------------------------------------
# Per-mode evaluation
# --------------------------------------------------------------------------

def evaluate_mode(df: pd.DataFrame, mode: str) -> dict:
    """Evaluate all (variant × policy) combos for a given mode.

    Returns: {
        "mode": mode,
        "N": int, "n_rigid": int, "n_flex": int,
        "stock_fixed": {R, RMSE, R_rigid, R_flex, pred},
        "results": [{variant, policy, R_in, RMSE_in,
                      CV: {R_mean, R_std, RMSE_mean, RMSE_std, preds_mean},
                      CV_rigid, CV_flex, coefs[7], intercept, s_coef_sign},
                     ...],
        "best": {...},
        "y": y, "is_flex": is_flex
    }
    """
    y = df["dg_exp"].to_numpy(dtype=np.float64)
    is_flex = (df["irmsd"] > IRMSD_CUTOFF).to_numpy()
    n_rigid = int((~is_flex).sum()); n_flex = int(is_flex.sum())
    X_stock = df[list(STOCK_FEATURES)].to_numpy(dtype=np.float64)

    # Stock FIXED baseline (unchanged).
    pred_fixed = X_stock @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed, rmse_fixed = R_RMSE(pred_fixed, y)
    r_fixed_rigid, _ = R_RMSE(pred_fixed[~is_flex], y[~is_flex])
    r_fixed_flex, _ = R_RMSE(pred_fixed[is_flex], y[is_flex])

    # Stock REFIT CV baseline (for reference; same CV splits as augmented).
    cv_stock = kfold_cv_R(X_stock, y, k=4, n_repeats=10, seed=0)

    results = []
    for variant in S_VARIANTS:
        s_col = df[variant].to_numpy(dtype=np.float64)
        if np.isnan(s_col).any():
            # Some complex had zero contacts. Skip this variant.
            print(f"[warn] {mode}/{variant}: NaN in S column — skipping")
            continue
        X_aug = np.concatenate([X_stock, s_col[:, None]], axis=1)
        for policy in POLICIES:
            label = policy[0]
            fitter = fit_from_policy(policy)
            # In-sample fit
            coefs, icept = fitter(X_aug, y)
            pred_in = X_aug @ coefs + icept
            r_in, rmse_in = R_RMSE(pred_in, y)
            # CV
            cv = kfold_cv_generic(X_aug, y, fitter, k=4, n_repeats=10, seed=0)
            preds_cv = cv["preds_mean"]
            r_cv_rigid, _ = R_RMSE(preds_cv[~is_flex], y[~is_flex])
            r_cv_flex, _ = R_RMSE(preds_cv[is_flex], y[is_flex])
            s_coef = float(coefs[6])
            results.append({
                "variant": variant, "policy": label,
                "R_in": r_in, "RMSE_in": rmse_in,
                "CV": cv, "CV_rigid": r_cv_rigid, "CV_flex": r_cv_flex,
                "coefs": coefs.tolist(), "intercept": icept,
                "s_coef": s_coef,
                "s_coef_sign": "+" if s_coef > 0 else ("-" if s_coef < 0 else "0"),
                "delta_R_cv_vs_stock_fixed": cv["R_mean"] - r_fixed,
            })

    # Find the best (highest CV R) combo.
    best = max(results, key=lambda r: r["CV"]["R_mean"]) if results else None

    return {
        "mode": mode, "N": len(y),
        "n_rigid": n_rigid, "n_flex": n_flex,
        "stock_fixed": {
            "R": r_fixed, "RMSE": rmse_fixed,
            "R_rigid": r_fixed_rigid, "R_flex": r_fixed_flex,
            "pred": pred_fixed,
        },
        "stock_refit_cv": cv_stock,
        "results": results,
        "best": best,
        "y": y, "is_flex": is_flex,
        "df": df,
    }


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def summarize_verdict(all_results: list[dict]) -> str:
    """HELPS / NO-HELP / MARGINAL based on best ΔR across modes."""
    best_deltas: list[float] = []
    sign_ok = True
    for res in all_results:
        if res["best"] is None:
            continue
        best = res["best"]
        best_deltas.append(best["delta_R_cv_vs_stock_fixed"])
        # Under the NNsign policy, s_coef will be ≥ 0. If under OLS it's
        # negative, flag that (prior may be wrong).
        if best["policy"] == "OLS" and best["s_coef"] < 0:
            sign_ok = False
    if not best_deltas:
        return "NO-HELP"
    max_delta = max(best_deltas)
    if max_delta >= DELTA_R_FLOOR:
        return "HELPS" if sign_ok else "HELPS (but PRIOR VIOLATED)"
    if max_delta >= 0.01:
        return "MARGINAL"
    return "NO-HELP"


def write_report(path: Path, mode_results: list[dict],
                 crystal_R: float, crystal_RMSE: float):
    verdict = summarize_verdict(mode_results)
    lines = [
        f"# Entropy surrogate on K81: **{verdict}**",
        "",
        "Augmentation:",
        "",
        "    ΔG_pred = PRODIGY_6(IC, NIS) + c · S(PAE) + Q,   (NNsign policy: c ≥ 0)",
        "",
        f"Crystal reference (`ba_val` vs `DG`): "
        f"R = {crystal_R:.3f}, RMSE = {crystal_RMSE:.2f} kcal/mol",
        "",
        f"Flexibility cutoff: iRMSD > {IRMSD_CUTOFF} Å (paper convention).",
        f"ΔR success floor: +{DELTA_R_FLOOR:.2f} over stock FIXED baseline "
        "(N=81 bootstrap-detectable).",
        "",
        "S-variants (contact mask = `min_heavy_dist ≤ 5.5`):",
        "",
        "- **S_mean**              = mean(pae_ab[contact_mask])",
        "- **S_max**               = max(pae_ab[contact_mask])",
        "- **S_sum**               = sum(pae_ab[contact_mask])",
        "- **S_contact_weighted**  = sum(pae_ab[contact_mask]) / sqrt(n_contacts)",
        "",
        "Coefficient policies:",
        "",
        "- **OLS**    — unconstrained `np.linalg.lstsq` on 7 augmented features",
        "- **NNsign** — `scipy.optimize.lsq_linear` with S-coef ≥ 0 "
        "(physical prior)",
        "",
    ]
    for res in mode_results:
        stock = res["stock_fixed"]
        cv_stock = res["stock_refit_cv"]
        lines += [
            f"## mode = {res['mode']}   "
            f"(N={res['N']}, rigid={res['n_rigid']}, flex={res['n_flex']})",
            "",
            "### Baselines",
            "",
            "| Baseline | R (all) | RMSE | R (rigid) | R (flex) |",
            "|---|---:|---:|---:|---:|",
            (f"| stock FIXED (stock coefs) | {stock['R']:+.3f} | "
             f"{stock['RMSE']:.2f} | {stock['R_rigid']:+.3f} | "
             f"{stock['R_flex']:+.3f} |"),
            (f"| stock 6-feat REFIT 4-fold CV ×10 | "
             f"{cv_stock['R_mean']:+.3f} ± {cv_stock['R_std']:.3f} | "
             f"{cv_stock['RMSE_mean']:.2f} ± {cv_stock['RMSE_std']:.2f} | "
             f"— | — |"),
            "",
            "### Augmented 7-feature results",
            "",
            "| variant | policy | R (CV mean ± std) | RMSE (CV) | R (in-sample) | "
            "ΔR vs stock FIXED | sign(c_S) |",
            "|---|---|---:|---:|---:|---:|:---:|",
        ]
        # Sort by CV R desc so the best is at the top per mode.
        for r in sorted(res["results"],
                         key=lambda x: -x["CV"]["R_mean"]):
            cv = r["CV"]
            lines.append(
                f"| {r['variant']} | {r['policy']} | "
                f"{cv['R_mean']:+.3f} ± {cv['R_std']:.3f} | "
                f"{cv['RMSE_mean']:.2f} ± {cv['RMSE_std']:.2f} | "
                f"{r['R_in']:+.3f} | "
                f"{r['delta_R_cv_vs_stock_fixed']:+.3f} | "
                f"{r['s_coef_sign']} |"
            )
        lines += [
            "",
            "### Stratified CV R per variant "
            f"(iRMSD > {IRMSD_CUTOFF} Å)",
            "",
            "| variant | policy | R (rigid, n=" f"{res['n_rigid']}) | "
            f"R (flex, n={res['n_flex']}) |",
            "|---|---|---:|---:|",
        ]
        for r in sorted(res["results"],
                         key=lambda x: -x["CV"]["R_mean"]):
            lines.append(
                f"| {r['variant']} | {r['policy']} | "
                f"{r['CV_rigid']:+.3f} | {r['CV_flex']:+.3f} |"
            )
        # Best variant coefs
        if res["best"] is not None:
            best = res["best"]
            lines += [
                "",
                f"### Coefficients of best variant "
                f"({best['variant']}, {best['policy']})",
                "",
                "| feature | stock | augmented fit | Δ |",
                "|---|---:|---:|---:|",
            ]
            for i, feat in enumerate(STOCK_FEATURES):
                st = COEFFS_STOCK[i]
                nw = best["coefs"][i]
                lines.append(f"| {feat} | {st:+.5f} | {nw:+.5f} | "
                             f"{nw - st:+.5f} |")
            # S coef
            lines.append(f"| {best['variant']} | — | "
                         f"{best['coefs'][6]:+.5f} | — |")
            lines.append(
                f"| intercept | {INTERCEPT_STOCK:+.3f} | "
                f"{best['intercept']:+.3f} | "
                f"{best['intercept'] - INTERCEPT_STOCK:+.3f} |"
            )
            lines.append("")
    lines += [
        "## Verdict key",
        "",
        "- **HELPS**: CV ΔR ≥ +0.03 over stock FIXED on at least one mode, "
        "and best-variant entropy coef is non-negative (prior holds).",
        "- **HELPS (but PRIOR VIOLATED)**: CV ΔR ≥ +0.03 but the "
        "unconstrained OLS S-coef is negative → feature correlates with DG "
        "in the wrong direction.",
        "- **MARGINAL**: CV ΔR in [+0.01, +0.03]; not bootstrap-detectable.",
        "- **NO-HELP**: CV ΔR < +0.01.",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_scatter(res: dict, out_dir: Path):
    y = res["y"]
    is_flex = res["is_flex"]
    pred_fixed = res["stock_fixed"]["pred"]
    if res["best"] is None:
        print(f"[plot] no augmented results for {res['mode']}; skipping scatter")
        return
    best = res["best"]
    pred_aug = best["CV"]["preds_mean"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharex=True, sharey=True)
    for ax, pred, ttl in zip(
        axes,
        (pred_fixed, pred_aug),
        ("stock FIXED coefs",
         f"augmented CV ({best['variant']}, {best['policy']})"),
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
    fig.savefig(out_dir / f"entropy_scatter_{res['mode']}.png", dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Grid CSV (one row per (pdb_id, mode))
# --------------------------------------------------------------------------

def write_grid_csv(path: Path, dfs: list[pd.DataFrame]):
    cols = (
        ["pdb_id", "mode", "dg_exp", "ba_val", "irmsd",
         "ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c",
         "n_contacts"]
        + list(S_VARIANTS)
    )
    merged = pd.concat([df[cols] for df in dfs], ignore_index=True)
    merged.to_csv(path, index=False)


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
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    out_dir = (
        Path(args.out_dir) if args.out_dir
        else BOLTZ_ROOT / "pae_calibration" / "entropy_surrogate"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    modes = (["msa_only", "template_msa"] if args.mode == "both"
             else [args.mode])

    truth = json.loads(DATASET_JSON.read_text())
    pdbs = sorted(truth)
    dg_exp_all = np.array([float(truth[p]["DG"]) for p in pdbs])
    ba_val_all = np.array([float(truth[p]["ba_val"]) for p in pdbs])
    crystal_R, crystal_RMSE = R_RMSE(ba_val_all, dg_exp_all)
    print(f"[ref] crystal  R={crystal_R:+.3f}  RMSE={crystal_RMSE:.2f}")

    mode_dfs: list[pd.DataFrame] = []
    mode_results: list[dict] = []
    for mode in modes:
        print(f"\n=== mode = {mode} ===")
        df = build_features(mode)
        mode_dfs.append(df)

        res = evaluate_mode(df, mode)
        mode_results.append(res)

        stock = res["stock_fixed"]
        print(f"  N={res['N']}  rigid={res['n_rigid']}  flex={res['n_flex']}")
        print(f"  stock FIXED   R={stock['R']:+.3f} RMSE={stock['RMSE']:.2f}  "
              f"(R_rigid={stock['R_rigid']:+.3f}, "
              f"R_flex={stock['R_flex']:+.3f})")
        for r in sorted(res["results"],
                         key=lambda x: -x["CV"]["R_mean"]):
            cv = r["CV"]
            print(
                f"  [aug] {r['variant']:22s} {r['policy']:6s}  "
                f"CV R={cv['R_mean']:+.3f}±{cv['R_std']:.3f}  "
                f"RMSE={cv['RMSE_mean']:.2f}  "
                f"in-sample R={r['R_in']:+.3f}  "
                f"ΔR={r['delta_R_cv_vs_stock_fixed']:+.3f}  "
                f"c_S={r['s_coef']:+.5f} ({r['s_coef_sign']})"
            )
        if res["best"] is not None:
            b = res["best"]
            print(f"  BEST → {b['variant']} / {b['policy']}  "
                  f"CV R = {b['CV']['R_mean']:+.3f}, "
                  f"ΔR = {b['delta_R_cv_vs_stock_fixed']:+.3f}")

        plot_scatter(res, out_dir)

    # CSV across all modes
    write_grid_csv(out_dir / "entropy_features.csv", mode_dfs)

    # Report
    write_report(out_dir / "report.md", mode_results, crystal_R, crystal_RMSE)

    verdict = summarize_verdict(mode_results)
    print(f"\n[verdict] Entropy surrogate on K81: {verdict}")
    print(f"[done] {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
