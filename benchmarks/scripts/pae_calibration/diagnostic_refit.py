#!/usr/bin/env python3
"""Experiment A — does refitting coefficients close the Boltz→crystal gap?

Diagnostic for the hypothesis: the 0.62 (Boltz stock) → 0.74 (crystal) R gap
comes from coefficient miscalibration, not loss of signal in the IC/NIS
features.

Loads the existing ``prodigy_scores.csv`` (stock IC counts + NIS% for each
Boltz prediction) and evaluates three coefficient policies on each mode:

    FIXED      — stock PRODIGY coefs (2015 crystal fit, ``NIS_COEFFICIENTS``)
    REFIT-IN   — OLS refit on all 81 Boltz complexes, report in-sample R
    REFIT-CV   — 4-fold CV × 10 repeats (paper convention)

Stratification uses the paper's iRMSD > 1.0 Å cutoff (not the 2.2 Å Vreven
convention) — the Kastritis 81 split is roughly 41 rigid / 40 flexible.

Usage:
    python diagnostic_refit.py                # both modes, all 81
    python diagnostic_refit.py --mode msa_only
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
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BOLTZ_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"
PRODIGY_CSV = BOLTZ_ROOT / "prodigy_scores.csv"

# 2015 PRODIGY IC-NIS coefficients (from src/.../scoring.py).
COEFFS_STOCK = np.array(
    [-0.09459, -0.10007, 0.19577, -0.22671, 0.18681, 0.13810],
    dtype=np.float64,
)
INTERCEPT_STOCK = -15.9433

# Paper's flexibility cutoff (Kastritis et al. 2011; also Vangone & Bonvin
# 2015 eLife), distinct from Vreven's 2.2 Å threshold.
IRMSD_CUTOFF = 1.0

FEATURE_ORDER = ("ic_cc", "ic_ca", "ic_pp", "ic_pa", "nis_a", "nis_c")


# --------------------------------------------------------------------------
# Data loading
# --------------------------------------------------------------------------

def load_rows(mode: str) -> list[dict]:
    out = []
    with PRODIGY_CSV.open() as f:
        for row in csv.DictReader(f):
            if row["mode"] != mode:
                continue
            r = dict(row)
            for k in FEATURE_ORDER:
                r[k] = float(r[k])
            r["dg_exp"] = float(r["dg_exp"])
            r["dg_pred_boltz"] = float(r["dg_pred_boltz"])
            r["dg_prodigy_baseline"] = float(r["dg_prodigy_baseline"])
            out.append(r)
    return out


def load_irmsd() -> dict[str, float]:
    d = json.loads(DATASET_JSON.read_text())
    return {k: float(v["iRMSD"]) for k, v in d.items()}


def build_X_y(rows: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    pdb = [r["pdb_id"] for r in rows]
    X = np.stack([
        [r["ic_cc"], r["ic_ca"], r["ic_pp"], r["ic_pa"],
         np.clip(r["nis_a"], 0, 100), np.clip(r["nis_c"], 0, 100)]
        for r in rows
    ]).astype(np.float64)
    y = np.array([r["dg_exp"] for r in rows], dtype=np.float64)
    return X, y, pdb


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def R_RMSE(pred: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    if np.std(pred) == 0:
        return float("nan"), float("nan")
    r = pearsonr(pred, y)[0]
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return r, rmse


def fit_ols(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Return (coefs[6], intercept)."""
    X_aug = np.concatenate([X, np.ones((X.shape[0], 1))], axis=1)
    beta, *_ = np.linalg.lstsq(X_aug, y, rcond=None)
    return beta[:-1], float(beta[-1])


def kfold_cv_R(X: np.ndarray, y: np.ndarray, k: int = 4,
               n_repeats: int = 10, seed: int = 0) -> dict:
    """4-fold CV × 10 repeats — paper convention."""
    rng = np.random.default_rng(seed)
    N = len(y)
    R_vals: list[float] = []; RMSE_vals: list[float] = []
    preds_all = np.zeros((n_repeats, N))
    for rep in range(n_repeats):
        perm = rng.permutation(N)
        folds = np.array_split(perm, k)
        preds = np.zeros(N)
        for fi in range(k):
            test = folds[fi]
            train = np.setdiff1d(np.arange(N), test, assume_unique=False)
            coefs, icept = fit_ols(X[train], y[train])
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
# Core evaluation for one mode
# --------------------------------------------------------------------------

def evaluate_mode(mode: str, irmsd_map: dict[str, float]) -> dict:
    rows = load_rows(mode)
    if not rows:
        raise SystemExit(f"No rows for mode={mode} in {PRODIGY_CSV}")
    X, y, pdb = build_X_y(rows)
    irmsd = np.array([irmsd_map[p] for p in pdb])
    is_flex = irmsd > IRMSD_CUTOFF
    n_rigid = int((~is_flex).sum()); n_flex = int(is_flex.sum())

    # 1. FIXED — stock coefficients
    pred_fixed = X @ COEFFS_STOCK + INTERCEPT_STOCK
    r_fixed_all, rmse_fixed_all = R_RMSE(pred_fixed, y)
    r_fixed_rigid, _ = R_RMSE(pred_fixed[~is_flex], y[~is_flex])
    r_fixed_flex,  _ = R_RMSE(pred_fixed[is_flex],  y[is_flex])

    # 2. REFIT-IN — OLS on all 81, report in-sample
    coefs_refit, icept_refit = fit_ols(X, y)
    pred_refit = X @ coefs_refit + icept_refit
    r_refit_in_all,   rmse_refit_in_all   = R_RMSE(pred_refit, y)
    r_refit_in_rigid, _ = R_RMSE(pred_refit[~is_flex], y[~is_flex])
    r_refit_in_flex,  _ = R_RMSE(pred_refit[is_flex],  y[is_flex])

    # 3. REFIT-CV — 4-fold × 10 repeats
    cv = kfold_cv_R(X, y, k=4, n_repeats=10, seed=0)
    preds_cv = cv["preds_mean"]
    r_cv_rigid, _ = R_RMSE(preds_cv[~is_flex], y[~is_flex])
    r_cv_flex,  _ = R_RMSE(preds_cv[is_flex],  y[is_flex])

    # Paper-style report
    return {
        "mode": mode, "N": len(y),
        "n_rigid": n_rigid, "n_flex": n_flex,
        "stock_coefs_rmse_kcal": float(
            np.sqrt(np.mean((np.array([r["dg_pred_boltz"] for r in rows]) - y) ** 2))
        ),
        "fixed": {
            "R_all": r_fixed_all, "RMSE_all": rmse_fixed_all,
            "R_rigid": r_fixed_rigid, "R_flex": r_fixed_flex,
        },
        "refit_in": {
            "R_all": r_refit_in_all, "RMSE_all": rmse_refit_in_all,
            "R_rigid": r_refit_in_rigid, "R_flex": r_refit_in_flex,
            "coefs": coefs_refit.tolist(), "intercept": icept_refit,
        },
        "refit_cv": {
            "R_mean": cv["R_mean"], "R_std": cv["R_std"],
            "RMSE_mean": cv["RMSE_mean"], "RMSE_std": cv["RMSE_std"],
            "R_rigid": r_cv_rigid, "R_flex": r_cv_flex,
        },
        "y": y, "preds_cv": preds_cv, "pred_fixed": pred_fixed,
        "pred_refit": pred_refit, "is_flex": is_flex, "pdb": pdb,
    }


# --------------------------------------------------------------------------
# Report + plot
# --------------------------------------------------------------------------

def write_report(path: Path, results: list[dict], crystal_R: float,
                 crystal_RMSE: float):
    lines = [
        "# Experiment A — diagnostic refit",
        "",
        f"Paper baseline (crystal, `ba_val`): R = {crystal_R:.3f},  "
        f"RMSE = {crystal_RMSE:.2f} kcal/mol",
        "Paper published on crystal (4-fold CV × 10 repeats): R = −0.73, "
        "RMSE = 1.89",
        "",
        f"Flexibility stratum: iRMSD > {IRMSD_CUTOFF} Å (paper convention).",
        "",
    ]
    for res in results:
        lines += [
            f"## mode = {res['mode']}   N = {res['N']}   "
            f"(rigid={res['n_rigid']},  flex={res['n_flex']})",
            "",
            "| Coefficient policy | R (all) | RMSE (all) | R (rigid) | R (flex) |",
            "|---|---:|---:|---:|---:|",
        ]
        for label, key in (("FIXED (stock coefs)", "fixed"),
                            ("REFIT in-sample",    "refit_in"),
                            ("REFIT 4-fold CV ×10","refit_cv")):
            d = res[key]
            r_all = d.get("R_all", d.get("R_mean"))
            rmse_all = d.get("RMSE_all", d.get("RMSE_mean"))
            r_r = d["R_rigid"]
            r_f = d["R_flex"]
            extra = ""
            if key == "refit_cv":
                extra = f" ± {d['R_std']:.03f}"
            lines.append(
                f"| {label} | {r_all:+.3f}{extra} | "
                f"{rmse_all:.2f} | {r_r:+.3f} | {r_f:+.3f} |"
            )
        # Refit coefs table
        coefs = res["refit_in"]["coefs"]
        icept = res["refit_in"]["intercept"]
        lines += [
            "",
            "### Refit coefficients (vs stock)",
            "",
            "| feature | stock | refit | Δ |",
            "|---|---:|---:|---:|",
        ]
        for name, stock, new in zip(FEATURE_ORDER, COEFFS_STOCK, coefs):
            lines.append(f"| {name} | {stock:+.5f} | {new:+.5f} | "
                         f"{new - stock:+.5f} |")
        lines.append(
            f"| intercept | {INTERCEPT_STOCK:+.3f} | {icept:+.3f} | "
            f"{icept - INTERCEPT_STOCK:+.3f} |"
        )
        lines.append("")
    lines += [
        "## Verdict",
        "",
        "- **If REFIT in-sample R ≫ FIXED R**: features carry signal, just"
        " mis-calibrated — experiment B (augmented selection) might help.",
        "- **If REFIT CV R ≈ FIXED R**: features themselves lost the signal"
        " on Boltz — must add new features (ipTM, mean PAE, pLDDT).",
        "- **If REFIT in-sample ≫ REFIT CV**: overfit warning; refit is memorising"
        " noise, can't generalise.",
    ]
    path.write_text("\n".join(lines) + "\n")


def plot_scatters(results: list[dict], out_dir: Path):
    for res in results:
        y = res["y"]
        pred_fixed = res["pred_fixed"]
        pred_refit_cv = res["preds_cv"]
        is_flex = res["is_flex"]
        fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharex=True,
                                  sharey=True)
        for ax, pred, title in zip(
            axes,
            (pred_fixed, pred_refit_cv),
            ("FIXED stock coefs", "REFIT 4-fold CV (mean over 10 repeats)"),
        ):
            r, rmse = R_RMSE(pred, y)
            ax.scatter(y[~is_flex], pred[~is_flex], alpha=0.6, s=22,
                       color="C0", label=f"rigid (n={(~is_flex).sum()})")
            ax.scatter(y[is_flex], pred[is_flex], alpha=0.7, s=28,
                       color="C3", marker="^",
                       label=f"flex (n={is_flex.sum()})")
            lo = min(y.min(), pred.min()) - 1
            hi = max(y.max(), pred.max()) + 1
            ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
            ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
            ax.set_title(f"{title}\nR={r:.2f}  RMSE={rmse:.2f}  ({res['mode']})")
            ax.set_xlabel("ΔG_exp (kcal/mol)")
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
        axes[0].set_ylabel("ΔG_pred (kcal/mol)")
        fig.tight_layout()
        fig.savefig(out_dir / f"refit_scatter_{res['mode']}.png", dpi=120)
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
               else BOLTZ_ROOT / "pae_calibration" / "diagnostic_refit")
    out_dir.mkdir(parents=True, exist_ok=True)

    irmsd_map = load_irmsd()

    # Crystal reference — `ba_val` from dataset.json vs DG (across all 81).
    truth = json.loads(DATASET_JSON.read_text())
    pdbs = sorted(truth)
    dg_exp_all = np.array([float(truth[p]["DG"]) for p in pdbs])
    ba_val_all = np.array([float(truth[p]["ba_val"]) for p in pdbs])
    crystal_R, crystal_RMSE = R_RMSE(ba_val_all, dg_exp_all)
    print(f"[ref] crystal (ba_val vs DG)  R={crystal_R:+.3f}  "
          f"RMSE={crystal_RMSE:.2f}")

    modes = (["msa_only", "template_msa"] if args.mode == "both"
             else [args.mode])
    results = []
    for mode in modes:
        print(f"\n=== mode = {mode} ===")
        res = evaluate_mode(mode, irmsd_map)
        results.append(res)

        f = res["fixed"]
        r = res["refit_in"]
        c = res["refit_cv"]
        print(f"  N={res['N']}  rigid={res['n_rigid']}  flex={res['n_flex']}")
        print(f"  FIXED          R_all={f['R_all']:+.3f} "
              f"RMSE={f['RMSE_all']:.2f}   "
              f"R_rigid={f['R_rigid']:+.3f}  R_flex={f['R_flex']:+.3f}")
        print(f"  REFIT in-sample R_all={r['R_all']:+.3f} "
              f"RMSE={r['RMSE_all']:.2f}   "
              f"R_rigid={r['R_rigid']:+.3f}  R_flex={r['R_flex']:+.3f}")
        print(f"  REFIT 4-fold CV R_all={c['R_mean']:+.3f}±{c['R_std']:.3f} "
              f"RMSE={c['RMSE_mean']:.2f}±{c['RMSE_std']:.2f}   "
              f"R_rigid={c['R_rigid']:+.3f}  R_flex={c['R_flex']:+.3f}")

    write_report(out_dir / "report.md", results, crystal_R, crystal_RMSE)
    plot_scatters(results, out_dir)
    print(f"\n[done] {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
