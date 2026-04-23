#!/usr/bin/env python3
"""Threshold-τ PAE gate calibration (complement to ``quick_pae_calib.py``).

Alternative parametrisation to the linear-α additive gate:

    contact_ij  =  1( min_heavy_atom_dist_ij ≤ d_cut )  ∧  ( PAE_ij ≤ τ )

τ → ∞  recovers stock PRODIGY.
τ  < ∞ drops contacts with high PAE regardless of distance — useful when
       the linear gate's distance inflation is the problem.

Fixed d_cut = 5.5 Å (no inflation), sweep τ over a decaying grid.

Reuses loaders / IC classification / bootstrap / stratification helpers from
``quick_pae_calib`` so this file stays small and consistent.

Outputs (under ``pae_calibration/<mode>/threshold/``):
    threshold_grid.csv        (pdb, tau, ic_cc..ic_pa, dg_pred_*)
    stage_B_curve.png         Pearson R + RMSE vs τ  (B1 fixed, B2 LOO)
    stratified_R_curve.png    R vs τ, rigid / medium / flexible strata
    scatter_best.png          3-panel crystal | stock | PAE-best
    summary.md                table with stratified best-τ + perm p
    comparison_alpha_tau.png  side-by-side: linear-α best vs threshold-τ best

Usage:
    python threshold_pae_calib.py --mode msa_only
    python threshold_pae_calib.py --mode msa_only --tau-grid 30,20,15,12,10,8,6,5,4,3
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).parent))

from quick_pae_calib import (  # noqa: E402
    BOLTZ_ROOT, COEFFS, D_CUT_DEFAULT, INTERCEPT, IRMSD_BINS_DEFAULT,
    Complex, bootstrap_R_RMSE, classify_ic, classify_stratum,
    load_complex, load_dataset_truth, load_stock_prodigy,
)

DEFAULT_TAU_GRID = "30,20,15,12,10,8,6,5,4,3"


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------

def contacts_threshold(min_dist: np.ndarray, pae_ab: np.ndarray,
                       tau: float, d_cut: float) -> np.ndarray:
    """Contact mask with PAE threshold gate (independent of distance)."""
    return (min_dist <= d_cut) & (pae_ab <= tau)


def sweep_ic_tau(complexes: list[Complex], tau_grid: np.ndarray,
                 d_cut: float, pae_override: list[np.ndarray] | None = None
                 ) -> np.ndarray:
    """[N, T, 4] IC counts sweeping τ with fixed d_cut."""
    ic = np.zeros((len(complexes), len(tau_grid), 4), dtype=np.int32)
    for ci, comp in enumerate(complexes):
        pae = pae_override[ci] if pae_override is not None else comp.pae_ab
        geom_ok = comp.min_dist <= d_cut          # [N_t, N_b]
        # Precompute PAE threshold masks: [T, N_t, N_b]
        for ti, tau in enumerate(tau_grid):
            contact = geom_ok & (pae <= float(tau))
            ic[ci, ti] = classify_ic(contact, comp.char_t, comp.char_b)
    return ic


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------

def _features_at_tau(complexes: list[Complex], ic_tau: np.ndarray) -> np.ndarray:
    nis = np.array([[c.nis_a, c.nis_c] for c in complexes], dtype=np.float64)
    return np.concatenate([ic_tau.astype(np.float64),
                           np.clip(nis, 0, 100)], axis=1)


def _metrics_1d(dg_pred: np.ndarray, dg_exp: np.ndarray) -> dict:
    T = dg_pred.shape[1]
    R = np.zeros(T); RMSE = np.zeros(T); MAE = np.zeros(T)
    for ti in range(T):
        x = dg_pred[:, ti]
        R[ti] = (pearsonr(x, dg_exp)[0] if np.std(x) > 0 else np.nan)
        RMSE[ti] = float(np.sqrt(np.mean((x - dg_exp) ** 2)))
        MAE[ti] = float(np.mean(np.abs(x - dg_exp)))
    return {"dg_pred": dg_pred, "dg_exp": dg_exp, "R": R, "RMSE": RMSE, "MAE": MAE}


def stage_b_fixed(complexes: list[Complex], ic_sweep: np.ndarray) -> dict:
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    N, T, _ = ic_sweep.shape
    dg_pred = np.zeros((N, T), dtype=np.float64)
    for ti in range(T):
        X = _features_at_tau(complexes, ic_sweep[:, ti])
        dg_pred[:, ti] = X @ COEFFS + INTERCEPT
    return _metrics_1d(dg_pred, dg_exp)


def stage_b_refit_loo(complexes: list[Complex], ic_sweep: np.ndarray) -> dict:
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    N, T, _ = ic_sweep.shape
    dg_pred = np.zeros((N, T), dtype=np.float64)
    for ti in range(T):
        X = _features_at_tau(complexes, ic_sweep[:, ti])
        X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)
        for i in range(N):
            mask = np.arange(N) != i
            beta, *_ = np.linalg.lstsq(X_aug[mask], dg_exp[mask], rcond=None)
            dg_pred[i, ti] = X_aug[i] @ beta
    return _metrics_1d(dg_pred, dg_exp)


def stage_a_loss(complexes: list[Complex], ic_sweep: np.ndarray) -> np.ndarray:
    """[T, 5] — (total_MSE, cc, ca, pp, pa) MSE over complexes."""
    crystal = np.array([
        [c.ic_cc_crystal, c.ic_ca_crystal, c.ic_pp_crystal, c.ic_pa_crystal]
        for c in complexes
    ], dtype=np.int32)
    res = ic_sweep - crystal[:, None, :]
    per_chan = np.mean(res ** 2, axis=0)
    total = per_chan.sum(axis=-1)
    return np.concatenate([total[:, None], per_chan], axis=-1)


def permutation_null(complexes: list[Complex], tau_grid: np.ndarray,
                     d_cut: float, n_perm: int = 50, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    T = len(tau_grid)
    perm_R = np.zeros((n_perm, T))
    for pi in range(n_perm):
        shuffled = [
            comp.pae_ab.flat[rng.permutation(comp.pae_ab.size)
                             ].reshape(comp.pae_ab.shape)
            for comp in complexes
        ]
        ic_perm = sweep_ic_tau(complexes, tau_grid, d_cut,
                               pae_override=shuffled)
        for ti in range(T):
            X = _features_at_tau(complexes, ic_perm[:, ti])
            dg_pred = X @ COEFFS + INTERCEPT
            perm_R[pi, ti] = (pearsonr(dg_pred, dg_exp)[0]
                              if np.std(dg_pred) > 0 else np.nan)
    return {"perm_R": perm_R}


# --------------------------------------------------------------------------
# Stratification
# --------------------------------------------------------------------------

def stratified_R(complexes: list[Complex], dg_pred: np.ndarray,
                 dg_exp: np.ndarray) -> dict[str, np.ndarray]:
    """Return {stratum: [T]} Pearson R arrays, masking strata with <3 items."""
    strata = np.array([classify_stratum(c.irmsd) for c in complexes])
    out: dict[str, np.ndarray] = {}
    T = dg_pred.shape[1]
    for name, _ in IRMSD_BINS_DEFAULT:
        idx = np.where(strata == name)[0]
        if len(idx) < 3:
            out[name] = np.full(T, np.nan)
            continue
        R = np.zeros(T)
        for ti in range(T):
            x = dg_pred[idx, ti]
            R[ti] = (pearsonr(x, dg_exp[idx])[0]
                     if np.std(x) > 0 else np.nan)
        out[name] = R
    return out


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_stage_b_curve(b1: dict, b2: dict, tau_grid: np.ndarray,
                       crystal_ref: dict, stock_ref: dict, path: Path):
    fig, (ax_R, ax_E) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax_R.plot(tau_grid, b1["R"], "o-", color="C0", label="B1 fixed coeffs")
    ax_R.plot(tau_grid, b2["R"], "s-", color="C2", label="B2 LOO-refit")
    ax_R.axhline(crystal_ref["R"], color="green", ls="--", alpha=0.7,
                 label=f"crystal {crystal_ref['R']:.2f}")
    ax_R.axhline(stock_ref["R"], color="orange", ls="--", alpha=0.7,
                 label=f"Boltz-stock {stock_ref['R']:.2f}")
    ax_R.invert_xaxis()
    ax_R.set_xlabel("τ  (PAE threshold, Å)")
    ax_R.set_ylabel("Pearson R")
    ax_R.set_title("Stage B · R vs τ   (d_cut = 5.5 Å fixed)")
    ax_R.legend(fontsize=8); ax_R.grid(alpha=0.3)

    ax_E.plot(tau_grid, b1["RMSE"], "o-", color="C0", label="B1 fixed coeffs")
    ax_E.plot(tau_grid, b2["RMSE"], "s-", color="C2", label="B2 LOO-refit")
    ax_E.axhline(crystal_ref["RMSE"], color="green", ls="--", alpha=0.7,
                 label=f"crystal {crystal_ref['RMSE']:.2f}")
    ax_E.axhline(stock_ref["RMSE"], color="orange", ls="--", alpha=0.7,
                 label=f"Boltz-stock {stock_ref['RMSE']:.2f}")
    ax_E.invert_xaxis()
    ax_E.set_xlabel("τ  (PAE threshold, Å)")
    ax_E.set_ylabel("RMSE (kcal/mol)")
    ax_E.set_title("Stage B · RMSE vs τ")
    ax_E.legend(fontsize=8); ax_E.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_stratified(strata_R: dict[str, np.ndarray], tau_grid: np.ndarray,
                    n_per_stratum: dict[str, int], path: Path):
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ("rigid", "medium", "flexible"):
        R = strata_R.get(name)
        n = n_per_stratum.get(name, 0)
        if R is None or np.all(np.isnan(R)):
            continue
        ax.plot(tau_grid, R, "o-", label=f"{name} (N={n})")
    ax.invert_xaxis()
    ax.set_xlabel("τ  (PAE threshold, Å)")
    ax.set_ylabel("Pearson R (B1 fixed)")
    ax.set_title("Stage B stratified by iRMSD")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_scatter_best(complexes: list[Complex], b1: dict, best_ti: int,
                      tau_grid: np.ndarray, path: Path):
    dg_exp = np.array([c.dg_exp for c in complexes])
    dg_crystal = np.array([c.ba_val for c in complexes])
    dg_stock = np.array([c.ic_stock.get("dg_pred_boltz", np.nan)
                         for c in complexes])
    dg_pae = b1["dg_pred"][:, best_ti]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, pred, ttl in zip(
        axes, (dg_crystal, dg_stock, dg_pae),
        ("Crystal (ba_val)", "Boltz stock (τ=∞)",
         f"Boltz + PAE threshold (τ={tau_grid[best_ti]:.1f} Å)"),
    ):
        valid = ~np.isnan(pred)
        r = pearsonr(pred[valid], dg_exp[valid])[0]
        rmse = np.sqrt(np.mean((pred[valid] - dg_exp[valid]) ** 2))
        ax.scatter(dg_exp[valid], pred[valid], alpha=0.6, s=22)
        lo = min(dg_exp.min(), np.nanmin(pred)) - 1
        hi = max(dg_exp.max(), np.nanmax(pred)) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{ttl}\nR={r:.2f}  RMSE={rmse:.2f}")
        ax.set_xlabel("ΔG_exp (kcal/mol)"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("ΔG_pred (kcal/mol)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_comparison(complexes: list[Complex], b1: dict, best_ti: int,
                    tau_grid: np.ndarray, mode: str, path: Path):
    """One-figure side-by-side: threshold-τ sweep + overlay of linear-α best."""
    linear_csv = BOLTZ_ROOT / "pae_calibration" / mode / "calib_grid.csv"
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(tau_grid, b1["R"], "o-", color="C0",
            label="threshold-τ (this run)")
    if linear_csv.exists():
        import pandas as pd
        df = pd.read_csv(linear_csv)
        agg = df.groupby(["alpha", "d_cut"]).apply(
            lambda g: (pearsonr(g["dg_pred_b1_fixed"], g["dg_exp"])[0]
                       if g["dg_pred_b1_fixed"].std() > 0 else np.nan),
            include_groups=False,
        )
        linear_R_max = float(np.nanmax(agg.values))
        stock_R = float(agg.loc[(0.0, 5.5)])
        ax.axhline(linear_R_max, color="C3", ls=":", lw=2,
                   label=f"linear-α best (R={linear_R_max:.3f})")
        ax.axhline(stock_R, color="orange", ls="--", alpha=0.7,
                   label=f"Boltz stock (R={stock_R:.3f})")
    ax.invert_xaxis()
    ax.set_xlabel("τ  (PAE threshold, Å)  ← tighter")
    ax.set_ylabel("Pearson R (B1 fixed)")
    ax.set_title(f"Parametrisation comparison — {mode}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_grid_csv(path: Path, complexes: list[Complex],
                   ic_sweep: np.ndarray, tau_grid: np.ndarray,
                   b1: dict, b2: dict):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb_id", "mode", "tau", "ic_cc", "ic_ca", "ic_pp", "ic_pa",
            "dg_pred_b1_fixed", "dg_pred_b2_loo", "dg_exp", "ba_val_crystal",
            "irmsd", "stratum",
        ])
        for ci, comp in enumerate(complexes):
            for ti, tau in enumerate(tau_grid):
                ic = ic_sweep[ci, ti]
                w.writerow([
                    comp.pdb_id, comp.mode, f"{tau:.3f}",
                    int(ic[0]), int(ic[1]), int(ic[2]), int(ic[3]),
                    f"{b1['dg_pred'][ci, ti]:.3f}",
                    f"{b2['dg_pred'][ci, ti]:.3f}",
                    f"{comp.dg_exp:.3f}", f"{comp.ba_val:.3f}",
                    f"{comp.irmsd:.3f}", classify_stratum(comp.irmsd),
                ])


def write_summary(path: Path, complexes: list[Complex],
                  tau_grid: np.ndarray, b1: dict, b2: dict,
                  best_ti_b1: int, best_ti_b2: int,
                  R_CI_b1, RMSE_CI_b1, R_CI_b2, RMSE_CI_b2,
                  perm_p: float,
                  crystal_ref: dict, stock_ref: dict,
                  strata_R: dict, n_per_stratum: dict):
    def _ci(c): return f"[{c[0]:.2f}, {c[1]:.2f}]"
    lines = [
        f"# Threshold-τ PAE calibration — mode={complexes[0].mode}, "
        f"N={len(complexes)}",
        "",
        f"τ grid (Å): {', '.join(f'{t:g}' for t in tau_grid)}",
        "d_cut: 5.5 Å (fixed)",
        "",
        "## ΔG prediction",
        "",
        "| Config | τ | R | RMSE | R 95% CI | RMSE 95% CI |",
        "|---|---:|---:|---:|---|---|",
        (f"| Crystal (`ba_val`) | — | {crystal_ref['R']:.3f} | "
         f"{crystal_ref['RMSE']:.3f} | — | — |"),
        (f"| Boltz stock (τ=∞) | ∞ | {stock_ref['R']:.3f} | "
         f"{stock_ref['RMSE']:.3f} | — | — |"),
        (f"| Boltz+PAE · B1 fixed | {tau_grid[best_ti_b1]:.1f} | "
         f"{b1['R'][best_ti_b1]:.3f} | {b1['RMSE'][best_ti_b1]:.3f} | "
         f"{_ci(R_CI_b1)} | {_ci(RMSE_CI_b1)} |"),
        (f"| Boltz+PAE · B2 LOO-refit | {tau_grid[best_ti_b2]:.1f} | "
         f"{b2['R'][best_ti_b2]:.3f} | {b2['RMSE'][best_ti_b2]:.3f} | "
         f"{_ci(R_CI_b2)} | {_ci(RMSE_CI_b2)} |"),
        "",
        "## Stratified R (B1 fixed)",
        "",
        "| stratum | N | τ=∞ | best τ | best R | ΔR vs stock |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    stock_ti = int(np.argmax(tau_grid))   # highest τ = loosest = stock-like
    for name in ("rigid", "medium", "flexible"):
        R = strata_R.get(name)
        n = n_per_stratum.get(name, 0)
        if R is None or np.all(np.isnan(R)):
            lines.append(f"| {name} | {n} | — | — | — | — |"); continue
        best_ti = int(np.nanargmax(R))
        R_stock = float(R[stock_ti])
        R_best = float(R[best_ti])
        lines.append(
            f"| {name} | {n} | {R_stock:+.3f} | "
            f"{tau_grid[best_ti]:.1f} | {R_best:+.3f} | "
            f"{R_best - R_stock:+.3f} |"
        )
    lines += [
        "",
        "## Permutation null",
        "",
        f"- p(R_null ≥ R_real at τ={tau_grid[best_ti_b1]:.1f}) = **{perm_p:.3f}**",
    ]
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--d-cut", type=float, default=D_CUT_DEFAULT)
    ap.add_argument("--tau-grid", default=DEFAULT_TAU_GRID,
                    help="comma-separated τ values in Å")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    tau_grid = np.array([float(t) for t in args.tau_grid.split(",")])
    tau_grid = np.sort(tau_grid)[::-1]     # descending (∞→tight)

    out_dir = (Path(args.out_dir) if args.out_dir
               else BOLTZ_ROOT / "pae_calibration" / args.mode / "threshold")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] mode={args.mode}  d_cut={args.d_cut}  "
          f"τ grid={list(tau_grid)}  bootstrap={args.bootstrap}  "
          f"n_perm={args.n_perm}")
    print(f"[cfg] out_dir={out_dir.relative_to(ROOT)}")

    truth = load_dataset_truth()
    stock = load_stock_prodigy()
    pdb_ids = sorted(truth.keys())
    if args.limit > 0:
        pdb_ids = pdb_ids[:args.limit]

    t0 = time.time()
    complexes = [c for c in (load_complex(p, args.mode, truth, stock)
                              for p in pdb_ids) if c is not None]
    if not complexes:
        print("[fatal] no complexes loaded"); sys.exit(1)
    print(f"[data] loaded {len(complexes)}/{len(pdb_ids)} in "
          f"{time.time() - t0:.1f}s")

    t0 = time.time()
    ic_sweep = sweep_ic_tau(complexes, tau_grid, args.d_cut)
    print(f"[sweep] {len(complexes)}×{len(tau_grid)} IC in "
          f"{time.time() - t0:.1f}s")

    b1 = stage_b_fixed(complexes, ic_sweep)
    b2 = stage_b_refit_loo(complexes, ic_sweep)
    print(f"[stageB] B1 best R={np.nanmax(b1['R']):.3f} @ "
          f"τ={tau_grid[np.nanargmax(b1['R'])]:.1f}  "
          f"B1 best RMSE={np.nanmin(b1['RMSE']):.3f} @ "
          f"τ={tau_grid[np.nanargmin(b1['RMSE'])]:.1f}")
    print(f"         B2 best R={np.nanmax(b2['R']):.3f} @ "
          f"τ={tau_grid[np.nanargmax(b2['R'])]:.1f}")

    best_ti_b1 = int(np.nanargmax(b1["R"]))
    best_ti_b2 = int(np.nanargmax(b2["R"]))
    R_CI_b1, RMSE_CI_b1 = bootstrap_R_RMSE(
        b1["dg_pred"][:, best_ti_b1], b1["dg_exp"], n=args.bootstrap)
    R_CI_b2, RMSE_CI_b2 = bootstrap_R_RMSE(
        b2["dg_pred"][:, best_ti_b2], b2["dg_exp"], n=args.bootstrap)
    print(f"[boot] B1 R CI {R_CI_b1}  RMSE CI {RMSE_CI_b1}")
    print(f"[boot] B2 R CI {R_CI_b2}  RMSE CI {RMSE_CI_b2}")

    t0 = time.time()
    perm = permutation_null(complexes, tau_grid, args.d_cut,
                             n_perm=args.n_perm, seed=1)
    null_Rs = perm["perm_R"][:, best_ti_b1]
    null_Rs = null_Rs[~np.isnan(null_Rs)]
    perm_p = (float((null_Rs >= b1["R"][best_ti_b1]).mean())
              if len(null_Rs) else float("nan"))
    print(f"[perm] {args.n_perm} in {time.time() - t0:.1f}s  "
          f"p(R_null ≥ R_real) = {perm_p:.3f}")

    # Reference numbers
    dg_exp = b1["dg_exp"]
    dg_crystal = np.array([c.ba_val for c in complexes])
    dg_stock_vec = np.array([c.ic_stock.get("dg_pred_boltz", np.nan)
                              for c in complexes])
    crystal_R = pearsonr(dg_crystal, dg_exp)[0]
    crystal_RMSE = float(np.sqrt(np.mean((dg_crystal - dg_exp) ** 2)))
    sv = ~np.isnan(dg_stock_vec)
    stock_R = pearsonr(dg_stock_vec[sv], dg_exp[sv])[0]
    stock_RMSE = float(np.sqrt(np.mean((dg_stock_vec[sv] - dg_exp[sv]) ** 2)))

    # Stratification
    strata_R = stratified_R(complexes, b1["dg_pred"], dg_exp)
    strata = [classify_stratum(c.irmsd) for c in complexes]
    n_per_stratum = {
        s: sum(1 for x in strata if x == s)
        for s in ("rigid", "medium", "flexible")
    }
    print(f"[strata] {n_per_stratum}")
    for name, R in strata_R.items():
        if np.all(np.isnan(R)):
            continue
        best_ti = int(np.nanargmax(R))
        stock_ti = int(np.argmax(tau_grid))
        print(f"         {name:<8s}  best R={R[best_ti]:+.3f} @ "
              f"τ={tau_grid[best_ti]:.1f}   stock R={R[stock_ti]:+.3f}   "
              f"ΔR={R[best_ti] - R[stock_ti]:+.3f}")

    # Plots + writers
    plot_stage_b_curve(b1, b2, tau_grid,
                       {"R": crystal_R, "RMSE": crystal_RMSE},
                       {"R": stock_R, "RMSE": stock_RMSE},
                       out_dir / "stage_B_curve.png")
    plot_stratified(strata_R, tau_grid, n_per_stratum,
                    out_dir / "stratified_R_curve.png")
    plot_scatter_best(complexes, b1, best_ti_b1, tau_grid,
                      out_dir / "scatter_best.png")
    plot_comparison(complexes, b1, best_ti_b1, tau_grid, args.mode,
                    out_dir / "comparison_alpha_tau.png")
    write_grid_csv(out_dir / "threshold_grid.csv", complexes, ic_sweep,
                   tau_grid, b1, b2)
    write_summary(out_dir / "summary.md", complexes, tau_grid, b1, b2,
                  best_ti_b1, best_ti_b2,
                  R_CI_b1, RMSE_CI_b1, R_CI_b2, RMSE_CI_b2, perm_p,
                  {"R": crystal_R, "RMSE": crystal_RMSE},
                  {"R": stock_R, "RMSE": stock_RMSE},
                  strata_R, n_per_stratum)
    print(f"[done] artefacts in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
