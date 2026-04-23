#!/usr/bin/env python3
"""Stratify the PAE calibration grid by iRMSD (rigid / medium / flexible).

Reads the ``calib_grid.csv`` produced by ``quick_pae_calib.py`` and
``dataset.json`` (for ``iRMSD``). Expresses the suspicion from the v2 run —
that the 62-complex rigid subset washes out any PAE signal localised to the
19 semi-/fully-flexible complexes.

Strata (standard docking-benchmark thresholds):
    rigid     iRMSD < 1.5 Å
    medium    1.5 ≤ iRMSD < 2.2 Å
    flexible  iRMSD ≥ 2.2 Å

Usage:
    python stratify_pae_calib.py --mode msa_only
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
BOLTZ_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"

IRMSD_BINS = [
    ("rigid",    (0.0, 1.5)),
    ("medium",   (1.5, 2.2)),
    ("flexible", (2.2, 10.0)),
]


def load_irmsd() -> dict[str, float]:
    d = json.loads(DATASET_JSON.read_text())
    return {k: float(v["iRMSD"]) for k, v in d.items()}


def classify(irmsd: float) -> str:
    for name, (lo, hi) in IRMSD_BINS:
        if lo <= irmsd < hi:
            return name
    return "flexible"


def metrics_surface(df: pd.DataFrame, pred_col: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (alpha_grid, d_cut_grid, R, RMSE) from a df of a single stratum."""
    alphas = np.sort(df["alpha"].unique())
    d_cuts = np.sort(df["d_cut"].unique())
    R = np.zeros((len(alphas), len(d_cuts)))
    RMSE = np.zeros((len(alphas), len(d_cuts)))
    for ai, a in enumerate(alphas):
        for di, d in enumerate(d_cuts):
            g = df[(df["alpha"] == a) & (df["d_cut"] == d)]
            if len(g) < 3 or g[pred_col].std() == 0:
                R[ai, di] = np.nan; RMSE[ai, di] = np.nan
                continue
            R[ai, di] = pearsonr(g[pred_col], g["dg_exp"])[0]
            RMSE[ai, di] = float(np.sqrt(((g[pred_col] - g["dg_exp"]) ** 2).mean()))
    return alphas, d_cuts, R, RMSE


def _heatmap(ax, data, alphas, d_cuts, title, cbar_label, cmap="viridis",
             mark_max=False):
    im = ax.pcolormesh(alphas, d_cuts, data.T, cmap=cmap, shading="auto")
    ax.set_xlabel("α"); ax.set_ylabel("d_cut (Å)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(cbar_label, fontsize=9)
    if mark_max and not np.all(np.isnan(data)):
        ai, di = np.unravel_index(np.nanargmax(data), data.shape)
        ax.plot(alphas[ai], d_cuts[di], "r+", ms=14, mew=2,
                label=f"max @ α={alphas[ai]:.2f}, d={d_cuts[di]:.2f}")
        ax.legend(fontsize=8, loc="upper right")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    in_dir = BOLTZ_ROOT / "pae_calibration" / args.mode
    csv = in_dir / "calib_grid.csv"
    if not csv.exists():
        raise SystemExit(
            f"{csv.relative_to(ROOT)} not found — run quick_pae_calib.py first"
        )
    out_dir = Path(args.out_dir) if args.out_dir else in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv)
    irmsd = load_irmsd()
    df["irmsd"] = df["pdb_id"].map(irmsd)
    df["stratum"] = df["irmsd"].map(classify)
    counts = df.drop_duplicates("pdb_id").groupby("stratum").size()
    print(f"[strata] N per stratum: {counts.to_dict()}")

    # Full vs per-stratum surfaces for both coef policies.
    strata_order = ["rigid", "medium", "flexible"]
    surfaces = {"all": metrics_surface(df, "dg_pred_b1_fixed")}
    for s in strata_order:
        surfaces[s] = metrics_surface(df[df["stratum"] == s], "dg_pred_b1_fixed")

    # Report best per stratum.
    lines = [
        f"# PAE calibration stratified by iRMSD — mode={args.mode}",
        "",
        f"Strata (N): rigid={counts.get('rigid', 0)}, "
        f"medium={counts.get('medium', 0)}, flexible={counts.get('flexible', 0)}",
        "",
        "## Best (α\\*, d_cut\\*) by stratum — B1 fixed coeffs",
        "",
        "| Stratum | N | best α | best d_cut | R | RMSE | R @ stock (α=0, d=5.5) | ΔR vs stock |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ["all"] + strata_order:
        alphas, d_cuts, R, RMSE = surfaces[name]
        if np.all(np.isnan(R)):
            continue
        ai, di = np.unravel_index(np.nanargmax(R), R.shape)
        a_s = int(np.argmin(np.abs(alphas - 0.0)))
        d_s = int(np.argmin(np.abs(d_cuts - 5.5)))
        R_stock = R[a_s, d_s]
        R_best = R[ai, di]
        dR = R_best - R_stock
        n_rows = (counts.get(name, counts.sum()) if name != "all"
                  else int(counts.sum()))
        lines.append(
            f"| {name} | {n_rows} | {alphas[ai]:.2f} | {d_cuts[di]:.2f} | "
            f"{R_best:.3f} | {RMSE[ai, di]:.2f} | {R_stock:.3f} | "
            f"{dR:+.3f} |"
        )

    summary_path = out_dir / "stratified_summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"[write] {summary_path.relative_to(ROOT)}")

    # Heatmap grid: rows = strata, cols = (R, RMSE) for B1.
    fig, axes = plt.subplots(len(strata_order) + 1, 2, figsize=(11, 14))
    for row, name in enumerate(["all"] + strata_order):
        alphas, d_cuts, R, RMSE = surfaces[name]
        n_label = (int(counts.get(name, counts.sum())) if name != "all"
                   else int(counts.sum()))
        _heatmap(axes[row, 0], R, alphas, d_cuts,
                 f"{name} (N={n_label}) — Pearson R",
                 "R", cmap="viridis", mark_max=True)
        _heatmap(axes[row, 1], RMSE, alphas, d_cuts,
                 f"{name} (N={n_label}) — RMSE (kcal/mol)",
                 "RMSE", cmap="viridis_r", mark_max=False)
    fig.tight_layout()
    heat_path = out_dir / "stratified_heatmaps.png"
    fig.savefig(heat_path, dpi=120)
    plt.close(fig)
    print(f"[write] {heat_path.relative_to(ROOT)}")

    # Focused 1-D line plot: R vs α at d_cut=5.5 for each stratum — the
    # clean visual for "does PAE help on flexible?"
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    for name in ["all"] + strata_order:
        alphas, d_cuts, R, _ = surfaces[name]
        d55_i = int(np.argmin(np.abs(d_cuts - 5.5)))
        n_label = (int(counts.get(name, counts.sum())) if name != "all"
                   else int(counts.sum()))
        ax.plot(alphas, R[:, d55_i], "o-",
                label=f"{name} (N={n_label})")
    ax.axvline(0, color="gray", lw=0.8, alpha=0.5)
    ax.set_xlabel("α  (linear PAE coefficient)")
    ax.set_ylabel("Pearson R (B1 fixed, d_cut=5.5 Å)")
    ax.set_title(f"PAE calibration by stratum — {args.mode}")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout()
    curve_path = out_dir / "stratified_R_curves.png"
    fig.savefig(curve_path, dpi=120)
    plt.close(fig)
    print(f"[write] {curve_path.relative_to(ROOT)}")

    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
