#!/usr/bin/env python3
"""3-panel Boltz evaluation plot (Kastritis 81 or Vreven BM5.5).

Panel 1  Confidence vs truth:
    X = TM-score (Boltz-predicted CIF  vs  crystal PDB; USalign, truth)
    Y = ipTM (from Boltz-2's confidence JSON)
    Series: Boltz mode (msa_only / template+msa)

Panel 2  Standard PRODIGY  --  crystal baseline vs Boltz structures:
    X = dG_exp (experimental, from dataset.json 'DG')
    Y = dG via standard PRODIGY (IC-NIS) on the given structure
    Series:
      * crystal                  PRODIGY on crystal PDB  (dataset.json 'ba_val', literature)
      * Boltz msa_only           PRODIGY on Boltz structure (computed by step 5b, tinygrad)
      * Boltz template+msa       PRODIGY on Boltz structure (computed by step 5b, tinygrad)

Panel 3  PAE-aware PRODIGY  (placeholder):
    Same axes as panel 2, but Y is PAE-gated PRODIGY on the Boltz structure,
    consuming pae_input_model_0.npz per complex. Implemented in Phase 2 of
    docs/PAE.md -- not yet filled in.

Inputs:
  benchmarks/output/kastritis_81_boltz/tm_scores.csv       (step 5)
  benchmarks/output/kastritis_81_boltz/prodigy_scores.csv  (step 5b)

Outputs:
  benchmarks/output/kastritis_81_boltz/boltz_eval.png
  benchmarks/output/kastritis_81_boltz/boltz_eval.pdf
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).parent))
from dataset_registry import AVAILABLE, get_paths  # noqa: E402

# Global font bump -- prior version was squished.
plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   14,
    "axes.labelsize":   12,
    "legend.fontsize":  10,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "figure.titlesize": 15,
})

ROOT = Path(__file__).resolve().parents[3]

MODE_STYLES = {
    "msa_only":     {"color": "#1f77b4", "marker": "o", "label": "Boltz msa_only"},
    "template_msa": {"color": "#d62728", "marker": "^", "label": "Boltz template+msa"},
}
CRYSTAL_STYLE = {"color": "#555555", "marker": "s", "label": "crystal"}

# Ultra-explicit provenance — "where does the Y value actually come from?"
DG_LABELS = {
    "crystal":      "crystal PDB  [ΔG from dataset.json 'ba_val']",
    "msa_only":     "Boltz msa_only  [ΔG recomputed on predicted CIF]",
    "template_msa": "Boltz template+msa  [ΔG recomputed on predicted CIF]",
}


def load_rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open()))


def safe_float(x: str | None) -> float | None:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except ValueError:
        return None


def panel_iptm_vs_tm(ax, tm_rows: list[dict]) -> None:
    ax.plot([0, 1], [0, 1], ls="--", color="grey", lw=1.0, alpha=0.5, label="y = x")
    ax.axhline(0.6, ls=":", color="grey", lw=0.7, alpha=0.4)
    ax.axvline(0.5, ls=":", color="grey", lw=0.7, alpha=0.4)

    for mode, style in MODE_STYLES.items():
        pts = [
            (safe_float(r["tm_ref_crystal"]), safe_float(r["iptm"]))
            for r in tm_rows if r["mode"] == mode
        ]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        rho, _ = spearmanr(xs, ys) if len(xs) >= 3 else (float("nan"), 1.0)
        ax.scatter(xs, ys, c=style["color"], marker=style["marker"],
                   alpha=0.75, s=60, edgecolors="white", linewidths=0.6,
                   label=f"{style['label']}   n={len(xs)}  ρ={rho:.2f}")

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("TM-score:  Boltz prediction  vs  crystal PDB\n(truth, via USalign)")
    ax.set_ylabel("ipTM\n(from Boltz-2 confidence JSON)")
    ax.set_title("1. Boltz structural confidence vs truth")
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def _scatter_series(ax, xs, ys, style, label_prefix: str) -> tuple[float, float, float, int]:
    """Add one series to a dG scatter; return (R, rho, RMSE, n) for the legend."""
    r_p, _ = pearsonr(xs, ys) if len(xs) >= 3 else (float("nan"), 1.0)
    rho, _ = spearmanr(xs, ys) if len(xs) >= 3 else (float("nan"), 1.0)
    rmse = float(np.sqrt(np.mean((np.array(xs) - np.array(ys)) ** 2)))
    ax.scatter(
        xs, ys,
        c=style["color"], marker=style["marker"],
        alpha=0.75, s=60, edgecolors="white", linewidths=0.6,
        label=f"{label_prefix}\n    n={len(xs)}   R={r_p:.2f}   RMSE={rmse:.2f} kcal/mol",
    )
    return r_p, rho, rmse, len(xs)


def panel_dg_scatter(
    ax,
    prodigy_rows: list[dict],
    y_col: str,
    title: str,
    ylabel: str,
    include_crystal_baseline: bool = True,
) -> None:
    """dG-pred (y) vs dG-exp (x) scatter. Three series when
    include_crystal_baseline: crystal + 2 Boltz modes.
    """
    all_dg: list[float] = []

    if include_crystal_baseline:
        # PRODIGY on crystal -- from dataset.json's ba_val, replicated per mode
        # in prodigy_rows; dedupe by pdb_id.
        seen: set[str] = set()
        pts = []
        for r in prodigy_rows:
            if r["pdb_id"] in seen:
                continue
            x, y = safe_float(r.get("dg_exp")), safe_float(r.get("dg_prodigy_baseline"))
            if x is not None and y is not None:
                pts.append((x, y))
                seen.add(r["pdb_id"])
        if pts:
            xs, ys = zip(*pts)
            _scatter_series(ax, xs, ys, CRYSTAL_STYLE, DG_LABELS["crystal"])
            all_dg.extend(xs); all_dg.extend(ys)

    for mode, style in MODE_STYLES.items():
        pts = [
            (safe_float(r["dg_exp"]), safe_float(r.get(y_col)))
            for r in prodigy_rows if r["mode"] == mode
        ]
        pts = [(x, y) for x, y in pts if x is not None and y is not None]
        if not pts:
            continue
        xs, ys = zip(*pts)
        _scatter_series(ax, xs, ys, style, DG_LABELS[mode])
        all_dg.extend(xs); all_dg.extend(ys)

    if all_dg:
        lo, hi = min(all_dg), max(all_dg)
        pad = 0.05 * (hi - lo)
        lo -= pad; hi += pad
    else:
        lo, hi = -20.0, -3.0
    ax.plot([lo, hi], [lo, hi], ls="--", color="grey", lw=0.8, alpha=0.5, label="y = x")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel("ΔG experimental  (kcal/mol)\n[from dataset.json 'DG',  truth]")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(loc="upper left", frameon=True)
    ax.grid(True, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")


def panel_placeholder(ax) -> None:
    ax.text(
        0.5, 0.5,
        "ΔG  PAE-aware PRODIGY\n"
        "on Boltz-predicted CIF\n\n"
        "TODO — Phase 2 of docs/PAE.md\n"
        "(contacts_pae.py gated by\n"
        "pae_input_model_0.npz per complex)",
        ha="center", va="center", fontsize=12, color="#555",
        transform=ax.transAxes,
    )
    ax.set_xlabel("ΔG experimental  (kcal/mol)\n[from dataset.json 'DG',  truth]")
    ax.set_ylabel("ΔG  PAE-aware PRODIGY on Boltz CIF  (kcal/mol)\n[to be computed — Phase 2]")
    ax.set_title("3. PAE-aware PRODIGY vs truth  (placeholder)")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linestyle((0, (4, 4)))
        spine.set_color("#999")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    args = ap.parse_args()
    paths = get_paths(args.dataset)
    out_root = paths.output_root
    tm_csv = out_root / "tm_scores.csv"
    prodigy_csv = out_root / "prodigy_scores.csv"

    if not tm_csv.exists():
        print(f"[fatal] missing {tm_csv}. Run 05_mmalign_tm.py first."); return 2
    if not prodigy_csv.exists():
        print(f"[fatal] missing {prodigy_csv}. Run 05b_prodigy_on_boltz.py first."); return 2

    tm_rows = load_rows(tm_csv)
    prodigy_rows = load_rows(prodigy_csv)

    fig, axes = plt.subplots(1, 3, figsize=(24, 9))
    panel_iptm_vs_tm(axes[0], tm_rows)
    panel_dg_scatter(
        axes[1],
        prodigy_rows,
        y_col="dg_pred_boltz",
        title="2. Standard PRODIGY:  crystal baseline  vs  Boltz-predicted",
        ylabel="ΔG standard PRODIGY  (kcal/mol)\n[see legend for which structure each point is scored on]",
        include_crystal_baseline=paths.has_prodigy_baseline,
    )
    panel_placeholder(axes[2])

    fig.suptitle(
        f"Boltz-2 on {paths.display}  —  structure (panel 1)  &  affinity (panels 2, 3)",
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97), w_pad=3.0)
    png = out_root / "boltz_eval.png"
    pdf = out_root / "boltz_eval.pdf"
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[{paths.display}]")
    print(f"Wrote {png}")
    print(f"Wrote {pdf}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
