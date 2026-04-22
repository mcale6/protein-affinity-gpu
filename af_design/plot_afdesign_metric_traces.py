#!/usr/bin/env python3
"""Plot all AFDesign metric traces over design steps as one colored-line figure.

Reads ``trajectory.json`` + ``bsa_history.json`` + ``summary.json`` from a
run directory and produces a single PNG where every metric we optimize on
(or diagnose with) is a differently-coloured line against the step axis.
Stage boundaries (logits → soft → hard) are marked with dashed verticals.

Two y-axes:
  - Left: probability-like AF scores (pLDDT, i_ptm, i_pae) + confidences
  - Right: absolute scales (loss, ba_val, BSA) — each series drawn with
    its own scale using a shared right axis that is min-max-normalised
    per series before plotting (raw value printed in the legend label).

Example::

    python af_design/plot_afdesign_metric_traces.py \
        --run-dir benchmarks/output/af-8hgo-prod \
        --output benchmarks/output/af-8hgo-prod/metric_traces.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# (log_key, display_label, axis = "prob" or "abs", colour)
METRIC_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("plddt",  "pLDDT",       "prob", "#2980b9"),
    ("i_ptm",  "i_ptm",       "prob", "#8e44ad"),
    ("i_pae",  "i_pae",       "prob", "#d35400"),
    ("ptm",    "ptm",         "prob", "#7f8c8d"),
    ("loss",   "loss",        "abs",  "#c0392b"),
    ("ba_val", "ba_val",      "abs",  "#1abc9c"),
    ("i_con",  "i_con",       "abs",  "#f39c12"),
)


def _series(log: list[dict], key: str) -> np.ndarray:
    out = np.full(len(log), np.nan, dtype=float)
    for i, row in enumerate(log):
        val = row.get(key)
        if val is None:
            continue
        try:
            out[i] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def _minmax(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    log = json.loads((run_dir / "trajectory.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())

    bsa_path = run_dir / "bsa_history.json"
    bsa = (
        np.asarray(json.loads(bsa_path.read_text()), dtype=float)
        if bsa_path.exists() else np.array([], dtype=float)
    )

    n = len(log)
    steps = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax2 = ax.twinx()

    for key, label, scale, color in METRIC_SPECS:
        raw = _series(log, key)
        if np.all(np.isnan(raw)):
            continue
        last_valid = raw[np.isfinite(raw)]
        last_val = float(last_valid[-1]) if last_valid.size else float("nan")
        max_val = float(last_valid.max()) if last_valid.size else float("nan")
        min_val = float(last_valid.min()) if last_valid.size else float("nan")

        legend_label = (
            f"{label}  last={last_val:.3g}  range=[{min_val:.2g}, {max_val:.2g}]"
        )

        if scale == "prob":
            ax.plot(steps, raw, color=color, linewidth=1.4, label=legend_label)
        else:
            ax2.plot(steps, _minmax(raw), color=color, linewidth=1.4,
                     linestyle="--", label=legend_label)

    if bsa.size:
        max_bsa = float(bsa.max())
        last_bsa = float(bsa[-1])
        legend_label = (
            f"BSA (Å², norm.)  last={last_bsa:.0f}  max={max_bsa:.0f}"
        )
        ax2.plot(steps[:bsa.size], _minmax(bsa), color="#27ae60",
                 linewidth=1.4, linestyle="--", label=legend_label)

    stages = summary.get("stage_schedule", {})
    if stages.get("mode") == "three_stage":
        l = int(stages.get("logits_iters", 0))
        s = int(stages.get("soft_iters", 0))
        for boundary, tag in [(l, "→ soft"), (l + s, "→ hard")]:
            ax.axvline(boundary, color="black", linestyle=":", alpha=0.4)
            ax.text(boundary, 1.0, tag, rotation=90, fontsize=8,
                    va="top", ha="right", alpha=0.6,
                    transform=ax.get_xaxis_transform())

    ax.set_xlabel("design step")
    ax.set_ylabel("probability-scale metrics")
    ax2.set_ylabel("abs-scale metrics (min-max normalised per series)")
    ax.set_ylim(-0.05, 1.05)
    ax2.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8, ncol=2,
              framealpha=0.85)

    run_name = summary.get("run_name", run_dir.name)
    ax.set_title(f"AFDesign metric traces — {run_name}  ({n} steps)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
