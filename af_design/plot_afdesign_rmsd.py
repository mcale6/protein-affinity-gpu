#!/usr/bin/env python3
"""Plot binder backbone RMSD vs iteration for AFDesign Modal runs.

Reads ``binder_ca_history.json`` files produced by
``benchmarks/modal_afdesign_ba_val.py`` and plots the per-iteration CA RMSD
against the final frame of each run.

Usage:
    python benchmarks/af_design/plot_afdesign_rmsd.py \
      --soft benchmarks/output/af-soft/binder_ca_history.json \
      --hardish benchmarks/output/af-hardish/binder_ca_history.json \
      --output benchmarks/output/af_rmsd.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_frames(path: Path) -> np.ndarray:
    """Load CA history as ``[n_steps, binder_len, 3]`` array."""
    data = json.loads(Path(path).read_text())
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{path}: expected [n_steps, binder_len, 3], got {arr.shape}")
    return arr


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """CA RMSD between frames ``a`` and ``b`` after optimal superposition."""
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    h = a_c.T @ b_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    a_rot = a_c @ r
    return float(np.sqrt(np.mean(np.sum((a_rot - b_c) ** 2, axis=1))))


def rmsd_curve(frames: np.ndarray, reference: np.ndarray | None = None) -> np.ndarray:
    """Per-iteration RMSD relative to ``reference`` (defaults to last frame)."""
    ref = frames[-1] if reference is None else reference
    return np.array([_kabsch_rmsd(frame, ref) for frame in frames], dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft", type=Path, required=True, help="soft run binder_ca_history.json")
    parser.add_argument("--hardish", type=Path, required=True, help="hard-ish run binder_ca_history.json")
    parser.add_argument("--output", type=Path, required=True, help="output PNG")
    parser.add_argument(
        "--reference",
        choices=("last-per-run", "soft-last"),
        default="last-per-run",
        help="'last-per-run' (default) compares each run to its own final frame; "
        "'soft-last' compares both to the soft run's final frame.",
    )
    args = parser.parse_args()

    soft = _load_frames(args.soft)
    hardish = _load_frames(args.hardish)

    if args.reference == "soft-last":
        ref = soft[-1]
        soft_rmsd = rmsd_curve(soft, ref)
        hardish_rmsd = rmsd_curve(hardish, ref)
        ref_label = "soft final"
    else:
        soft_rmsd = rmsd_curve(soft)
        hardish_rmsd = rmsd_curve(hardish)
        ref_label = "each run's final"

    fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    soft_iters = np.arange(len(soft_rmsd))
    hardish_iters = np.arange(len(hardish_rmsd))
    ax.plot(soft_iters, soft_rmsd, marker="o", linewidth=1.6, label=f"soft (n={len(soft)})")
    ax.plot(hardish_iters, hardish_rmsd, marker="s", linewidth=1.6, label=f"hard-ish (n={len(hardish)})")

    ax.set_xlabel("iteration")
    ax.set_ylabel("binder CA RMSD (Å)")
    ax.set_title(f"AFDesign binder CA RMSD vs {ref_label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
