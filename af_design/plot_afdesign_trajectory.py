#!/usr/bin/env python3
"""Render an AFDesign trajectory as an animated GIF.

Uses the per-step metrics from ``trajectory.json`` plus the BSA trace from
``bsa_history.json`` (both written by ``af_design/modal_afdesign_ba_val.py``).
Each frame highlights the current step across four panels: loss, pLDDT,
i_ptm, and BSA.

Example::

    python af_design/plot_afdesign_trajectory.py \
        --run-dir benchmarks/output/af-8hgo-egfr-hotspot-120 \
        --output benchmarks/output/af-8hgo-egfr-hotspot-120/traj.gif
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


def _load_series(log: list[dict], key: str) -> list[float]:
    out: list[float] = []
    for row in log:
        val = row.get(key)
        try:
            out.append(float(val) if val is not None else float("nan"))
        except (TypeError, ValueError):
            out.append(float("nan"))
    return out


def _load_bsa(run_dir: Path) -> list[float]:
    path = run_dir / "bsa_history.json"
    if not path.exists():
        return []
    try:
        raw = json.loads(path.read_text())
        return [float(x) for x in raw]
    except (json.JSONDecodeError, TypeError, ValueError):
        return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
                        help="Directory containing trajectory.json + bsa_history.json")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output GIF path")
    parser.add_argument("--stride", type=int, default=1,
                        help="Frame stride (default 1 = every step)")
    parser.add_argument("--fps", type=int, default=10)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    log = json.loads((run_dir / "trajectory.json").read_text())
    n_steps = len(log)

    metrics = {
        "loss":  _load_series(log, "loss"),
        "plddt": _load_series(log, "plddt"),
        "i_ptm": _load_series(log, "i_ptm"),
        "ba_val": _load_series(log, "ba_val"),
    }
    bsa = _load_bsa(run_dir)

    steps = np.arange(1, n_steps + 1)
    frame_indices = list(range(0, n_steps, max(1, args.stride)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    ax_loss, ax_plddt, ax_bsa, ax_ba = axes.flat

    def _setup(ax, series, title, ylabel, color):
        arr = np.array(series, dtype=float)
        ax.plot(steps[:len(arr)], arr, color=color, alpha=0.4, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        marker, = ax.plot([], [], "o", color=color, markersize=7)
        vline = ax.axvline(0, color=color, alpha=0.3)
        return arr, marker, vline

    loss_arr, loss_m, loss_v = _setup(ax_loss, metrics["loss"], "loss", "loss", "#c0392b")
    plddt_arr, plddt_m, plddt_v = _setup(ax_plddt, metrics["plddt"], "pLDDT", "pLDDT", "#2980b9")
    bsa_arr, bsa_m, bsa_v = _setup(ax_bsa, bsa, "BSA (Å²)", "BSA", "#27ae60")
    ba_arr, ba_m, ba_v = _setup(ax_ba, metrics["ba_val"], "ba_val (PRODIGY ΔG proxy)",
                                 "ba_val", "#8e44ad")

    title = fig.suptitle(f"AFDesign trajectory — {run_dir.name} (step 0)")

    def update(frame_idx: int):
        step_num = frame_idx + 1
        for arr, marker, vline in [
            (loss_arr, loss_m, loss_v),
            (plddt_arr, plddt_m, plddt_v),
            (bsa_arr, bsa_m, bsa_v),
            (ba_arr, ba_m, ba_v),
        ]:
            if frame_idx < len(arr) and not np.isnan(arr[frame_idx]):
                marker.set_data([step_num], [arr[frame_idx]])
            else:
                marker.set_data([], [])
            vline.set_xdata([step_num, step_num])
        title.set_text(f"AFDesign trajectory — {run_dir.name} (step {step_num})")
        return loss_m, plddt_m, bsa_m, ba_m, loss_v, plddt_v, bsa_v, ba_v, title

    anim = animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=1000 // args.fps, blit=False
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=animation.PillowWriter(fps=args.fps))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
