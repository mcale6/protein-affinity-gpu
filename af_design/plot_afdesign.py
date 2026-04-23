#!/usr/bin/env python3
"""Unified AFDesign plot CLI.

All subcommands read the artifact JSONs written by
``af_design/modal_afdesign_ba_val.py`` — ``trajectory.json``,
``summary.json``, ``bsa_history.json``, ``binder_ca_history.json``.

Subcommands::

    traces    single-run per-step metric overlay (PNG)
    rmsd      two-run binder Cα RMSD + BSA curves (PNG)
    animate   single-run animated GIF (loss / pLDDT / i_ptm / BSA / ba_val)
    compare   two-run best_metrics bar chart + post-hoc filter gate (PNG)

Examples::

    python af_design/plot_afdesign.py traces \\
        --run-dir benchmarks/output/afdesign_april2026/af-8hgo-prod \\
        --output  benchmarks/output/afdesign_april2026/af-8hgo-prod/metric_traces.png

    python af_design/plot_afdesign.py rmsd \\
        --soft-dir    benchmarks/output/afdesign_april2026/af-soft \\
        --hardish-dir benchmarks/output/afdesign_april2026/af-hardish \\
        --output      benchmarks/output/afdesign_april2026/af_traj.png \\
        --metric both

    python af_design/plot_afdesign.py animate \\
        --run-dir benchmarks/output/afdesign_april2026/af-cascade-prod \\
        --output  benchmarks/output/afdesign_april2026/af-cascade-prod/traj.gif

    python af_design/plot_afdesign.py compare \\
        --before benchmarks/output/afdesign_april2026/af-hardish/summary.json \\
        --after  benchmarks/output/afdesign_april2026/af-cascade-prod/summary.json \\
        --output benchmarks/output/afdesign_april2026/af_before_after.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------------------
# Shared loaders
# ---------------------------------------------------------------------------

def _load_log(run_dir: Path) -> list[dict]:
    return json.loads((run_dir / "trajectory.json").read_text())


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


def _load_bsa(run_dir: Path) -> np.ndarray:
    path = run_dir / "bsa_history.json"
    if not path.exists():
        return np.array([], dtype=float)
    try:
        return np.asarray(json.loads(path.read_text()), dtype=float)
    except (json.JSONDecodeError, TypeError, ValueError):
        return np.array([], dtype=float)


def _load_frames(path: Path) -> np.ndarray:
    arr = np.asarray(json.loads(path.read_text()), dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{path}: expected [n_steps, binder_len, 3], got {arr.shape}")
    return arr


def _minmax(arr: np.ndarray) -> np.ndarray:
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr
    lo, hi = float(finite.min()), float(finite.max())
    if hi == lo:
        return np.full_like(arr, 0.5)
    return (arr - lo) / (hi - lo)


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    a_c = a - a.mean(axis=0, keepdims=True)
    b_c = b - b.mean(axis=0, keepdims=True)
    h = a_c.T @ b_c
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1.0, 1.0, d]) @ vt
    return float(np.sqrt(np.mean(np.sum((a_c @ r - b_c) ** 2, axis=1))))


def _rmsd_curve(frames: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.array([_kabsch_rmsd(frame, reference) for frame in frames], dtype=np.float64)


# ---------------------------------------------------------------------------
# traces — single-run per-step metric overlay
# ---------------------------------------------------------------------------

# (log_key, display_label, axis = "prob" or "abs", colour)
_METRIC_SPECS: tuple[tuple[str, str, str, str], ...] = (
    ("plddt",  "pLDDT",  "prob", "#2980b9"),
    ("i_ptm",  "i_ptm",  "prob", "#8e44ad"),
    ("i_pae",  "i_pae",  "prob", "#d35400"),
    ("ptm",    "ptm",    "prob", "#7f8c8d"),
    ("loss",   "loss",   "abs",  "#c0392b"),
    ("ba_val", "ba_val", "abs",  "#1abc9c"),
    ("i_con",  "i_con",  "abs",  "#f39c12"),
)


def cmd_traces(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    log = _load_log(run_dir)
    summary = json.loads((run_dir / "summary.json").read_text())
    bsa = _load_bsa(run_dir)

    n = len(log)
    steps = np.arange(1, n + 1)
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    ax2 = ax.twinx()

    stages = summary.get("stage_schedule", {})
    ba_active_start = 1
    if stages.get("mode") == "adaptive":
        ba_active_start = int(stages.get("phase_a_iters", 0)) + 1
    elif stages.get("mode") == "three_stage":
        ba_active_start = int(stages.get("logits_iters", 0)) + 1

    for key, label, scale, color in _METRIC_SPECS:
        raw = _series(log, key)
        if key == "ba_val" and ba_active_start > 1:
            raw = raw.copy()
            raw[: ba_active_start - 1] = np.nan
        if np.all(np.isnan(raw)):
            continue
        valid = raw[np.isfinite(raw)]
        last_val = float(valid[-1]) if valid.size else float("nan")
        legend = (
            f"{label}  last={last_val:.3g}  "
            f"range=[{float(valid.min()):.2g}, {float(valid.max()):.2g}]"
        )
        if scale == "prob":
            ax.plot(steps, raw, color=color, linewidth=1.4, label=legend)
        else:
            ax2.plot(steps, _minmax(raw), color=color, linewidth=1.4,
                     linestyle="--", label=legend)

    if bsa.size:
        legend = f"BSA (Å², norm.)  last={float(bsa[-1]):.0f}  max={float(bsa.max()):.0f}"
        ax2.plot(steps[: bsa.size], _minmax(bsa), color="#27ae60",
                 linewidth=1.4, linestyle="--", label=legend)

    boundaries: list[tuple[int, str]] = []
    if stages.get("mode") == "three_stage":
        l = int(stages.get("logits_iters", 0))
        s = int(stages.get("soft_iters", 0))
        boundaries = [(l, "→ soft"), (l + s, "→ hard")]
    elif stages.get("mode") == "adaptive":
        a = int(stages.get("phase_a_iters", 0))
        b = int(stages.get("phase_b_iters", 0))
        boundaries = [(a, "ba_val on"), (a + b, "→ hard")]
    for boundary, tag in boundaries:
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


# ---------------------------------------------------------------------------
# rmsd — two-run binder Cα RMSD + BSA curves
# ---------------------------------------------------------------------------

def _plot_two_run(ax, curves: dict[str, np.ndarray], ylabel: str, title: str) -> None:
    for label, arr, marker in (
        ("soft", curves["soft"], "o"),
        ("hardish", curves["hardish"], "s"),
    ):
        display = "hard-ish" if label == "hardish" else label
        ax.plot(np.arange(len(arr)), arr, marker=marker, linewidth=1.6,
                label=f"{display} (n={len(arr)})")
    ax.set_xlabel("iteration")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")


def cmd_rmsd(args: argparse.Namespace) -> None:
    soft_frames = _load_frames(args.soft_dir / "binder_ca_history.json")
    hard_frames = _load_frames(args.hardish_dir / "binder_ca_history.json")

    if args.reference == "soft-last":
        ref = soft_frames[-1]
        ref_label = "soft final"
        rmsd = {
            "soft": _rmsd_curve(soft_frames, ref),
            "hardish": _rmsd_curve(hard_frames, ref),
        }
    else:
        ref_label = "each run's final"
        rmsd = {
            "soft": _rmsd_curve(soft_frames, soft_frames[-1]),
            "hardish": _rmsd_curve(hard_frames, hard_frames[-1]),
        }

    bsa = {"soft": _load_bsa(args.soft_dir), "hardish": _load_bsa(args.hardish_dir)}
    if args.metric in ("bsa", "both") and (bsa["soft"].size == 0 or bsa["hardish"].size == 0):
        missing = [k for k, v in bsa.items() if v.size == 0]
        print(f"note: {missing} has no bsa_history.json — BSA panel will be empty")

    if args.metric == "rmsd":
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        _plot_two_run(ax, rmsd, "binder Cα RMSD (Å)", f"binder Cα RMSD vs {ref_label}")
    elif args.metric == "bsa":
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        _plot_two_run(ax, bsa, "buried surface area (Å²)", "BSA vs iteration")
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.2, 7.8), sharex=True, constrained_layout=True,
        )
        _plot_two_run(ax_top, rmsd, "binder Cα RMSD (Å)",
                      f"binder Cα RMSD vs {ref_label}")
        _plot_two_run(ax_bot, bsa, "buried surface area (Å²)", "BSA vs iteration")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


# ---------------------------------------------------------------------------
# animate — single-run animated GIF
# ---------------------------------------------------------------------------

def cmd_animate(args: argparse.Namespace) -> None:
    run_dir = args.run_dir.expanduser().resolve()
    log = _load_log(run_dir)
    n_steps = len(log)
    series = {
        "loss":   _series(log, "loss"),
        "plddt":  _series(log, "plddt"),
        "i_ptm":  _series(log, "i_ptm"),
        "ba_val": _series(log, "ba_val"),
    }
    bsa = _load_bsa(run_dir)

    steps = np.arange(1, n_steps + 1)
    frame_indices = list(range(0, n_steps, max(1, args.stride)))

    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    ax_loss, ax_plddt, ax_bsa, ax_ba = axes.flat

    def _setup(ax, arr, title, ylabel, color):
        arr = np.asarray(arr, dtype=float)
        ax.plot(steps[: len(arr)], arr, color=color, alpha=0.4, linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("step")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.3)
        marker, = ax.plot([], [], "o", color=color, markersize=7)
        vline = ax.axvline(0, color=color, alpha=0.3)
        return arr, marker, vline

    panels = [
        _setup(ax_loss,  series["loss"],   "loss",  "loss",  "#c0392b"),
        _setup(ax_plddt, series["plddt"],  "pLDDT", "pLDDT", "#2980b9"),
        _setup(ax_bsa,   bsa,              "BSA (Å²)", "BSA", "#27ae60"),
        _setup(ax_ba,    series["ba_val"], "ba_val (PRODIGY ΔG proxy)", "ba_val", "#8e44ad"),
    ]
    title = fig.suptitle(f"AFDesign trajectory — {run_dir.name} (step 0)")

    def update(frame_idx: int):
        step_num = frame_idx + 1
        for arr, marker, vline in panels:
            if frame_idx < len(arr) and not np.isnan(arr[frame_idx]):
                marker.set_data([step_num], [arr[frame_idx]])
            else:
                marker.set_data([], [])
            vline.set_xdata([step_num, step_num])
        title.set_text(f"AFDesign trajectory — {run_dir.name} (step {step_num})")
        return (*[p[1] for p in panels], *[p[2] for p in panels], title)

    anim = animation.FuncAnimation(
        fig, update, frames=frame_indices, interval=1000 // args.fps, blit=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    anim.save(str(args.output), writer=animation.PillowWriter(fps=args.fps))
    print(f"wrote {args.output}")


# ---------------------------------------------------------------------------
# compare — two-run best_metrics bar chart + filter gate
# ---------------------------------------------------------------------------

_COMPARE_METRIC_ORDER: tuple[str, ...] = (
    "plddt", "i_ptm", "ptm", "pae", "i_pae", "i_con", "con",
    "ba_val", "bsa", "ipSAE", "loss",
)
_FILTER_ORDER: tuple[str, ...] = ("plddt", "i_ptm", "i_pae", "i_con", "ipSAE")


def _pull_metric(summary: dict, key: str) -> float | None:
    value = summary.get("best_metrics", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _filter_vals(filters: dict) -> list[int]:
    return [
        1 if isinstance(filters.get(k), dict) and filters[k].get("pass") else 0
        for k in _FILTER_ORDER
    ]


def _pass_count(filters: dict) -> tuple[int, int]:
    entries = [v for k, v in filters.items() if k != "all_pass" and isinstance(v, dict)]
    return sum(1 for v in entries if v.get("pass")), len(entries)


def cmd_compare(args: argparse.Namespace) -> None:
    before = json.loads(args.before.read_text())
    after = json.loads(args.after.read_text())

    keys = [
        k for k in _COMPARE_METRIC_ORDER
        if _pull_metric(before, k) is not None or _pull_metric(after, k) is not None
    ]
    x = np.arange(len(keys))
    width = 0.4
    b_vals = [_pull_metric(before, k) for k in keys]
    a_vals = [_pull_metric(after, k) for k in keys]

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.6, 8.4),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )

    def _bars(ax, values, offset, label, color):
        heights = [0.0 if v is None else v for v in values]
        bars = ax.bar(x + offset, heights, width, label=label, color=color)
        for bar, raw in zip(bars, values):
            if raw is None:
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                continue
            ax.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                    f"{raw:.3g}", ha="center", va="bottom", fontsize=8)

    _bars(ax_top, b_vals, -width / 2,
          f"before ({before.get('run_name', '?')})", "#7a9cc6")
    _bars(ax_top, a_vals, +width / 2,
          f"after ({after.get('run_name', '?')})", "#c67a7a")
    ax_top.set_xticks(x)
    ax_top.set_xticklabels(keys, rotation=30, ha="right")
    ax_top.set_ylabel("best_metrics value")
    ax_top.set_title("AFDesign best_metrics — before vs after")
    ax_top.axhline(0.0, color="black", linewidth=0.5)
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_top.legend(loc="best", fontsize=9)

    before_filters = before.get("filters") or {}
    after_filters = after.get("filters") or {}
    bp, bt = _pass_count(before_filters)
    ap, at = _pass_count(after_filters)
    fx = np.arange(len(_FILTER_ORDER))
    ax_bot.bar(fx - width / 2, _filter_vals(before_filters), width,
               label=f"before ({bp}/{bt} pass)", color="#7a9cc6")
    ax_bot.bar(fx + width / 2, _filter_vals(after_filters), width,
               label=f"after ({ap}/{at} pass)", color="#c67a7a")
    ax_bot.set_xticks(fx)
    ax_bot.set_xticklabels(list(_FILTER_ORDER), rotation=30, ha="right")
    ax_bot.set_ylim(0, 1.2)
    ax_bot.set_yticks([0, 1])
    ax_bot.set_yticklabels(["fail", "pass"])
    ax_bot.set_title("post-hoc filter gate")
    ax_bot.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_bot.legend(loc="upper right", fontsize=9)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = ap.add_subparsers(dest="cmd", required=True)

    p_traces = subs.add_parser(
        "traces", help="Single-run per-step metric overlay (PNG)",
    )
    p_traces.add_argument("--run-dir", type=Path, required=True)
    p_traces.add_argument("--output", type=Path, required=True)
    p_traces.set_defaults(func=cmd_traces)

    p_rmsd = subs.add_parser(
        "rmsd", help="Two-run binder Cα RMSD + BSA curves (PNG)",
    )
    p_rmsd.add_argument("--soft-dir", type=Path, required=True)
    p_rmsd.add_argument("--hardish-dir", type=Path, required=True)
    p_rmsd.add_argument("--output", type=Path, required=True)
    p_rmsd.add_argument("--metric", choices=("rmsd", "bsa", "both"), default="both")
    p_rmsd.add_argument("--reference",
                        choices=("last-per-run", "soft-last"),
                        default="last-per-run")
    p_rmsd.set_defaults(func=cmd_rmsd)

    p_anim = subs.add_parser(
        "animate", help="Single-run animated GIF (loss/pLDDT/i_ptm/BSA/ba_val)",
    )
    p_anim.add_argument("--run-dir", type=Path, required=True)
    p_anim.add_argument("--output", type=Path, required=True)
    p_anim.add_argument("--stride", type=int, default=1)
    p_anim.add_argument("--fps", type=int, default=10)
    p_anim.set_defaults(func=cmd_animate)

    p_cmp = subs.add_parser(
        "compare", help="Two-run best_metrics bar chart + filter gate (PNG)",
    )
    p_cmp.add_argument("--before", type=Path, required=True)
    p_cmp.add_argument("--after", type=Path, required=True)
    p_cmp.add_argument("--output", type=Path, required=True)
    p_cmp.set_defaults(func=cmd_compare)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
