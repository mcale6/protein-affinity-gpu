#!/usr/bin/env python3
"""Compare AFDesign best_metrics across two runs (before vs after).

Reads two ``summary.json`` files produced by
``af_design/modal_afdesign_ba_val.py`` and renders a grouped bar chart
across shared ``best_metrics`` keys plus the new BSA / ipSAE / filter
signals.

Usage:
    python af_design/plot_afdesign_compare.py \
        --before benchmarks/output/af-hardish/summary.json \
        --after  benchmarks/output/af-cascade-prod/summary.json \
        --output benchmarks/output/af_before_after.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

METRIC_ORDER: tuple[str, ...] = (
    "plddt",
    "i_ptm",
    "ptm",
    "pae",
    "i_pae",
    "i_con",
    "con",
    "ba_val",
    "bsa",
    "ipSAE",
    "loss",
)

FILTER_ORDER: tuple[str, ...] = ("plddt", "i_ptm", "i_pae", "i_con", "ipSAE")


def _load_summary(path: Path) -> dict:
    data = json.loads(Path(path).read_text())
    return data


def _pull(summary: dict, key: str) -> float | None:
    value = summary.get("best_metrics", {}).get(key)
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _plot_metrics(ax, before: dict, after: dict) -> None:
    keys = [
        k for k in METRIC_ORDER
        if _pull(before, k) is not None or _pull(after, k) is not None
    ]
    x = np.arange(len(keys))
    width = 0.4

    before_vals = [_pull(before, k) for k in keys]
    after_vals = [_pull(after, k) for k in keys]

    def _bars(values, offset, label, color):
        heights = [0.0 if v is None else v for v in values]
        bars = ax.bar(x + offset, heights, width, label=label, color=color)
        for bar, raw in zip(bars, values):
            if raw is None:
                bar.set_hatch("//")
                bar.set_alpha(0.35)
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{raw:.3g}",
                ha="center", va="bottom", fontsize=8,
            )

    _bars(before_vals, -width / 2, f"before ({before.get('run_name', '?')})", "#7a9cc6")
    _bars(after_vals,  +width / 2, f"after ({after.get('run_name', '?')})", "#c67a7a")

    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right")
    ax.set_ylabel("best_metrics value")
    ax.set_title("AFDesign best_metrics — before vs after")
    ax.axhline(0.0, color="black", linewidth=0.5)
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="best", fontsize=9)


def _plot_filter_gate(ax, before: dict, after: dict) -> None:
    before_filters = (before.get("filters") or {})
    after_filters = (after.get("filters") or {})

    def _pass_count(f: dict) -> tuple[int, int]:
        entries = [v for k, v in f.items() if k != "all_pass" and isinstance(v, dict)]
        passed = sum(1 for v in entries if v.get("pass"))
        return passed, len(entries)

    before_pass, before_total = _pass_count(before_filters)
    after_pass, after_total = _pass_count(after_filters)

    x = np.arange(len(FILTER_ORDER))
    width = 0.4

    def _vals(f: dict) -> list[int]:
        out: list[int] = []
        for key in FILTER_ORDER:
            entry = f.get(key)
            if isinstance(entry, dict) and entry.get("pass"):
                out.append(1)
            else:
                out.append(0)
        return out

    ax.bar(
        x - width / 2, _vals(before_filters), width,
        label=f"before ({before_pass}/{before_total} pass)", color="#7a9cc6",
    )
    ax.bar(
        x + width / 2, _vals(after_filters), width,
        label=f"after ({after_pass}/{after_total} pass)", color="#c67a7a",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(list(FILTER_ORDER), rotation=30, ha="right")
    ax.set_ylim(0, 1.2)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["fail", "pass"])
    ax.set_title("post-hoc filter gate")
    ax.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper right", fontsize=9)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True,
                        help="summary.json of the baseline run")
    parser.add_argument("--after", type=Path, required=True,
                        help="summary.json of the new run")
    parser.add_argument("--output", type=Path, required=True,
                        help="output PNG")
    args = parser.parse_args()

    before = _load_summary(args.before)
    after = _load_summary(args.after)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(9.6, 8.4),
        gridspec_kw={"height_ratios": [3, 1]},
        constrained_layout=True,
    )
    _plot_metrics(ax_top, before, after)
    _plot_filter_gate(ax_bot, before, after)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
