#!/usr/bin/env python3
"""Merge one or more ``results.csv`` files (produced by ``benchmark.py`` and/or
``modal_benchmark.py``) and render a three-panel comparison figure.

Does not run predictions. The local + GPU runners each emit their own CSV;
this script is the join point — rows are merged on ``pdb_id``, backend names
are auto-detected from ``<name>_status`` columns, and the figure shows:

1. Warm-mean wall time vs atom14 atom count, one line per backend.
2. Per-structure SASA-sum scatter — CPU on the x-axis, every other backend
   a coloured series with Pearson r annotated.
3. Heatmap of per-backend Pearson r vs CPU across scalar metrics (ΔG, NIS
   channels, interface contacts).

Usage
-----

    python benchmarks/plot_results.py \\
        benchmarks/output/local/results.csv \\
        benchmarks/output/gpu/results.csv \\
        --output-dir benchmarks/output/combined
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from benchmarks.sasa.sasa_benchmark import BACKENDS  # noqa: E402

LOGGER = logging.getLogger(__name__)

HEATMAP_METRICS = [
    "ba_val",
    "contacts_ic",
    "contacts_charged",
    "contacts_polar",
    "contacts_aliphatic",
    "nis_aliphatic",
    "nis_charged",
    "nis_polar",
]

_BACKEND_COLORS = {
    "cpu":             "#2ca02c",
    "jax-single":      "#9467bd",
    "jax-batch":       "#1f77b4",
    "jax-scan":        "#17becf",
    "tinygrad-single": "#d62728",
    "tinygrad-batch":  "#ff7f0e",
}


# --- CSV loading ---------------------------------------------------------

def _detect_backends_from_columns(columns: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    detected: list[str] = []
    for col in columns:
        if not col.endswith("_status"):
            continue
        name = col[: -len("_status")]
        if name in seen:
            continue
        seen.add(name)
        detected.append(name)
    canonical = [b for b in BACKENDS if b in seen]
    extras = [b for b in detected if b not in BACKENDS]
    return canonical + extras


def _coerce_row(row: dict[str, str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in row.items():
        if value in (None, ""):
            out[key] = None
            continue
        if key == "n_atoms_atom14":
            try:
                out[key] = int(value)
            except ValueError:
                out[key] = None
            continue
        try:
            out[key] = float(value)
        except ValueError:
            out[key] = value
    return out


def load_rows_from_csvs(csv_paths: list[Path]) -> tuple[list[dict[str, Any]], list[str]]:
    merged: dict[str, dict[str, Any]] = {}
    all_backends: list[str] = []
    for csv_path in csv_paths:
        with Path(csv_path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            columns = reader.fieldnames or []
            for name in _detect_backends_from_columns(columns):
                if name not in all_backends:
                    all_backends.append(name)
            for raw in reader:
                coerced = _coerce_row(raw)
                key = coerced.get("pdb_id")
                if not key:
                    continue
                if key not in merged:
                    merged[key] = coerced
                else:
                    for field, value in coerced.items():
                        if value is None:
                            continue
                        existing = merged[key].get(field)
                        if existing in (None, ""):
                            merged[key][field] = value
    canonical = [b for b in BACKENDS if b in all_backends]
    extras = [b for b in all_backends if b not in BACKENDS]
    return list(merged.values()), canonical + extras


# --- Statistics ----------------------------------------------------------

def _pearson(x: np.ndarray, y: np.ndarray) -> float | None:
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return None
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 2:
        return None
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def _ok_rows(rows: list[dict[str, Any]], name: str) -> list[dict[str, Any]]:
    return [r for r in rows if r.get(f"{name}_status") == "ok"]


def _display_name(name: str) -> str:
    return BACKENDS[name].display if name in BACKENDS else name


# --- Figure --------------------------------------------------------------

def _import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dep
        raise ImportError("Plot generation requires 'matplotlib'.") from exc
    return plt


def plot_figure(
    rows: list[dict[str, Any]], backends: list[str], output_path: Path
) -> None:
    plt = _import_pyplot()
    if not rows:
        LOGGER.warning("No rows to plot.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(19, 5.6), constrained_layout=True)
    ax_timing, ax_sasa, ax_heat = axes

    # --- Subplot 1: timing vs atom count -------------------------------
    import matplotlib.ticker as mticker
    for name in backends:
        ok = _ok_rows(rows, name)
        points = [
            (
                r.get("n_atoms_atom14"),
                r.get(f"{name}_warm_mean_seconds"),
                r.get(f"{name}_warm_std_seconds"),
            )
            for r in ok
            if r.get("n_atoms_atom14") is not None
            and r.get(f"{name}_warm_mean_seconds") is not None
        ]
        if not points:
            continue
        points.sort(key=lambda p: p[0])
        xs = np.asarray([p[0] for p in points], dtype=float)
        ys = np.asarray([p[1] for p in points], dtype=float)
        stds = np.asarray(
            [p[2] if p[2] is not None else np.nan for p in points], dtype=float,
        )
        colds = np.asarray(
            [r.get(f"{name}_cold_seconds") for r in ok
             if r.get(f"{name}_cold_seconds") is not None],
            dtype=float,
        )
        label = _display_name(name)
        if colds.size and name != "cpu":
            label += f"  (compile {np.median(colds):.2f}s)"
        color = _BACKEND_COLORS.get(name)
        if np.any(np.isfinite(stds)):
            ax_timing.errorbar(
                xs, ys, yerr=np.where(np.isfinite(stds), stds, 0.0),
                marker="o", linewidth=1.4, markersize=5,
                capsize=3, elinewidth=1.0, color=color,
                ecolor=color, alpha=0.95,
                label=label,
            )
        else:
            ax_timing.plot(
                xs, ys,
                marker="o", linewidth=1.4, markersize=5,
                color=color, label=label,
            )
    ax_timing.set_xlabel("Atom14 atoms (padded, n_residues × 14)")
    ax_timing.set_ylabel("Warm wall time (s, mean ± std over repeats)")
    ax_timing.set_yscale("log")
    ax_timing.set_title("Timing vs atom count")
    ax_timing.yaxis.set_major_locator(mticker.LogLocator(base=10.0, numticks=15))
    ax_timing.yaxis.set_minor_locator(
        mticker.LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1, numticks=15)
    )
    ax_timing.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax_timing.yaxis.set_minor_formatter(
        mticker.FuncFormatter(lambda v, _: f"{v:g}" if v in (0.2, 0.3, 0.5, 2, 3, 5) else "")
    )
    ax_timing.tick_params(axis="y", which="minor", labelsize=7)
    ax_timing.grid(alpha=0.25, which="major")
    ax_timing.grid(alpha=0.12, which="minor")
    ax_timing.legend(fontsize=8, loc="upper left")

    # --- Subplot 2: per-structure SASA-sum, CPU vs each backend -------
    if "cpu" in backends:
        cpu_ok = _ok_rows(rows, "cpu")
        for name in backends:
            if name == "cpu":
                continue
            paired = [r for r in cpu_ok if r.get(f"{name}_status") == "ok"]
            if not paired:
                continue
            xs = np.asarray([r.get("cpu_sasa_sum") for r in paired], dtype=float)
            ys = np.asarray([r.get(f"{name}_sasa_sum") for r in paired], dtype=float)
            r_value = _pearson(xs, ys)
            mask = np.isfinite(xs) & np.isfinite(ys)
            mae = float(np.mean(np.abs(xs[mask] - ys[mask]))) if mask.any() else None
            label = _display_name(name)
            if r_value is not None:
                label += f"  (r={r_value:.4f}"
                if mae is not None:
                    label += f", MAE={mae:,.0f} Å²"
                label += ")"
            ax_sasa.scatter(
                xs, ys, s=48, alpha=0.85,
                color=_BACKEND_COLORS.get(name),
                label=label, zorder=2,
            )
        all_x = [
            r.get("cpu_sasa_sum") for r in cpu_ok if r.get("cpu_sasa_sum") is not None
        ]
        if all_x:
            lo, hi = float(min(all_x)), float(max(all_x))
            pad = max((hi - lo) * 0.05, 1.0)
            ax_sasa.plot(
                [lo - pad, hi + pad], [lo - pad, hi + pad],
                linestyle="--", linewidth=1.0, color="#6c757d", zorder=1,
            )
    ax_sasa.set_xlabel("CPU total SASA (Å²)")
    ax_sasa.set_ylabel("Backend total SASA (Å²)")
    ax_sasa.set_title("SASA agreement (per structure)")
    ax_sasa.grid(alpha=0.25)
    ax_sasa.legend(fontsize=8, loc="upper left")

    # --- Subplot 3: heatmap of Pearson r vs CPU across metrics --------
    if "cpu" in backends and len(backends) > 1:
        non_cpu = [b for b in backends if b != "cpu"]
        matrix = np.full((len(non_cpu), len(HEATMAP_METRICS)), np.nan)
        cpu_ok = _ok_rows(rows, "cpu")
        for i, name in enumerate(non_cpu):
            paired = [r for r in cpu_ok if r.get(f"{name}_status") == "ok"]
            if not paired:
                continue
            for j, metric in enumerate(HEATMAP_METRICS):
                xs = np.asarray(
                    [r.get(f"cpu_{metric}") for r in paired], dtype=float
                )
                ys = np.asarray(
                    [r.get(f"{name}_{metric}") for r in paired], dtype=float
                )
                value = _pearson(xs, ys)
                if value is not None:
                    matrix[i, j] = value
        im = ax_heat.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=-1.0, vmax=1.0)
        ax_heat.set_xticks(range(len(HEATMAP_METRICS)))
        ax_heat.set_xticklabels(HEATMAP_METRICS, rotation=40, ha="right", fontsize=8)
        ax_heat.set_yticks(range(len(non_cpu)))
        ax_heat.set_yticklabels([_display_name(b) for b in non_cpu], fontsize=9)
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                cell = matrix[i, j]
                text = "n/a" if np.isnan(cell) else f"{cell:.3f}"
                ax_heat.text(
                    j, i, text,
                    ha="center", va="center", fontsize=8,
                    color="black" if not np.isnan(cell) and abs(cell) < 0.6 else "white",
                )
        ax_heat.set_title("Pearson r vs CPU")
        fig.colorbar(im, ax=ax_heat, fraction=0.04, pad=0.04)
    else:
        ax_heat.set_axis_off()
        ax_heat.text(
            0.5, 0.5,
            "No CPU baseline available\n— heatmap skipped",
            ha="center", va="center", fontsize=10, transform=ax_heat.transAxes,
        )

    fig.suptitle("Backend comparison", fontsize=14)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    LOGGER.info("Wrote figure to %s", output_path)


# --- CLI -----------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Merge one or more benchmark results.csv files (from "
            "benchmarks/benchmark.py or benchmarks/modal_benchmark.py) and "
            "render the three-panel comparison figure. Does not run predictions."
        ),
    )
    parser.add_argument(
        "csv_paths", type=Path, nargs="+",
        help="One or more results.csv files to merge on pdb_id.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benchmarks/output/combined",
        help="Directory where comparison_figure.png will be written.",
    )
    parser.add_argument(
        "--figure-name", default="comparison_figure.png",
        help="Filename for the generated figure.",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(message)s",
    )

    missing = [p for p in args.csv_paths if not Path(p).exists()]
    if missing:
        parser.error(f"CSV not found: {missing}")

    rows, backends = load_rows_from_csvs(list(args.csv_paths))
    if not rows:
        parser.error("No rows loaded from the supplied CSV(s).")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig_path = args.output_dir / args.figure_name
    plot_figure(rows, backends, fig_path)

    LOGGER.info(
        "Plotted %d structures across backends: %s", len(rows), backends
    )
    print(str(fig_path))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
