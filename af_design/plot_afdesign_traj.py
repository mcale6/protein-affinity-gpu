#!/usr/bin/env python3
"""Plot AFDesign trajectory metrics (binder Cα RMSD, BSA) vs iteration.

Reads artifact JSONs produced by ``af_design/modal_afdesign_ba_val.py``:

- ``binder_ca_history.json`` — per-iteration ``[binder_len, 3]`` Cα frames.
- ``bsa_history.json``       — per-iteration buried surface area (Å²).

Usage:
    python af_design/plot_afdesign_traj.py \
      --soft-dir benchmarks/output/af-soft \
      --hardish-dir benchmarks/output/af-hardish \
      --output benchmarks/output/af_traj.png \
      --metric both
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def _load_frames(path: Path) -> np.ndarray:
    """Load Cα history as ``[n_steps, binder_len, 3]`` array."""
    data = json.loads(Path(path).read_text())
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"{path}: expected [n_steps, binder_len, 3], got {arr.shape}")
    return arr


def _load_bsa(path: Path) -> np.ndarray:
    """Load per-iteration BSA as a 1-D float array (Å²)."""
    data = json.loads(Path(path).read_text())
    arr = np.asarray(data, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{path}: expected flat list of floats, got shape {arr.shape}")
    return arr


def _kabsch_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    """Cα RMSD between frames ``a`` and ``b`` after optimal superposition."""
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


def _resolve_inputs(args: argparse.Namespace) -> dict[str, dict[str, Path]]:
    """Return ``{"soft": {"rmsd": path, "bsa": path}, "hardish": {...}}``.

    Accepts either ``--soft-dir/--hardish-dir`` (preferred) or the legacy
    ``--soft/--hardish`` flags that pointed directly at ``binder_ca_history.json``.
    """
    resolved: dict[str, dict[str, Path]] = {}
    for label, dir_arg, file_arg in (
        ("soft", args.soft_dir, args.soft),
        ("hardish", args.hardish_dir, args.hardish),
    ):
        if dir_arg is not None:
            base = Path(dir_arg)
            resolved[label] = {
                "rmsd": base / "binder_ca_history.json",
                "bsa": base / "bsa_history.json",
            }
        elif file_arg is not None:
            ca_path = Path(file_arg)
            resolved[label] = {
                "rmsd": ca_path,
                "bsa": ca_path.with_name("bsa_history.json"),
            }
        else:
            raise SystemExit(
                f"missing {label!r} inputs — pass --{label}-dir (preferred) "
                f"or --{label} pointing at binder_ca_history.json"
            )
    return resolved


def _compute_rmsd_curves(
    paths: dict[str, dict[str, Path]], reference_mode: str
) -> tuple[dict[str, np.ndarray], str]:
    soft = _load_frames(paths["soft"]["rmsd"])
    hardish = _load_frames(paths["hardish"]["rmsd"])
    if reference_mode == "soft-last":
        ref = soft[-1]
        curves = {"soft": rmsd_curve(soft, ref), "hardish": rmsd_curve(hardish, ref)}
        label = "soft final"
    else:
        curves = {"soft": rmsd_curve(soft), "hardish": rmsd_curve(hardish)}
        label = "each run's final"
    return curves, label


def _compute_bsa_curves(paths: dict[str, dict[str, Path]]) -> dict[str, np.ndarray]:
    return {
        "soft": _load_bsa(paths["soft"]["bsa"]),
        "hardish": _load_bsa(paths["hardish"]["bsa"]),
    }


def _plot_rmsd(ax, curves: dict[str, np.ndarray], ref_label: str) -> None:
    ax.plot(
        np.arange(len(curves["soft"])), curves["soft"],
        marker="o", linewidth=1.6, label=f"soft (n={len(curves['soft'])})",
    )
    ax.plot(
        np.arange(len(curves["hardish"])), curves["hardish"],
        marker="s", linewidth=1.6, label=f"hard-ish (n={len(curves['hardish'])})",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("binder Cα RMSD (Å)")
    ax.set_title(f"binder Cα RMSD vs {ref_label}")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")


def _plot_bsa(ax, curves: dict[str, np.ndarray]) -> None:
    ax.plot(
        np.arange(len(curves["soft"])), curves["soft"],
        marker="o", linewidth=1.6, label=f"soft (n={len(curves['soft'])})",
    )
    ax.plot(
        np.arange(len(curves["hardish"])), curves["hardish"],
        marker="s", linewidth=1.6, label=f"hard-ish (n={len(curves['hardish'])})",
    )
    ax.set_xlabel("iteration")
    ax.set_ylabel("buried surface area (Å²)")
    ax.set_title("BSA vs iteration")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--soft-dir", type=Path, default=None,
                        help="soft run output dir (contains binder_ca_history.json / bsa_history.json)")
    parser.add_argument("--hardish-dir", type=Path, default=None,
                        help="hard-ish run output dir")
    parser.add_argument("--soft", type=Path, default=None,
                        help="legacy: path to soft run's binder_ca_history.json")
    parser.add_argument("--hardish", type=Path, default=None,
                        help="legacy: path to hard-ish run's binder_ca_history.json")
    parser.add_argument("--output", type=Path, required=True, help="output PNG")
    parser.add_argument(
        "--metric",
        choices=("rmsd", "bsa", "both"),
        default="both",
        help="which metric(s) to plot (default: both, two-panel)",
    )
    parser.add_argument(
        "--reference",
        choices=("last-per-run", "soft-last"),
        default="last-per-run",
        help="'last-per-run' (default) compares each run to its own final frame; "
        "'soft-last' compares both to the soft run's final frame.",
    )
    args = parser.parse_args()

    paths = _resolve_inputs(args)

    if args.metric == "rmsd":
        rmsd_curves, ref_label = _compute_rmsd_curves(paths, args.reference)
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        _plot_rmsd(ax, rmsd_curves, ref_label)
    elif args.metric == "bsa":
        bsa_curves = _compute_bsa_curves(paths)
        fig, ax = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
        _plot_bsa(ax, bsa_curves)
    else:
        rmsd_curves, ref_label = _compute_rmsd_curves(paths, args.reference)
        bsa_curves = _compute_bsa_curves(paths)
        fig, (ax_top, ax_bot) = plt.subplots(
            2, 1, figsize=(7.2, 7.8), sharex=True, constrained_layout=True
        )
        _plot_rmsd(ax_top, rmsd_curves, ref_label)
        _plot_bsa(ax_bot, bsa_curves)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
