#!/usr/bin/env python3
"""Pick the higher-ipTM Boltz diffusion sample per complex.

With ``--diffusion_samples 2``, Boltz writes two independent rollouts per
complex (``input_model_0.cif`` + ``input_model_1.cif``, with matching
pae / pde / plddt / confidence files). Downstream scripts (05, 05b,
augmented_refit) only look at ``*_model_0*``. This script:

 1. Reads ``confidence_input_model_0.json`` and
    ``confidence_input_model_1.json`` for each complex
 2. If sample 1 has higher ``iptm``, **renames the files so that sample 1
    becomes the new sample 0** (and vice versa) — five file types each
    (cif, pae.npz, pde.npz, plddt.npz, confidence.json)
 3. Writes a ``best_sample.csv`` in the dataset's output root recording
    which original sample index is now at position 0

Idempotent: re-running doesn't churn files if the best sample is already
at index 0. Uses atomic three-way rename via ``.tmp`` suffix.

Usage:
    python select_best_sample.py --dataset vreven
    python select_best_sample.py --dataset vreven --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from dataset_registry import AVAILABLE, get_paths  # noqa: E402


FILE_STEMS = (
    "input_model_{n}.cif",
    "pae_input_model_{n}.npz",
    "pde_input_model_{n}.npz",
    "plddt_input_model_{n}.npz",
    "confidence_input_model_{n}.json",
)


def pair_files(pred_dir: Path, sample_a: int, sample_b: int
                ) -> list[tuple[Path, Path]]:
    pairs = []
    for stem in FILE_STEMS:
        a = pred_dir / stem.format(n=sample_a)
        b = pred_dir / stem.format(n=sample_b)
        if not a.exists() or not b.exists():
            raise FileNotFoundError(
                f"missing pair: {a.name} / {b.name} under {pred_dir}"
            )
        pairs.append((a, b))
    return pairs


def swap_samples(pred_dir: Path) -> None:
    """Swap sample 0 and sample 1 files in place via three-way rename."""
    pairs = pair_files(pred_dir, 0, 1)
    for a, b in pairs:
        tmp = a.with_suffix(a.suffix + ".tmp_swap")
        a.rename(tmp)
        b.rename(a)
        tmp.rename(b)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, choices=AVAILABLE)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    paths = get_paths(args.dataset)

    mapping: list[dict] = []
    n_swapped = 0
    n_missing = 0
    n_total = 0

    modes = [d.name for d in paths.output_root.iterdir()
             if d.is_dir() and not d.name.startswith((".", "pae_calibration"))
             and not d.name.endswith(".csv") and d.name != "pae_calibration"]
    print(f"[{paths.display}]  modes found: {modes}")

    for mode in modes:
        mode_dir = paths.output_root / mode
        if not mode_dir.is_dir():
            continue
        for pred_dir in sorted(mode_dir.iterdir()):
            if not pred_dir.is_dir():
                continue
            # Navigate to the Boltz predictions subdir
            pred_input = next(pred_dir.rglob("predictions/input"), None)
            if pred_input is None:
                continue
            pdb_id = pred_dir.name.removesuffix(f"_{mode}")
            n_total += 1

            conf0 = pred_input / "confidence_input_model_0.json"
            conf1 = pred_input / "confidence_input_model_1.json"
            if not conf0.exists() or not conf1.exists():
                n_missing += 1
                mapping.append({
                    "pdb_id": pdb_id, "mode": mode,
                    "iptm_sample_0": "", "iptm_sample_1": "",
                    "best_source": "0 (no sample 1)", "swapped": False,
                })
                continue

            iptm0 = json.loads(conf0.read_text()).get("iptm")
            iptm1 = json.loads(conf1.read_text()).get("iptm")
            best = 1 if (iptm1 is not None and iptm0 is not None
                          and iptm1 > iptm0) else 0
            swap = best == 1
            if swap and not args.dry_run:
                swap_samples(pred_input)
                n_swapped += 1
            elif swap:
                n_swapped += 1
            mapping.append({
                "pdb_id": pdb_id, "mode": mode,
                "iptm_sample_0": f"{iptm0:.4f}" if iptm0 is not None else "",
                "iptm_sample_1": f"{iptm1:.4f}" if iptm1 is not None else "",
                "best_source": f"{best}",
                "swapped": swap,
            })
            action = "SWAP" if swap else "keep"
            print(f"[{action}] {pdb_id:>6s}/{mode:<13s}  "
                  f"iptm0={iptm0:.3f}  iptm1={iptm1:.3f}  "
                  f"best=#{best}")

    out_csv = paths.output_root / "best_sample.csv"
    if not args.dry_run:
        with out_csv.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "pdb_id", "mode", "iptm_sample_0", "iptm_sample_1",
                "best_source", "swapped",
            ])
            w.writeheader()
            w.writerows(mapping)

    print(f"\n[summary] total={n_total}  swapped={n_swapped}  "
          f"single-sample only={n_missing}")
    print(f"  → best sample is now at index 0 for every complex")
    if not args.dry_run:
        print(f"  mapping: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
