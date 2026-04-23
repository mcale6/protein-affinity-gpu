"""Dataset registry for the Boltz-pipeline scripts.

Centralises every path that differs between calibration benchmarks, so
scripts 03–06 can take a single ``--dataset {kastritis, vreven}`` flag
instead of duplicating path constants.

## iRMSD conventions (important — see docs/PAE.md)

There are **two** iRMSDs floating around the pipeline; don't confuse them:

1. **Benchmark iRMSD** — bound vs unbound conformational change
   (the flexibility indicator published in Kastritis 2011 / Vreven 2015).
   Lives in ``<manifest>.irmsd``. Used for *stratification* (rigid / medium /
   difficult). Depends on which unbound reference the benchmark chose, so
   Kastritis and Vreven can disagree on a small number of PDBs (e.g. 1ACB:
   Kastritis 1.08 Å vs Vreven 2.26 Å).

2. **Measured iRMSD** — Boltz-prediction vs crystal, computed per-run by
   USalign in step 05. Reflects *prediction quality* for a single AF run.
   Distinct concept, separate column (``irmsd_boltz`` or similar in the
   per-run CSVs).

When comparing results across datasets use the *matching* benchmark iRMSD
(Vreven's for Vreven analysis, Kastritis's for Kastritis analysis) — they
represent different reference choices but answer the same question.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

ROOT = Path(__file__).resolve().parents[3]

DatasetName = Literal["kastritis", "vreven"]


@dataclass(frozen=True)
class DatasetPaths:
    """Paths + metadata for a Boltz-pipeline calibration target."""

    name: str
    display: str                 # human-readable label for logs/plots
    manifest: Path               # must have: pdb_id, seq_target, seq_binder, len_*, dg_exp
    cleaned_dir: Path            # {pdb_id}_AB.pdb + {pdb_id}_AB.cif land here (shared w/ step 1)
    yaml_root: Path              # {mode}/{pdb_id}.yaml written by step 3
    output_root: Path            # boltz predictions + downstream CSVs/plots
    irmsd_source: str            # provenance of the stratification iRMSD column
    has_prodigy_baseline: bool   # True if manifest has ba_val_prodigy column


_DATASETS: dict[str, DatasetPaths] = {
    "kastritis": DatasetPaths(
        name="kastritis",
        display="Kastritis 81",
        manifest=ROOT / "benchmarks/datasets/kastritis_81/manifest.csv",
        cleaned_dir=ROOT / "benchmarks/downloads/kastritis_81/cleaned",
        yaml_root=ROOT / "benchmarks/downloads/kastritis_81_boltz_inputs",
        output_root=ROOT / "benchmarks/output/kastritis_81_boltz",
        irmsd_source="Kastritis 2011 (bound vs unbound heavy-atom)",
        has_prodigy_baseline=True,
    ),
    "vreven": DatasetPaths(
        name="vreven",
        display="Vreven BM5.5 (affinity subset, N=106)",
        manifest=ROOT / "benchmarks/datasets/vreven_bm55/manifest_boltz.csv",
        cleaned_dir=ROOT / "benchmarks/downloads/vreven_bm55/cleaned",
        yaml_root=ROOT / "benchmarks/downloads/vreven_bm55_boltz_inputs",
        output_root=ROOT / "benchmarks/output/vreven_bm55_boltz",
        irmsd_source="Vreven 2015 Table_BM5.5.xlsx (bound vs unbound Cα)",
        has_prodigy_baseline=False,
    ),
}

AVAILABLE = tuple(_DATASETS.keys())


def get_paths(name: str) -> DatasetPaths:
    if name not in _DATASETS:
        raise SystemExit(
            f"unknown dataset {name!r}; available: {', '.join(AVAILABLE)}"
        )
    return _DATASETS[name]
