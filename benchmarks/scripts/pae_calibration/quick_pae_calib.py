#!/usr/bin/env python3
"""Linear-α PAE gate calibration on Kastritis 81 Boltz predictions.

See ``docs/PAE.md`` Phase 2 for the full plan and data wireframe.

Linear-α gate — joint sweep over α and d_cut:
    contact_ij  =  1( min_heavy_atom_dist_ij + α·PAE_ij  ≤  d_cut )

The v1 single-d_cut run (commit landing this script) showed that α\*=0 at
d_cut=5.5 — adding PAE shrinks the contact set below what matches crystal.
v2 adds d_cut as a second sweep axis: a larger d_cut admits contacts that
a positive α would push past the threshold, so the joint optimum (α\*, d\*)
may be far from the 1-D α\*=0 result.

Two stages:
    A. Match PDB  — minimise Σ(IC_pae − IC_crystal)² vs dataset.json targets.
    B. Match ΔG   — LOO-CV Pearson/RMSE vs experimental DG, two coef policies:
                      B1 fixed stock PRODIGY coefficients,
                      B2 LOO-refit 6 coeffs + intercept per (α, d_cut).

+ Bootstrap (n=500) over the 81 complexes for 95% CIs on (α\*, d\*), R, RMSE.
+ Permutation null: within-complex shuffle of PAE entries, evaluated at (α\*, d\*).
+ Atom14-vs-heavy-atom parity check vs prodigy_scores.csv at (α=0, d=5.5).

Usage:
    python quick_pae_calib.py --mode msa_only                  # full 81
    python quick_pae_calib.py --mode msa_only --limit 3        # smoke test
    python quick_pae_calib.py --mode msa_only \\
        --alpha-grid 0,1.5,16 --d-cut-grid 5.5,8.0,6
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from Bio.PDB import MMCIFParser  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from protein_affinity_gpu.contacts_pae import slice_pae_inter  # noqa: E402


# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------

D_CUT_DEFAULT = 5.5

# PRODIGY "ic" classification (see utils/residue_classification.py).
# Index: 0 = Aliphatic, 1 = Charged, 2 = Polar.
IC_CHAR_IDX: dict[str, int] = {
    "ALA": 0, "CYS": 0, "GLY": 0, "PHE": 0, "ILE": 0,
    "LEU": 0, "MET": 0, "PRO": 0, "TRP": 0, "VAL": 0, "TYR": 0,
    "GLU": 1, "ASP": 1, "HIS": 1, "LYS": 1, "ARG": 1,
    "ASN": 2, "GLN": 2, "SER": 2, "THR": 2,
}

# PRODIGY IC-NIS published coefficients (scoring.py NIS_COEFFICIENTS).
COEFFS = np.array(
    [-0.09459, -0.10007, 0.19577, -0.22671, 0.18681, 0.13810],
    dtype=np.float64,
)
INTERCEPT = -15.9433

# atom14 packs all standard-AA heavy atoms into 14 slots. All 20 AAs fit
# (max is Trp at 14). If CIF parsing produces more, we truncate + warn.
ATOM14_MAX = 14

BOLTZ_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"
PRODIGY_CSV = BOLTZ_ROOT / "prodigy_scores.csv"
DATASET_NAME = "kastritis"                    # set via ``set_dataset()``


def set_dataset(name: str) -> None:
    """Swap the module-level paths/constants to ``kastritis`` or ``vreven``.

    Call this at the start of any script that uses ``load_dataset_truth``,
    ``load_stock_prodigy``, ``find_cif_and_pae`` or ``load_complex`` before
    the first load. Idempotent; re-calling with the same name is a no-op.

    Vreven has no ``dataset.json`` — the truth source is the Boltz-ready
    manifest (``manifest_boltz.csv``). ``load_dataset_truth`` handles both.
    """
    global BOLTZ_ROOT, DATASET_JSON, PRODIGY_CSV, DATASET_NAME
    if name == "kastritis":
        BOLTZ_ROOT = ROOT / "benchmarks/output/kastritis_81_boltz"
        DATASET_JSON = ROOT / "benchmarks/datasets/kastritis_81/dataset.json"
    elif name == "vreven":
        BOLTZ_ROOT = ROOT / "benchmarks/output/vreven_bm55_boltz"
        DATASET_JSON = ROOT / "benchmarks/datasets/vreven_bm55/manifest_boltz.csv"
    else:
        raise ValueError(f"unknown dataset: {name!r}")
    PRODIGY_CSV = BOLTZ_ROOT / "prodigy_scores.csv"
    DATASET_NAME = name

# Standard docking-benchmark iRMSD strata (also used by stratify_pae_calib
# and threshold_pae_calib to agree on stratum boundaries).
IRMSD_BINS_DEFAULT = [
    ("rigid",    (0.0, 1.5)),
    ("medium",   (1.5, 2.2)),
    ("flexible", (2.2, 10.0)),
]


def classify_stratum(irmsd: float,
                     bins: list = IRMSD_BINS_DEFAULT) -> str:
    for name, (lo, hi) in bins:
        if lo <= irmsd < hi:
            return name
    return bins[-1][0]    # fall-through → highest stratum


# --------------------------------------------------------------------------
# Complex dataclass
# --------------------------------------------------------------------------

@dataclass
class Complex:
    pdb_id: str
    mode: str
    min_dist: np.ndarray     # [N_t, N_b] precomputed min heavy-atom distance
    char_t: np.ndarray       # [N_t] int in {0,1,2}
    char_b: np.ndarray       # [N_b] int in {0,1,2}
    pae_ab: np.ndarray       # [N_t, N_b] inter-chain PAE
    ic_cc_crystal: int
    ic_ca_crystal: int
    ic_pp_crystal: int
    ic_pa_crystal: int
    nis_a: float
    nis_c: float
    dg_exp: float
    ba_val: float            # PRODIGY-on-crystal baseline
    irmsd: float             # iRMSD from dataset.json (flexibility class)
    ic_stock: dict           # stock PRODIGY-on-Boltz (from prodigy_scores.csv)


# --------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------

def parse_boltz_cif(cif_path: Path, chain_t: str = "A", chain_b: str = "B"):
    """Return per-chain (positions, mask, chars) for heavy atoms only."""
    parser = MMCIFParser(QUIET=True)
    struct = parser.get_structure("x", str(cif_path))
    model = next(iter(struct))

    def _extract(chain_id: str):
        chain = model[chain_id]
        pos_list, mask_list, char_list = [], [], []
        for res in chain:
            hetflag, _, _ = res.id
            if hetflag.strip():
                continue
            resname = res.get_resname()
            if resname not in IC_CHAR_IDX:
                continue
            atom_pos = np.zeros((ATOM14_MAX, 3), dtype=np.float32)
            atom_msk = np.zeros(ATOM14_MAX, dtype=bool)
            heavy = [a for a in res if (a.element or "").strip() != "H"]
            if len(heavy) > ATOM14_MAX:
                heavy = heavy[:ATOM14_MAX]
            for i, atom in enumerate(heavy):
                atom_pos[i] = atom.get_coord()
                atom_msk[i] = True
            pos_list.append(atom_pos)
            mask_list.append(atom_msk)
            char_list.append(IC_CHAR_IDX[resname])
        if not pos_list:
            raise ValueError(f"empty chain {chain_id} in {cif_path}")
        return (
            np.stack(pos_list),
            np.stack(mask_list),
            np.array(char_list, dtype=np.int8),
        )

    return _extract(chain_t), _extract(chain_b)


def load_pae_npz(npz_path: Path) -> np.ndarray:
    """Load Boltz-format PAE (single-key ``pae`` [L, L])."""
    return np.load(npz_path)["pae"].astype(np.float32)


def find_cif_and_pae(mode: str, pdb_id: str) -> tuple[Path, Path] | None:
    pred = (BOLTZ_ROOT / mode / f"{pdb_id}_{mode}"
            / "boltz_results_input" / "predictions" / "input")
    cif = next(pred.glob("*_model_0.cif"), None)
    pae = next(pred.glob("pae_*_model_0.npz"), None)
    if cif is None or pae is None:
        return None
    return cif, pae


def load_dataset_truth() -> dict:
    """Return ``{pdb_id: {DG, ba_val, iRMSD, nis_a, nis_c, CC, AC, PP, AP, ...}}``.

    Kastritis ships a pre-computed JSON with crystal IC/NIS values. Vreven
    has only a Boltz-ready manifest; crystal IC/NIS are unavailable so
    those fields are stubbed with Boltz-mode values from
    ``prodigy_scores.csv`` (msa_only) as a first approximation — this
    matches how the downstream scripts use ``nis_a`` / ``nis_c`` as
    complex-level features rather than literal crystal statistics.
    """
    if DATASET_JSON.suffix == ".json":
        return json.loads(DATASET_JSON.read_text())
    # Vreven: CSV manifest + Boltz NIS substitution.
    boltz_nis: dict[str, tuple[float, float]] = {}
    if PRODIGY_CSV.exists():
        with PRODIGY_CSV.open() as f:
            for row in csv.DictReader(f):
                if row["mode"] != "msa_only":
                    continue
                boltz_nis[row["pdb_id"]] = (
                    float(row["nis_a"]), float(row["nis_c"]),
                )
    out: dict = {}
    with DATASET_JSON.open() as f:
        for row in csv.DictReader(f):
            pid = row["pdb_id"]
            nis = boltz_nis.get(pid, (0.0, 0.0))
            ba_val_raw = row.get("ba_val_prodigy", "") or ""
            ba_val = float(ba_val_raw) if ba_val_raw.strip() else float("nan")
            out[pid] = {
                "DG": float(row["dg_exp"]),
                "ba_val": ba_val,
                "iRMSD": float(row["irmsd"]),
                "nis_a": nis[0],
                "nis_c": nis[1],
                # Crystal IC not available for Vreven without running PRODIGY
                # on each bound crystal — stub as 0.
                "CC": 0, "AC": 0, "PP": 0, "AP": 0, "AA": 0, "CP": 0,
                "Functional_class": row.get("functional_class", ""),
                "BSA": float(row.get("bsa") or 0) if row.get("bsa") else 0.0,
            }
    return out


def load_stock_prodigy() -> dict:
    out: dict = {}
    with PRODIGY_CSV.open() as f:
        for row in csv.DictReader(f):
            out[(row["pdb_id"], row["mode"])] = {
                "ic_cc": float(row["ic_cc"]),
                "ic_ca": float(row["ic_ca"]),
                "ic_pp": float(row["ic_pp"]),
                "ic_pa": float(row["ic_pa"]),
                "nis_a": float(row["nis_a"]),
                "nis_c": float(row["nis_c"]),
                "dg_pred_boltz": float(row["dg_pred_boltz"]),
            }
    return out


# --------------------------------------------------------------------------
# Kernel
# --------------------------------------------------------------------------

def min_heavy_dist(pos_t: np.ndarray, pos_b: np.ndarray,
                   mask_t: np.ndarray, mask_b: np.ndarray) -> np.ndarray:
    """[N_t, N_b] min pairwise heavy-atom distance. NumPy 5-D diff."""
    diff = pos_t[:, None, :, None, :] - pos_b[None, :, None, :, :]
    dist2 = np.sum(diff ** 2, axis=-1)
    atom_valid = mask_t[:, None, :, None] & mask_b[None, :, None, :]
    dist = np.sqrt(np.maximum(dist2, 0.0))
    dist = np.where(atom_valid, dist, np.inf)
    return np.min(dist, axis=(2, 3))


def contacts_alpha(min_dist: np.ndarray, pae_ab: np.ndarray,
                   alpha: float, d_cut: float) -> np.ndarray:
    return (min_dist + alpha * pae_ab) <= d_cut


def classify_ic(contacts: np.ndarray, char_t: np.ndarray,
                char_b: np.ndarray) -> tuple[int, int, int, int]:
    """PRODIGY 'ic' 4-channel sum: (cc, ca, pp, pa)."""
    cbool = contacts.astype(np.int32)

    def _sum(ti: int, bj: int) -> int:
        sel = (char_t[:, None] == ti) & (char_b[None, :] == bj)
        return int((cbool * sel).sum())

    ic_cc = _sum(1, 1)
    ic_pp = _sum(2, 2)
    ic_ca = _sum(0, 1) + _sum(1, 0)
    ic_pa = _sum(0, 2) + _sum(2, 0)
    return ic_cc, ic_ca, ic_pp, ic_pa


# --------------------------------------------------------------------------
# Complex loader
# --------------------------------------------------------------------------

def load_complex(pdb_id: str, mode: str,
                 truth: dict, stock: dict) -> Complex | None:
    paths = find_cif_and_pae(mode, pdb_id)
    if paths is None:
        print(f"[skip] {pdb_id}/{mode}: missing CIF or PAE")
        return None
    cif, pae = paths
    try:
        (pos_t, mask_t, char_t), (pos_b, mask_b, char_b) = parse_boltz_cif(cif)
    except (KeyError, ValueError) as exc:
        print(f"[skip] {pdb_id}: CIF parse failed — {exc}")
        return None
    pae_full = load_pae_npz(pae)
    n_t, n_b = pos_t.shape[0], pos_b.shape[0]
    if pae_full.shape[0] < n_t + n_b:
        print(f"[skip] {pdb_id}: PAE shape {pae_full.shape} < parsed "
              f"N_t+N_b = {n_t+n_b}")
        return None
    pae_ab = slice_pae_inter(pae_full, n_t, n_b, symmetrize=True).astype(np.float32)
    md = min_heavy_dist(pos_t, pos_b, mask_t, mask_b)

    t = truth[pdb_id]
    return Complex(
        pdb_id=pdb_id, mode=mode,
        min_dist=md, char_t=char_t, char_b=char_b, pae_ab=pae_ab,
        ic_cc_crystal=int(t["CC"]), ic_ca_crystal=int(t["AC"]),
        ic_pp_crystal=int(t["PP"]), ic_pa_crystal=int(t["AP"]),
        nis_a=float(t["nis_a"]), nis_c=float(t["nis_c"]),
        dg_exp=float(t["DG"]), ba_val=float(t["ba_val"]),
        irmsd=float(t["iRMSD"]),
        ic_stock=stock.get((pdb_id, mode), {}),
    )


# --------------------------------------------------------------------------
# Sweeps & metrics
# --------------------------------------------------------------------------

def sweep_ic(complexes: list[Complex], alpha_grid: np.ndarray,
             d_cut_grid: np.ndarray,
             pae_override: list[np.ndarray] | None = None
             ) -> np.ndarray:
    """[N, A_α, A_d, 4] IC counts over the 2-D (α, d_cut) grid.

    ``pae_override`` swaps in alternate PAE matrices per complex (used by
    permutation null).
    """
    N = len(complexes)
    Aa = len(alpha_grid); Ad = len(d_cut_grid)
    ic = np.zeros((N, Aa, Ad, 4), dtype=np.int32)
    for ci, comp in enumerate(complexes):
        pae = pae_override[ci] if pae_override is not None else comp.pae_ab
        # Precompute the effective distance for each alpha: [Aa, N_t, N_b].
        eff = comp.min_dist[None, :, :] + alpha_grid[:, None, None] * pae[None, :, :]
        for di, d_cut in enumerate(d_cut_grid):
            mask = eff <= float(d_cut)         # [Aa, N_t, N_b]
            for ai in range(Aa):
                ic[ci, ai, di] = classify_ic(
                    mask[ai], comp.char_t, comp.char_b
                )
    return ic


def sanity_check_stock(complexes: list[Complex], ic_sweep: np.ndarray,
                       alpha_grid: np.ndarray, d_cut_grid: np.ndarray,
                       tol: float = 3.0) -> list[tuple]:
    """Find entries where recomputed IC at (α≈0, d_cut≈5.5) differs from
    prodigy_scores.csv by > ``tol``. Probes atom14 vs heavy-atom parity.
    """
    a0 = int(np.argmin(np.abs(alpha_grid - 0.0)))
    d0 = int(np.argmin(np.abs(d_cut_grid - 5.5)))
    errs: list[tuple] = []
    channels = ("ic_cc", "ic_ca", "ic_pp", "ic_pa")
    for ci, comp in enumerate(complexes):
        if not comp.ic_stock:
            continue
        for chi, name in enumerate(channels):
            got = float(ic_sweep[ci, a0, d0, chi])
            want = comp.ic_stock[name]
            if abs(got - want) > tol:
                errs.append((comp.pdb_id, name, got, want))
    return errs


def stage_a_loss(complexes: list[Complex], ic_sweep: np.ndarray) -> np.ndarray:
    """[A_α, A_d, 5]: (total_MSE, cc, ca, pp, pa) MSE over complexes."""
    crystal = np.array([
        [c.ic_cc_crystal, c.ic_ca_crystal, c.ic_pp_crystal, c.ic_pa_crystal]
        for c in complexes
    ], dtype=np.int32)                            # [N, 4]
    res = ic_sweep - crystal[:, None, None, :]    # [N, A_α, A_d, 4]
    per_chan = np.mean(res ** 2, axis=0)          # [A_α, A_d, 4]
    total = per_chan.sum(axis=-1)                 # [A_α, A_d]
    return np.concatenate([total[..., None], per_chan], axis=-1)  # [A_α, A_d, 5]


def _features_at_alpha(complexes: list[Complex], ic_alpha: np.ndarray
                       ) -> np.ndarray:
    nis = np.array([[c.nis_a, c.nis_c] for c in complexes], dtype=np.float64)
    return np.concatenate([ic_alpha.astype(np.float64),
                           np.clip(nis, 0, 100)], axis=1)


def _metrics_2d(dg_pred: np.ndarray, dg_exp: np.ndarray) -> dict:
    """Compute R, RMSE, MAE across the last two axes (α, d_cut)."""
    Aa, Ad = dg_pred.shape[1:]
    R = np.zeros((Aa, Ad))
    for ai in range(Aa):
        for di in range(Ad):
            x = dg_pred[:, ai, di]
            if np.std(x) == 0:
                R[ai, di] = np.nan
            else:
                R[ai, di] = pearsonr(x, dg_exp)[0]
    resid = dg_pred - dg_exp[:, None, None]
    RMSE = np.sqrt(np.mean(resid ** 2, axis=0))
    MAE = np.mean(np.abs(resid), axis=0)
    return {"dg_pred": dg_pred, "dg_exp": dg_exp, "R": R, "RMSE": RMSE, "MAE": MAE}


def stage_b_fixed(complexes: list[Complex], ic_sweep: np.ndarray,
                  alpha_grid: np.ndarray, d_cut_grid: np.ndarray) -> dict:
    """Fixed PRODIGY coefficients → dg_pred [N, A_α, A_d], metrics [A_α, A_d]."""
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    N, Aa, Ad, _ = ic_sweep.shape
    dg_pred = np.zeros((N, Aa, Ad), dtype=np.float64)
    for ai in range(Aa):
        for di in range(Ad):
            X = _features_at_alpha(complexes, ic_sweep[:, ai, di])
            dg_pred[:, ai, di] = X @ COEFFS + INTERCEPT
    return _metrics_2d(dg_pred, dg_exp)


def stage_b_refit_loo(complexes: list[Complex], ic_sweep: np.ndarray,
                      alpha_grid: np.ndarray, d_cut_grid: np.ndarray) -> dict:
    """LOO-refit 7 params per (α, d_cut)."""
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    N, Aa, Ad, _ = ic_sweep.shape
    dg_pred = np.zeros((N, Aa, Ad), dtype=np.float64)
    for ai in range(Aa):
        for di in range(Ad):
            X = _features_at_alpha(complexes, ic_sweep[:, ai, di])
            X_aug = np.concatenate([X, np.ones((N, 1))], axis=1)
            for i in range(N):
                mask = np.arange(N) != i
                beta, *_ = np.linalg.lstsq(X_aug[mask], dg_exp[mask], rcond=None)
                dg_pred[i, ai, di] = X_aug[i] @ beta
    return _metrics_2d(dg_pred, dg_exp)


def bootstrap_R_RMSE(dg_pred: np.ndarray, dg_exp: np.ndarray,
                     n: int = 500, seed: int = 0) -> tuple[tuple, tuple]:
    rng = np.random.default_rng(seed)
    N = len(dg_exp)
    Rs, Es = [], []
    for _ in range(n):
        idx = rng.integers(0, N, size=N)
        Rs.append(pearsonr(dg_pred[idx], dg_exp[idx])[0])
        Es.append(np.sqrt(np.mean((dg_pred[idx] - dg_exp[idx]) ** 2)))
    return (
        (float(np.percentile(Rs, 2.5)), float(np.percentile(Rs, 97.5))),
        (float(np.percentile(Es, 2.5)), float(np.percentile(Es, 97.5))),
    )


def permutation_null(complexes: list[Complex], alpha_grid: np.ndarray,
                     d_cut_grid: np.ndarray,
                     n_perm: int = 50, seed: int = 0) -> dict:
    """Within-complex PAE shuffle. Breaks the (i,j)↔PAE_ij correspondence.

    Fast + shape-agnostic. Returns 2-D-per-permutation R / L_A tensors.
    """
    rng = np.random.default_rng(seed)
    dg_exp = np.array([c.dg_exp for c in complexes], dtype=np.float64)
    Aa = len(alpha_grid); Ad = len(d_cut_grid)
    perm_R = np.zeros((n_perm, Aa, Ad))
    perm_LA = np.zeros((n_perm, Aa, Ad))

    crystal = np.array([
        [c.ic_cc_crystal, c.ic_ca_crystal, c.ic_pp_crystal, c.ic_pa_crystal]
        for c in complexes
    ], dtype=np.int32)

    for pi in range(n_perm):
        idx_maps = [
            rng.permutation(c.pae_ab.size).astype(np.int64)
            for c in complexes
        ]
        shuffled_pae = [
            comp.pae_ab.flat[idx_map].reshape(comp.pae_ab.shape)
            for comp, idx_map in zip(complexes, idx_maps)
        ]
        ic_perm = sweep_ic(complexes, alpha_grid, d_cut_grid,
                           pae_override=shuffled_pae)
        # Sum squared residuals over channel axis, mean over complex axis.
        perm_LA[pi] = np.mean(
            np.sum((ic_perm - crystal[:, None, None, :]) ** 2, axis=-1),
            axis=0,
        )
        for ai in range(Aa):
            for di in range(Ad):
                X = _features_at_alpha(complexes, ic_perm[:, ai, di])
                dg_pred = X @ COEFFS + INTERCEPT
                if np.std(dg_pred) == 0:
                    perm_R[pi, ai, di] = np.nan
                else:
                    perm_R[pi, ai, di] = pearsonr(dg_pred, dg_exp)[0]

    return {"perm_R": perm_R, "perm_LA": perm_LA}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def _heatmap(ax, data: np.ndarray, alpha_grid: np.ndarray,
             d_cut_grid: np.ndarray, title: str, cbar_label: str,
             cmap: str = "viridis", mark_min: bool = True,
             mark_max: bool = False):
    """Shared heatmap helper. x = α, y = d_cut. data: [A_α, A_d]."""
    im = ax.pcolormesh(alpha_grid, d_cut_grid, data.T, cmap=cmap, shading="auto")
    ax.set_xlabel("α")
    ax.set_ylabel("d_cut (Å)")
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label, fontsize=9)
    if mark_min:
        ai, di = np.unravel_index(np.nanargmin(data), data.shape)
        ax.plot(alpha_grid[ai], d_cut_grid[di], "rx", ms=12, mew=2,
                label=f"min @ α={alpha_grid[ai]:.2f}, d={d_cut_grid[di]:.2f}")
        ax.legend(fontsize=8, loc="upper right")
    if mark_max:
        ai, di = np.unravel_index(np.nanargmax(data), data.shape)
        ax.plot(alpha_grid[ai], d_cut_grid[di], "r+", ms=14, mew=2,
                label=f"max @ α={alpha_grid[ai]:.2f}, d={d_cut_grid[di]:.2f}")
        ax.legend(fontsize=8, loc="upper right")


def plot_stage_a(loss_A: np.ndarray, alpha_grid: np.ndarray,
                 d_cut_grid: np.ndarray, path: Path):
    """5 panels: total MSE + 4 per-channel MSE heatmaps."""
    fig, axes = plt.subplots(1, 5, figsize=(22, 4.2))
    _heatmap(axes[0], loss_A[..., 0], alpha_grid, d_cut_grid,
             "Stage A · total IC MSE", "L_A(α, d_cut)")
    for i, lab in enumerate(("cc", "ca", "pp", "pa")):
        _heatmap(axes[i + 1], loss_A[..., i + 1], alpha_grid, d_cut_grid,
                 f"ic_{lab}", "MSE")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_stage_b(b1: dict, b2: dict, alpha_grid: np.ndarray,
                 d_cut_grid: np.ndarray,
                 crystal_ref: dict, stock_ref: dict, path: Path):
    """2x2 heatmap grid: rows = (B1, B2), cols = (R, RMSE).

    Reference numbers (crystal, Boltz-stock) are shown in the panel titles.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for row, (lbl, st) in enumerate((("B1 fixed coeffs", b1),
                                      ("B2 LOO-refit coeffs", b2))):
        _heatmap(
            axes[row, 0], st["R"], alpha_grid, d_cut_grid,
            f"{lbl} — Pearson R\n"
            f"(crystal={crystal_ref['R']:.2f}, stock={stock_ref['R']:.2f})",
            "R", cmap="viridis", mark_min=False, mark_max=True,
        )
        _heatmap(
            axes[row, 1], st["RMSE"], alpha_grid, d_cut_grid,
            f"{lbl} — RMSE (kcal/mol)\n"
            f"(crystal={crystal_ref['RMSE']:.2f}, stock={stock_ref['RMSE']:.2f})",
            "RMSE", cmap="viridis_r", mark_min=True, mark_max=False,
        )
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_scatter_best(complexes: list[Complex], b1: dict,
                      best_ai: int, best_di: int,
                      alpha_grid: np.ndarray, d_cut_grid: np.ndarray,
                      path: Path):
    dg_exp = np.array([c.dg_exp for c in complexes])
    dg_crystal = np.array([c.ba_val for c in complexes])
    dg_stock = np.array([c.ic_stock.get("dg_pred_boltz", np.nan)
                         for c in complexes])
    dg_pae = b1["dg_pred"][:, best_ai, best_di]
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharex=True, sharey=True)
    for ax, pred, ttl in zip(
        axes, (dg_crystal, dg_stock, dg_pae),
        ("Crystal (ba_val)", "Boltz stock (α=0, d=5.5)",
         f"Boltz + PAE (α={alpha_grid[best_ai]:.2f}, d={d_cut_grid[best_di]:.2f})"),
    ):
        valid = ~np.isnan(pred)
        r = pearsonr(pred[valid], dg_exp[valid])[0]
        rmse = np.sqrt(np.mean((pred[valid] - dg_exp[valid]) ** 2))
        ax.scatter(dg_exp[valid], pred[valid], alpha=0.6, s=22)
        lo = min(dg_exp.min(), np.nanmin(pred)) - 1
        hi = max(dg_exp.max(), np.nanmax(pred)) + 1
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_title(f"{ttl}\nR={r:.2f}  RMSE={rmse:.2f}")
        ax.set_xlabel("ΔG_exp (kcal/mol)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("ΔG_pred (kcal/mol)")
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def plot_null(loss_A_real: np.ndarray, perm: dict,
              alpha_grid: np.ndarray, d_cut_grid: np.ndarray,
              best_ai: int, best_di: int, real_R: float, path: Path):
    """3-panel null visualization:
       (a) real L_A heatmap
       (b) shuffled-mean L_A heatmap
       (c) histogram of null R at (α*, d_cut*)
    """
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    _heatmap(axes[0], loss_A_real[..., 0], alpha_grid, d_cut_grid,
             "Stage A · real PAE  (L_A)", "L_A")
    _heatmap(axes[1], perm["perm_LA"].mean(axis=0), alpha_grid, d_cut_grid,
             f"Stage A · shuffled PAE (mean, n={perm['perm_LA'].shape[0]})",
             "L_A")
    null_Rs = perm["perm_R"][:, best_ai, best_di]
    null_Rs = null_Rs[~np.isnan(null_Rs)]
    axes[2].hist(null_Rs, bins=20, color="C7", alpha=0.7,
                 label=f"null R(α*,d*)  n={len(null_Rs)}")
    axes[2].axvline(real_R, color="red", linestyle="--", lw=1.5,
                    label=f"real R = {real_R:.3f}")
    p = float((null_Rs >= real_R).mean()) if len(null_Rs) else float("nan")
    axes[2].set_xlabel(f"Pearson R at α={alpha_grid[best_ai]:.2f}, "
                       f"d_cut={d_cut_grid[best_di]:.2f}")
    axes[2].set_ylabel("count")
    axes[2].set_title(f"Permutation p = {p:.3f}")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


# --------------------------------------------------------------------------
# Writers
# --------------------------------------------------------------------------

def write_grid_csv(path: Path, complexes: list[Complex],
                   ic_sweep: np.ndarray, alpha_grid: np.ndarray,
                   d_cut_grid: np.ndarray, b1: dict, b2: dict):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "pdb_id", "mode", "alpha", "d_cut",
            "ic_cc", "ic_ca", "ic_pp", "ic_pa",
            "dg_pred_b1_fixed", "dg_pred_b2_loo", "dg_exp", "ba_val_crystal",
        ])
        for ci, comp in enumerate(complexes):
            for ai, alpha in enumerate(alpha_grid):
                for di, d_cut in enumerate(d_cut_grid):
                    ic = ic_sweep[ci, ai, di]
                    w.writerow([
                        comp.pdb_id, comp.mode,
                        f"{alpha:.4f}", f"{d_cut:.3f}",
                        int(ic[0]), int(ic[1]), int(ic[2]), int(ic[3]),
                        f"{b1['dg_pred'][ci, ai, di]:.3f}",
                        f"{b2['dg_pred'][ci, ai, di]:.3f}",
                        f"{comp.dg_exp:.3f}", f"{comp.ba_val:.3f}",
                    ])


def write_summary(path: Path, complexes: list[Complex],
                  alpha_grid: np.ndarray, d_cut_grid: np.ndarray,
                  loss_A: np.ndarray, b1: dict, b2: dict, perm: dict,
                  R_CIs: dict, RMSE_CIs: dict,
                  stage_a_best: tuple[float, float],
                  best_ad_b1: tuple[int, int], best_ad_b2: tuple[int, int],
                  perm_p_value: float, sanity_errs: list,
                  crystal_ref: dict, stock_ref: dict,
                  stock_ad: tuple[int, int]):
    def _ci(c): return f"[{c[0]:.2f}, {c[1]:.2f}]"
    a1, d1 = best_ad_b1
    a2, d2 = best_ad_b2
    a0, d0 = stock_ad
    Rp1 = b1["R"][a1, d1]; Ep1 = b1["RMSE"][a1, d1]
    Rp2 = b2["R"][a2, d2]; Ep2 = b2["RMSE"][a2, d2]
    Rst = b1["R"][a0, d0]; Est = b1["RMSE"][a0, d0]
    alpha_star, d_cut_star = stage_a_best
    lines = [
        f"# PAE calibration — mode={complexes[0].mode}, N={len(complexes)}",
        "",
        f"α grid: {alpha_grid[0]:.2f} … {alpha_grid[-1]:.2f}  "
        f"(n={len(alpha_grid)})",
        f"d_cut grid: {d_cut_grid[0]:.2f} … {d_cut_grid[-1]:.2f} Å  "
        f"(n={len(d_cut_grid)})",
        "",
        "## ΔG prediction",
        "",
        "| Config | α | d_cut | R | RMSE | R 95% CI | RMSE 95% CI |",
        "|---|---:|---:|---:|---:|---|---|",
        (f"| Crystal (`ba_val`) | — | — | {crystal_ref['R']:.3f} | "
         f"{crystal_ref['RMSE']:.3f} | — | — |"),
        (f"| Boltz stock (α=0, d=5.5) | 0 | 5.50 | {Rst:.3f} | {Est:.3f} | "
         "— | — |"),
        (f"| Boltz+PAE · B1 fixed | {alpha_grid[a1]:.2f} | "
         f"{d_cut_grid[d1]:.2f} | {Rp1:.3f} | {Ep1:.3f} | "
         f"{_ci(R_CIs['B1'])} | {_ci(RMSE_CIs['B1'])} |"),
        (f"| Boltz+PAE · B2 LOO-refit | {alpha_grid[a2]:.2f} | "
         f"{d_cut_grid[d2]:.2f} | {Rp2:.3f} | {Ep2:.3f} | "
         f"{_ci(R_CIs['B2'])} | {_ci(RMSE_CIs['B2'])} |"),
        "",
        "## Stage A — match crystal IC",
        "",
        f"- Joint minimum (α\\*, d_cut\\*): "
        f"**α={alpha_star:.2f}, d_cut={d_cut_star:.2f} Å**",
        f"- L_A at (α=0, d=5.5) stock: {loss_A[a0, d0, 0]:.2f}",
        f"- L_A at (α\\*, d_cut\\*) min: {np.nanmin(loss_A[..., 0]):.2f}",
        "",
        "## Permutation null",
        "",
        f"- Permutations: {perm['perm_R'].shape[0]}",
        f"- p(R_null ≥ R_real at B1 optimum α={alpha_grid[a1]:.2f}, "
        f"d={d_cut_grid[d1]:.2f}) = **{perm_p_value:.3f}**",
    ]
    if sanity_errs:
        lines += [
            "",
            "## ⚠ Atom14 parity warnings",
            "",
            f"{len(sanity_errs)} IC entries differ > tol from "
            "`prodigy_scores.csv` at (α=0, d=5.5) (first 10):",
            "",
            "| pdb | channel | recomputed | csv | Δ |",
            "|---|---|---:|---:|---:|",
        ]
        for pid, chan, got, want in sanity_errs[:10]:
            lines.append(f"| {pid} | {chan} | {got:.1f} | {want:.1f} | "
                         f"{got - want:+.1f} |")
    path.write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--mode", default="msa_only",
                    choices=["msa_only", "template_msa"])
    ap.add_argument("--alpha-grid", default="0,1.5,16",
                    help="'lo,hi,n' → np.linspace(lo, hi, n)")
    ap.add_argument("--d-cut-grid", default="5.5,8.0,6",
                    help="'lo,hi,n' → np.linspace(lo, hi, n)  Å")
    ap.add_argument("--limit", type=int, default=0,
                    help="process first N pdb_ids only (0 = all)")
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--n-perm", type=int, default=50)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    a_lo, a_hi, a_n = args.alpha_grid.split(",")
    alpha_grid = np.linspace(float(a_lo), float(a_hi), int(a_n))
    d_lo, d_hi, d_n = args.d_cut_grid.split(",")
    d_cut_grid = np.linspace(float(d_lo), float(d_hi), int(d_n))

    out_dir = (Path(args.out_dir) if args.out_dir
               else BOLTZ_ROOT / "pae_calibration" / args.mode)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[cfg] mode={args.mode}  "
          f"α∈[{a_lo},{a_hi}] n={a_n}  "
          f"d_cut∈[{d_lo},{d_hi}] Å n={d_n}  "
          f"bootstrap={args.bootstrap}  n_perm={args.n_perm}")
    print(f"[cfg] out_dir={out_dir.relative_to(ROOT)}")

    truth = load_dataset_truth()
    stock = load_stock_prodigy()
    pdb_ids = sorted(truth.keys())
    if args.limit > 0:
        pdb_ids = pdb_ids[:args.limit]

    t0 = time.time()
    complexes: list[Complex] = []
    for pid in pdb_ids:
        c = load_complex(pid, args.mode, truth, stock)
        if c is not None:
            complexes.append(c)
    if not complexes:
        print("[fatal] no complexes loaded"); sys.exit(1)
    print(f"[data] loaded {len(complexes)}/{len(pdb_ids)} complexes "
          f"in {time.time() - t0:.1f}s")

    t0 = time.time()
    ic_sweep = sweep_ic(complexes, alpha_grid, d_cut_grid)
    print(f"[sweep] {len(complexes)}×{len(alpha_grid)}×{len(d_cut_grid)} IC "
          f"in {time.time() - t0:.1f}s")

    sanity_errs = sanity_check_stock(
        complexes, ic_sweep, alpha_grid, d_cut_grid, tol=3.0,
    )
    if sanity_errs:
        n_tot = 4 * sum(1 for c in complexes if c.ic_stock)
        print(f"[WARN] {len(sanity_errs)}/{n_tot} IC mismatches at "
              f"(α≈0, d=5.5) — atom14 vs heavy-atom parity:")
        for pid, chan, got, want in sanity_errs[:6]:
            print(f"       {pid:6s} {chan}: recomp={got:.1f}  csv={want:.1f}")
    else:
        print("[ok] atom14 vs heavy-atom parity: all channels within tol "
              "at (α≈0, d=5.5)")

    loss_A = stage_a_loss(complexes, ic_sweep)
    ai_a, di_a = np.unravel_index(np.nanargmin(loss_A[..., 0]),
                                   loss_A[..., 0].shape)
    stage_a_best = (float(alpha_grid[ai_a]), float(d_cut_grid[di_a]))
    print(f"[stageA] argmin(L_A): α*={stage_a_best[0]:.3f} "
          f"d_cut*={stage_a_best[1]:.3f}  "
          f"L_A min={loss_A[ai_a, di_a, 0]:.2f}")

    t0 = time.time()
    b1 = stage_b_fixed(complexes, ic_sweep, alpha_grid, d_cut_grid)
    b2 = stage_b_refit_loo(complexes, ic_sweep, alpha_grid, d_cut_grid)
    print(f"[stageB] fixed + LOO-refit in {time.time() - t0:.1f}s")

    def _argmax2d(arr): return np.unravel_index(np.nanargmax(arr), arr.shape)

    best_ad_b1 = _argmax2d(b1["R"])
    best_ad_b2 = _argmax2d(b2["R"])
    a1, d1 = best_ad_b1; a2, d2 = best_ad_b2
    print(f"         B1 max R={b1['R'][a1, d1]:.3f} @ α={alpha_grid[a1]:.2f}"
          f", d={d_cut_grid[d1]:.2f}  "
          f"RMSE there={b1['RMSE'][a1, d1]:.3f}")
    print(f"         B2 max R={b2['R'][a2, d2]:.3f} @ α={alpha_grid[a2]:.2f}"
          f", d={d_cut_grid[d2]:.2f}  "
          f"RMSE there={b2['RMSE'][a2, d2]:.3f}")

    # Stock reference in the 2-D grid at (α≈0, d=5.5).
    a_stock = int(np.argmin(np.abs(alpha_grid - 0.0)))
    d_stock = int(np.argmin(np.abs(d_cut_grid - 5.5)))
    stock_ad = (a_stock, d_stock)

    R_CI_b1, RMSE_CI_b1 = bootstrap_R_RMSE(
        b1["dg_pred"][:, a1, d1], b1["dg_exp"], n=args.bootstrap)
    R_CI_b2, RMSE_CI_b2 = bootstrap_R_RMSE(
        b2["dg_pred"][:, a2, d2], b2["dg_exp"], n=args.bootstrap)
    print(f"[boot] B1 R 95% CI {R_CI_b1}  RMSE {RMSE_CI_b1}")
    print(f"[boot] B2 R 95% CI {R_CI_b2}  RMSE {RMSE_CI_b2}")

    t0 = time.time()
    perm = permutation_null(complexes, alpha_grid, d_cut_grid,
                            n_perm=args.n_perm, seed=1)
    null_R = perm["perm_R"][:, a1, d1]
    null_R = null_R[~np.isnan(null_R)]
    perm_p = (float((null_R >= b1["R"][a1, d1]).mean())
              if len(null_R) else float("nan"))
    print(f"[perm] {args.n_perm} in {time.time() - t0:.1f}s  "
          f"p(R_null ≥ R_real @ α={alpha_grid[a1]:.2f},"
          f"d={d_cut_grid[d1]:.2f}) = {perm_p:.3f}")

    dg_exp = np.array([c.dg_exp for c in complexes])
    dg_crystal = np.array([c.ba_val for c in complexes])
    dg_stock_vec = np.array([c.ic_stock.get("dg_pred_boltz", np.nan)
                             for c in complexes])
    crystal_R = pearsonr(dg_crystal, dg_exp)[0]
    crystal_RMSE = float(np.sqrt(np.mean((dg_crystal - dg_exp) ** 2)))
    sv = ~np.isnan(dg_stock_vec)
    stock_R = pearsonr(dg_stock_vec[sv], dg_exp[sv])[0]
    stock_RMSE = float(np.sqrt(np.mean((dg_stock_vec[sv] - dg_exp[sv]) ** 2)))

    plot_stage_a(loss_A, alpha_grid, d_cut_grid,
                 out_dir / "stage_A_ic.png")
    plot_stage_b(
        b1, b2, alpha_grid, d_cut_grid,
        {"R": crystal_R, "RMSE": crystal_RMSE},
        {"R": stock_R, "RMSE": stock_RMSE},
        out_dir / "stage_B_dg.png",
    )
    plot_scatter_best(complexes, b1, a1, d1, alpha_grid, d_cut_grid,
                      out_dir / "scatter_best.png")
    plot_null(loss_A, perm, alpha_grid, d_cut_grid,
              a1, d1, b1["R"][a1, d1], out_dir / "null_perm.png")

    write_grid_csv(out_dir / "calib_grid.csv", complexes, ic_sweep,
                   alpha_grid, d_cut_grid, b1, b2)
    write_summary(
        out_dir / "summary.md",
        complexes, alpha_grid, d_cut_grid, loss_A, b1, b2, perm,
        R_CIs={"B1": R_CI_b1, "B2": R_CI_b2},
        RMSE_CIs={"B1": RMSE_CI_b1, "B2": RMSE_CI_b2},
        stage_a_best=stage_a_best,
        best_ad_b1=best_ad_b1, best_ad_b2=best_ad_b2,
        perm_p_value=perm_p, sanity_errs=sanity_errs,
        crystal_ref={"R": crystal_R, "RMSE": crystal_RMSE},
        stock_ref={"R": stock_R, "RMSE": stock_RMSE},
        stock_ad=stock_ad,
    )
    print(f"[done] artefacts in {out_dir.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
