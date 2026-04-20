"""PRODIGY IC-NIS scoring — backend-agnostic via tensor methods.

JAX arrays and tinygrad Tensors share enough method API (`.sum`, `.reshape`,
`@`, `.clip`, `.exp`) that the scoring primitives work on both without
dispatching. Only ``stack_scalars`` / ``concat`` need a shim — they live in
:mod:`utils._array`.
"""
from __future__ import annotations

from dataclasses import dataclass

from tinygrad import Tensor as _TGTensor

from .utils._array import Array, exp, stack_scalars


@dataclass(frozen=True)
class NISCoefficients:
    """PRODIGY IC-NIS linear model coefficients."""

    ic_cc: float = -0.09459
    ic_ca: float = -0.10007
    ic_pp: float = 0.19577
    ic_pa: float = -0.22671
    p_nis_a: float = 0.18681
    p_nis_c: float = 0.13810
    intercept: float = -15.9433

    def as_tuple(self) -> tuple[float, ...]:
        """Coefficient vector in the order expected by :func:`score_ic_nis`."""
        return (self.ic_cc, self.ic_ca, self.ic_pp, self.ic_pa, self.p_nis_a, self.p_nis_c)


NIS_COEFFICIENTS = NISCoefficients()


def get_atom_radii(
    sequence_one_hot: Array,
    residue_radii_matrix: Array,
    atom_mask: Array | None = None,
) -> Array:
    """Project residue probabilities into per-atom vdW radii.

    ``atom_mask`` is optional — when the radii table already zeros padding
    slots (the atom14 library does), the mask is redundant. Kept for the
    atom37 path where padding carries nonzero radii.
    """
    radii = (sequence_one_hot @ residue_radii_matrix).reshape(-1)
    if atom_mask is None:
        return radii
    return radii * atom_mask.reshape(-1)


def calculate_relative_sasa(
    complex_sasa: Array,
    sequence_probabilities: Array,
    relative_sasa_array: Array,
    atoms_per_residue: int,
) -> Array:
    """Residue-level relative SASA = per-residue sum / reference total."""
    residue_sasa = complex_sasa.reshape(-1, atoms_per_residue).sum(axis=1)
    reference_sasa = sequence_probabilities @ relative_sasa_array
    return residue_sasa / (reference_sasa + 1e-8)


def calculate_nis_percentages(
    sasa_values: Array,
    sequence_probabilities: Array,
    character_matrix: Array,
    threshold: float = 0.05,
) -> Array:
    """Percentage of non-interacting surface per residue character."""
    residue_classes = sequence_probabilities @ character_matrix
    # `* 1.0` promotes bool → float on both jax and tinygrad without needing
    # a backend-specific ``.astype`` / ``.float()``.
    mask = (sasa_values >= threshold) * 1.0
    counts = (residue_classes * mask[:, None]).sum(axis=0)
    total = mask.sum() + 1e-8
    return 100.0 * counts / total


def score_ic_nis(
    ic_cc: Array,
    ic_ca: Array,
    ic_pp: Array,
    ic_pa: Array,
    p_nis_a: Array,
    p_nis_c: Array,
    coeffs: Array,
    intercept: Array,
) -> Array:
    """ΔG via the PRODIGY IC-NIS linear model."""
    p_nis_a = p_nis_a.clip(0.0, 100.0)
    p_nis_c = p_nis_c.clip(0.0, 100.0)
    inputs = stack_scalars(ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c)
    return (coeffs * inputs).sum() + intercept


def dg_to_kd(dg: Array, temperature: float = 25.0) -> Array:
    """Convert ΔG (kcal/mol) to Kd (M)."""
    gas_constant = 0.0019858775
    temperature_kelvin = temperature + 273.15
    return exp(dg.clip(-100.0, 100.0) / (gas_constant * temperature_kelvin))


def coefficient_tensors_tinygrad(
    coefficients: NISCoefficients = NIS_COEFFICIENTS,
) -> tuple[_TGTensor, _TGTensor]:
    """PRODIGY IC-NIS coefficient vector + intercept as tinygrad tensors."""
    coeffs = _TGTensor(list(coefficients.as_tuple()))
    intercept = _TGTensor([coefficients.intercept])
    return coeffs, intercept
