import jax.numpy as jnp
from tinygrad import Tensor as _TGTensor


NIS_CONSTANTS = {
    "ic_cc": -0.09459,
    "ic_ca": -0.10007,
    "ic_pp": 0.19577,
    "ic_pa": -0.22671,
    "p_nis_a": 0.18681,
    "p_nis_c": 0.13810,
    "intercept": -15.9433,
}


def get_atom_radii(sequence_one_hot: jnp.ndarray, residue_radii_matrix: jnp.ndarray) -> jnp.ndarray:
    """Project residue probabilities into atom radii."""
    return jnp.matmul(sequence_one_hot, residue_radii_matrix).reshape(-1)


def calculate_relative_sasa(
    complex_sasa: jnp.ndarray,
    sequence_probabilities: jnp.ndarray,
    relative_sasa_array: jnp.ndarray,
    atoms_per_residue: int,
) -> jnp.ndarray:
    """Calculate relative SASA values for each residue."""
    residue_sasa = complex_sasa.reshape(-1, atoms_per_residue).sum(axis=1)
    reference_sasa = jnp.matmul(sequence_probabilities, relative_sasa_array)
    return residue_sasa / (reference_sasa + 1e-8)


def calculate_nis_percentages(
    sasa_values: jnp.ndarray,
    sequence_probabilities: jnp.ndarray,
    character_matrix: jnp.ndarray,
    threshold: float = 0.05,
) -> jnp.ndarray:
    """Calculate non-interacting surface percentages."""
    residue_classes = sequence_probabilities @ character_matrix
    mask = sasa_values >= threshold
    counts = jnp.sum(residue_classes * mask[:, None], axis=0)
    total = jnp.sum(mask) + 1e-8
    return 100.0 * counts / total


def score_ic_nis(
    ic_cc: jnp.ndarray,
    ic_ca: jnp.ndarray,
    ic_pp: jnp.ndarray,
    ic_pa: jnp.ndarray,
    p_nis_a: jnp.ndarray,
    p_nis_c: jnp.ndarray,
    coeffs: jnp.ndarray,
    intercept: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate binding affinity from PRODIGY IC-NIS features."""
    p_nis_a = jnp.clip(p_nis_a, 0, 100)
    p_nis_c = jnp.clip(p_nis_c, 0, 100)
    inputs = jnp.array([ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c])
    return jnp.dot(coeffs, inputs) + intercept


def dg_to_kd(dg: jnp.ndarray, temperature: float = 25.0) -> jnp.ndarray:
    """Convert binding free energy (DG) to dissociation constant (Kd)."""
    gas_constant = 0.0019858775
    temperature_kelvin = temperature + 273.15
    dg_clipped = jnp.clip(dg, -100.0, 100.0)
    return jnp.exp(dg_clipped / (gas_constant * temperature_kelvin))


# --- Tinygrad variants ---------------------------------------------------

def get_atom_radii_tinygrad(sequence_one_hot, residue_radii_matrix):
    """Project residue one-hot probabilities into per-atom van der Waals radii."""
    return (sequence_one_hot @ residue_radii_matrix).reshape(-1)


def calculate_relative_sasa_tinygrad(
    complex_sasa,
    sequence_probabilities,
    relative_sasa_array,
    atoms_per_residue: int,
):
    """Residue-level relative SASA = per-residue sum / reference total (tinygrad)."""
    residue_sasa = complex_sasa.reshape(-1, atoms_per_residue).sum(axis=1)
    reference_sasa = sequence_probabilities @ relative_sasa_array
    return residue_sasa / (reference_sasa + 1e-8)


def calculate_nis_percentages_tinygrad(
    sasa_values,
    sequence_probabilities,
    character_matrix,
    threshold: float = 0.05,
):
    """Percentage of the non-interacting surface per residue character (tinygrad)."""
    residue_classes = sequence_probabilities @ character_matrix
    mask = (sasa_values >= threshold).float()
    counts = (residue_classes * mask[:, None]).sum(axis=0)
    total = mask.sum() + 1e-8
    return 100.0 * counts / total


def score_ic_nis_tinygrad(
    ic_cc,
    ic_ca,
    ic_pp,
    ic_pa,
    p_nis_a,
    p_nis_c,
    coeffs,
    intercept,
):
    """Compute ΔG via the PRODIGY IC-NIS linear model (tinygrad)."""
    p_nis_a = p_nis_a.clip(0.0, 100.0)
    p_nis_c = p_nis_c.clip(0.0, 100.0)
    inputs = _TGTensor.stack(ic_cc, ic_ca, ic_pp, ic_pa, p_nis_a, p_nis_c)
    return (coeffs * inputs).sum() + intercept


def dg_to_kd_tinygrad(dg, temperature: float = 25.0):
    """Convert free energy (kcal/mol) to dissociation constant (M) on tinygrad."""
    gas_constant = 0.0019858775
    temperature_kelvin = temperature + 273.15
    return dg.clip(-100.0, 100.0).div(gas_constant * temperature_kelvin).exp()


def coefficient_tensors_tinygrad():
    """Return the PRODIGY IC-NIS coefficient vector and intercept as tinygrad tensors."""
    coeffs = _TGTensor(
        [
            NIS_CONSTANTS["ic_cc"],
            NIS_CONSTANTS["ic_ca"],
            NIS_CONSTANTS["ic_pp"],
            NIS_CONSTANTS["ic_pa"],
            NIS_CONSTANTS["p_nis_a"],
            NIS_CONSTANTS["p_nis_c"],
        ]
    )
    intercept = _TGTensor([NIS_CONSTANTS["intercept"]])
    return coeffs, intercept
