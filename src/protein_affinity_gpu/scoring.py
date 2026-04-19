import jax.numpy as jnp


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
