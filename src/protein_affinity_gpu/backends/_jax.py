"""JAX ``BackendAdapter`` implementation."""
from __future__ import annotations

import subprocess
from functools import cached_property

import jax
import jax.numpy as jnp
import numpy as np

from ..contacts import calculate_residue_contacts
from ..sasa import (
    calculate_sasa_batch,
    calculate_sasa_batch_soft,
    generate_sphere_points,
)
from ..scoring import NIS_COEFFICIENTS
from ..utils._array import Array
from ..utils.residue_classification import ResidueClassification
from ..utils.residue_library import default_library as residue_library


class JAXAdapter:
    """Backend adapter for :mod:`jax`. Supports optional soft-SASA kernel."""

    def __init__(self, *, soft_sasa: bool = False, soft_beta: float = 10.0) -> None:
        self._soft_sasa = soft_sasa
        self._soft_beta = soft_beta

    @property
    def name(self) -> str:
        return jax.default_backend().upper()

    # --- Lazy constants ---
    @cached_property
    def radii_matrix_atom14(self) -> Array:
        return jnp.array(residue_library.radii_matrix_atom14)

    @cached_property
    def relative_sasa_array(self) -> Array:
        return jnp.array(ResidueClassification().relative_sasa_array)

    @cached_property
    def contact_class_matrix(self) -> Array:
        return jnp.array(ResidueClassification("ic").classification_matrix)

    @cached_property
    def nis_class_matrix(self) -> Array:
        return jnp.array(ResidueClassification("protorp").classification_matrix)

    @cached_property
    def coeffs(self) -> Array:
        return jnp.array(NIS_COEFFICIENTS.as_tuple())

    @cached_property
    def intercept(self) -> Array:
        return jnp.array([NIS_COEFFICIENTS.intercept])

    # --- Conversion / construction ---
    def from_numpy(self, x: np.ndarray) -> Array:
        return jnp.asarray(x)

    def to_numpy(self, x: Array) -> np.ndarray:
        return np.asarray(x)

    def one_hot(self, indices: np.ndarray, num_classes: int) -> Array:
        return jax.nn.one_hot(indices, num_classes=num_classes)

    def concat(self, tensors: list[Array], axis: int = 0) -> Array:
        return jnp.concatenate(list(tensors), axis=axis)

    def sphere_points(self, n: int) -> Array:
        return generate_sphere_points(n)

    # --- Kernels ---
    def estimate_block_size(self, n_atoms: int, sphere_points: int = 100) -> int:
        """Metal: empirical exp-decay fit. CPU / CUDA: target ~1GB float32 scratch."""
        if self.name == "METAL":
            amplitude = 6.8879e02
            decay = -2.6156e-04
            offset = 17.4525
            block_size = int(round(amplitude * np.exp(decay * n_atoms) + offset))
            max_block = min(250, int(5000 / np.sqrt(max(n_atoms, 1) / 1000)))
            return max(5, min(block_size, max_block))

        cpu_scratch_bytes = 1_000_000_000
        per_atom_bytes = sphere_points * max(n_atoms, 1) * 4
        block_size = max(32, cpu_scratch_bytes // per_atom_bytes)
        return int(min(block_size, n_atoms))

    def validate_size(self, n_atoms: int, sphere_points: int = 100) -> None:
        if self.name == "METAL":
            return
        max_atoms = self._estimate_max_atoms(sphere_points=sphere_points)
        if n_atoms > max_atoms:
            raise ValueError(f"Too many atoms for JAX backend: {n_atoms} > {max_atoms}")

    @staticmethod
    def _estimate_max_atoms(safety_factor: float = 0.8, sphere_points: int = 100) -> int:
        try:
            result = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,nounits,noheader",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
            _, total = [int(part.strip()) for part in result.split(",")]
            available_memory = total * 1_000_000
        except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
            return 100_000

        bytes_per_atom = 3 * 4 + sphere_points * 3 * 4 + sphere_points * 4 * 1000
        max_atoms = int((available_memory * safety_factor) / bytes_per_atom)
        rounded = int(str(max_atoms)[0] + "0" * max(len(str(max_atoms)) - 1, 0))
        return max(rounded, 1000)

    def residue_contacts(
        self,
        target_pos: Array,
        binder_pos: Array,
        target_mask: Array,
        binder_mask: Array,
        distance_cutoff: float,
    ) -> Array:
        return calculate_residue_contacts(
            target_pos, binder_pos, target_mask, binder_mask,
            distance_cutoff=distance_cutoff,
        )

    def sasa(
        self,
        coords: Array,
        vdw_radii: Array,
        mask: Array,
        sphere_points: Array,
        block_size: int | None,
    ) -> Array:
        sasa_fn = calculate_sasa_batch_soft if self._soft_sasa else calculate_sasa_batch
        kwargs = {"beta": self._soft_beta} if self._soft_sasa else {}
        return sasa_fn(
            coords=coords,
            vdw_radii=vdw_radii,
            mask=mask,
            block_size=block_size,
            sphere_points=sphere_points,
            **kwargs,
        )
