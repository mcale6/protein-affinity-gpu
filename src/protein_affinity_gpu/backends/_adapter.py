"""``BackendAdapter`` Protocol — the surface the shared pipeline calls into.

Members split into three groups:

- *Lazy constants* (``radii_matrix_atom14`` … ``intercept``): backend-native
  tensors reused across every prediction. Held on the adapter so each
  backend materializes them once per process.
- *Conversion / construction* (``from_numpy``, ``to_numpy``, ``one_hot``,
  ``concat``, ``sphere_points``): the API divergence points between
  ``jax.numpy`` and ``tinygrad.Tensor``.
- *Kernels* (``residue_contacts``, ``sasa``, ``estimate_block_size``,
  ``validate_size``): the backend-specific compute paths.
"""
from __future__ import annotations

from typing import Protocol

import numpy as np

from ..utils._array import Array


class BackendAdapter(Protocol):
    """Backend-specific primitives consumed by :func:`..predict._run_pipeline`."""

    name: str

    # --- Lazy backend-native constants ---
    radii_matrix_atom14: Array
    relative_sasa_array: Array
    contact_class_matrix: Array
    nis_class_matrix: Array
    coeffs: Array
    intercept: Array

    # --- Conversion / construction ---
    def from_numpy(self, x: np.ndarray) -> Array: ...
    def to_numpy(self, x: Array) -> np.ndarray: ...
    def one_hot(self, indices: np.ndarray, num_classes: int) -> Array: ...
    def concat(self, tensors: list[Array], axis: int = 0) -> Array: ...
    def sphere_points(self, n: int) -> Array: ...

    # --- Kernels ---
    def estimate_block_size(self, n_atoms: int, sphere_points: int = 100) -> int | None:
        """Block size for the batched SASA kernel, or ``None`` for the full
        (non-batched) path. Implementations choose based on device and size.
        """
        ...

    def validate_size(self, n_atoms: int, sphere_points: int = 100) -> None:
        """Raise if the structure is too large for this backend. No-op by default."""
        ...

    def residue_contacts(
        self,
        target_pos: Array,
        binder_pos: Array,
        target_mask: Array,
        binder_mask: Array,
        distance_cutoff: float,
    ) -> Array: ...

    def sasa(
        self,
        coords: Array,
        vdw_radii: Array,
        mask: Array,
        sphere_points: Array,
        block_size: int | None,
    ) -> Array: ...
