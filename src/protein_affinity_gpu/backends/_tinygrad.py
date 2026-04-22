"""Tinygrad ``BackendAdapter`` implementation — experimental surface.

The tinygrad pipeline lives on the experimental side; its SASA kernels are
maintained in :mod:`..sasa_experimental`.
"""
from __future__ import annotations

import os
from functools import cached_property
from typing import Literal

import numpy as np
from tinygrad import Device, Tensor

from ..contacts import calculate_residue_contacts_tinygrad
from ..sasa import calculate_sasa_batch_tinygrad, generate_sphere_points
from ..sasa_experimental import (
    calculate_sasa_batch_tinygrad_bucketed,
    calculate_sasa_tinygrad,
    calculate_sasa_tinygrad_neighbor,
)
from ..scoring import coefficient_tensors_tinygrad
from ..utils._array import Array
from ..utils.residue_classification import ResidueClassification
from ..utils.residue_library import default_library as residue_library

TinygradSasaMode = Literal["block", "single", "neighbor", "bucketed"]
_ACCELERATOR_DEVICES = {"METAL", "CUDA", "GPU"}


class TinygradAdapter:
    """Backend adapter for :mod:`tinygrad`.

    ``mode`` selects the SASA dispatch:

    - ``"block"`` (default on accelerators) — blocked kernel with per-shape
      ``TinyJit`` cache, bounded scratch.
    - ``"single"`` — fully-fused single TinyJit, ``[N, M, N]`` scratch;
      device-memory limited.
    - ``"neighbor"`` — single TinyJit using ``Tensor.topk`` to keep only
      the K nearest atoms per row, shrinking the buried-check scratch
      from ``[N, M, N]`` to ``[N, M, K]`` (~80× at K=64). Lossless when
      K covers the worst-case occlusion neighbor count.
    - ``"bucketed"`` — same blocked kernel as ``"block"`` but pads ``N``
      up to the next multiple of ``bucket_step`` before dispatch. The
      TinyJit cache keys on the bucketed shape, so across a batch of
      similarly-sized structures one compile services many inputs (the
      compile storm, not arithmetic, is the dominant cost on Apple Metal).
    """

    def __init__(
        self,
        *,
        mode: TinygradSasaMode = "block",
        k_neighbors: int = 64,
        bucket_step: int = 2048,
    ) -> None:
        self._mode = mode
        self._k_neighbors = k_neighbors
        self._bucket_step = bucket_step

    @property
    def name(self) -> str:
        override = os.environ.get("TINYGRAD_DEVICE")
        if override:
            return override.upper()
        return str(Device.DEFAULT).upper()

    # --- Lazy constants ---
    @cached_property
    def radii_matrix_atom14(self) -> Array:
        return Tensor(np.asarray(residue_library.radii_matrix_atom14, dtype=np.float32))

    @cached_property
    def relative_sasa_array(self) -> Array:
        return Tensor(np.asarray(ResidueClassification().relative_sasa_array, dtype=np.float32))

    @cached_property
    def contact_class_matrix(self) -> Array:
        return Tensor(np.asarray(ResidueClassification("ic").classification_matrix, dtype=np.float32))

    @cached_property
    def nis_class_matrix(self) -> Array:
        return Tensor(np.asarray(ResidueClassification("protorp").classification_matrix, dtype=np.float32))

    @cached_property
    def _coeffs_intercept(self) -> tuple[Tensor, Tensor]:
        return coefficient_tensors_tinygrad()

    @property
    def coeffs(self) -> Array:
        return self._coeffs_intercept[0]

    @property
    def intercept(self) -> Array:
        return self._coeffs_intercept[1]

    # --- Conversion / construction ---
    def from_numpy(self, x: np.ndarray) -> Array:
        return Tensor(np.ascontiguousarray(x))

    def to_numpy(self, x: Array) -> np.ndarray:
        return x.numpy()

    def one_hot(self, indices: np.ndarray, num_classes: int) -> Array:
        """Padding ``-1`` naturally produces a zero row via the one-hot compare."""
        return Tensor(np.asarray(indices, dtype=np.int64)).one_hot(num_classes).float()

    def concat(self, tensors: list[Array], axis: int = 0) -> Array:
        return Tensor.cat(*tensors, dim=axis)

    def sphere_points(self, n: int) -> Array:
        return Tensor(generate_sphere_points(n))

    # --- Kernels ---
    def estimate_block_size(self, n_atoms: int, sphere_points: int = 100) -> int | None:
        """Block size on accelerators; ``None`` on CPU → use the full (non-batched) kernel.

        Empirically on Apple Metal (M-series), throughput improves monotonically
        up to ``block ≈ 768`` for 1A2K-sized complexes. Past 768 Metal starts
        spilling the ``[block, 100, N]`` float32 scratch (~5GB) out of the fast
        L2/MMU path.
        """
        if self.name not in _ACCELERATOR_DEVICES:
            return None
        return min(768, n_atoms)

    def validate_size(self, n_atoms: int, sphere_points: int = 100) -> None:
        return None

    def residue_contacts(
        self,
        target_pos: Array,
        binder_pos: Array,
        target_mask: Array,
        binder_mask: Array,
        distance_cutoff: float,
    ) -> Array:
        return calculate_residue_contacts_tinygrad(
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
        common = dict(coords=coords, vdw_radii=vdw_radii, mask=mask, sphere_points=sphere_points)
        if self._mode == "neighbor":
            return calculate_sasa_tinygrad_neighbor(**common, k_neighbors=self._k_neighbors)
        if self._mode == "single" or block_size is None:
            return calculate_sasa_tinygrad(**common)
        if self._mode == "bucketed":
            return calculate_sasa_batch_tinygrad_bucketed(
                **common, block_size=block_size, bucket_step=self._bucket_step,
            )
        return calculate_sasa_batch_tinygrad(**common, block_size=block_size)
