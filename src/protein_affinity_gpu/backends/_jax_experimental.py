"""Experimental JAX ``BackendAdapter`` — adds soft / single / neighbor modes."""
from __future__ import annotations

from typing import Literal

from ..sasa import calculate_sasa_batch, calculate_sasa_batch_scan, calculate_sasa_jax
from ..sasa_experimental import calculate_sasa_jax_neighbor
from ..sasa_soft import calculate_sasa_batch_scan_soft, calculate_sasa_batch_soft, calculate_sasa_jax_soft
from ..utils._array import Array
from ._jax import JAXAdapter

ExperimentalSasaMode = Literal["block", "single", "scan", "neighbor"]


class JAXExperimentalAdapter(JAXAdapter):
    """``JAXAdapter`` with the full mode / soft-SASA matrix.

    - ``"block"`` / ``"scan"`` — same default kernels as :class:`JAXAdapter`,
      optionally the differentiable sigmoid variant via ``soft_sasa=True``.
    - ``"single"`` — fully-fused single ``@jit``. One XLA program, no Python
      loop; peak scratch ``[N, M, N]`` so limited by device memory.
    - ``"neighbor"`` — single fused ``@jit`` that uses ``lax.top_k`` to keep
      only the K nearest atoms per row; ``[N, M, K]`` scratch (~80× smaller
      than ``"single"`` at K=64). Inference-only; ``soft_sasa`` is ignored
      in this mode because ``top_k`` is not usefully differentiable.
    """

    def __init__(
        self,
        *,
        soft_sasa: bool = False,
        soft_beta: float = 10.0,
        mode: ExperimentalSasaMode = "block",
        k_neighbors: int = 64,
    ) -> None:
        # ``mode`` is a superset of the default adapter's modes; type-narrowing
        # at this layer is ExperimentalSasaMode rather than SasaMode.
        super().__init__(mode="block" if mode in {"single", "neighbor"} else mode)
        self._soft_sasa = soft_sasa
        self._soft_beta = soft_beta
        self._mode = mode  # override with the wider mode set
        self._k_neighbors = k_neighbors

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
            return calculate_sasa_jax_neighbor(**common, k_neighbors=self._k_neighbors)
        if self._mode == "single":
            if self._soft_sasa:
                return calculate_sasa_jax_soft(**common, beta=self._soft_beta)
            return calculate_sasa_jax(**common)
        if self._mode == "scan":
            sasa_fn = calculate_sasa_batch_scan_soft if self._soft_sasa else calculate_sasa_batch_scan
        else:
            sasa_fn = calculate_sasa_batch_soft if self._soft_sasa else calculate_sasa_batch
        kwargs = {"beta": self._soft_beta} if self._soft_sasa else {}
        return sasa_fn(block_size=block_size, **common, **kwargs)
