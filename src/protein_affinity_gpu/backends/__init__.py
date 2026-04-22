"""Backend adapters — one per tensor framework (jax, tinygrad).

The predictor pipeline in :mod:`..predict` runs against a :class:`BackendAdapter`;
each concrete adapter owns the backend-specific primitives (tensor
construction, one-hot, kernels, lazy constants). Adapters are lazy-imported
so that ``import protein_affinity_gpu`` doesn't pull jax *and* tinygrad.

The default ``JAXAdapter`` covers ``mode ∈ {"block", "scan"}``. The
experimental adapters (``JAXExperimentalAdapter`` with soft/single/neighbor
and ``TinygradAdapter``) live alongside but are only reached through
:mod:`..experimental`.
"""
from __future__ import annotations

from ._adapter import BackendAdapter

__all__ = ["BackendAdapter", "get_adapter"]


def get_adapter(backend: str, **kwargs) -> BackendAdapter:
    """Instantiate the adapter for ``backend``.

    ``backend`` values:

    - ``"jax"`` — default :class:`JAXAdapter` (block / scan modes only).
    - ``"jax-experimental"`` — :class:`JAXExperimentalAdapter` with soft /
      single / neighbor support.
    - ``"tinygrad"`` — :class:`TinygradAdapter` (experimental-only surface).
    """
    backend = backend.lower()
    if backend == "jax":
        from ._jax import JAXAdapter

        return JAXAdapter(**kwargs)
    if backend == "jax-experimental":
        from ._jax_experimental import JAXExperimentalAdapter

        return JAXExperimentalAdapter(**kwargs)
    if backend == "tinygrad":
        from ._tinygrad import TinygradAdapter

        return TinygradAdapter(**kwargs)
    raise ValueError(
        f"Unknown backend: {backend!r} "
        "(expected 'jax', 'jax-experimental', or 'tinygrad')"
    )
