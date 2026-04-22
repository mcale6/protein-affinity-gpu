"""Public API for the protein-affinity-gpu package.

Default surface: CPU (:func:`predict_binding_affinity`) and JAX block / scan
(:func:`predict_binding_affinity_jax`). Tinygrad and the extended JAX modes
(soft / single / neighbor) are opt-in through :mod:`protein_affinity_gpu.experimental`.
"""

from .results import ContactAnalysis, ProdigyResults
from .utils.structure import Protein, load_complex, load_structure
from .version import __version__

__all__ = [
    "__version__",
    "ContactAnalysis",
    "Protein",
    "ProdigyResults",
    "load_complex",
    "load_structure",
    "predict",
    "predict_binding_affinity",
    "predict_binding_affinity_jax",
]


def predict_binding_affinity(*args, **kwargs):
    """Run the CPU PRODIGY-compatible affinity predictor."""
    from .cpu import predict_binding_affinity as impl

    return impl(*args, **kwargs)


def predict_binding_affinity_jax(*args, **kwargs):
    """Run the JAX affinity predictor (default block / scan modes)."""
    from .predict import predict_binding_affinity_jax as impl

    return impl(*args, **kwargs)


def predict(*args, **kwargs):
    """Unified router — ``predict(struct_path, backend="cpu"|"jax", ...)``.

    Tinygrad and the extended JAX modes (soft / single / neighbor) live in
    :mod:`protein_affinity_gpu.experimental`.
    """
    from .predict import predict_binding_affinity as impl

    return impl(*args, **kwargs)
