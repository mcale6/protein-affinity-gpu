"""Public API for the protein-affinity-gpu package."""

from .results import ContactAnalysis, ProdigyResults
from .structure import Protein, load_complex, load_structure
from .version import __version__

__all__ = [
    "__version__",
    "ContactAnalysis",
    "Protein",
    "ProdigyResults",
    "load_complex",
    "load_structure",
    "predict_binding_affinity",
    "predict_binding_affinity_jax",
]


def predict_binding_affinity(*args, **kwargs):
    """Run the CPU PRODIGY-compatible affinity predictor."""
    from .cpu import predict_binding_affinity as impl

    return impl(*args, **kwargs)


def predict_binding_affinity_jax(*args, **kwargs):
    """Run the JAX affinity predictor."""
    from .jax import predict_binding_affinity_jax as impl

    return impl(*args, **kwargs)
