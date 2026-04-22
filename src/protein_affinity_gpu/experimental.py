"""Experimental predictor entry points — tinygrad + extended JAX modes.

These wrap :func:`.predict._run_pipeline` with experimental adapters
(:class:`.backends._jax_experimental.JAXExperimentalAdapter`,
:class:`.backends._tinygrad.TinygradAdapter`) that reach into
:mod:`.sasa_experimental`. The default surface in :mod:`.predict` covers
only JAX block / scan.
"""
from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from .predict import _run_pipeline
from .results import ProdigyResults


def predict_binding_affinity_jax_experimental(
    struct_path: str | Path,
    selection: str = "A,B",
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: Optional[str | Path] = ".",
    quiet: bool = True,
    soft_sasa: bool = False,
    soft_beta: float = 10.0,
    mode: Literal["block", "single", "scan", "neighbor"] = "block",
    k_neighbors: int = 64,
) -> ProdigyResults:
    """PRODIGY IC-NIS on JAX with the experimental SASA kernel set.

    ``soft_sasa=True`` swaps the hard Shrake–Rupley threshold for a sigmoid of
    sharpness ``soft_beta`` — meaningful gradients w.r.t. coords / radii at the
    cost of some accuracy (β→∞ recovers the hard kernel). Intended for
    training / differentiable design; leave off for straight inference.

    ``mode`` selects the SASA dispatch: ``"block"`` / ``"scan"`` (same as the
    default entry point), ``"single"`` (one fused ``@jit``, ``[N, M, N]``
    scratch), or ``"neighbor"`` (``lax.top_k`` keeps only the K nearest atoms
    per row — ``[N, M, K]`` scratch, ~80× smaller than ``"single"`` at K=64;
    inference-only).
    """
    from .backends._jax_experimental import JAXExperimentalAdapter

    return _run_pipeline(
        JAXExperimentalAdapter(
            soft_sasa=soft_sasa, soft_beta=soft_beta,
            mode=mode, k_neighbors=k_neighbors,
        ),
        struct_path=struct_path,
        selection=selection,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        temperature=temperature,
        sphere_points=sphere_points,
        save_results=save_results,
        output_dir=output_dir,
        quiet=quiet,
    )


def predict_binding_affinity_tinygrad(
    struct_path: str | Path,
    selection: str = "A,B",
    distance_cutoff: float = 5.5,
    acc_threshold: float = 0.05,
    temperature: float = 25.0,
    sphere_points: int = 100,
    save_results: bool = False,
    output_dir: Optional[str | Path] = ".",
    quiet: bool = True,
    mode: Literal["block", "single", "neighbor", "bucketed"] = "block",
    k_neighbors: int = 64,
    bucket_step: int = 2048,
) -> ProdigyResults:
    """Run the PRODIGY IC-NIS pipeline on tinygrad (experimental surface).

    ``mode`` selects the SASA dispatch: ``"block"`` (default) uses the
    blocked TinyJit kernel, ``"single"`` is the fully-fused kernel,
    ``"neighbor"`` keeps only the K nearest atoms per row via
    ``Tensor.topk`` — ~80× scratch reduction at K=64, lossless when
    K covers the worst-case occlusion neighbor count. ``"bucketed"`` pads
    ``N`` up to the next multiple of ``bucket_step`` before dispatch so
    the TinyJit cache is keyed on a handful of shapes — one compile
    amortises across many structures instead of per-structure compiles.
    """
    from .backends._tinygrad import TinygradAdapter

    return _run_pipeline(
        TinygradAdapter(mode=mode, k_neighbors=k_neighbors, bucket_step=bucket_step),
        struct_path=struct_path,
        selection=selection,
        distance_cutoff=distance_cutoff,
        acc_threshold=acc_threshold,
        temperature=temperature,
        sphere_points=sphere_points,
        save_results=save_results,
        output_dir=output_dir,
        quiet=quiet,
    )


__all__ = [
    "predict_binding_affinity_jax_experimental",
    "predict_binding_affinity_tinygrad",
]
