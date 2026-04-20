"""Tiny dispatch shim for the handful of tensor ops whose API differs
between ``jax.numpy`` and ``tinygrad.Tensor``, plus a NumPy-aware JSON
encoder used whenever we serialize pipeline outputs.

Everything else in the scoring/contacts pipeline uses tensor *methods*
(`.sum()`, `.reshape()`, `@`, `.clip()`, `.exp()`) which are already
identical across the two backends — no shim needed for those.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, TypeAlias, Union

import numpy as np
from tinygrad import Tensor as _TGTensor

if TYPE_CHECKING:
    import jax

    Array: TypeAlias = Union[np.ndarray, jax.Array, _TGTensor]
else:
    Array: TypeAlias = Any


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that understands numpy scalars/arrays and anything with ``tolist``."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if hasattr(obj, "tolist"):
            return obj.tolist()
        return super().default(obj)


def to_numpy(x: Array) -> np.ndarray:
    """Materialize a jax / tinygrad / numpy array as a plain ``np.ndarray``.

    Single dispatch point for the ``complex_sasa.numpy()`` / ``np.asarray(jax)``
    split that otherwise leaks into every predictor. ``.numpy()`` on tinygrad
    triggers realization; ``np.asarray`` on jax is zero-copy on CPU, a host
    copy on GPU/Metal. Numpy arrays pass through untouched.
    """
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, _TGTensor):
        return x.numpy()
    return np.asarray(x)


def concat(tensors: list[Array], axis: int = 0) -> Array:
    """``jnp.concatenate([a, b], axis=k)`` / ``Tensor.cat(a, b, dim=k)``."""
    first = tensors[0]
    if isinstance(first, _TGTensor):
        return _TGTensor.cat(*tensors, dim=axis)
    import jax.numpy as jnp

    return jnp.concatenate(list(tensors), axis=axis)


def stack_scalars(*xs: Array) -> Array:
    """Stack 0-d tensors into a 1-d vector — ``jnp.stack`` takes a list,
    ``Tensor.stack`` takes varargs."""
    first = xs[0]
    if isinstance(first, _TGTensor):
        return _TGTensor.stack(*xs)
    import jax.numpy as jnp

    return jnp.stack(list(xs))


def exp(x: Array) -> Array:
    """``jnp.exp(x)`` / ``Tensor.exp()`` — jax arrays have no ``.exp`` method."""
    if isinstance(x, _TGTensor):
        return x.exp()
    import jax.numpy as jnp

    return jnp.exp(x)
