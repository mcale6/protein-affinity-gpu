"""Stable differentiable scoring helpers for design-time objectives."""
from __future__ import annotations

import jax

from .utils._array import Array


def calculate_nis_percentages_soft(
    sasa_values: Array,
    sequence_probabilities: Array,
    character_matrix: Array,
    threshold: float = 0.05,
    beta: float = 20.0,
) -> Array:
    """Soft percentage of non-interacting surface per residue character.

    Replaces the hard ``sasa >= threshold`` gate with a sigmoid of sharpness
    ``beta`` so the NIS contribution remains differentiable.
    """
    residue_classes = sequence_probabilities @ character_matrix
    exposed = jax.nn.sigmoid(beta * (sasa_values - threshold))
    counts = (residue_classes * exposed[:, None]).sum(axis=0)
    total = exposed.sum() + 1e-8
    return 100.0 * counts / total


__all__ = ["calculate_nis_percentages_soft"]
