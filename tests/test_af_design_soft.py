from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from protein_affinity_gpu.af_design import add_ba_val_loss
from protein_affinity_gpu.contacts import calculate_residue_contacts
from protein_affinity_gpu.contacts_soft import calculate_residue_contacts_soft
from protein_affinity_gpu.sasa import generate_sphere_points
from protein_affinity_gpu.sasa_experimental import calculate_sasa_batch_scan_soft as experimental_scan_soft
from protein_affinity_gpu.sasa_soft import calculate_sasa_batch_scan_soft
from protein_affinity_gpu.scoring import calculate_nis_percentages
from protein_affinity_gpu.scoring_soft import calculate_nis_percentages_soft

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")


def test_soft_modules_import_smoke():
    import protein_affinity_gpu.af_design as af_design
    import protein_affinity_gpu.contacts_soft as contacts_soft
    import protein_affinity_gpu.scoring_soft as scoring_soft
    import protein_affinity_gpu.sasa_soft as sasa_soft

    assert callable(af_design.add_ba_val_loss)
    assert callable(contacts_soft.calculate_residue_contacts_soft)
    assert callable(scoring_soft.calculate_nis_percentages_soft)
    assert callable(sasa_soft.calculate_sasa_batch_scan_soft)


def test_soft_contacts_shape_range_and_hard_limit():
    target_pos = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float32)
    binder_pos = jnp.asarray(
        [
            [[5.0, 0.0, 0.0]],
            [[6.5, 0.0, 0.0]],
        ],
        dtype=jnp.float32,
    )
    target_mask = jnp.ones((1, 1), dtype=jnp.float32)
    binder_mask = jnp.ones((2, 1), dtype=jnp.float32)

    soft_contacts = calculate_residue_contacts_soft(
        target_pos,
        binder_pos,
        target_mask,
        binder_mask,
        distance_cutoff=5.5,
        beta=100.0,
    )
    hard_contacts = calculate_residue_contacts(
        target_pos,
        binder_pos,
        target_mask,
        binder_mask,
        distance_cutoff=5.5,
    )

    assert soft_contacts.shape == hard_contacts.shape
    assert bool(jnp.all(soft_contacts >= 0.0))
    assert bool(jnp.all(soft_contacts <= 1.0))
    np.testing.assert_allclose(np.asarray(soft_contacts), np.asarray(hard_contacts, dtype=np.float32), atol=1e-2)


def test_soft_nis_matches_hard_threshold_at_large_beta():
    sasa_values = jnp.asarray([0.01, 0.08, 0.09], dtype=jnp.float32)
    sequence_probabilities = jnp.eye(3, dtype=jnp.float32)
    character_matrix = jnp.eye(3, dtype=jnp.float32)

    soft = calculate_nis_percentages_soft(
        sasa_values,
        sequence_probabilities,
        character_matrix,
        threshold=0.05,
        beta=200.0,
    )
    hard = calculate_nis_percentages(
        sasa_values,
        sequence_probabilities,
        character_matrix,
        threshold=0.05,
    )

    assert soft.shape == hard.shape
    assert np.isfinite(np.asarray(soft)).all()
    np.testing.assert_allclose(np.asarray(soft), np.asarray(hard), atol=0.1)


def test_add_ba_val_loss_appends_callback_and_sets_default_weight():
    class DummyModel:
        def __init__(self):
            self.opt = {"weights": {}}
            self._callbacks = {"model": {"loss": []}}
            self._binder_len = 2

    model = DummyModel()
    add_ba_val_loss(model)

    assert len(model._callbacks["model"]["loss"]) == 1
    assert model.opt["weights"]["ba_val"] == 0.0
    assert inspect.signature(add_ba_val_loss).parameters["binder_seq_mode"].default == "soft"


def test_add_ba_val_loss_rejects_invalid_binder_seq_mode():
    class DummyModel:
        def __init__(self):
            self.opt = {"weights": {}}
            self._callbacks = {"model": {"loss": []}}
            self._binder_len = 2

    with pytest.raises(ValueError, match="binder_seq_mode"):
        add_ba_val_loss(DummyModel(), binder_seq_mode="not-a-mode")


def test_sasa_experimental_reexports_stable_soft_kernel():
    assert experimental_scan_soft is calculate_sasa_batch_scan_soft


def test_modal_afdesign_script_imports_stable_helper():
    source = Path("af_design/modal_afdesign_ba_val.py").read_text()
    assert "from protein_affinity_gpu.af_design import add_ba_val_loss" in source
    assert "def add_ba_val_loss(" not in source


def test_soft_contacts_have_nonzero_gradients():
    target_pos = jnp.asarray([[[0.0, 0.0, 0.0]]], dtype=jnp.float32)
    binder_pos = jnp.asarray([[[5.4, 0.0, 0.0]]], dtype=jnp.float32)
    target_mask = jnp.ones((1, 1), dtype=jnp.float32)
    binder_mask = jnp.ones((1, 1), dtype=jnp.float32)

    def loss_fn(x):
        return calculate_residue_contacts_soft(
            target_pos,
            x,
            target_mask,
            binder_mask,
            distance_cutoff=5.5,
            beta=8.0,
        ).sum()

    grad = jax.grad(loss_fn)(binder_pos)
    assert np.max(np.abs(np.asarray(grad))) > 0.0


def test_soft_nis_has_nonzero_gradients():
    sasa_values = jnp.asarray([0.02, 0.055, 0.09], dtype=jnp.float32)
    sequence_probabilities = jnp.eye(3, dtype=jnp.float32)
    character_matrix = jnp.eye(3, dtype=jnp.float32)

    def loss_fn(x):
        return calculate_nis_percentages_soft(
            x,
            sequence_probabilities,
            character_matrix,
            threshold=0.05,
            beta=20.0,
        )[1]

    grad = jax.grad(loss_fn)(sasa_values)
    assert np.max(np.abs(np.asarray(grad))) > 0.0


def test_soft_sasa_has_nonzero_gradients():
    coords = jnp.asarray(
        [
            [0.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    vdw_radii = jnp.asarray([1.7, 1.7], dtype=jnp.float32)
    mask = jnp.ones((2,), dtype=jnp.float32)
    sphere_points = jnp.asarray(generate_sphere_points(32), dtype=jnp.float32)

    def loss_fn(x):
        return calculate_sasa_batch_scan_soft(
            coords=x,
            vdw_radii=vdw_radii,
            mask=mask,
            block_size=2,
            sphere_points=sphere_points,
            beta=10.0,
        ).sum()

    grad = jax.grad(loss_fn)(coords)
    assert np.max(np.abs(np.asarray(grad))) > 0.0
