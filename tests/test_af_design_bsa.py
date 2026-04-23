"""Smoke tests for the BSA logging path added to the AFDesign Modal pipeline.

Two independent slices:

1. The hard-SASA-based BSA computation pattern itself — synthetic two-monomer
   systems, no ColabDesign/Modal required. Validates that isolating each
   monomer via its mask and subtracting ``SASA(complex)`` recovers positive
   BSA when the monomers are in contact and ≈0 when they are far apart.

2. The ``af_design/plot_afdesign.py rmsd`` subcommand — checks that
   ``--metric`` accepts ``rmsd``, ``bsa``, and ``both`` against synthetic
   JSON artifacts.
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("jax") is None, reason="jax not installed")


def _hard_bsa(pos_flat, radii_flat, mask_a, mask_b, sphere_points, block_size):
    """Reference BSA via the hard JAX SASA kernel (mirrors the Modal callback)."""
    from protein_affinity_gpu.sasa import calculate_sasa_batch_scan

    sasa_complex = calculate_sasa_batch_scan(
        pos_flat, radii_flat, mask_a + mask_b, block_size, sphere_points
    )
    sasa_a = calculate_sasa_batch_scan(
        pos_flat, radii_flat, mask_a, block_size, sphere_points
    )
    sasa_b = calculate_sasa_batch_scan(
        pos_flat, radii_flat, mask_b, block_size, sphere_points
    )
    return float(sasa_a.sum() + sasa_b.sum() - sasa_complex.sum())


def test_bsa_math_positive_on_contact_and_zero_when_separated():
    import jax.numpy as jnp
    from protein_affinity_gpu.sasa import generate_sphere_points

    # Two 3-atom "monomers": each a tight triangle of carbons.
    monomer_a = np.array(
        [[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [0.75, 1.3, 0.0]], dtype=np.float32
    )
    monomer_b_contact = monomer_a + np.array([3.0, 0.0, 0.0], dtype=np.float32)
    monomer_b_far = monomer_a + np.array([60.0, 0.0, 0.0], dtype=np.float32)

    radii = np.full(6, 1.70, dtype=np.float32)  # 6 carbons
    sphere_points = jnp.asarray(generate_sphere_points(64), dtype=jnp.float32)

    mask_a = jnp.asarray([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)
    mask_b = jnp.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float32)

    def bsa_for(pos):
        pos_flat = jnp.asarray(pos.reshape(-1, 3), dtype=jnp.float32)
        radii_flat = jnp.asarray(radii, dtype=jnp.float32)
        return _hard_bsa(pos_flat, radii_flat, mask_a, mask_b, sphere_points, block_size=6)

    pos_contact = np.concatenate([monomer_a, monomer_b_contact], axis=0)
    pos_far = np.concatenate([monomer_a, monomer_b_far], axis=0)

    bsa_contact = bsa_for(pos_contact)
    bsa_far = bsa_for(pos_far)

    assert bsa_contact > 10.0, f"in-contact BSA should be clearly positive, got {bsa_contact}"
    assert abs(bsa_far) < 1.0, f"far-apart BSA should vanish, got {bsa_far}"
    assert bsa_contact > bsa_far


def test_plot_afdesign_rmsd_renders_all_metric_modes(tmp_path):
    """Smoke-test the consolidated plot CLI across ``rmsd | bsa | both``."""
    soft_dir = tmp_path / "soft"
    hardish_dir = tmp_path / "hardish"
    soft_dir.mkdir()
    hardish_dir.mkdir()

    # 4 steps, 8-residue binder — synthetic but shape-valid.
    rng = np.random.default_rng(0)
    soft_frames = rng.standard_normal((4, 8, 3)).tolist()
    hardish_frames = rng.standard_normal((4, 8, 3)).tolist()
    (soft_dir / "binder_ca_history.json").write_text(json.dumps(soft_frames))
    (hardish_dir / "binder_ca_history.json").write_text(json.dumps(hardish_frames))
    (soft_dir / "bsa_history.json").write_text(json.dumps([0.0, 120.5, 310.2, 480.7]))
    (hardish_dir / "bsa_history.json").write_text(json.dumps([0.0, 90.1, 250.4, 400.0]))

    script = Path(__file__).resolve().parents[1] / "af_design" / "plot_afdesign.py"
    assert script.exists(), f"missing plot script: {script}"

    for metric in ("rmsd", "bsa", "both"):
        output = tmp_path / f"out_{metric}.png"
        proc = subprocess.run(
            [
                sys.executable, str(script), "rmsd",
                "--soft-dir", str(soft_dir),
                "--hardish-dir", str(hardish_dir),
                "--output", str(output),
                "--metric", metric,
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, f"metric={metric} failed: {proc.stderr}"
        assert output.exists() and output.stat().st_size > 0
