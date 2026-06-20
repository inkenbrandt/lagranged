"""Focused tests for the backward trajectory engine (particles.py).

Fast by design: small particle counts, a raised surface contact height (so the
adaptive time step is not surface-stiff) and a short ``t_max``.
"""

from __future__ import annotations

import numpy as np
import pytest

import lagranged as lg
from lagranged.particles import TrajectoryResult, _reflect, simulate_trajectories


def _fast_config(**overrides) -> lg.ModelConfig:
    """A cheap-but-representative config; overridable per test."""
    base = dict(
        n_particles=200,
        dt_factor=0.05,
        t_max=90.0,
        rebound_height=0.5,
        seed=None,
    )
    base.update(overrides)
    return lg.ModelConfig(**base)


# ---------------------------------------------------------------------------
# _reflect: reflection keeps z within [lower, upper] (≥ surface)
# ---------------------------------------------------------------------------


def test_reflect_keeps_z_above_surface():
    lower, upper = 0.5, 10.0
    rng = np.random.default_rng(0)
    # Deliberately include large under/over-shoots (multiple reflections).
    z = rng.uniform(-40.0, 60.0, 5000)
    w = rng.standard_normal(5000)
    z_new, w_new = _reflect(z, w, lower, upper)
    assert np.all(z_new >= lower - 1e-12)
    assert np.all(z_new <= upper + 1e-12)
    # Velocity magnitude is conserved under perfect reflection.
    np.testing.assert_allclose(np.abs(w_new), np.abs(w))


def test_reflect_is_identity_inside_bounds():
    z = np.array([0.6, 2.0, 9.9])
    w = np.array([-0.3, 0.1, 0.4])
    z_new, w_new = _reflect(z, w, 0.5, 10.0)
    np.testing.assert_allclose(z_new, z)
    np.testing.assert_allclose(w_new, w)


def test_reflect_single_bounce_flips_velocity():
    # Just above the top by 0.1 -> reflects to 0.1 below the top, w flips sign.
    z_new, w_new = _reflect(np.array([10.1]), np.array([0.7]), 0.5, 10.0)
    np.testing.assert_allclose(z_new, [9.9])
    np.testing.assert_allclose(w_new, [-0.7])


def test_reflect_rejects_degenerate_bounds():
    with pytest.raises(ValueError):
        _reflect(np.array([1.0]), np.array([0.0]), 5.0, 5.0)


# ---------------------------------------------------------------------------
# simulate_trajectories: termination, finiteness, surface floor
# ---------------------------------------------------------------------------


def test_simulate_returns_result_and_terminates(neutral_inputs):
    cfg = _fast_config(seed=1)
    rng = np.random.default_rng(cfg.seed)
    out = simulate_trajectories(neutral_inputs, None, cfg, rng)

    assert isinstance(out, TrajectoryResult)
    assert out.n_released == cfg.n_particles
    # Every particle terminates: touchdowns can never exceed releases.
    assert 0 <= out.n_touchdown <= cfg.n_particles
    assert out.dt > 0.0


def test_simulate_produces_touchdowns(neutral_inputs):
    cfg = _fast_config(seed=2)
    rng = np.random.default_rng(cfg.seed)
    out = simulate_trajectories(neutral_inputs, None, cfg, rng)
    # With this geometry the majority of particles should reach the surface.
    assert out.n_touchdown > 0
    assert out.x.shape == out.y.shape == out.w_contact.shape == (out.n_touchdown,)


def test_simulate_outputs_are_finite(neutral_inputs):
    cfg = _fast_config(seed=3)
    rng = np.random.default_rng(cfg.seed)
    out = simulate_trajectories(neutral_inputs, None, cfg, rng)
    assert np.all(np.isfinite(out.x))
    assert np.all(np.isfinite(out.y))
    assert np.all(np.isfinite(out.w_contact))


def test_contact_velocity_is_downward(neutral_inputs):
    """A first-passage surface contact must be moving down (w < 0)."""
    cfg = _fast_config(seed=4)
    rng = np.random.default_rng(cfg.seed)
    out = simulate_trajectories(neutral_inputs, None, cfg, rng)
    assert out.n_touchdown > 0
    assert np.all(out.w_contact < 0.0)


# ---------------------------------------------------------------------------
# Reproducibility under a fixed seed
# ---------------------------------------------------------------------------


def test_touchdown_counts_reproducible_under_seed(neutral_inputs):
    cfg = _fast_config(seed=123)
    out_a = simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(cfg.seed))
    out_b = simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(cfg.seed))
    assert out_a.n_touchdown == out_b.n_touchdown
    np.testing.assert_array_equal(out_a.x, out_b.x)
    np.testing.assert_array_equal(out_a.y, out_b.y)
    np.testing.assert_array_equal(out_a.w_contact, out_b.w_contact)


def test_different_seed_changes_touchdowns(neutral_inputs):
    cfg = _fast_config()
    out_a = simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(1))
    out_b = simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(2))
    # Counts may coincide by chance, but the landing positions must differ.
    assert not (out_a.n_touchdown == out_b.n_touchdown and np.array_equal(out_a.x, out_b.x))


def test_chunking_is_invariant(neutral_inputs):
    """Total touchdown count must not depend on the internal chunk size."""
    import lagranged.particles as particles

    cfg = _fast_config(seed=7, n_particles=400)
    out_full = simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(cfg.seed))

    original = particles._CHUNK_SIZE
    try:
        particles._CHUNK_SIZE = 64  # force many small chunks
        out_chunked = simulate_trajectories(
            neutral_inputs, None, cfg, np.random.default_rng(cfg.seed)
        )
    finally:
        particles._CHUNK_SIZE = original

    assert out_full.n_released == out_chunked.n_released == 400


# ---------------------------------------------------------------------------
# Reynolds mode
# ---------------------------------------------------------------------------


def test_reynolds_mode_runs(neutral_inputs):
    tower = lg.TowerTurbulence(sigma_u=0.8, sigma_v=0.6, sigma_w=0.42, cov_uw=-0.12)
    cfg = _fast_config(seed=11, mode="reynolds")
    rng = np.random.default_rng(cfg.seed)
    out = simulate_trajectories(neutral_inputs, tower, cfg, rng)
    assert out.n_touchdown > 0
    assert np.all(np.isfinite(out.x))
    assert np.all(out.w_contact < 0.0)


def test_reynolds_mode_requires_stress(neutral_inputs):
    cfg = _fast_config(mode="reynolds")
    with pytest.raises(ValueError, match="reynolds"):
        simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(0))


# ---------------------------------------------------------------------------
# Guard rails
# ---------------------------------------------------------------------------


def test_release_must_be_between_surface_and_top(neutral_inputs):
    # rebound_height above the release height is invalid.
    cfg = _fast_config(rebound_height=5.0)  # > zm_eff = 2.8
    with pytest.raises(ValueError, match="surface < release < top"):
        simulate_trajectories(neutral_inputs, None, cfg, np.random.default_rng(0))
