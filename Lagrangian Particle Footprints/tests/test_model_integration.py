"""End-to-end integration tests for the :class:`FootprintModel` pipeline.

Fast by design: small particle counts, a raised surface contact height and a
short ``t_max`` (matching tests/test_particles.py), so the whole
validate → simulate → touchdown → grid → contour chain runs quickly while still
exercising every stage.
"""

from __future__ import annotations

import numpy as np
import pytest

import lagranged as lg
from lagranged._rng import get_rng
from lagranged.particles import simulate_trajectories
from lagranged.touchdown import detect_touchdowns


def _fast_config(**overrides) -> lg.ModelConfig:
    """Cheap-but-representative config; overridable per test."""
    base = dict(n_particles=400, dt_factor=0.05, t_max=90.0, rebound_height=0.5, seed=0)
    base.update(overrides)
    return lg.ModelConfig(**base)


def test_density_integrates_to_one(neutral_inputs, small_grid):
    """The normalized density satisfies ∫∫ f dx dy ≈ 1."""
    cfg = _fast_config(seed=1)
    result = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg).run()
    assert result.n_touchdowns > 0  # otherwise nothing to normalize
    integral = float(result.density.sum()) * small_grid.dx * small_grid.dy
    assert integral == pytest.approx(1.0, abs=1e-9)


def test_x_peak_positive_and_finite(neutral_inputs, small_grid):
    """The crosswind-integrated peak sits upwind of the receptor (x > 0)."""
    cfg = _fast_config(seed=2)
    result = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg).run()
    assert np.isfinite(result.x_peak)
    assert result.x_peak > 0.0
    assert np.isfinite(result.mc_noise)
    assert result.mc_noise > 0.0


def test_same_seed_gives_identical_arrays(neutral_inputs, small_grid):
    """A fixed seed fully determines the run (bit-for-bit reproducible)."""
    cfg = _fast_config(seed=3)
    r_a = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg).run()
    r_b = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg).run()
    np.testing.assert_array_equal(r_a.density, r_b.density)
    np.testing.assert_array_equal(r_a.x, r_b.x)
    np.testing.assert_array_equal(r_a.y, r_b.y)
    assert r_a.x_peak == r_b.x_peak
    assert r_a.n_touchdowns == r_b.n_touchdowns


def test_total_weighted_touchdowns_conserved(neutral_inputs, small_grid):
    """The pipeline carries the touchdown weights through unchanged.

    Reproducing the simulate → detect stages independently under the same seed
    must yield the same touchdown count and total weight the model recorded.
    """
    cfg = _fast_config(seed=4)
    result = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg).run()

    traj = simulate_trajectories(neutral_inputs, None, cfg, get_rng(cfg.seed))
    _, _, weight = detect_touchdowns(traj.x, traj.y, traj.w_contact, sigma_w0=traj.sigma_w_surface)
    assert result.n_touchdowns == traj.n_touchdown
    assert result.meta["total_weight"] == pytest.approx(float(weight.sum()))


def test_total_weight_is_grid_independent(neutral_inputs):
    """Touchdown mass depends on the physics, not the accumulation grid."""
    cfg = _fast_config(seed=5)
    fine = lg.DomainGrid(nx=100, ny=100, dx=2.0, dy=2.0, x0=-100.0, y0=-100.0)
    coarse = lg.DomainGrid(nx=25, ny=25, dx=8.0, dy=8.0, x0=-100.0, y0=-100.0)
    r_fine = lg.FootprintModel(neutral_inputs, grid=fine, config=cfg).run()
    r_coarse = lg.FootprintModel(neutral_inputs, grid=coarse, config=cfg).run()
    assert r_fine.n_touchdowns == r_coarse.n_touchdowns
    assert r_fine.meta["total_weight"] == pytest.approx(r_coarse.meta["total_weight"])


def test_reynolds_mode_runs_end_to_end(neutral_inputs, small_grid):
    """``mode='reynolds'`` is honored: measured turbulence drives a full run."""
    tower = lg.TowerTurbulence(sigma_u=0.8, sigma_v=0.6, sigma_w=0.42, cov_uw=-0.12)
    cfg = _fast_config(seed=6, mode="reynolds")
    result = lg.FootprintModel(neutral_inputs, grid=small_grid, turbulence=tower, config=cfg).run()
    assert result.meta["mode"] == "reynolds"
    integral = float(result.density.sum()) * small_grid.dx * small_grid.dy
    assert integral == pytest.approx(1.0, abs=1e-9)
    assert result.x_peak > 0.0
