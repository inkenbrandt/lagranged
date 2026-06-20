"""Tests for touchdown gridding and cumulative contours."""

from __future__ import annotations

import numpy as np

from lagranged.config import DomainGrid
from lagranged.contours import cumulative_levels
from lagranged.gridding import accumulate, grid_cell_centers


def test_cell_centers_shape_and_spacing():
    grid = DomainGrid(nx=10, ny=5, dx=2.0, dy=3.0)
    x, y = grid_cell_centers(grid)
    assert x.shape == (10,)
    assert y.shape == (5,)
    assert np.allclose(np.diff(x), 2.0)
    assert np.allclose(np.diff(y), 3.0)


def test_accumulate_normalizes_to_unit_integral():
    grid = DomainGrid(nx=20, ny=20, dx=1.0, dy=1.0, x0=0.0, y0=0.0)
    rng = np.random.default_rng(0)
    x = rng.uniform(0, 20, size=5000)
    y = rng.uniform(0, 20, size=5000)
    w = rng.uniform(0.5, 1.5, size=5000)
    dens = accumulate(x, y, w, grid)
    integral = dens.sum() * grid.dx * grid.dy
    assert integral == 1.0 or abs(integral - 1.0) < 1e-9


def test_cumulative_levels_monotonic():
    dens = np.zeros((10, 10))
    dens[5, 5] = 10.0
    dens[5, 4] = 5.0
    dens[4, 5] = 2.0
    levels = cumulative_levels(dens, fractions=(0.5, 0.8, 0.9), cell_area=1.0)
    # Higher cumulative fraction => lower (or equal) density threshold.
    assert levels[0.5] >= levels[0.8] >= levels[0.9]
