"""Accumulate weighted touchdowns into a normalized 2-D footprint density.

Bins weighted contacts onto a :class:`lagranged.config.DomainGrid`, optionally
smooths with a kernel, and normalizes so the density integrates to 1. Also
estimates Monte-Carlo noise / effective sample size.
"""

from __future__ import annotations

import numpy as np

from .config import DomainGrid


def grid_cell_centers(grid: DomainGrid) -> tuple[np.ndarray, np.ndarray]:
    """Return (x, y) cell-center coordinate vectors for ``grid`` [m]."""
    x = grid.x0 + (np.arange(grid.nx) + 0.5) * grid.dx
    y = grid.y0 + (np.arange(grid.ny) + 0.5) * grid.dy
    return x, y


def accumulate(x_td, y_td, w_td, grid: DomainGrid) -> np.ndarray:
    """Bin weighted touchdowns into a (ny, nx) density that integrates to 1.

    Parameters
    ----------
    x_td, y_td:
        Touchdown coordinates in the upwind model frame [m].
    w_td:
        Per-contact weights.
    grid:
        Target accumulation grid.
    """
    x_edges = grid.x0 + np.arange(grid.nx + 1) * grid.dx
    y_edges = grid.y0 + np.arange(grid.ny + 1) * grid.dy
    h, _, _ = np.histogram2d(y_td, x_td, bins=[y_edges, x_edges], weights=w_td)
    total = h.sum() * grid.dx * grid.dy
    if total > 0:
        h = h / total
    return h
