"""Cumulative source-area contours.

Given a normalized density, finds the iso-level enclosing a requested
cumulative contribution fraction (e.g. 50/80/90 %) by sorting cells in
descending order and accumulating until the target fraction is reached.
"""

from __future__ import annotations

import numpy as np


def cumulative_levels(
    density: np.ndarray,
    fractions=(0.5, 0.8, 0.9),
    cell_area: float = 1.0,
) -> dict[float, float]:
    """Return ``{fraction: density_level}`` enclosing each cumulative fraction.

    The level is the density value of the lowest-density cell still inside the
    source area that accounts for ``fraction`` of the total contribution.
    """
    flat = np.sort(density.ravel())[::-1]
    csum = np.cumsum(flat) * cell_area
    total = csum[-1] if csum.size else 0.0
    levels: dict[float, float] = {}
    for f in fractions:
        if total <= 0:
            levels[f] = float("nan")
            continue
        idx = int(np.searchsorted(csum, f * total))
        idx = min(idx, flat.size - 1)
        levels[f] = float(flat[idx])
    return levels
