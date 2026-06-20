"""Matplotlib views of footprint results.

All functions accept an optional ``ax`` and return the :class:`~matplotlib.axes.Axes`;
none of them call ``plt.show()`` so they compose cleanly in scripts and notebooks.
"""

from __future__ import annotations

from .results import FootprintResult


def plot_footprint(result: FootprintResult, ax=None, **kwargs):
    """Filled pcolormesh of the footprint density in the upwind model frame."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    mesh = ax.pcolormesh(result.x, result.y, result.density, shading="auto", **kwargs)
    ax.set_xlabel("along-wind distance x [m]")
    ax.set_ylabel("cross-wind distance y [m]")
    ax.set_aspect("equal")
    ax.figure.colorbar(mesh, ax=ax, label="footprint density [m$^{-2}$]")
    return ax


def plot_contours(result: FootprintResult, ax=None, **kwargs):
    """Overlay cumulative source-area contour levels on the density field."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()
    ax.contour(
        result.x,
        result.y,
        result.density,
        levels=sorted(v for v in result.contours.values() if v == v),  # drop NaN
        **kwargs,
    )
    ax.set_xlabel("along-wind distance x [m]")
    ax.set_ylabel("cross-wind distance y [m]")
    ax.set_aspect("equal")
    return ax
