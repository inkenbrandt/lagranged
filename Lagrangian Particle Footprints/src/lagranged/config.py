"""Numerical and geometry configuration objects.

Kept separate from the physics inputs (:mod:`lagranged.inputs`) so the same met
record can be re-gridded or re-sampled freely.
"""

from __future__ import annotations

from dataclasses import dataclass

from .constants import C0_DEFAULT


@dataclass(frozen=True)
class ModelConfig:
    """Numerical / Monte-Carlo settings for a footprint run."""

    n_particles: int = 50_000
    """Number of backward particles released."""
    C0: float = C0_DEFAULT
    """Kolmogorov/Langevin constant (literature 3–6)."""
    dt_factor: float = 0.02
    """Time step as a fraction of the minimum Lagrangian timescale: dt = dt_factor·min(τ_L)."""
    t_max: float = 1200.0
    """Maximum backward integration time per particle [s]."""
    mode: str = "param"
    """``"param"`` (parameterized σ_i) or ``"reynolds"`` (full 3-D correlated)."""
    rebound_height: float | None = None
    """Surface contact height [m]; defaults to ≈ z0 when ``None``."""
    seed: int | None = None
    """RNG seed for reproducibility."""
    bl_reflection: bool = True
    """Reflect particles at the boundary-layer top z = h."""

    def __post_init__(self) -> None:
        if self.n_particles <= 0:
            raise ValueError("n_particles must be positive.")
        if self.C0 <= 0:
            raise ValueError("C0 must be positive.")
        if self.mode not in ("param", "reynolds"):
            raise ValueError(f"mode must be 'param' or 'reynolds', got {self.mode!r}.")


@dataclass(frozen=True)
class DomainGrid:
    """Rectangular accumulation grid in the upwind model frame (receptor at origin)."""

    nx: int
    ny: int
    dx: float
    dy: float
    x0: float = 0.0
    """Grid origin x in the model (upwind) frame [m]."""
    y0: float = 0.0
    """Grid origin y in the model (upwind) frame [m]."""
    origin_xy: tuple[float, float] | None = None
    """Tower easting/northing for georeferencing (optional)."""
    crs: str | None = None
    """Coordinate reference system, e.g. ``"EPSG:32612"`` (optional)."""

    def __post_init__(self) -> None:
        if self.nx <= 0 or self.ny <= 0:
            raise ValueError("nx and ny must be positive.")
        if self.dx <= 0 or self.dy <= 0:
            raise ValueError("dx and dy must be positive.")
