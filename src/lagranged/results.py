"""The :class:`FootprintResult` container and serialization helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .config import DomainGrid, ModelConfig
from .inputs import FootprintInputs


@dataclass
class FootprintResult:
    """Output of a footprint computation.

    The ``density`` field is the source-weight grid normalized so that
    ``sum(density * dx * dy) == 1`` (within Monte-Carlo tolerance).
    """

    density: np.ndarray  # (ny, nx) footprint density, integrates to 1
    x: np.ndarray  # cell-center x coordinates, upwind frame [m]
    y: np.ndarray  # cell-center y coordinates, upwind frame [m]
    contours: dict[float, Any] = field(default_factory=dict)  # {0.8: level/paths}
    x_peak: float = float("nan")  # along-wind peak distance [m]
    n_touchdowns: int = 0
    mc_noise: float = float("nan")  # Monte-Carlo noise estimate
    inputs: FootprintInputs | None = None
    config: ModelConfig | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict view (arrays kept as numpy)."""
        return {
            "density": self.density,
            "x": self.x,
            "y": self.y,
            "contours": self.contours,
            "x_peak": self.x_peak,
            "n_touchdowns": self.n_touchdowns,
            "mc_noise": self.mc_noise,
            "meta": self.meta,
        }

    def to_xarray(self):
        """Return the density grid as a labeled :class:`xarray.DataArray`."""
        import xarray as xr

        return xr.DataArray(
            self.density,
            coords={"y": self.y, "x": self.x},
            dims=("y", "x"),
            name="footprint",
            attrs={"long_name": "flux footprint density", "units": "m-2", **self.meta},
        )

    def to_geotiff(self, path: str, grid: DomainGrid) -> None:
        """Write a georeferenced GeoTIFF (requires the ``geo`` extra)."""
        from . import geo

        geo.write_geotiff(self, grid, path)

    def contour_gdf(self, grid: DomainGrid):
        """Return contour polygons as a GeoDataFrame (requires the ``geo`` extra)."""
        from . import geo

        return geo.contours_to_gdf(self, grid)
