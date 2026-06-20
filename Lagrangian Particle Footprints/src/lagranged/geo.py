"""Geospatial rotation, georeferencing, raster and vector export.

The footprint is computed in the *upwind model frame* (receptor at the origin,
``+x`` pointing **upwind** — the direction the source area extends — and ``+y``
the lateral offset).  This module rotates that frame into geographic orientation
using ``inputs.wind_dir`` and georeferences it to the tower easting/northing
(``grid.origin_xy``) in ``grid.crs``.

Rotation is encoded **losslessly in the GeoTIFF affine geotransform** (a rotated
raster) rather than by resampling the array.  That keeps the exported pixel
values — and therefore the footprint sum — bit-for-bit identical to the model
output while still placing every cell at its correct geographic location.

Wind-direction convention
--------------------------
``wind_dir`` follows the meteorological convention: the compass bearing (degrees
clockwise from North) the wind blows *from*.  The footprint extends upwind, so
the model ``+x`` axis points along that bearing.  A compass bearing ``β`` is the
unit vector ``(E, N) = (sin β, cos β)``; mapping the model ``+x`` axis onto it is
a counter-clockwise planar rotation by ``α = 90° − β`` (see :func:`rotate_xy`).

Requires the optional geospatial stack: ``pip install lagranged[geo]``
(rioxarray, rasterio, geopandas, shapely, pyproj).
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .config import DomainGrid
from .contours import cumulative_levels
from .results import FootprintResult

if TYPE_CHECKING:  # imported lazily at runtime; only needed for static typing
    import geopandas as gpd
    from affine import Affine

__all__ = ["rotate_xy", "model_to_geographic", "write_geotiff", "contours_to_gdf"]

_GEO_HINT = "Geospatial export needs the 'geo' extra: pip install lagranged[geo]"

# Cumulative source-area fractions exported as contour polygons (largest last so
# the resulting polygons are nested: 50% area ⊂ 80% area ⊂ 90% area).
_CONTOUR_FRACTIONS: tuple[float, ...] = (0.5, 0.8, 0.9)


def _require_geo() -> None:
    """Raise a helpful :class:`ImportError` if the optional geo stack is absent."""
    try:
        import geopandas  # noqa: F401
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError as exc:  # pragma: no cover - depends on install
        raise ImportError(_GEO_HINT) from exc


# --------------------------------------------------------------------------- #
# Rotation helpers (pure numpy — importable without the geo extra)
# --------------------------------------------------------------------------- #
def rotate_xy(
    x: np.ndarray | float,
    y: np.ndarray | float,
    angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Rotate planar coordinates counter-clockwise by ``angle_deg`` degrees.

    Applies the standard 2-D rotation::

        x' = x·cos θ − y·sin θ
        y' = x·sin θ + y·cos θ

    Works element-wise on arrays or scalars.  The transform is orthogonal, so it
    round-trips exactly: ``rotate_xy(*rotate_xy(x, y, θ), -θ) == (x, y)`` to
    floating-point precision.
    """
    theta = np.radians(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x * c - y * s, x * s + y * c


def _geo_angle(wind_dir: float) -> float:
    """CCW rotation [deg] mapping the model (upwind-``x``) frame onto ``(E, N)``.

    With ``α = 90° − wind_dir`` the model ``+x`` axis maps to the compass bearing
    ``wind_dir`` (i.e. ``(E, N) = (sin wind_dir, cos wind_dir)``).
    """
    return 90.0 - wind_dir


def model_to_geographic(
    x: np.ndarray | float,
    y: np.ndarray | float,
    wind_dir: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
) -> tuple[np.ndarray, np.ndarray]:
    """Map upwind-frame coordinates ``(x, y)`` to geographic ``(easting, northing)``.

    Rotates by ``wind_dir`` (see module docstring) and translates by the tower
    location ``origin_xy``.
    """
    e0, n0 = origin_xy
    e, n = rotate_xy(x, y, _geo_angle(wind_dir))
    return e + e0, n + n0


def _grid_affine(grid: DomainGrid, wind_dir: float, origin_xy: tuple[float, float]) -> Affine:
    """Build the rotated raster affine mapping pixel ``(col, row)`` → ``(E, N)``.

    ``col`` indexes the model ``x`` (grid columns) and ``row`` the model ``y``
    (grid rows), matching the ``(ny, nx)`` density layout.  The transform encodes
    the wind-direction rotation and the translation to ``origin_xy`` so the
    GeoTIFF georeferences losslessly without resampling.
    """
    from affine import Affine

    alpha = _geo_angle(wind_dir)
    # World location of the (col=0, row=0) corner of the grid.
    e0, n0 = model_to_geographic(grid.x0, grid.y0, wind_dir, origin_xy)
    # (E, N) world step for +1 column (+dx in model x) and +1 row (+dy in model y).
    a, d = rotate_xy(grid.dx, 0.0, alpha)
    b, e = rotate_xy(0.0, grid.dy, alpha)
    return Affine(float(a), float(b), float(e0), float(d), float(e), float(n0))


def _resolve_georef(
    result: FootprintResult, grid: DomainGrid
) -> tuple[float, tuple[float, float], str | None]:
    """Pull ``wind_dir``, ``origin_xy`` and ``crs``, warning on missing georef."""
    if result.inputs is None:
        raise ValueError("result.inputs is required to georeference the footprint (need wind_dir).")
    wind_dir = float(result.inputs.wind_dir)
    if grid.origin_xy is None:
        warnings.warn(
            "grid.origin_xy is None; placing the tower at (0, 0) in the grid CRS.",
            stacklevel=3,
        )
        origin_xy = (0.0, 0.0)
    else:
        origin_xy = grid.origin_xy
    return wind_dir, origin_xy, grid.crs


def write_geotiff(result: FootprintResult, grid: DomainGrid, path: str) -> None:
    """Write a georeferenced GeoTIFF of the footprint density.

    The upwind-frame density is rotated into geographic orientation with
    ``result.inputs.wind_dir`` and georeferenced to ``grid.origin_xy`` /
    ``grid.crs`` via a rotated affine geotransform (no resampling — the stored
    pixel values, and hence the footprint sum, are preserved exactly).
    """
    _require_geo()
    import rasterio

    wind_dir, origin_xy, crs = _resolve_georef(result, grid)
    if crs is None:
        warnings.warn("grid.crs is None; writing GeoTIFF without a CRS.", stacklevel=2)

    density = np.ascontiguousarray(result.density, dtype="float64")
    ny, nx = density.shape
    transform = _grid_affine(grid, wind_dir, origin_xy)

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=ny,
        width=nx,
        count=1,
        dtype="float64",
        crs=crs,
        transform=transform,
        compress="deflate",
    ) as dst:
        dst.write(density, 1)
        dst.set_band_description(1, "flux footprint density [m-2]")
        dst.update_tags(
            wind_dir=f"{wind_dir:g}",
            frame="upwind model frame (x=upwind) rotated into geographic orientation",
            normalization="integrates to 1 over the grid",
        )


def contours_to_gdf(result: FootprintResult, grid: DomainGrid) -> gpd.GeoDataFrame:
    """Return 50/80/90 % cumulative source-area polygons as a GeoDataFrame.

    For each fraction the density iso-level enclosing that cumulative
    contribution is found (:func:`lagranged.contours.cumulative_levels`); the
    cells at or above it are polygonized with :func:`rasterio.features.shapes`
    using the same rotated affine as :func:`write_geotiff`, so the returned
    polygons live in ``grid.crs``.  Because higher fractions use lower density
    thresholds the masks — and therefore the polygons — are strictly nested
    (50 % ⊆ 80 % ⊆ 90 %).
    """
    _require_geo()
    import geopandas as gpd
    from rasterio import features
    from shapely.geometry import shape
    from shapely.ops import unary_union

    wind_dir, origin_xy, crs = _resolve_georef(result, grid)
    transform = _grid_affine(grid, wind_dir, origin_xy)

    density = np.ascontiguousarray(result.density, dtype="float64")
    cell_area = grid.dx * grid.dy
    levels = cumulative_levels(density, _CONTOUR_FRACTIONS, cell_area=cell_area)

    records: list[dict[str, object]] = []
    for frac in _CONTOUR_FRACTIONS:
        level = levels[frac]
        geom = None
        if np.isfinite(level):
            mask = density >= level
            if mask.any():
                polys = [
                    shape(g)
                    for g, _ in features.shapes(
                        mask.astype(np.uint8), mask=mask, transform=transform
                    )
                ]
                if polys:
                    geom = unary_union(polys)
                    if not geom.is_valid:  # repair any self-touching boundaries
                        geom = geom.buffer(0)
        records.append({"fraction": frac, "level": level, "geometry": geom})

    return gpd.GeoDataFrame(records, geometry="geometry", crs=crs)
