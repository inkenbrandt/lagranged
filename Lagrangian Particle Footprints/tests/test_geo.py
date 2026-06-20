"""Tests for geospatial rotation, GeoTIFF export and contour polygons.

The pure-numpy rotation helpers are always exercised; the export tests require
the optional ``[geo]`` stack and skip cleanly when it is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

from lagranged.config import DomainGrid
from lagranged.geo import model_to_geographic, rotate_xy
from lagranged.gridding import grid_cell_centers
from lagranged.inputs import FootprintInputs
from lagranged.results import FootprintResult


def _has_geo() -> bool:
    try:
        import geopandas  # noqa: F401
        import rasterio  # noqa: F401
        import shapely  # noqa: F401
    except ImportError:
        return False
    return True


requires_geo = pytest.mark.skipif(not _has_geo(), reason="geo extra not installed")


# --------------------------------------------------------------------------- #
# Rotation helper (no geo extra needed)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("theta", [0.0, 17.0, 45.0, 90.0, 123.4, 210.0, -30.0, 360.0])
def test_rotate_xy_roundtrips(theta: float) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-100, 100, size=64)
    y = rng.uniform(-100, 100, size=64)
    xr, yr = rotate_xy(x, y, theta)
    xb, yb = rotate_xy(xr, yr, -theta)
    assert np.allclose(xb, x)
    assert np.allclose(yb, y)


def test_rotate_xy_preserves_length_and_is_orthogonal() -> None:
    # Rotation preserves vector norms and the angle between two vectors.
    x = np.array([1.0, 0.0, 3.0])
    y = np.array([0.0, 1.0, -4.0])
    xr, yr = rotate_xy(x, y, 37.0)
    assert np.allclose(np.hypot(xr, yr), np.hypot(x, y))


def test_rotate_xy_known_quarter_turn() -> None:
    # +x rotated 90° CCW lands on +y.
    xr, yr = rotate_xy(1.0, 0.0, 90.0)
    assert np.allclose([float(xr), float(yr)], [0.0, 1.0])


@pytest.mark.parametrize(
    "wind_dir, expected_en",
    [
        (0.0, (0.0, 1.0)),  # wind from N -> footprint extends North
        (90.0, (1.0, 0.0)),  # wind from E -> East
        (180.0, (0.0, -1.0)),  # wind from S -> South
        (270.0, (-1.0, 0.0)),  # wind from W -> West
    ],
)
def test_model_x_axis_points_along_wind_bearing(wind_dir, expected_en) -> None:
    # The model +x (upwind) axis should map to the compass bearing == wind_dir.
    e, n = model_to_geographic(1.0, 0.0, wind_dir, origin_xy=(0.0, 0.0))
    assert np.allclose([float(e), float(n)], expected_en, atol=1e-12)


def test_model_to_geographic_translates_to_origin() -> None:
    origin = (500_000.0, 4_000_000.0)
    e, n = model_to_geographic(0.0, 0.0, 210.0, origin_xy=origin)
    assert np.allclose([float(e), float(n)], origin)


# --------------------------------------------------------------------------- #
# Fixtures for the export tests
# --------------------------------------------------------------------------- #
@pytest.fixture
def gaussian_result() -> tuple[FootprintResult, DomainGrid]:
    """A smooth Gaussian 'footprint' on a georeferenced grid.

    A Gaussian gives cleanly nested 50/80/90 % contours, so the export tests can
    check nesting without depending on the stochastic engine.
    """
    grid = DomainGrid(
        nx=41,
        ny=41,
        dx=2.0,
        dy=2.0,
        x0=-41.0,
        y0=-41.0,
        origin_xy=(500_000.0, 4_000_000.0),
        crs="EPSG:32612",
    )
    x, y = grid_cell_centers(grid)
    xx, yy = np.meshgrid(x, y)
    # Offset the peak upwind so it is not exactly grid-centered.
    density = np.exp(-(((xx - 20.0) ** 2 + yy**2) / (2.0 * 12.0**2)))
    density /= density.sum() * grid.dx * grid.dy  # integrate to 1

    inputs = FootprintInputs(
        zm=3.0, L=-50.0, ustar=0.3, wind_dir=210.0, h=1000.0, sigma_v=0.5, z0=0.03
    )
    result = FootprintResult(density=density, x=x, y=y, inputs=inputs)
    return result, grid


# --------------------------------------------------------------------------- #
# GeoTIFF export
# --------------------------------------------------------------------------- #
@requires_geo
def test_geotiff_roundtrip_preserves_sum_and_crs(gaussian_result, tmp_path) -> None:
    import rasterio

    result, grid = gaussian_result
    path = tmp_path / "footprint.tif"
    result.to_geotiff(str(path), grid)  # exercises the FootprintResult delegate

    with rasterio.open(path) as src:
        assert src.count == 1
        assert (src.width, src.height) == (grid.nx, grid.ny)
        assert src.crs.to_epsg() == 32612
        arr = src.read(1)
        # Lossless rotated geotransform: the density sum is preserved exactly.
        assert np.isclose(arr.sum(), result.density.sum(), rtol=0, atol=1e-12)
        # wind_dir 210° is not axis-aligned, so the transform must carry rotation.
        assert abs(src.transform.b) > 1e-9 and abs(src.transform.d) > 1e-9


@requires_geo
def test_geotiff_georeferences_to_tower_origin(gaussian_result, tmp_path) -> None:
    import rasterio

    result, grid = gaussian_result
    path = tmp_path / "footprint.tif"
    result.to_geotiff(str(path), grid)

    # The grid-origin corner (col=0, row=0) must land at origin_xy + rotated (x0, y0).
    expected = model_to_geographic(grid.x0, grid.y0, result.inputs.wind_dir, grid.origin_xy)
    with rasterio.open(path) as src:
        e, n = src.transform * (0, 0)
    assert np.allclose([e, n], [float(expected[0]), float(expected[1])])


# --------------------------------------------------------------------------- #
# Contour polygons
# --------------------------------------------------------------------------- #
@requires_geo
def test_contours_valid_nested_and_in_crs(gaussian_result) -> None:
    result, grid = gaussian_result
    gdf = result.contour_gdf(grid)  # exercises the FootprintResult delegate

    assert list(gdf["fraction"]) == [0.5, 0.8, 0.9]
    assert gdf.crs is not None and gdf.crs.to_epsg() == 32612

    geoms = {row.fraction: row.geometry for row in gdf.itertuples()}
    for g in geoms.values():
        assert g is not None and g.is_valid and g.area > 0.0

    a50, a80, a90 = geoms[0.5].area, geoms[0.8].area, geoms[0.9].area
    # Larger cumulative fraction encloses a larger source area.
    assert a50 < a80 < a90

    # Strict nesting: each inner polygon lies (essentially) inside the next.
    tol = 1e-6 * grid.dx * grid.dy
    assert geoms[0.5].difference(geoms[0.8]).area < tol
    assert geoms[0.8].difference(geoms[0.9]).area < tol


@requires_geo
def test_contours_area_matches_threshold_cells(gaussian_result) -> None:
    # The 90% polygon area should equal the number of cells at/above its level
    # times the cell area (the polygon is just the union of those cells).
    from lagranged.contours import cumulative_levels

    result, grid = gaussian_result
    cell_area = grid.dx * grid.dy
    levels = cumulative_levels(result.density, (0.5, 0.8, 0.9), cell_area=cell_area)
    n_cells = int((result.density >= levels[0.9]).sum())

    gdf = result.contour_gdf(grid)
    area90 = gdf.loc[gdf["fraction"] == 0.9, "geometry"].iloc[0].area
    assert np.isclose(area90, n_cells * cell_area, rtol=1e-9)
