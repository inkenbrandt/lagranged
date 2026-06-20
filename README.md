# lagranged

Eddy-covariance flux footprints via a **backward Lagrangian stochastic (bLS)**
particle model.

Particles are released at the receptor (tower measurement height) and integrated
backward in time; their contacts with the ground ("touchdowns") build a 2-D
source-weight / probability-density field that, once normalized, is the flux
footprint. The package supports both simple FFP-style inputs and richer
eddy-covariance tower inputs (measured σ_u/σ_v/σ_w and the full Reynolds-stress
covariance matrix).

> ⚠️ **Research prototype.** Not a validated replacement for published footprint
> models (Kljun FFP, Kormann–Meixner). Approximate parameterizations emit
> `warnings`; see [`docs/limitations.md`](docs/limitations.md).
>
> 🚧 **Scaffold status.** The package structure, input/config objects, MOST
> profiles, gridding, contour, and plotting layers are in place and tested. The
> stochastic trajectory engine (`stochastic` / `particles` / `touchdown`) is
> stubbed and raises `NotImplementedError` when invoked.

## Install

Modern `src`-layout package; install in editable mode:

```bash
pip install -e .            # core: numpy, scipy, pandas, xarray, matplotlib
pip install -e ".[geo]"     # + GeoTIFF / vector export (rasterio, geopandas, ...)
pip install -e ".[dev]"     # + pytest, hypothesis, ruff, black, mypy
pip install -e ".[all]"     # everything
```

## Quick check

```python
import lagranged
print(lagranged.__version__)   # 0.2.0
```

## Usage

```python
import lagranged as lg

grid = lg.DomainGrid(nx=400, ny=400, dx=2.0, dy=2.0, x0=-400, y0=-400)

result = lg.compute_footprint(
    zm=3.0, z0=0.03, d=0.2, L=-50.0, ustar=0.35,
    umean=2.4, wind_dir=210.0, h=1000.0, sigma_v=0.6,
    grid=grid, n_particles=100_000, seed=42,
)

ax = lg.plot_footprint(result)
result.to_xarray().to_netcdf("footprint.nc")
```

See [`docs/`](docs/) for theory, the API reference, and limitations, and
[`examples/`](examples/) for runnable examples.

## Project layout

```
src/lagranged/    package source (inputs, config, profiles, turbulence,
                  stochastic, particles, touchdown, gridding, contours,
                  model, results, geo, plotting, io, cli)
tests/            pytest suite (smoke, profiles, gridding, ...)
examples/         scripts and (planned) notebooks
docs/             index, theory, api, quickstart, limitations
```

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check . && black --check . && mypy
```

## License

MIT — see [LICENSE](LICENSE).
