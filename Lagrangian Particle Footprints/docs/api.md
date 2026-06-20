# API reference

Everything below is generated directly from the source docstrings with
[mkdocstrings](https://mkdocstrings.github.io/). The curated public surface is
what `import lagranged as lg` exposes:

| Symbol | Kind | Purpose |
|--------|------|---------|
| [`FootprintInputs`][lagranged.inputs.FootprintInputs] | dataclass | FFP-style physics inputs for one period. |
| [`TowerTurbulence`][lagranged.inputs.TowerTurbulence] | dataclass | Optional measured σ_i, covariances, fluxes. |
| [`ReynoldsStress`][lagranged.inputs.ReynoldsStress] | dataclass | Validated symmetric 3×3 velocity covariance. |
| [`ModelConfig`][lagranged.config.ModelConfig] | dataclass | Particle count, C0, dt policy, mode, seed. |
| [`DomainGrid`][lagranged.config.DomainGrid] | dataclass | Accumulation grid + optional georeferencing. |
| [`FootprintModel`][lagranged.model.FootprintModel] | class | Orchestrator; `.run() -> FootprintResult`. |
| [`FootprintResult`][lagranged.results.FootprintResult] | dataclass | Density grid, contours, summary stats, export. |
| [`compute_footprint`][lagranged.model.compute_footprint] | function | One-call convenience wrapper. |
| [`run_batch`][lagranged.model.run_batch] | function | Footprint per row of an EC table. |
| [`plot_footprint`][lagranged.plotting.plot_footprint], [`plot_contours`][lagranged.plotting.plot_contours] | functions | Matplotlib views (return `Axes`). |
| [`lg.io`][lagranged.io] | submodule | Tabular EC ingestion (lazy import). |
| [`lg.geo`][lagranged.geo] | submodule | Raster/vector export (lazy import; `[geo]` extra). |

## Object workflow

```python
import lagranged as lg

inputs = lg.FootprintInputs(zm=3, z0=0.03, d=0.2, L=-50, ustar=0.35,
                            umean=2.4, wind_dir=210, h=1000, sigma_v=0.6)
turb   = lg.TowerTurbulence(sigma_u=0.7, sigma_v=0.6, sigma_w=0.4,
                            cov_uw=-0.12, epsilon=0.08)
grid   = lg.DomainGrid(nx=400, ny=400, dx=2, dy=2)
model  = lg.FootprintModel(inputs, grid=grid, turbulence=turb,
                           config=lg.ModelConfig(mode="reynolds", seed=42))
result = model.run()
```

## Export

```python
result.to_xarray().to_netcdf("footprint.nc")   # core
result.to_geotiff("footprint.tif", grid)        # needs lagranged[geo]
gdf = result.contour_gdf(grid)                   # needs lagranged[geo]
```

---

## Inputs

::: lagranged.inputs.FootprintInputs

::: lagranged.inputs.TowerTurbulence

::: lagranged.inputs.ReynoldsStress

## Configuration

::: lagranged.config.ModelConfig

::: lagranged.config.DomainGrid

## Running a footprint

::: lagranged.model.FootprintModel

::: lagranged.model.compute_footprint

::: lagranged.model.run_batch

## Results

::: lagranged.results.FootprintResult

## Plotting

::: lagranged.plotting.plot_footprint

::: lagranged.plotting.plot_contours

## Tabular I/O — `lagranged.io`

::: lagranged.io

## Geospatial export — `lagranged.geo`

The geospatial helpers require the optional `[geo]` extra
(`pip install "lagranged[geo]"`). They are reachable as `lg.geo` (lazily
imported) once that stack is installed.

::: lagranged.geo

---

## Internals — physics modules

These modules implement the equations in [Theory & assumptions](theory.md). They
are not part of the curated public API and their signatures may change, but they
are documented here because they carry the core micrometeorology.

### Input validation — `lagranged.validation`

::: lagranged.validation

### Mean wind & stability functions — `lagranged.profiles`

::: lagranged.profiles

### Turbulence statistics — `lagranged.turbulence`

::: lagranged.turbulence

### Langevin integrator — `lagranged.stochastic`

::: lagranged.stochastic

### Particle release & advection — `lagranged.particles`

::: lagranged.particles

### Touchdown weighting — `lagranged.touchdown`

::: lagranged.touchdown

### Gridding — `lagranged.gridding`

::: lagranged.gridding

### Cumulative source-area contours — `lagranged.contours`

::: lagranged.contours
