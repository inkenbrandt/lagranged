# Quickstart

```python
import lagranged as lg

print(lg.__version__)

grid = lg.DomainGrid(nx=400, ny=400, dx=2.0, dy=2.0, x0=-400, y0=-400)

result = lg.compute_footprint(
    zm=3.0, z0=0.03, d=0.2, L=-50.0, ustar=0.35,
    umean=2.4, wind_dir=210.0, h=1000.0, sigma_v=0.6,
    grid=grid, n_particles=100_000, seed=42,
)

result.x_peak                 # along-wind peak distance
ax = lg.plot_footprint(result)
```

Explicit object workflow and batch/geo export are described in the
[API reference](api.md).

!!! note "Research prototype"
    `compute_footprint` now runs the full backward-stochastic pipeline
    end-to-end (validate → simulate → touchdown → grid → contour). It remains a
    research-grade approximation: when measured turbulence is not supplied it
    falls back to parameterized profiles and emits `warnings`. See
    [Limitations & caveats](limitations.md).
