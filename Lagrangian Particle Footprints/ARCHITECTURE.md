# `lagranged` — Architecture Design

A Python package for estimating eddy-covariance (EC) flux footprints with a **backward Lagrangian stochastic (bLS) particle model**. Particles are released at the receptor (tower measurement height) and integrated backward in time; their contacts with the ground ("touchdowns") build a 2-D source-weight / probability-density field. The flux footprint is the source-weight field normalized so it integrates to 1.

The design targets a research-grade prototype, not a validated replacement for published models (Kljun FFP, Kormann–Meixner). Approximate parameterizations carry explicit `warnings`.

References staged in the repo: Wilson (1996, bLS), Minier & Pozorski (2014, PDF/Langevin methods), and a 2004 *Boundary-Layer Meteorology* article. The Langevin core follows Thomson (1987) well-mixed theory; touchdown counting follows Flesch, Wilson & Yee (1995/2004).

---

## 1. Package directory structure

```
lagranged/
├── pyproject.toml              # build + deps (PEP 621), tool config
├── README.md
├── LICENSE
├── CHANGELOG.md
├── CITATION.cff
├── src/
│   └── lagranged/
│       ├── __init__.py         # version + curated public exports
│       ├── constants.py        # κ, C0, g, defaults
│       ├── inputs.py           # FootprintInputs, TowerTurbulence, ReynoldsStress
│       ├── config.py           # ModelConfig, DomainGrid
│       ├── validation.py       # range checks, physical-consistency warnings
│       ├── profiles.py         # MOST mean-wind profile, ψ_m/ψ_h stability fns
│       ├── turbulence.py       # σ_i parameterizations, Lagrangian timescales, ε
│       ├── stochastic.py       # Langevin / well-mixed drift integrator (1-D & 3-D)
│       ├── particles.py        # release + trajectory orchestration
│       ├── touchdown.py        # touchdown detection + per-contact weights
│       ├── gridding.py         # 2-D density accumulation + smoothing
│       ├── contours.py         # cumulative source-area contours (e.g. 80 %)
│       ├── model.py            # FootprintModel orchestrator → FootprintResult
│       ├── results.py          # FootprintResult; xarray/dict serialization
│       ├── geo.py              # rotation, georeferencing, GeoTIFF, polygons
│       ├── plotting.py         # matplotlib views
│       ├── io.py               # tabular EC ingestion (pandas), batch helpers
│       ├── cli.py              # `lagranged` command-line entry point
│       └── _rng.py             # seeded RNG, reproducibility
├── tests/
│   ├── conftest.py
│   ├── test_profiles.py
│   ├── test_turbulence.py
│   ├── test_stochastic_wellmixed.py
│   ├── test_touchdown.py
│   ├── test_gridding.py
│   ├── test_contours.py
│   ├── test_model_integration.py
│   ├── test_geo.py
│   ├── test_inputs_validation.py
│   └── data/                   # small golden arrays + sample EC csv
├── examples/
│   ├── 01_synthetic_neutral.ipynb
│   ├── 02_stability_sweep.ipynb
│   ├── 03_tower_csv_ingest.ipynb
│   ├── 04_batch_geotiff_export.ipynb
│   ├── 05_measured_turbulence_reynolds.ipynb
│   └── scripts/run_single.py
└── docs/
    ├── index.md
    ├── theory.md               # equations, assumptions, caveats
    ├── api.md
    ├── quickstart.md
    └── limitations.md
```

`src/` layout (not flat) so tests run against the installed package and import errors surface early.

---

## 2. Module responsibilities

| Module | Responsibility | Key dependencies |
|---|---|---|
| `constants.py` | Physical constants (von Kármán κ≈0.40, C0≈3–6, g) and tunable defaults in one place. | — |
| `inputs.py` | Immutable input objects: `FootprintInputs` (FFP-style), `TowerTurbulence` (optional measured), `ReynoldsStress` (covariance matrix). Light derived properties only (e.g. `zm_eff = zm - d`). | dataclasses |
| `config.py` | `ModelConfig` (particle count, time step policy, C0, seed, BL treatment) and `DomainGrid` (extent, resolution, origin, CRS). Separates *physics inputs* from *numerical/output choices*. | dataclasses |
| `validation.py` | Pure functions validating ranges and physical consistency (e.g. `zm > z0`, `zm < h`, near-neutral when `\|L\|` large). Emits `warnings.warn` for approximate regimes; raises only on impossible values. | warnings |
| `profiles.py` | Monin–Obukhov mean wind `U(z)` and the integrated stability functions ψ_m, ψ_h (Businger–Dyer / Paulson stable & unstable branches). | numpy, scipy |
| `turbulence.py` | σ_u, σ_v, σ_w as functions of u\*, z/L, z, h (convective + mechanical parts); Lagrangian timescales T_Li(z); dissipation ε(z) from C0 and σ_w; gradient profiles `dσ²/dz` needed by the drift term. Honors measured values when supplied. | numpy, scipy |
| `stochastic.py` | The Langevin engine. Increments velocity with a deterministic drift (well-mixed correction term) plus a random kick scaled by `(C0 ε)^½`. Supports a fast 1-D-w / parameterized-uv mode and a full 3-D correlated mode driven by the Reynolds-stress matrix (Thomson 1987 generalized form). Adaptive `dt = min(τ_L)/N`. | numpy |
| `particles.py` | Releases N particles backward from `zm_eff`, advances position from velocity, applies reflection at the surface (z0 / small absorbing-reflecting height), terminates on touchdown or `t_max`/leaving domain. Vectorized over particles; optional numba/chunking. | numpy |
| `touchdown.py` | Detects ground contacts and assigns each a weight (the bLS estimator weight ∝ 1/\|w_touchdown\|, capped); accumulates landing (x,y) positions and weights. | numpy |
| `gridding.py` | Bins weighted touchdowns into the source-weight grid; optional kernel smoothing; normalizes to a footprint density integrating to 1; computes effective sample size / Monte-Carlo noise estimate. | numpy, scipy |
| `contours.py` | Cumulative contribution: sorts cells descending, finds isolines enclosing requested fractions (50/80/90 %); returns level values and contour paths. | numpy, scipy/matplotlib |
| `geo.py` | Rotates the model grid (x = upwind) into geographic orientation using `wind_dir`, georeferences to tower easting/northing + CRS, writes GeoTIFF via rioxarray/rasterio, builds contour polygons via shapely and a `GeoDataFrame`. | xarray, rioxarray, rasterio, geopandas, shapely, pyproj |
| `model.py` | `FootprintModel` orchestrator: validate → build profiles/turbulence → simulate → touchdown → grid → contour → assemble `FootprintResult`. Single seam everything composes through. | (all above) |
| `results.py` | `FootprintResult` container; `.to_xarray()`, `.to_dict()`, summary stats (peak distance x_peak, contour areas). | numpy, xarray |
| `plotting.py` | `plot_footprint`, `plot_contours`, `plot_profiles`, optional basemap overlay. Returns `matplotlib` Axes; never calls `show()`. | matplotlib |
| `io.py` | Read EC tabular output (EddyPro/AmeriFlux-style CSV) into rows of inputs; column-mapping config; `iter_records()` / `run_batch()`. | pandas |
| `cli.py` | `lagranged run config.yaml` and `lagranged batch data.csv --out dir/`; thin wrapper over the public API. | argparse/typer |
| `_rng.py` | Central `numpy.random.Generator` factory keyed on `ModelConfig.seed` for reproducibility. | numpy |

---

## 3. Core data classes

Frozen `@dataclass` objects (or pydantic if runtime coercion is wanted later). Units are SI and documented on every field.

```python
@dataclass(frozen=True)
class FootprintInputs:
    """FFP-style minimum inputs (one EC averaging period)."""
    zm: float            # measurement height above ground [m]
    z0: float | None     # roughness length [m] (None → derive from U & u*)
    d: float = 0.0       # displacement height [m]
    L: float             # Obukhov length [m] (sign convention: <0 unstable)
    ustar: float         # friction velocity [m s-1]
    umean: float | None  # mean wind speed at zm [m s-1] (optional, for σ_v check)
    wind_dir: float      # mean wind direction [deg from N, met convention]
    h: float             # boundary-layer height z_i [m]
    sigma_v: float       # lateral velocity std dev [m s-1]

    @property
    def zm_eff(self) -> float:        # height above displacement
        return self.zm - self.d


@dataclass(frozen=True)
class TowerTurbulence:
    """Optional measured turbulence + scalar fluxes (overrides parameterizations)."""
    sigma_u: float | None = None
    sigma_v: float | None = None
    sigma_w: float | None = None
    cov_uv:  float | None = None
    cov_uw:  float | None = None   # ≈ -u*² in aligned coords
    cov_vw:  float | None = None
    tke:     float | None = None
    epsilon: float | None = None   # dissipation rate [m² s-3]
    H:       float | None = None   # sensible heat flux [W m-2]
    LE:      float | None = None   # latent heat flux [W m-2]
    co2_flux: float | None = None  # [µmol m-2 s-1]
    qc_flag:  int | None = None    # EC quality flag (0=best)

    def reynolds_stress(self) -> "ReynoldsStress | None":
        """Assemble 3×3 covariance matrix if the six components are present."""


@dataclass(frozen=True)
class ReynoldsStress:
    """Symmetric 3×3 velocity covariance ⟨u_i' u_j'⟩; drives the 3-D drift term."""
    matrix: np.ndarray   # (3,3), positive semi-definite (validated)

    @classmethod
    def from_components(cls, su, sv, sw, cuv, cuw, cvw) -> "ReynoldsStress": ...
    @property
    def inverse(self) -> np.ndarray: ...


@dataclass(frozen=True)
class ModelConfig:
    n_particles: int = 50_000
    C0: float = 4.0              # Kolmogorov/Langevin constant (3–6)
    dt_factor: float = 0.02      # dt = dt_factor · min(τ_L)
    t_max: float = 1200.0        # max backward integration time [s]
    mode: str = "param"          # "param" | "reynolds" (full 3-D correlated)
    rebound_height: float | None = None  # touchdown surface (default ≈ z0)
    seed: int | None = None
    bl_reflection: bool = True   # reflect at z = h


@dataclass(frozen=True)
class DomainGrid:
    nx: int; ny: int
    dx: float; dy: float         # cell size [m]
    x0: float = 0.0; y0: float   # model origin (receptor at 0,0 upwind frame)
    origin_xy: tuple | None = None  # tower easting/northing for georef
    crs: str | None = None          # e.g. "EPSG:32612"


@dataclass
class FootprintResult:
    density: np.ndarray          # (ny,nx) footprint, ∫=1
    x: np.ndarray; y: np.ndarray # cell-center coords, upwind frame [m]
    contours: dict[float, ...]   # {0.8: paths/level, ...}
    x_peak: float                # along-wind peak distance [m]
    n_touchdowns: int
    mc_noise: float              # Monte-Carlo noise estimate
    inputs: FootprintInputs
    config: ModelConfig
    meta: dict
    def to_xarray(self) -> "xr.DataArray": ...
    def to_geotiff(self, path, grid: DomainGrid) -> None: ...
    def contour_gdf(self, grid: DomainGrid) -> "gpd.GeoDataFrame": ...
```

Design notes: inputs are split into *physics* (`FootprintInputs`/`TowerTurbulence`) vs *numerics & geometry* (`ModelConfig`/`DomainGrid`) so a single met record can be re-run at different resolutions or particle counts without touching the physics. `TowerTurbulence` is fully optional; when fields are `None`, `turbulence.py` falls back to parameterizations and warns.

---

## 4. Public API

```python
import lagranged as lg

# --- one-call convenience ---
result = lg.compute_footprint(
    zm=3.0, z0=0.03, d=0.2, L=-50.0, ustar=0.35,
    umean=2.4, wind_dir=210.0, h=1000.0, sigma_v=0.6,
    grid=lg.DomainGrid(nx=400, ny=400, dx=2.0, dy=2.0),
    n_particles=100_000, seed=42,
)
result.x_peak                      # along-wind peak
ax = lg.plot_footprint(result)

# --- explicit object workflow ---
inputs = lg.FootprintInputs(zm=3, z0=0.03, d=0.2, L=-50, ustar=0.35,
                            umean=2.4, wind_dir=210, h=1000, sigma_v=0.6)
turb   = lg.TowerTurbulence(sigma_u=0.7, sigma_v=0.6, sigma_w=0.4,
                            cov_uw=-0.12, epsilon=0.08)
model  = lg.FootprintModel(inputs, turbulence=turb,
                           config=lg.ModelConfig(mode="reynolds", seed=42),
                           grid=lg.DomainGrid(nx=400, ny=400, dx=2, dy=2))
result = model.run()

# --- geospatial export ---
result.to_geotiff("footprint.tif", grid)          # georeferenced raster
gdf = result.contour_gdf(grid)                     # 50/80/90 % polygons
result.to_xarray().to_netcdf("footprint.nc")

# --- batch over an EC time series ---
df = lg.io.read_ec_csv("tower.csv", mapping=lg.io.EDDYPRO_MAP)
results = lg.run_batch(df, grid=grid, config=lg.ModelConfig(seed=42),
                       progress=True)             # → list/dict of FootprintResult
```

Curated `lagranged/__init__.py` exports: `FootprintInputs`, `TowerTurbulence`, `ReynoldsStress`, `ModelConfig`, `DomainGrid`, `FootprintModel`, `FootprintResult`, `compute_footprint`, `run_batch`, `plot_footprint`, `plot_contours`, and the `io`/`geo` submodules. Everything else is internal.

CLI mirrors this: `lagranged run --config run.yaml` and `lagranged batch tower.csv --grid 400x400@2m --out results/`.

---

## 5. Test strategy

Physics models need *behavioral* and *statistical* tests, not just I/O checks.

**Deterministic unit tests**
- `profiles`: ψ_m → 0 as `L→∞`; recovers the log law in neutral conditions; `U(z0+d)=0`; monotonic in z.
- `turbulence`: σ_w/u\* ≈ 1.25 ± neutral limit; σ_i increase with instability; measured values pass through unchanged; ε > 0.
- `validation`: warns on `zm` near `h`, raises on `zm ≤ z0+d`, rejects non-PSD Reynolds matrices.
- `geo`: rotation round-trips (rotate by θ then −θ ≈ identity); GeoTIFF read-back preserves sum and CRS; contour polygons are valid and nested.

**Statistical / physics tests** (seeded, tolerance-based)
- **Well-mixed criterion** (the key correctness test): particles initialized uniformly in a horizontally homogeneous domain must stay statistically uniform in concentration over time (Thomson 1987). Asserts the integrator's drift term is consistent with the turbulence fields.
- **Touchdown mass**: total weighted touchdowns conserved; footprint density integrates to 1 within Monte-Carlo tolerance.
- **Convergence**: peak location and 80 % area stabilize as `n_particles` grows (halving noise ∝ N^−½).
- **Stability scaling**: x_peak moves closer to the tower as conditions become more unstable, farther when stable — sign/monotonicity sanity vs. published behavior.

**Regression / golden tests**
- Stored small `(ny,nx)` footprint arrays for fixed seed + inputs in `tests/data/`; assert allclose. Guards against silent numerical drift.

**Determinism & property tests**
- Same seed → identical arrays; different seed → statistically equivalent (KS on x_peak across replicates).
- `hypothesis` over plausible input ranges to confirm no NaN/inf escapes and validation triggers correctly.

**Cross-model sanity (non-blocking)**
- An `examples`/`tests/slow` comparison of x_peak and 80 % distance against a Kljun-FFP reference for a few neutral/unstable cases — documented as a sanity band, not an exact-match assertion. Mark `@pytest.mark.slow`.

Tooling: `pytest`, `pytest-cov`, `hypothesis`; small N and fixed seeds keep the suite fast, with `@slow` markers for convergence/cross-model runs. Target ≥85 % line coverage on physics modules.

---

## 6. Example notebooks / scripts

1. **`01_synthetic_neutral.ipynb`** — minimal FFP-style inputs, neutral case; run, plot density + contours, read off x_peak. The "hello world".
2. **`02_stability_sweep.ipynb`** — fix everything but `L`; show footprint contracting/expanding across stable→unstable; small-multiples figure.
3. **`03_tower_csv_ingest.ipynb`** — load an EddyPro/AmeriFlux CSV with `lg.io`, map columns, filter on QC flags, run one period.
4. **`04_batch_geotiff_export.ipynb`** — `run_batch` over a day of records, aggregate a climatological footprint, export georeferenced GeoTIFF + contour shapefile, overlay on a basemap with geopandas.
5. **`05_measured_turbulence_reynolds.ipynb`** — supply σ_u,v,w and the full covariance set; run `mode="reynolds"`; compare against the parameterized run to show the effect of measured anisotropy.
6. **`scripts/run_single.py`** — headless equivalent of notebook 1 for CLI/CI smoke testing.

Each notebook is also exported to `docs/` and exercised in CI via `nbmake` (or `jupyter nbconvert --execute`) so examples can't rot.

---

## 7. Dependencies

**Runtime (required)**
- `numpy` — arrays, vectorized particle integration.
- `scipy` — interpolation, special functions, contour/area helpers.
- `pandas` — tabular EC ingestion and batch records.
- `xarray` — labeled footprint grids, NetCDF I/O.
- `matplotlib` — plotting.

**Geospatial (required for export)**
- `rioxarray` + `rasterio` — GeoTIFF write, CRS handling.
- `geopandas` + `shapely` — contour polygons / vector output.
- `pyproj` — coordinate transforms (pulled in by the above).

**Optional / extras**
- `numba` (`lagranged[fast]`) — JIT the inner Langevin loop; pure-numpy fallback kept.
- `typer` (`lagranged[cli]`) — nicer CLI if preferred over argparse.
- `tqdm` — batch progress bars.

**Dev (`lagranged[dev]`)**
- `pytest`, `pytest-cov`, `hypothesis`, `nbmake` — testing.
- `ruff`, `black`, `mypy` — lint/format/type-check.
- `sphinx` or `mkdocs-material` — docs.

`pyproject.toml` declares core deps minimally and gates the heavier geospatial/accel stacks behind extras, so `pip install lagranged` works in a plain scientific environment and `pip install lagranged[geo,fast,dev]` pulls the rest.

---

## Key design principles

- **Single composition seam.** Everything flows through `FootprintModel.run()`; `compute_footprint` and the CLI are thin wrappers. One place to reason about correctness.
- **Physics ⊥ numerics ⊥ geometry.** Separating `FootprintInputs` from `ModelConfig`/`DomainGrid` lets the same met record be re-gridded or re-sampled freely.
- **Measured-overrides-parameterized.** `TowerTurbulence`/`ReynoldsStress` cleanly supersede the MOST parameterizations; absence triggers documented fallbacks with `warnings`.
- **Reproducibility.** All randomness routes through one seeded `Generator`.
- **Honest caveats.** Approximations (BL-height treatment, stability-function branches, touchdown-weight capping, well-mixed assumption in inhomogeneous terrain) raise `warnings` and are listed in `docs/limitations.md`. This is a research prototype, not a validated operational footprint model.
