# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/) and this project adheres to
[Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.2.0] — 2026-06-20

First functional release: the backward Lagrangian stochastic engine and the
features listed as "not yet implemented" in 0.1.0 are now in place. This remains
a **research-grade prototype**, not a validated operational model — see
[`docs/limitations.md`](docs/limitations.md).

### Added
- Backward Lagrangian stochastic engine: Thomson (1987) well-mixed drift
  (`stochastic`), backward particle integration with perfect boundary-layer
  reflection (`particles`), and first-passage touchdown detection with a
  floored `1/|w|` estimator weight (`touchdown`).
- Turbulence parameterizations (`turbulence`): MOST σ_u/σ_v/σ_w profiles,
  Lagrangian timescales, dissipation, σ² gradients, and convective (w\*)
  contributions in unstable conditions.
- `run_batch` (`model`) for one footprint per row of an eddy-covariance CSV.
- `lagranged` command-line interface (`cli`) with `run` (single config) and
  `batch` (per-row CSV) subcommands and grid-spec parsing (e.g. `400x400@2m`).
- Geospatial export (`geo`, optional `[geo]` extra): `write_geotiff` and
  `contours_to_gdf`, plus model→geographic rotation and affine georeferencing;
  surfaced on `FootprintResult` as `to_geotiff` / `contour_gdf`.
- `FootprintResult` diagnostics: `x_peak`, `n_touchdowns`, and `mc_noise`
  (inverse square-root of the Kish effective sample size), with `to_dict` /
  `to_xarray` views.
- Input validation and edge-of-validity runtime warnings
  (`validation.validate_inputs`).
- Deterministic seeding through a single `numpy.random.Generator`
  (`_rng`, `ModelConfig.seed`).

### Tests & docs
- Regression goldens for neutral/stable/unstable cases
  (`tests/data/golden`, `scripts/regenerate_goldens.py`) and a non-blocking
  Kljun FFP (2015) cross-model sanity band (`@pytest.mark.slow`).
- New test modules: stochastic, well-mixed, particles, touchdown, turbulence,
  physics behavior, model integration, geo, batch/CLI, regression, cross-model.
- Expanded documentation: theory, limitations, quickstart, and API reference;
  `mkdocs-material` site config and `nox` sessions.

## [0.1.0] — 2026-06-18

### Added
- Initial package scaffold: `src/lagranged/` with PEP 621 `pyproject.toml`,
  `src` layout, and editable-install support.
- Input/config dataclasses: `FootprintInputs`, `TowerTurbulence`,
  `ReynoldsStress`, `ModelConfig`, `DomainGrid`.
- Functional `profiles` (MOST mean wind, ψ_m/ψ_h), `gridding`, and `contours`.
- `FootprintModel` orchestrator, `compute_footprint`, plotting, lazy `io`/`geo`.
- Smoke, profile, and gridding tests; docs and example skeletons.
