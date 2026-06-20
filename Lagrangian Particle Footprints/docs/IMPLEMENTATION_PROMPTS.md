# Claude Code implementation prompts

A dependency-ordered sequence of prompts to finish `lagranged` from its current
scaffold. Paste them into Claude Code (VS Code) **one at a time**, in order, and
review/commit between each. Every prompt ends with acceptance criteria so Claude
verifies its own work before you move on.

**Workflow tips**
- Start each session with: *"Read ARCHITECTURE.md, README.md, and the relevant
  module before writing code. This is a research-grade prototype — add
  `warnings` where assumptions are approximate."*
- Commit after each prompt passes (`git add -A && git commit`).
- Run `pytest -q` and `ruff check . && black --check . && mypy` after each.
- If a prompt is too large for one pass, tell Claude *"do step N only."*

---

## Phase 0 — Environment & guardrails

```
Set up the dev environment and CI guardrails for the lagranged package.
1. Confirm `pip install -e ".[dev]"` succeeds; fix any pyproject issues.
2. Verify `import lagranged; print(lagranged.__version__)` prints 0.1.0 and that
   `pytest -q` passes the existing scaffold tests.
3. Add a pre-commit config (ruff, black, mypy) and a GitHub Actions workflow
   (.github/workflows/ci.yml) that runs ruff, black --check, mypy, and pytest on
   Python 3.10–3.12.
Acceptance: clean `pytest`, clean `ruff`/`black`/`mypy`, CI workflow file present
and syntactically valid (validate the YAML).
```

---

## Phase 1 — Turbulence parameterizations

```
Implement src/lagranged/turbulence.py. Read profiles.py and constants.py first.
Goal: height- and stability-dependent turbulence fields the Langevin engine needs.
Implement:
- sigma_profiles(z, ustar, L, h): σ_u, σ_v, σ_w(z) blending mechanical (∝ u*) and
  convective parts; use a documented surface-layer formulation (cite Kljun et al.
  2004 / Rannik et al. 2003 in the docstring). Recover the neutral ratios from
  constants.py as |z/L|→0.
- lagrangian_timescales(z, sigma_w, ustar, L): T_Lu, T_Lv, T_Lw(z) [s].
- gradient terms dσ_i²/dz needed by the well-mixed drift (analytic or finite-diff).
- Keep dissipation() consistent with ε = 2 σ_w² / (C0 T_L).
- When measured TowerTurbulence values are supplied, pass them through unchanged.
Emit a `warnings.warn` when falling back to parameterized values.
Write tests/test_turbulence.py: σ_w/u* ≈ 1.25 in the neutral limit; σ_i grow with
instability; ε > 0; measured values pass through; gradients match finite-diff.
Acceptance: new tests pass; existing tests still pass; ruff/black/mypy clean.
```

---

## Phase 2 — Stochastic Langevin engine

```
Implement src/lagranged/stochastic.py — the core well-mixed Langevin integrator.
Read turbulence.py, constants.py, _rng.py, and docs/theory.md first.
Implement:
- step_1d(...): advance vertical velocity w with the Thomson (1987) 1-D
  well-mixed drift term a_w (including the dσ_w²/dz correction) plus the random
  kick (C0·ε)^½·dW. Parameterized horizontal velocities for the "param" mode.
- step_3d(...): full 3-D correlated Gaussian form driven by the inverse
  Reynolds-stress matrix (Thomson 1987 generalized; ReynoldsStress.inverse).
All randomness must come from a passed-in numpy Generator (no global state).
Vectorize over particles (operate on (N,) / (N,3) arrays).
Document the well-mixed condition and cite Thomson (1987) in docstrings.
Acceptance: functions return finite arrays of correct shape for N particles;
add a fast unit test that one step preserves array shapes and is deterministic
under a fixed seed. Full physics validation comes in Phase 4.
```

---

## Phase 3 — Particle release, advection, touchdown

```
Implement src/lagranged/particles.py and src/lagranged/touchdown.py.
Read stochastic.py, config.py, inputs.py first.
particles.simulate_trajectories(inputs, turbulence, config, rng):
- Release config.n_particles backward from inputs.zm_eff with velocities sampled
  from the local σ_i (or Reynolds stress).
- Adaptive dt = config.dt_factor · min(τ_L). Integrate position from velocity each
  step via the stochastic step functions.
- Perfect reflection at the surface contact height (config.rebound_height or ≈z0);
  optional reflection at z = h when config.bl_reflection.
- Terminate a particle on touchdown, on leaving the domain, or at config.t_max.
- Return touchdown (x, y) positions in the upwind frame and the data needed for
  weighting (e.g. w at contact). Chunk over particles to bound memory.
touchdown.detect_touchdowns(...): assign each contact the bLS weight ∝ 1/|w_td|,
capped (Flesch, Wilson & Yee 1995/2004); return x_td, y_td, weight arrays.
Add focused tests: reflection keeps z ≥ surface; particles terminate; weights are
finite and positive; touchdown counts are reproducible under a fixed seed.
Acceptance: tests pass; ruff/black/mypy clean.
```

---

## Phase 4 — Well-mixed criterion test (key correctness gate)

```
Add tests/test_stochastic_wellmixed.py — the central correctness test for the
Langevin engine (Thomson 1987 well-mixed criterion).
Initialize particles uniformly distributed in a horizontally homogeneous, bounded
domain with the package's turbulence fields. Integrate forward/backward and assert
the concentration (particle density weighted appropriately) stays statistically
uniform over time within Monte-Carlo tolerance — i.e. an initially well-mixed
tracer remains well-mixed. Use a fixed seed and a documented tolerance; mark the
larger-N variant @pytest.mark.slow.
If the test fails, the drift term in stochastic.py is inconsistent with the
turbulence fields — fix stochastic.py/turbulence.py until it passes, and explain
the fix.
Acceptance: well-mixed test passes at the default N; slow high-N variant passes.
```

---

## Phase 5 — Wire the model orchestrator

```
Implement the full pipeline in src/lagranged/model.py FootprintModel.run():
validate → build profiles/turbulence → simulate trajectories → detect touchdowns
→ grid (gridding.accumulate) → contours (contours.cumulative_levels) → assemble a
FootprintResult. Populate density, x, y, contours, x_peak (along-wind peak of the
crosswind-integrated density), n_touchdowns, and mc_noise (a Monte-Carlo noise /
effective-sample-size estimate). Remove the NotImplementedError.
Make compute_footprint() run end-to-end. Honor mode="param" vs "reynolds".
Update examples/scripts/run_single.py so it now produces and saves a real figure.
Add tests/test_model_integration.py: density integrates to 1 within tolerance;
x_peak is positive and finite; same seed → identical arrays; total weighted
touchdowns conserved.
Acceptance: run_single.py produces footprint_neutral.png; integration tests pass;
all prior tests still pass.
```

---

## Phase 6 — Physics validation & convergence

```
Add tests/test_physics_behavior.py (seeded, tolerance-based):
- Convergence: x_peak and the 80% source area stabilize as n_particles grows;
  Monte-Carlo noise falls ∝ N^(−1/2).
- Stability scaling: holding all else fixed, x_peak moves closer to the tower as
  conditions become more unstable (L→0−) and farther when stable — assert the
  sign/monotonicity, not exact values.
- Touchdown mass conservation across grids of different resolution.
Mark expensive cases @pytest.mark.slow. Document each expected behavior with a
reference. If any behavior is wrong, investigate and fix the underlying module.
Acceptance: behavior tests pass; document any tolerances chosen and why.
```

---

## Phase 7 — Geospatial export

```
Implement src/lagranged/geo.py (requires the [geo] extra; guard imports with the
existing _require_geo helper).
- write_geotiff(result, grid, path): rotate the upwind-frame density into
  geographic orientation using inputs.wind_dir, georeference using
  grid.origin_xy + grid.crs, and write a GeoTIFF via rioxarray/rasterio.
- contours_to_gdf(result, grid): build 50/80/90% cumulative source-area polygons
  (shapely) and return a GeoDataFrame in the grid CRS.
Add a rotation helper and prove it round-trips (rotate by θ then −θ ≈ identity).
Add tests/test_geo.py (skip if geo extra not installed): GeoTIFF read-back
preserves the sum and CRS; contour polygons are valid and properly nested.
Acceptance: geo tests pass when [geo] installed and skip cleanly otherwise.
```

---

## Phase 8 — Batch ingestion & CLI

```
Implement run_batch() in model.py and finish the CLI in cli.py.
- run_batch(df, grid, config, progress): iterate rows via io.iter_records, build
  FootprintInputs per row, run each, and return a list/dict of FootprintResult.
  Optional tqdm progress bar; skip rows failing QC with a warning.
- cli `run --config run.yaml --out f.tif`: load a YAML/JSON config, run one
  footprint, export. cli `batch data.csv --grid 400x400@2m --out dir/`: parse the
  grid spec, run run_batch, write per-record outputs.
Add tests: run_batch over a tiny synthetic DataFrame returns N results; CLI parses
a grid spec correctly; `lagranged --version` works; `lagranged run` on a sample
config produces an output file (use tmp_path).
Acceptance: batch + CLI tests pass.
```

---

## Phase 9 — Regression / golden tests

```
Add tests/test_regression.py with stored golden footprint arrays.
Generate small (ny,nx) density arrays for 2–3 fixed seed + input combinations
(neutral, unstable, stable), save them under tests/data/, and assert np.allclose
on re-run. Add a documented procedure (and a make/nox target or script) to
regenerate goldens intentionally when the physics changes.
Add a hypothesis-based property test over plausible input ranges confirming no
NaN/inf escapes and that validation triggers correctly.
Acceptance: regression + property tests pass; regeneration procedure documented.
```

---

## Phase 10 — Example notebooks

```
Create the example notebooks described in ARCHITECTURE.md §6 under examples/:
01_synthetic_neutral, 02_stability_sweep, 03_tower_csv_ingest,
04_batch_geotiff_export, 05_measured_turbulence_reynolds. Keep particle counts
small so they execute in CI. Include a tiny synthetic EddyPro/AmeriFlux-style CSV
under tests/data/ for notebook 03. Wire `nbmake` into pytest/CI so notebooks are
executed and can't rot.
Acceptance: `pytest --nbmake examples/*.ipynb` passes; notebooks render figures.
```

---

## Phase 11 — Documentation site

```
Finish the docs. Fill in any thin sections of docs/theory.md (equations,
assumptions) and docs/limitations.md. Add mkdocs.yml (mkdocs-material) with an
auto-generated API reference via mkdocstrings that replaces the hand-written
table in docs/api.md. Add a docs build step to CI.
Acceptance: `mkdocs build --strict` succeeds with no warnings.
```

---

## Phase 12 — Cross-model sanity check (non-blocking)

```
Add a slow, non-blocking comparison (tests/test_cross_model.py, @pytest.mark.slow)
of x_peak and the 80% distance against a Kljun FFP reference for a few
neutral/unstable cases. Treat it as a documented sanity band, not an exact-match
assertion — print the comparison and assert only that results fall within a wide,
justified tolerance. Summarize findings in docs/limitations.md.
Acceptance: sanity test runs under `-m slow`; tolerances and rationale documented.
```

---

## Final pass

```
Do a release-readiness review: bump version to 0.2.0 in __init__.py and
pyproject.toml, update CHANGELOG.md, run the full suite including slow tests,
confirm ruff/black/mypy/mkdocs all clean, and verify `pip wheel .` builds an
installable wheel. List anything still approximate or unvalidated in
docs/limitations.md.
Acceptance: full suite green; wheel builds; CHANGELOG and limitations current.
```
