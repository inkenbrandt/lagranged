# Limitations & caveats

`lagranged` is a **research-grade prototype**, not a validated operational
footprint model. This page lists what it assumes, where those assumptions break,
and how the code signals trouble. Read it before drawing any scientific
conclusion from an output.

!!! warning "Not a validated model"
    Cross-check every result against an established model (e.g. Kljun FFP or
    Kormann–Meixner) before using it. Approximate regimes raise Python
    `warnings` at runtime — do not silence them blindly.

## Physical assumptions

- **Surface-layer similarity.** The MOST mean-wind profile
  ([`profiles.mean_wind`][lagranged.profiles.mean_wind]) and the $\sigma_i$
  parameterizations ([`turbulence.sigma_profiles`][lagranged.turbulence.sigma_profiles])
  assume horizontally homogeneous, stationary, flat-terrain surface-layer
  conditions. A warning is emitted ([`validation.validate_inputs`][lagranged.validation.validate_inputs])
  when the effective measurement height exceeds $0.8\,h$, where surface-layer
  scaling is weak.
- **Gaussian, well-mixed turbulence.** The Thomson (1987) drift used in
  [`stochastic`][lagranged.stochastic] is exactly well-mixed only for Gaussian
  velocity statistics and the idealized, horizontally homogeneous turbulence
  fields the package builds. Skewed convective turbulence and inhomogeneous or
  complex terrain violate the well-mixed condition, biasing the trajectory
  statistics.
- **Stability-function branches.** The Businger–Dyer (unstable) and linear
  (stable) forms of $\psi_m$, $\phi_m$ are standard but approximate, and the
  linear stable branch in particular degrades in very stable conditions
  ($z/L \gtrsim 1$).
- **Boundary-layer treatment.** Perfect reflection at $z = h$
  ([`particles`][lagranged.particles]) is a crude stand-in for entrainment-zone
  physics; there is no explicit entrainment, capping inversion, or above-BL
  decay.
- **First-passage estimator.** Each particle contributes at most one touchdown
  (first surface contact), which is a simplification of the full
  multiple-reflection bLS estimator. Near-source contributions are the most
  affected.
- **Touchdown-weight capping.** The $1/\lvert w \rvert$ estimator weight
  ([`touchdown.detect_touchdowns`][lagranged.touchdown.detect_touchdowns]) is
  floored for grazing contacts ($\lvert w \rvert \to 0$) to keep it finite. This
  slightly biases the near-field (large-weight) contributions; the floor scales
  with the near-surface $\sigma_w$ when available.

## Numerical caveats

- **Monte-Carlo noise.** Results carry sampling noise that falls only as
  $N^{-1/2}$. Inspect `result.mc_noise` (the inverse square-root of the Kish
  effective sample size) and confirm convergence by increasing `n_particles`.
  Tail contours (90 %) and `x_peak` in the far field need the most particles.
- **Adaptive time step.** A single global $dt = \texttt{dt\_factor}\cdot
  \min(\tau_L)$ is derived from the launch region. Reducing `dt_factor`
  tightens the integration but raises cost roughly linearly; the default
  ($0.02$) is a pragmatic compromise, not a convergence-proven value.
- **Trajectory-time truncation.** Each particle is integrated for at most
  `config.t_max` ([`particles`][lagranged.particles]); particles that have not
  touched down or left the domain by then are dropped and contribute no
  touchdown. The longest-lived trajectories are the farthest upwind, so this
  biases the far-field tail low (and is one reason the cross-model `x80` ratios
  below are conservative). Check the touched-down fraction
  `result.meta["n_touchdown"] / result.meta["n_released"]` and raise `t_max`
  until `x_peak` and the source-area distances stop drifting.
- **Domain truncation.** Particles leaving a large horizontal cap, or the grid,
  are dropped. Choose a [`DomainGrid`][lagranged.config.DomainGrid] large enough
  to contain the source area you care about; touchdowns outside the grid are not
  counted, which can clip the far field.
- **Roughness fallback.** When `z0` is not supplied a fixed fallback roughness
  is used; pass a measured `z0` for quantitative work.
- **No accelerated backend yet.** The `fast` extra
  (`pip install lagranged[fast]`, numba) is declared in packaging but is **not
  yet wired into any integration path** — installing it currently changes
  nothing. The integrator is pure NumPy, so large `n_particles` runs can be slow;
  budget runtime accordingly rather than expecting JIT speedups.

## Reproducibility

All randomness flows from a single seeded `numpy.random.Generator`
([`config.ModelConfig.seed`][lagranged.config.ModelConfig]). A fixed seed makes a
run bit-for-bit reproducible on the same platform; results across NumPy versions
or architectures may differ at the floating-point level.

## What is *not* validated

The following have **not** been checked against measurements or a reference
model and should be treated as provisional:

- Absolute footprint magnitudes and the precise shape of the source-weight
  field.
- Stability scaling of `x_peak` and the source-area distances beyond the
  qualitative sign of the trend (closer when unstable, farther when stable).
- The convective ($w_*$) contributions to $\sigma_i$ in strongly unstable
  conditions.
- GeoTIFF/polygon georeferencing accuracy under unusual CRS or near-polar
  projections.
- Eddy-covariance CSV ingestion ([`io`][lagranged.io]) assumes EddyPro /
  AmeriFlux-style column **names, units, and sign conventions** and applies a
  default column mapping: matching columns are *renamed*, not validated or
  unit-checked, so a differently exported file can silently feed wrong values
  into `run_batch`. Confirm the mapping against your processor's output.

## Cross-model sanity check (Kljun FFP)

A non-blocking comparison against the analytical Flux Footprint Prediction (FFP)
parameterization of **Kljun et al. (2015)** lives in
`tests/test_cross_model.py` (`@pytest.mark.slow`). It is a *documented sanity
band*, **not** a validation or exact-match assertion: the two models are
structurally different (a backward Lagrangian first-passage estimator with
parameterized MOST turbulence here, versus a regression fit to a Lagrangian
ensemble there), so disagreement by a factor of a few is expected and acceptable.

**What is compared.** Two scalar descriptors, for three neutral/unstable cases
(`L = -10, -50, -10000` m) sharing one met record (`zm = 3 m`, `d = 0.2 m`,
`z0 = 0.03 m`, `u* = 0.35 m s⁻¹`, `h = 1000 m`):

- `x_peak` — the along-wind peak of the crosswind-integrated footprint;
- `x80` — the upwind distance enclosing 80 % of the crosswind-integrated
  footprint.

Both are taken as the **median over three seeds**, because a single-realization
`x_peak` (the argmax of a broad, flat profile) is a noisy statistic.

**Findings** (model ÷ Kljun ratios, at the test's operating point — modest
`N = 4000` and a truncated `t_max = 300 s`):

| case | `L` [m] | `x_peak` ratio | `x80` ratio |
|------|--------:|---------------:|------------:|
| unstable      | −10     | 0.79 | 1.31 |
| mild-unstable | −50     | 2.98 | 1.97 |
| neutral       | −10⁴    | 2.03 | 3.23 |

The model's peak sits within ~0.8–3× of FFP's, and its 80 % distance ~1.3–3.2×
farther: `lagranged` places more weight in the far upwind tail than FFP. The
ordering is plausible (positive, finite, same order of magnitude) but the
agreement is only qualitative — consistent with everything in *"What is not
validated"* above.

**Tolerance & rationale.** The test asserts only that each ratio lies in the wide
band **[0.2, 5.0]** (a factor of five either way). That band is chosen to absorb,
together: the first-passage vs. full-ensemble estimator difference; parameterized
vs. fitted turbulence scaling; the `t_max` truncation that biases the model's
distances low; and Monte-Carlo / argmax-quantization noise at the modest `N`
used. Observed ratios (0.79–3.23) sit comfortably inside with ≥1.5× margin, so the
check rejects order-of-magnitude divergence or a sign error without being flaky.
Treat agreement within this band as the current bar for plausibility, nothing
stronger.

## How problems surface

`lagranged` favors loud failures over silent ones:

- **`ValueError`** for physically impossible inputs (non-positive `zm`, `ustar`,
  `h`, `sigma_v`; `zm_eff` below `z0` or above `h`; `mode="reynolds"` without a
  full Reynolds stress).
- **`warnings.warn`** for edge-of-validity regimes (upper-BL measurement height,
  missing measured turbulence falling back to parameterizations, out-of-range
  wind direction, missing georeferencing).

If a result looks wrong, check the emitted warnings first.
