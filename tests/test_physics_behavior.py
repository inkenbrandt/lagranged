"""Physics-behavior tests for the end-to-end footprint model (Phase 6).

These are *behavioral* tests: seeded, tolerance-based assertions about how the
footprint responds to particle count, atmospheric stability and grid resolution,
rather than exact-value checks. They guard the qualitative physics that a flux
footprint must obey while exercising the full public pipeline
(:class:`lagranged.model.FootprintModel`) and the underlying trajectory/touchdown
engine.

Three behaviors are covered (the Phase-6 acceptance set):

1. **Monte-Carlo convergence.** As the number of released particles ``N`` grows,
   the along-wind peak ``x_peak`` and the 80 % source area stabilize, and the
   model's Monte-Carlo noise estimate falls as ``N**(-1/2)``. The ``N**(-1/2)``
   law is the standard error of any Monte-Carlo estimator (Sawford 1985; Wilson &
   Sawford 1996, §5): the model reports ``mc_noise = 1/sqrt(N_eff)`` with the Kish
   effective sample size ``N_eff = (Σw)²/Σw²``, which grows ``∝ N`` because the
   touchdown-weight distribution is independent of ``N``.
2. **Stability scaling.** Holding everything else fixed, the footprint peak moves
   *toward* the tower as conditions become more unstable (``L → 0⁻``: stronger
   buoyant mixing brings particles to the surface sooner) and *away* from it as
   conditions become stable (Kljun et al. 2002, Fig. 4; Leclerc & Foken 2014,
   §3.4). We assert the *sign / monotonicity* of this shift, not its magnitude.
3. **Touchdown mass conservation.** Re-gridding the same touchdown ensemble at
   different resolutions neither creates nor destroys weight: the captured mass is
   identical across resolutions and every normalized density integrates to 1.

Choice of metrics and tolerances
--------------------------------
``x_peak`` is the argmax of the crosswind-integrated density. A footprint's
crosswind-integrated profile ``f(x)`` is broad and nearly flat near its maximum,
so the argmax of a *single* finite-``N`` realization is a noisy statistic — it can
jump by tens of metres between seeds — even though its central tendency converges
cleanly (verified empirically: the seed-to-seed std of ``x_peak`` falls from
~10 m at N=1000 to ~4 m at N=16000 on a 4 m grid). Consequently:

* the convergence test measures the *shrinking seed-to-seed scatter* of
  ``x_peak`` rather than its absolute value, and
* the stability test compares the *median over seeds* (robust to the occasional
  far-out argmax), asserting only the ordering between well-separated stability
  classes.

The 80 % source **area** is estimated by counting grid cells inside the 80 %
cumulative-contribution contour. This histogram estimator is biased low at small
``N`` (unoccupied cells in the tail are missing) and converges *from below* as
``N`` grows; the bias shrinks faster on a coarse grid, so the area-convergence
check uses a deliberately coarse (25 m) grid where the per-quadrupling change is
already <10 % at N=16000. Every tolerance below is justified inline.

Expensive (high-``N`` / multi-seed) cases are marked ``@pytest.mark.slow``.

References
----------
Sawford BL (1985) Lagrangian statistical simulation of concentration mean and
    fluctuation fields. J Climate Appl Meteorol 24:1152-1166.
Wilson JD, Sawford BL (1996) Review of Lagrangian stochastic models for
    trajectories in the turbulent atmosphere. Boundary-Layer Meteorol 78:191-210.
Kljun N, Rotach MW, Schmid HP (2002) A three-dimensional backward Lagrangian
    footprint model for a wide range of boundary-layer stratifications.
    Boundary-Layer Meteorol 103:205-226.
Leclerc MY, Foken T (2014) Footprints in Micrometeorology and Ecology. Springer.
"""

from __future__ import annotations

import numpy as np
import pytest

import lagranged as lg
from lagranged._rng import get_rng
from lagranged.contours import cumulative_levels
from lagranged.gridding import accumulate, grid_cell_centers
from lagranged.particles import simulate_trajectories
from lagranged.touchdown import detect_touchdowns

# --- A fixed near-neutral base state; only the field under test is varied. ------
_BASE = dict(
    zm=3.0,
    z0=0.03,
    d=0.2,
    ustar=0.35,
    umean=2.4,
    wind_dir=210.0,
    h=1000.0,
    sigma_v=0.6,
)
_NEUTRAL_L = -10_000.0  # |z/L| ~ 0: effectively neutral


def _inputs(L: float) -> lg.FootprintInputs:
    return lg.FootprintInputs(L=L, **_BASE)


def _config(n_particles: int, seed: int, t_max: float = 400.0) -> lg.ModelConfig:
    """Cheap-but-representative numerics shared by all behavior tests.

    A raised ``rebound_height`` keeps the adaptive time step off the (stiff)
    surface, matching tests/test_particles.py and tests/test_model_integration.py.
    """
    return lg.ModelConfig(
        n_particles=n_particles,
        dt_factor=0.05,
        t_max=t_max,
        rebound_height=0.5,
        seed=seed,
    )


def _touchdowns(L: float, n_particles: int, seed: int, t_max: float = 400.0):
    """Simulate one backward ensemble and return weighted touchdowns.

    Returns ``(x_td, y_td, weight)`` exactly as :meth:`FootprintModel.run` derives
    them, so a single simulation can feed several gridded metrics without paying
    to re-integrate the (dominant) trajectory cost.
    """
    cfg = _config(n_particles, seed, t_max)
    traj = simulate_trajectories(_inputs(L), None, cfg, get_rng(seed))
    return detect_touchdowns(traj.x, traj.y, traj.w_contact, sigma_w0=traj.sigma_w_surface)


def _mc_noise(weight: np.ndarray) -> float:
    """Monte-Carlo noise ``1/sqrt(N_eff)`` with the Kish effective sample size.

    Mirrors :meth:`FootprintModel.run`; ``N_eff = (Σw)²/Σw²``.
    """
    total = float(weight.sum())
    sum_w2 = float(np.square(weight).sum())
    return 1.0 / np.sqrt(total**2 / sum_w2)


def _x_peak(x_td, y_td, weight, grid: lg.DomainGrid) -> float:
    """Along-wind peak of the crosswind-integrated density (mirrors the model)."""
    x, _ = grid_cell_centers(grid)
    density = accumulate(x_td, y_td, weight, grid)
    fx = density.sum(axis=0) * grid.dy
    return float(x[int(np.argmax(fx))])


def _source_area(x_td, y_td, weight, grid: lg.DomainGrid, frac: float = 0.8) -> float:
    """Area [m²] of the cells inside the ``frac`` cumulative-source contour."""
    density = accumulate(x_td, y_td, weight, grid)
    cell_area = grid.dx * grid.dy
    level = cumulative_levels(density, (frac,), cell_area=cell_area)[frac]
    return float((density >= level).sum()) * cell_area


# Fine grid for x_peak (4 m, like the package default); wide enough for the
# heavy upwind tail of the touchdown distribution.
_FINE_GRID = lg.DomainGrid(nx=300, ny=160, dx=4.0, dy=4.0, x0=-40.0, y0=-320.0)
# Coarse grid for the 80 % source area (see module docstring: the histogram
# area estimator converges from below much faster on a coarse grid).
_COARSE_GRID = lg.DomainGrid(nx=40, ny=40, dx=25.0, dy=25.0, x0=-75.0, y0=-500.0)


# ===========================================================================
# 1. Monte-Carlo convergence
# ===========================================================================


@pytest.mark.slow
def test_montecarlo_convergence():
    """x_peak scatter and the 80 % area stabilize; mc_noise falls as N**(-1/2).

    A single ensemble is simulated for each (N, seed); the three metrics are
    derived from it (one simulation feeds all three — the trajectory integration
    dominates cost). Particle counts span a 16x range so the N**(-1/2) trend is
    unambiguous; four seeds give a stable median/scatter at each N. A reduced
    ``t_max`` truncates the far upwind tail equally for every (N, seed), so it
    shifts the estimands slightly but leaves these *relative* trends intact while
    keeping the test affordable.

    Tolerances
    ----------
    * **mc_noise law** — fit ``log(median mc_noise)`` against ``log N``; the slope
      must lie in ``(-0.70, -0.30)`` (ideal -0.5). The ±0.2 band absorbs the
      scatter from the heavy-tailed ``1/|w|`` weights (which makes ``N_eff`` noisy)
      while still rejecting the no-convergence (slope 0) and over-fast (slope -1)
      hypotheses; observed ≈ -0.44. The median over seeds is used because a single
      near-grazing contact can transiently inflate ``Σw²`` and spike ``mc_noise``.
    * **x_peak scatter** — the seed-to-seed std at the largest N must be both
      smaller than at the smallest N and below ``2·dx`` (8 m, i.e. argmax pinned to
      within ~two grid cells). Observed ~10 m → ~4 m.
    * **80 % area** — the estimate grows monotonically with N (converges from
      below) and its per-quadrupling relative change both shrinks and drops below
      10 % at the largest N. Observed relative change ~0.11 → ~0.06.
    """
    n_values = (1000, 4000, 16000)
    seeds = range(4)
    t_max = 200.0

    mc_median: dict[int, float] = {}
    xpeak_std: dict[int, float] = {}
    area_mean: dict[int, float] = {}
    for n in n_values:
        mc, xp, area = [], [], []
        for s in seeds:
            x_td, y_td, w = _touchdowns(_NEUTRAL_L, n, s, t_max=t_max)
            mc.append(_mc_noise(w))
            xp.append(_x_peak(x_td, y_td, w, _FINE_GRID))
            area.append(_source_area(x_td, y_td, w, _COARSE_GRID))
        mc_median[n] = float(np.median(mc))
        xpeak_std[n] = float(np.std(xp))
        area_mean[n] = float(np.mean(area))

    lo, hi = n_values[0], n_values[-1]

    # --- mc_noise ∝ N**(-1/2) ------------------------------------------------
    assert mc_median[hi] < mc_median[lo], "mc_noise must fall as N grows"
    slope = float(np.polyfit(np.log(n_values), [np.log(mc_median[n]) for n in n_values], 1)[0])
    assert -0.70 < slope < -0.30, f"mc_noise ~ N^{slope:.2f}, expected ~ N^(-1/2)"

    # --- x_peak scatter shrinks toward the grid resolution -------------------
    assert xpeak_std[hi] < xpeak_std[lo], "x_peak scatter must shrink with N"
    assert (
        xpeak_std[hi] < 2.0 * _FINE_GRID.dx
    ), f"x_peak still scatters by {xpeak_std[hi]:.1f} m (> 2 cells) at N={hi}"

    # --- 80 % source area converges from below -------------------------------
    mid = n_values[1]
    assert (
        area_mean[lo] < area_mean[mid] < area_mean[hi]
    ), "80 % source area should increase monotonically (converge from below)"
    rel_change_early = (area_mean[mid] - area_mean[lo]) / area_mean[mid]
    rel_change_late = (area_mean[hi] - area_mean[mid]) / area_mean[hi]
    # Successive per-quadrupling changes must shrink and fall below 10 %.
    assert rel_change_late < rel_change_early, "80 % area increments are not shrinking"
    assert (
        rel_change_late < 0.10
    ), f"80 % area still changing by {rel_change_late:.1%} per 4x N at N={hi}"


# ===========================================================================
# 2. Stability scaling
# ===========================================================================


def test_peak_closer_to_tower_when_unstable():
    """Fast directional check: unstable peak sits closer than the stable peak.

    The most robust single-realization statement (it held for every seed in
    calibration): with everything else fixed, the along-wind peak under strongly
    *unstable* conditions is closer to the tower than under strongly *stable* ones.
    A single fixed seed suffices because the two regimes are separated by many grid
    cells; we still demand the gap exceed one cell so it cannot be a quantization
    artifact. See Kljun et al. (2002), Leclerc & Foken (2014).
    """
    seed, n = 1, 1500
    x_u, y_u, w_u = _touchdowns(L=-10.0, n_particles=n, seed=seed, t_max=300.0)
    x_s, y_s, w_s = _touchdowns(L=20.0, n_particles=n, seed=seed, t_max=300.0)
    peak_unstable = _x_peak(x_u, y_u, w_u, _FINE_GRID)
    peak_stable = _x_peak(x_s, y_s, w_s, _FINE_GRID)
    assert peak_stable - peak_unstable > _FINE_GRID.dx, (
        f"expected stable peak farther than unstable; got "
        f"unstable={peak_unstable:.1f} m, stable={peak_stable:.1f} m"
    )


@pytest.mark.slow
def test_x_peak_monotonic_with_stability():
    """x_peak increases monotonically from unstable → neutral → stable.

    Holding ``zm, u*, h, σ_v, z0`` fixed and varying only ``L`` across three
    well-separated stability classes, the *median* along-wind peak over seeds must
    increase as the atmosphere becomes more stable. The median (not the mean)
    guards against the occasional far-out argmax on the broad stable-profile peak.

    Tolerances
    ----------
    We assert ordering only (per the prompt: sign/monotonicity, not values) plus a
    floor on the total unstable→stable shift of ``2·dx`` (8 m) so the trend clearly
    exceeds grid quantization. N=4000 is needed here: at lower N a single
    ``1/|w|``-weighted far touchdown can spike a tail cell and send the argmax of
    the broad stable profile hundreds of metres out, which even the median cannot
    absorb. Observed medians (5 seeds, N=4000): ~6 m (L=-10), ~22 m (neutral),
    ~34 m (L=+15).
    """
    seeds = range(5)
    peaks = {}
    for label, L in (("unstable", -10.0), ("neutral", _NEUTRAL_L), ("stable", 15.0)):
        vals = [_x_peak(*_touchdowns(L, 4000, s, t_max=250.0), grid=_FINE_GRID) for s in seeds]
        peaks[label] = float(np.median(vals))

    assert (
        peaks["unstable"] < peaks["neutral"] < peaks["stable"]
    ), f"x_peak not monotonic in stability: {peaks}"
    assert (
        peaks["stable"] - peaks["unstable"] > 2.0 * _FINE_GRID.dx
    ), f"stability shift {peaks['stable'] - peaks['unstable']:.1f} m too small to trust"


# ===========================================================================
# 3. Touchdown mass conservation across grid resolutions
# ===========================================================================


def test_touchdown_mass_conserved_across_resolutions():
    """Re-gridding the same touchdowns at different resolutions conserves mass.

    One ensemble is simulated and accumulated onto three grids that share an
    *identical* extent (chosen to contain every touchdown) but differ in cell size
    by 4x. Two invariants must hold at every resolution:

    1. The captured weight equals the full touchdown weight — no mass is created or
       lost by changing resolution (exact: each contact lands in exactly one bin).
    2. The normalized density integrates to 1 (the probability-mass normalization,
       to floating-point tolerance).
    """
    x_td, y_td, weight = _touchdowns(_NEUTRAL_L, n_particles=2000, seed=0)
    total_weight = float(weight.sum())
    assert x_td.size > 0

    # Shared extent containing all touchdowns, snapped so a 16 m cell tiles it
    # exactly (hence 4 m and 8 m cells do too — every grid has identical edges).
    coarsest = 16.0
    x0 = np.floor((x_td.min() - 1.0) / coarsest) * coarsest
    y0 = np.floor((y_td.min() - 1.0) / coarsest) * coarsest
    span_x = np.ceil((x_td.max() + 1.0 - x0) / coarsest) * coarsest
    span_y = np.ceil((y_td.max() + 1.0 - y0) / coarsest) * coarsest

    for dx in (4.0, 8.0, 16.0):
        grid = lg.DomainGrid(
            nx=int(round(span_x / dx)),
            ny=int(round(span_y / dx)),
            dx=dx,
            dy=dx,
            x0=x0,
            y0=y0,
        )
        # Captured weight from the raw (un-normalized) histogram on the same edges.
        x_edges = grid.x0 + np.arange(grid.nx + 1) * grid.dx
        y_edges = grid.y0 + np.arange(grid.ny + 1) * grid.dy
        raw, _, _ = np.histogram2d(y_td, x_td, bins=[y_edges, x_edges], weights=weight)
        # rel=1e-9 covers the reordered floating-point summation, not physics.
        assert float(raw.sum()) == pytest.approx(
            total_weight, rel=1e-9
        ), f"weight lost/created at dx={dx} m"

        density = accumulate(x_td, y_td, weight, grid)
        integral = float(density.sum()) * grid.dx * grid.dy
        assert integral == pytest.approx(1.0, abs=1e-9), f"density not normalized at dx={dx} m"
