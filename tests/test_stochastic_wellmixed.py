"""Well-mixed criterion test for the Langevin engine (Thomson 1987).

This is the central correctness gate for :mod:`lagranged.stochastic`. Thomson's
(1987) *well-mixed condition* is the defining admissibility criterion for a
Lagrangian stochastic model: if an ensemble of particles is initially
distributed according to the Eulerian PDF of the turbulence — here a spatially
**uniform number density** with velocities drawn from the local Gaussian
:math:`\\mathcal N(0, \\sigma_w^2(z))` — then the drift term must keep it that
way for all time. An initially well-mixed tracer must stay well-mixed.

The criterion is a *forward-time* statement about the Fokker–Planck equation of
the velocity Langevin process, so we integrate the model **forward** with
:func:`lagranged.stochastic.step_1d` in a horizontally homogeneous, vertically
bounded domain with reflecting top and bottom walls (perfect elastic reflection,
the package's :func:`lagranged.particles._reflect`). Because the domain is
closed, no particles are lost and the exact stationary distribution is the
uniform one we start from.

Why this discriminates the drift
--------------------------------
The vertical drift in ``step_1d`` is the Thomson (1987) 1-D inhomogeneous form

.. math::
    a_w = -\\frac{C_0\\varepsilon}{2\\sigma_w^2}\\,w
          + \\tfrac12\\,\\frac{\\partial\\sigma_w^2}{\\partial z}
            \\left(1 + \\frac{w^2}{\\sigma_w^2}\\right).

The second (gradient-correction) term is exactly what prevents the classic
spurious accumulation of particles in regions of *low* :math:`\\sigma_w`. We pick
strongly **unstable** conditions so that :math:`\\sigma_w(z)` varies markedly
across the domain (≈ +50 % from bottom to top); with the correction term the
density stays flat, but if the drift were inconsistent with the turbulence
fields (e.g. the gradient term were dropped or had the wrong sign) particles
would pile up by tens of percent and the test would fail loudly.

Tolerances are Monte-Carlo and documented at each assertion; the larger-N
variant is marked ``@pytest.mark.slow``.

References
----------
Thomson DJ (1987) Criteria for the selection of stochastic models of particle
    trajectories in turbulent flows. J Fluid Mech 180:529–556.
Wilson JD, Sawford BL (1996) Review of Lagrangian stochastic models for
    trajectories in the turbulent atmosphere. Boundary-Layer Meteorol 78:191–210.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from lagranged.constants import C0_DEFAULT
from lagranged.particles import _reflect
from lagranged.stochastic import step_1d
from lagranged.turbulence import (
    dissipation,
    lagrangian_timescales,
    sigma_profiles,
    sigma_squared_gradients,
)

# --- Domain & atmospheric state -------------------------------------------------
# Strongly unstable so sigma_w(z) has a pronounced vertical gradient: this is the
# regime in which the Thomson gradient-correction term actually matters, making
# the test a sharp probe of drift/turbulence consistency.
USTAR = 0.35  # friction velocity [m s-1]
H_BL = 1000.0  # boundary-layer height [m]
L_OBUKHOV = -50.0  # Obukhov length [m] (unstable)
Z_LO, Z_HI = 5.0, 45.0  # closed, reflecting domain [m]
C0 = C0_DEFAULT

# Numerics: dt as a fraction of the smallest Lagrangian timescale in the domain.
DT_FACTOR = 0.02


def _domain_min_tau_L() -> float:
    """Smallest Lagrangian timescale over the domain (sets a stable ``dt``)."""
    z = np.linspace(Z_LO, Z_HI, 256)
    _, _, sw = sigma_profiles(z, USTAR, L_OBUKHOV, H_BL)
    t_lu, t_lv, t_lw = lagrangian_timescales(z, sw, USTAR, L_OBUKHOV, C0)
    return float(min(t_lu.min(), t_lv.min(), t_lw.min()))


def _mixing_time(dt: float) -> float:
    """Crude vertical mixing time, used only to size the run length.

    Mid-domain eddy diffusivity ``K ≈ σ_w² T_Lw`` gives a diffusive crossing time
    ``(Z_HI - Z_LO)² / K`` for the layer. The run is expressed as a number of
    these so the test stays well-sized if the domain or stability is retuned.
    """
    z_mid = 0.5 * (Z_LO + Z_HI)
    _, _, sw = sigma_profiles(z_mid, USTAR, L_OBUKHOV, H_BL)
    _, _, t_lw = lagrangian_timescales(z_mid, sw, USTAR, L_OBUKHOV, C0)
    k_eddy = float(sw.item() ** 2 * t_lw.item())
    return (Z_HI - Z_LO) ** 2 / k_eddy


def _run_well_mixed(
    n_particles: int,
    n_mixing_times: float,
    seed: int,
    nbins: int = 10,
) -> dict:
    """Integrate an initially well-mixed ensemble forward in a closed domain.

    Particles start uniform in ``z`` with equilibrium velocities
    ``w ~ N(0, σ_w(z)²)`` (and likewise for ``u, v``). Each step advances the
    velocity with :func:`step_1d` using the package's turbulence fields, advects
    ``z = z + w dt``, and reflects perfectly at both walls. The number density is
    time-averaged over the final two-thirds of the run (a one-mixing-time burn-in
    is discarded) to suppress Monte-Carlo noise without hiding any systematic,
    drift-induced redistribution.

    Returns a dict with the normalized time-averaged density per height bin, the
    final-snapshot histogram, and the final ``(z, w)`` arrays.
    """
    rng = np.random.default_rng(seed)
    dt = DT_FACTOR * _domain_min_tau_L()
    n_steps = int(np.ceil(n_mixing_times * _mixing_time(dt) / dt))

    # Initial condition: the exact stationary state (uniform density, local
    # equilibrium velocity variance). A well-mixed model must preserve it.
    z = rng.uniform(Z_LO, Z_HI, n_particles)
    su0, sv0, sw0 = sigma_profiles(z, USTAR, L_OBUKHOV, H_BL)
    u = rng.standard_normal(n_particles) * su0
    v = rng.standard_normal(n_particles) * sv0
    w = rng.standard_normal(n_particles) * sw0

    edges = np.linspace(Z_LO, Z_HI, nbins + 1)
    accum = np.zeros(nbins)
    n_accum = 0
    burn_in = n_steps // 3

    for step in range(n_steps):
        su, sv, sw = sigma_profiles(z, USTAR, L_OBUKHOV, H_BL)
        _, _, t_lw = lagrangian_timescales(z, sw, USTAR, L_OBUKHOV, C0)
        eps = dissipation(sw, t_lw, C0)
        _, _, dsw2_dz = sigma_squared_gradients(z, USTAR, L_OBUKHOV, H_BL)

        u, v, w = step_1d(
            u,
            v,
            w,
            sigma_u=su,
            sigma_v=sv,
            sigma_w=sw,
            dsigma_w2_dz=dsw2_dz,
            epsilon=eps,
            dt=dt,
            rng=rng,
            C0=C0,
        )
        z = z + w * dt
        z, w = _reflect(z, w, Z_LO, Z_HI)

        if step >= burn_in:
            accum += np.histogram(z, bins=edges)[0]
            n_accum += 1

    density = accum / (accum.sum() / nbins)  # normalize so a flat profile == 1
    final_counts = np.histogram(z, bins=edges)[0]
    return {
        "dt": dt,
        "n_steps": n_steps,
        "density": density,
        "final_counts": final_counts,
        "edges": edges,
        "z": z,
        "w": w,
    }


def _assert_uniform(out: dict, n_particles: int, *, max_dev: float) -> None:
    """Assert the density is flat (time-averaged) and not rejected as uniform.

    Two complementary checks:

    1. Time-averaged number density: every height bin within ``max_dev`` of the
       flat value 1. This is the directly interpretable well-mixed statement.
    2. Final-snapshot chi-square: at a single instant the ``N`` particle heights
       are independent, so the bin counts are exactly multinomial under the
       uniform null and a chi-square goodness-of-fit test is valid. We require it
       *not* be rejected at the 99.9 % level — a wide margin that a correct model
       clears easily (statistic ≈ dof) while an inconsistent drift fails by
       orders of magnitude.
    """
    density = out["density"]
    assert np.all(np.isfinite(density))
    dev = float(np.max(np.abs(density - 1.0)))
    assert dev < max_dev, f"time-averaged density deviates by {dev:.3f} (> {max_dev}); {density}"

    counts = out["final_counts"]
    expected = n_particles / counts.size
    chi2 = float(np.sum((counts - expected) ** 2 / expected))
    dof = counts.size - 1
    critical = float(stats.chi2.ppf(0.999, dof))
    assert chi2 < critical, f"final snapshot rejected as uniform: chi2={chi2:.1f} > {critical:.1f}"


# ---------------------------------------------------------------------------
# Default-N well-mixed test
# ---------------------------------------------------------------------------


def test_well_mixed_density_stays_uniform():
    """An initially uniform tracer stays uniform under forward integration.

    Default N (10 000), run for 3 vertical mixing times. The 5 % tolerance sits
    well above the observed Monte-Carlo scatter (≈ 2 %) and far below the tens of
    percent an inconsistent drift would produce.
    """
    out = _run_well_mixed(n_particles=10_000, n_mixing_times=3.0, seed=12345, nbins=10)
    _assert_uniform(out, 10_000, max_dev=0.05)


def test_well_mixed_velocity_variance_matches_sigma():
    """The other half of well-mixedness: local velocity variance is preserved.

    After integration, the variance of ``w`` among particles in each height bin
    must match the local :math:`\\sigma_w^2(z)`. Checked to 12 % (relative),
    comfortably above per-bin sampling noise at this N.
    """
    out = _run_well_mixed(n_particles=20_000, n_mixing_times=3.0, seed=2024, nbins=8)
    z, w, edges = out["z"], out["w"], out["edges"]
    idx = np.clip(np.digitize(z, edges) - 1, 0, edges.size - 2)
    for b in range(edges.size - 1):
        in_bin = idx == b
        assert in_bin.sum() > 200  # enough samples for a stable variance
        z_center = 0.5 * (edges[b] + edges[b + 1])
        _, _, sw_c = sigma_profiles(z_center, USTAR, L_OBUKHOV, H_BL)
        ratio = float(np.var(w[in_bin]) / sw_c.item() ** 2)
        assert ratio == pytest.approx(1.0, rel=0.12), f"bin {b}: var(w)/σ_w² = {ratio:.3f}"


def test_well_mixed_is_reproducible_under_seed():
    """A fixed seed fully determines the run (no global RNG state)."""
    a = _run_well_mixed(n_particles=4_000, n_mixing_times=1.5, seed=7, nbins=10)
    b = _run_well_mixed(n_particles=4_000, n_mixing_times=1.5, seed=7, nbins=10)
    np.testing.assert_array_equal(a["final_counts"], b["final_counts"])
    np.testing.assert_array_equal(a["density"], b["density"])


# ---------------------------------------------------------------------------
# Slow, high-N variant: tighter tolerance
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_well_mixed_density_stays_uniform_high_n():
    """High-N variant (50 000 particles, 4 mixing times, 12 bins).

    The larger ensemble shrinks the Monte-Carlo floor, so we tighten the
    time-averaged tolerance to 2 %. This is the stringent acceptance test for the
    well-mixed criterion.
    """
    out = _run_well_mixed(n_particles=50_000, n_mixing_times=4.0, seed=98765, nbins=12)
    _assert_uniform(out, 50_000, max_dev=0.02)
