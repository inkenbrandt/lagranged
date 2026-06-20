"""Cross-model sanity comparison against the Kljun (2015) FFP (Phase 12).

This is a **non-blocking sanity band**, not a validation test. It compares two
scalar footprint descriptors — the along-wind peak ``x_peak`` and the 80 %
along-wind distance ``x80`` (the upwind distance enclosing 80 % of the
crosswind-integrated footprint) — between this package and the analytical
Flux Footprint Prediction (FFP) parameterization of Kljun et al. (2015) for a
few neutral/unstable cases.

The two models are *structurally different* (a backward Lagrangian first-passage
estimator with parameterized MOST turbulence here, versus a regression fit to a
Lagrangian ensemble there), so they are **not expected to agree to better than a
factor of a few**. We therefore print the full comparison and assert only that
each ratio falls inside a wide, justified band — enough to catch order-of-
magnitude divergence or a sign error, not to certify agreement. See
``docs/limitations.md`` for the recorded findings.

The reference
-------------
We embed the published FFP crosswind-integrated parameterization (no external
dependency). For a measurement height ``zm`` above the displacement plane, with
roughness ``z0``, boundary-layer height ``h`` and Obukhov length ``L``, FFP scales
the upwind distance ``x`` to a non-dimensional ``X*`` (Kljun et al. 2015, Eqs.
7 & 25; their reference implementation ``calc_footprint_FFP.py``)::

    X* = (x / zm) * (1 - zm/h) / (ln(zm/z0) - psi_M)

where the denominator ``ln(zm/z0) - psi_M`` is the MOST term ``k * U(zm) / u*``.
The crosswind-integrated, non-dimensional footprint is::

    F*_y = a (X* - d)^b * exp(-c / (X* - d)),   X* > d

with the published constants ``a = 1.4524``, ``b = -1.9914``, ``c = 1.4622``,
``d = 0.1359``. Its peak is at ``X*_peak = d - c/b ≈ 0.8702``; the dimensional
distances follow by inverting the scaling, ``x = X* * zm * (ln(zm/z0) - psi_M)
/ (1 - zm/h)``. For ``psi_M`` we use FFP's own stability correction (Kljun et al.
2015; Paulson 1970): the Businger–Dyer integral with ``gamma_M = 19`` for the
unstable/neutral branch and ``-5.3 zm/L`` for the stable branch. (The package's
own profiles use ``gamma_M = 16`` / ``-5 zm/L``; the difference is a few percent
in ``psi_M``, far inside the sanity band, and keeping FFP's published constants
makes this an *independent* reference rather than a circular one.)

Tolerances & rationale
----------------------
Metric robustness first: a single-realization ``x_peak`` is the argmax of a broad,
nearly flat crosswind-integrated profile and is a notoriously noisy statistic
(it can jump tens of metres between seeds — see tests/test_physics_behavior.py).
We therefore compare the **median over a few seeds** of both descriptors. ``x80``
is a cumulative quantile and is already far steadier than the argmax.

Band: each ratio ``model / Kljun`` must lie in ``[0.2, 5.0]`` — a factor of five
either way. This is deliberately wide. It must absorb, simultaneously:

* the first-passage (one touchdown per particle) vs. full-ensemble estimator
  difference, which shifts mass between near and far field;
* parameterized MOST turbulence here vs. FFP's fitted scaling;
* the reduced ``t_max`` used to keep this test affordable, which truncates the far
  upwind tail and biases the model's distances low relative to a converged run;
* Monte-Carlo noise at the modest ``N`` used here, and argmax/grid quantization.

Observed ratios at the configured operating point are ~0.8–3.0 (``x_peak``) and
~1.3–3.2 (``x80``) — comfortably inside the band with ≥1.5x margin to its edges,
so the assertion rejects gross divergence without being flaky. The full table is
printed (visible with ``pytest -s`` or on failure) and summarized in
``docs/limitations.md``.

Run with::

    pytest tests/test_cross_model.py -m slow -s

References
----------
Kljun N, Calanca P, Rotach MW, Schmid HP (2015) A simple two-dimensional
    parameterisation for Flux Footprint Prediction (FFP). Geosci Model Dev
    8:3695-3713. doi:10.5194/gmd-8-3695-2015.
Paulson CA (1970) The mathematical representation of wind speed and temperature
    profiles in the unstable atmospheric surface layer. J Appl Meteorol 9:857-861.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import lagranged as lg
from lagranged._rng import get_rng
from lagranged.gridding import accumulate, grid_cell_centers
from lagranged.particles import simulate_trajectories
from lagranged.touchdown import detect_touchdowns

# --- Kljun et al. (2015) FFP crosswind-integrated parameterization -----------
_FFP_A = 1.4524
_FFP_B = -1.9914
_FFP_C = 1.4622
_FFP_D = 0.1359
# Note: the MOST term (ln(zm/z0) - psi_M) below equals k*U(zm)/u* with the von
# Kármán constant k = 0.4 folded in, so k does not appear explicitly here.


def _ffp_psi_m(zm: float, L: float) -> float:
    """FFP stability correction psi_M(zm/L) (Kljun et al. 2015; Paulson 1970).

    Unstable/neutral branch uses the Businger–Dyer integral with gamma_M = 19;
    the stable branch is the linear ``-5.3 zm/L`` form, matching FFP's reference
    code. ``L`` very large (|L| >= 1e5) is treated as neutral.
    """
    if L <= 0 or L >= 1.0e5:  # unstable or (effectively) neutral
        x = (1.0 - 19.0 * zm / L) ** 0.25 if L < 0 else 1.0
        return (
            np.log((1.0 + x**2) / 2.0)
            + 2.0 * np.log((1.0 + x) / 2.0)
            - 2.0 * np.arctan(x)
            + np.pi / 2.0
        )
    return -5.3 * zm / L  # stable


def _ffp_scale(zm: float, z0: float, h: float, L: float) -> float:
    """Dimensional length that maps scaled X* to upwind distance x.

    ``x = X* * scale`` with ``scale = zm * (ln(zm/z0) - psi_M) / (1 - zm/h)``.
    """
    return zm * (np.log(zm / z0) - _ffp_psi_m(zm, L)) / (1.0 - zm / h)


def _ffp_xpeak(zm: float, z0: float, h: float, L: float) -> float:
    """FFP along-wind peak distance [m]: X*_peak = d - c/b, scaled to x."""
    xstar_peak = _FFP_D - _FFP_C / _FFP_B
    return xstar_peak * _ffp_scale(zm, z0, h, L)


def _ffp_xr(zm: float, z0: float, h: float, L: float, frac: float = 0.8) -> float:
    """FFP along-wind distance enclosing ``frac`` of the footprint [m].

    Integrates the scaled crosswind-integrated footprint ``F*_y(X*)`` and inverts
    its CDF at ``frac``, then maps to a dimensional distance.
    """
    xstar = np.linspace(_FFP_D + 1.0e-6, 200.0, 400_000)
    fstar = _FFP_A * (xstar - _FFP_D) ** _FFP_B * np.exp(-_FFP_C / (xstar - _FFP_D))
    cdf = np.cumsum(fstar)
    cdf /= cdf[-1]
    xstar_r = float(xstar[int(np.searchsorted(cdf, frac))])
    return xstar_r * _ffp_scale(zm, z0, h, L)


# --- Shared operating point (a single met record, varying only L) ------------
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
_ZM_EFF = _BASE["zm"] - _BASE["d"]  # FFP "zm" is height above the displacement plane

# Cases the FFP regression covers well: neutral and (mildly→strongly) unstable.
_CASES: tuple[tuple[str, float], ...] = (
    ("unstable", -10.0),
    ("mild-unstable", -50.0),
    ("neutral", -10_000.0),
)
_SEEDS = range(3)  # median over a few seeds tames the noisy argmax; ~20 s/case
_N_PARTICLES = 4000
_T_MAX = 300.0  # truncates the far tail to stay affordable (documented bias)

# Grid wide/long enough to contain the 80 % distance for every case above.
_GRID = lg.DomainGrid(nx=300, ny=200, dx=3.0, dy=3.0, x0=-60.0, y0=-300.0)

# Sanity band on the model/Kljun ratio (see module docstring for the rationale).
_BAND_LO, _BAND_HI = 0.2, 5.0


def _model_metrics(L: float, seed: int) -> tuple[float, float]:
    """Return ``(x_peak, x80)`` [m] from one seeded backward ensemble.

    Mirrors :meth:`lagranged.model.FootprintModel.run`: simulate → touchdown →
    grid → crosswind-integrate. ``x80`` is the upwind distance (from the receptor)
    enclosing 80 % of the crosswind-integrated density.
    """
    cfg = lg.ModelConfig(
        n_particles=_N_PARTICLES,
        dt_factor=0.05,
        t_max=_T_MAX,
        rebound_height=0.5,  # keep dt off the stiff surface (cf. physics tests)
        seed=seed,
    )
    inputs = lg.FootprintInputs(L=L, **_BASE)
    traj = simulate_trajectories(inputs, None, cfg, get_rng(seed))
    x_td, y_td, w = detect_touchdowns(traj.x, traj.y, traj.w_contact, sigma_w0=traj.sigma_w_surface)

    x, _ = grid_cell_centers(_GRID)
    density = accumulate(x_td, y_td, w, _GRID)
    fx = density.sum(axis=0) * _GRID.dy

    x_peak = float(x[int(np.argmax(fx))])

    order = np.argsort(x)
    xs, fxs = x[order], fx[order]
    upwind = xs > 0.0
    xs, fxs = xs[upwind], fxs[upwind]
    cdf = np.cumsum(fxs)
    cdf /= cdf[-1]
    x80 = float(xs[int(np.searchsorted(cdf, 0.8))])
    return x_peak, x80


def test_ffp_reference_is_self_consistent():
    """Fast guard on the embedded FFP reference (no model run).

    Catches a typo in the parameterization without paying for a simulation:
    the scaled peak is ``d - c/b``, distances are positive and finite, and the
    80 % distance lies upwind of the peak.
    """
    assert _FFP_D - _FFP_C / _FFP_B == pytest.approx(0.8702, abs=1e-3)
    xpeak = _ffp_xpeak(_ZM_EFF, _BASE["z0"], _BASE["h"], -10_000.0)
    x80 = _ffp_xr(_ZM_EFF, _BASE["z0"], _BASE["h"], -10_000.0, 0.8)
    assert np.isfinite(xpeak) and xpeak > 0.0
    assert np.isfinite(x80) and x80 > xpeak  # 80 % distance is beyond the peak


@pytest.mark.slow
def test_cross_model_sanity_band():
    """x_peak and x80 agree with Kljun FFP within a wide, documented band.

    Non-blocking sanity check: prints the model-vs-reference comparison for each
    case and asserts only that every ratio lies in ``[0.2, 5.0]`` (factor of 5).
    See the module docstring and ``docs/limitations.md`` for the rationale.
    """
    header = (
        f"{'case':14s} {'L':>9s} "
        f"{'K_xpeak':>9s} {'M_xpeak':>9s} {'r_peak':>7s}   "
        f"{'K_x80':>9s} {'M_x80':>9s} {'r_x80':>7s}"
    )
    print("\nCross-model sanity comparison (lagranged vs Kljun 2015 FFP)")
    print(header)
    print("-" * len(header))

    ratios: list[tuple[str, str, float]] = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parameterized-turbulence notice
        for label, L in _CASES:
            xpeaks, x80s = [], []
            for s in _SEEDS:
                xp, x8 = _model_metrics(L, s)
                xpeaks.append(xp)
                x80s.append(x8)
            m_xpeak = float(np.median(xpeaks))
            m_x80 = float(np.median(x80s))
            k_xpeak = _ffp_xpeak(_ZM_EFF, _BASE["z0"], _BASE["h"], L)
            k_x80 = _ffp_xr(_ZM_EFF, _BASE["z0"], _BASE["h"], L, 0.8)
            r_peak = m_xpeak / k_xpeak
            r_x80 = m_x80 / k_x80
            print(
                f"{label:14s} {L:>9.1f} "
                f"{k_xpeak:9.2f} {m_xpeak:9.2f} {r_peak:7.2f}   "
                f"{k_x80:9.2f} {m_x80:9.2f} {r_x80:7.2f}"
            )
            ratios.append((label, "x_peak", r_peak))
            ratios.append((label, "x80", r_x80))

    out_of_band = [
        (label, metric, r) for label, metric, r in ratios if not (_BAND_LO <= r <= _BAND_HI)
    ]
    assert not out_of_band, (
        "model/Kljun ratio outside the sanity band " f"[{_BAND_LO}, {_BAND_HI}]: {out_of_band}"
    )
