"""Particle release and backward trajectory orchestration.

Releases ``config.n_particles`` particles backward from the receptor at
``inputs.zm_eff`` and integrates their motion with the Langevin velocity steps
in :mod:`lagranged.stochastic`.  Each particle drifts *upwind* with the MOST mean
wind while diffusing turbulently; the first time it reaches the surface contact
height (``config.rebound_height`` ≈ ``z0``) the contact is recorded as a
*touchdown* and the particle terminates.  Particles are reflected at the
boundary-layer top ``z = h`` when ``config.bl_reflection`` is set, and terminate
on touchdown, on leaving the domain, or at ``config.t_max``.

Frame
-----
Work is done in the *upwind model frame*: the receptor sits at the origin, ``x``
increases **upwind** (the direction the footprint extends), ``y`` is the lateral
offset and ``z`` is height above the displacement plane.  The rotation into
geographic coordinates by ``wind_dir`` happens downstream (gridding/geo), so the
wind direction is not used here.

The vertical coordinate is the Monin–Obukhov similarity height (height above the
displacement plane ``d``); it is passed directly to the turbulence profiles and,
with ``d`` added back, to :func:`lagranged.profiles.mean_wind`.

This is a first-passage bLS estimator: each particle contributes at most one
touchdown, weighted ∝ ``1/|w|`` downstream in :mod:`lagranged.touchdown`
(Flesch, Wilson & Yee 1995/2004).  It is a research-grade approximation, not the
full multiple-reflection bLS estimator.

References
----------
Flesch TK, Wilson JD, Yee E (1995) Backward-time Lagrangian stochastic dispersion
    models and their application to estimate gaseous emissions. J Appl Meteorol
    34:1320–1332.
Flesch TK, Wilson JD, Harper LA, Crenna BP, Sharpe RR (2004) Deducing ground-to-air
    emissions from observed trace gas concentrations: a field trial. J Appl
    Meteorol 43:487–502.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import ModelConfig
from .inputs import FootprintInputs, TowerTurbulence
from .profiles import mean_wind
from .stochastic import step_1d, step_3d
from .turbulence import (
    dissipation,
    lagrangian_timescales,
    sigma_profiles,
    sigma_squared_gradients,
)

__all__ = ["TrajectoryResult", "simulate_trajectories"]

# Roughness used only when ``inputs.z0`` is None (documented fallback) [m].
_Z0_FALLBACK: float = 0.01
# Particles are integrated in blocks of this many to bound peak memory.
_CHUNK_SIZE: int = 20_000
# Sanity cap on horizontal travel [m]; particles beyond it are treated as having
# left the domain (they never contribute a touchdown).
_X_DOMAIN_CAP: float = 1.0e6
# Number of levels used to find min(τ_L) for the adaptive time step.
_DT_PROFILE_LEVELS: int = 64


@dataclass(frozen=True)
class TrajectoryResult:
    """Touchdown positions and contact data from a backward ensemble.

    The ``(x, y)`` positions and ``w_contact`` arrays all have length
    ``n_touchdown``; ``(x, y)`` are in the upwind model frame [m] and
    ``w_contact`` is the vertical velocity at the surface contact [m s-1] (used
    for the ``1/|w|`` bLS weight in :mod:`lagranged.touchdown`).
    """

    x: np.ndarray
    y: np.ndarray
    w_contact: np.ndarray
    n_released: int
    n_touchdown: int
    dt: float
    sigma_w_surface: float


def _reflect(
    z: np.ndarray, w: np.ndarray, lower: float, upper: float
) -> tuple[np.ndarray, np.ndarray]:
    """Fold ``z`` into ``[lower, upper]`` by perfect elastic reflection.

    Each reflection flips the sign of the corresponding vertical velocity ``w``.
    Overshoots of any size are handled by triangle-wave folding, so the returned
    heights always satisfy ``lower <= z <= upper`` (to within floating point).

    Parameters
    ----------
    z, w:
        Heights and vertical velocities, same shape.
    lower, upper:
        Reflecting boundaries, ``lower < upper``.

    Returns
    -------
    z_new, w_new : np.ndarray
        Reflected heights (inside ``[lower, upper]``) and velocities.
    """
    span = upper - lower
    if span <= 0.0:
        raise ValueError(f"upper ({upper}) must exceed lower ({lower}).")
    # Triangle wave: fold (z - lower) with period 2*span.
    q = np.mod(z - lower, 2.0 * span)
    descending = q > span  # odd number of reflections -> flip velocity
    folded = np.where(descending, 2.0 * span - q, q)
    z_new = lower + folded
    w_new = np.where(descending, -w, w)
    return z_new, w_new


def _adaptive_dt(
    z_surf: float,
    z_hi: float,
    inp: FootprintInputs,
    config: ModelConfig,
    sigma_w_const: float | None,
) -> float:
    """Global time step ``dt = dt_factor · min(τ_L)`` over the occupied layer.

    The smallest Lagrangian timescale occurs near the surface (where ε is largest
    and σ_w smallest), so a single global ``dt`` derived from it keeps the
    integration stable everywhere particles actually travel.  The profile is
    evaluated over ``[z_surf, z_hi]`` (the launch region); τ_L grows with height
    above the surface, while σ_w → 0 right at the BL top would otherwise drive a
    spurious τ_L → 0 the particles never reach.
    """
    z = np.geomspace(z_surf, z_hi, _DT_PROFILE_LEVELS)
    if config.mode == "reynolds":
        assert sigma_w_const is not None
        sw = np.full_like(z, sigma_w_const)
    else:
        _, _, sw = sigma_profiles(z, inp.ustar, inp.L, inp.h)
    t_lu, t_lv, t_lw = lagrangian_timescales(z, sw, inp.ustar, inp.L, config.C0)
    tau_min = float(min(t_lu.min(), t_lv.min(), t_lw.min()))
    dt = config.dt_factor * tau_min
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError(f"computed dt={dt!r} is not positive; check inputs.")
    return dt


def simulate_trajectories(
    inputs: FootprintInputs,
    turbulence: TowerTurbulence | None,
    config: ModelConfig,
    rng: np.random.Generator,
) -> TrajectoryResult:
    """Run the backward ensemble and collect surface touchdowns.

    Parameters
    ----------
    inputs:
        Physics inputs for the averaging period.
    turbulence:
        Optional measured turbulence.  Required (with σ_u, σ_v, σ_w) when
        ``config.mode == "reynolds"``; ignored by the parameterized ``"param"``
        mode, which uses the MOST profiles in :mod:`lagranged.turbulence`.
    config:
        Numerical / Monte-Carlo settings.
    rng:
        Seeded :class:`numpy.random.Generator`; a fixed seed fully determines the
        run (release velocities and every Langevin kick are drawn from it).

    Returns
    -------
    TrajectoryResult
        Touchdown ``(x, y)`` in the upwind frame plus the contact ``w`` needed
        for weighting.
    """
    z0_eff = inputs.z0 if inputs.z0 is not None else _Z0_FALLBACK
    z_surf = config.rebound_height if config.rebound_height is not None else z0_eff
    z_release = inputs.zm_eff
    z_top = inputs.h  # similarity height of the BL top (displacement d ≪ h, neglected)
    if z_surf <= 0.0:
        raise ValueError(f"surface contact height must be positive, got {z_surf}.")
    if not z_surf < z_release < z_top:
        raise ValueError(f"need surface < release < top, got {z_surf} < {z_release} < {z_top}.")

    ustar, L, h, d = inputs.ustar, inputs.L, inputs.h, inputs.d
    c0 = config.C0
    mode = config.mode

    # --- Mode-specific turbulence setup ---
    tau: np.ndarray | None = None
    tau_inv: np.ndarray | None = None
    sigma_w_const: float | None = None
    su0 = sv0 = sw0 = 0.0
    if mode == "reynolds":
        stress = turbulence.reynolds_stress() if turbulence is not None else None
        if stress is None:
            raise ValueError("mode='reynolds' requires a TowerTurbulence with σ_u, σ_v, σ_w.")
        tau = stress.matrix
        tau_inv = stress.inverse
        sigma_w_const = float(np.sqrt(tau[2, 2]))
        sigma_w_surface = sigma_w_const
    else:
        su0, sv0, sw0 = (float(s.item()) for s in sigma_profiles(z_release, ustar, L, h))
        sigma_w_surface = float(sigma_profiles(z_surf, ustar, L, h)[2].item())

    dt = _adaptive_dt(z_surf, z_release, inputs, config, sigma_w_const)
    max_steps = int(np.ceil(config.t_max / dt))
    wind_floor = z0_eff * (1.0 + 1e-6)

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    w_parts: list[np.ndarray] = []

    n = config.n_particles
    for start in range(0, n, _CHUNK_SIZE):
        m = min(_CHUNK_SIZE, n - start)

        x = np.zeros(m)
        y = np.zeros(m)
        z = np.full(m, z_release)
        if mode == "reynolds":
            assert tau is not None
            vel = rng.multivariate_normal(np.zeros(3), tau, size=m)
            u, v, w = vel[:, 0].copy(), vel[:, 1].copy(), vel[:, 2].copy()
        else:
            u = rng.standard_normal(m) * su0
            v = rng.standard_normal(m) * sv0
            w = rng.standard_normal(m) * sw0

        step = 0
        while m > 0 and step < max_steps:
            # --- Advance turbulent velocity one Langevin step ---
            if mode == "reynolds":
                assert tau_inv is not None and sigma_w_const is not None
                sw = np.full(m, sigma_w_const)
                _, _, t_lw = lagrangian_timescales(z, sw, ustar, L, c0)
                eps = dissipation(sw, t_lw, c0)
                vel = step_3d(
                    np.column_stack([u, v, w]),
                    tau_inv=tau_inv,
                    epsilon=eps,
                    dt=dt,
                    rng=rng,
                    C0=c0,
                )
                u, v, w = vel[:, 0], vel[:, 1], vel[:, 2]
            else:
                su, sv, sw = sigma_profiles(z, ustar, L, h)
                _, _, t_lw = lagrangian_timescales(z, sw, ustar, L, c0)
                eps = dissipation(sw, t_lw, c0)
                _, _, dsw2 = sigma_squared_gradients(z, ustar, L, h)
                u, v, w = step_1d(
                    u,
                    v,
                    w,
                    sigma_u=su,
                    sigma_v=sv,
                    sigma_w=sw,
                    dsigma_w2_dz=dsw2,
                    epsilon=eps,
                    dt=dt,
                    rng=rng,
                    C0=c0,
                )

            # --- Advance position; mean wind advects upwind (+x) ---
            big_u = mean_wind(np.maximum(z, wind_floor) + d, ustar, z0_eff, L, d)
            x_new = x + (big_u + u) * dt
            y_new = y + v * dt
            z_new = z + w * dt

            # --- Surface contact = touchdown (first passage), then terminate ---
            td = z_new < z_surf
            if np.any(td):
                drop = np.maximum(z[td] - z_new[td], 1e-30)
                frac = (z[td] - z_surf) / drop
                x_parts.append(x[td] + frac * (x_new[td] - x[td]))
                y_parts.append(y[td] + frac * (y_new[td] - y[td]))
                w_parts.append(w[td])

            # --- Leaving the domain (no touchdown) ---
            left = (np.abs(x_new) > _X_DOMAIN_CAP) | (np.abs(y_new) > _X_DOMAIN_CAP)
            if not config.bl_reflection:
                left = left | (z_new > z_top)

            keep = ~td & ~left
            x, y, z = x_new[keep], y_new[keep], z_new[keep]
            u, v, w = u[keep], v[keep], w[keep]

            # --- Reflect survivors at the BL top (kept inside [z_surf, z_top]) ---
            if config.bl_reflection and z.size and np.any(z > z_top):
                z, w = _reflect(z, w, z_surf, z_top)

            m = int(z.size)
            step += 1

    if x_parts:
        x_td = np.concatenate(x_parts)
        y_td = np.concatenate(y_parts)
        w_td = np.concatenate(w_parts)
    else:
        x_td = np.empty(0, dtype=float)
        y_td = np.empty(0, dtype=float)
        w_td = np.empty(0, dtype=float)

    return TrajectoryResult(
        x=x_td,
        y=y_td,
        w_contact=w_td,
        n_released=n,
        n_touchdown=int(x_td.size),
        dt=dt,
        sigma_w_surface=sigma_w_surface,
    )
