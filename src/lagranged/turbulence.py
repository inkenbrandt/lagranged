"""Turbulence parameterizations: σ_u, σ_v, σ_w, Lagrangian timescales, ε.

Measured values from :class:`lagranged.inputs.TowerTurbulence` take precedence;
when absent, the MOST-based parameterizations here are used and a
:func:`warnings.warn` is emitted by :func:`get_turbulence`.

References
----------
Kljun N, Calanca P, Rotach MW, Schmid HP (2004) A simple parameterisation for
    flux footprint predictions. Boundary-Layer Meteorol 112:503–523.
    https://doi.org/10.1023/B:BOUN.0000030653.71031.96
Lenschow DH, Wyngaard JC, Pennell WT (1980) Mean-field and second-moment budgets
    in a baroclinic, convective boundary layer. J Atmos Sci 37:1313–1326.
Panofsky HA, Tennekes H, Lenschow DH, Wyngaard JC (1977) The characteristics of
    turbulent velocity components in the surface layer under convective conditions.
    Boundary-Layer Meteorol 11:355–361.
Rannik Ü, Markkanen T, Raittila J, Hari P, Vesala T (2003) Turbulence statistics
    above and within two Scots pine forests during the SMEAR campaigns.
    Agric For Meteorol 114:231–252.
Rodean HC (1996) Stochastic Lagrangian Models of Turbulent Diffusion.
    Meteorological Monographs 26. AMS, Boston.
Stull RB (1988) An Introduction to Boundary Layer Meteorology.
    Kluwer Academic, Dordrecht.
Thomson DJ (1987) Criteria for the selection of stochastic models of particle
    trajectories in turbulent flows. J Fluid Mech 180:529–556.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from .constants import (
    BUSINGER_BETA_M,
    BUSINGER_GAMMA_M,
    C0_DEFAULT,
    SIGMA_U_OVER_USTAR_NEUTRAL,
    SIGMA_V_OVER_USTAR_NEUTRAL,
    SIGMA_W_OVER_USTAR_NEUTRAL,
    VON_KARMAN,
)

if TYPE_CHECKING:
    from .inputs import TowerTurbulence


def sigma_neutral(ustar: float) -> tuple[float, float, float]:
    """Neutral-limit (σ_u, σ_v, σ_w) from surface-layer similarity ratios."""
    return (
        SIGMA_U_OVER_USTAR_NEUTRAL * ustar,
        SIGMA_V_OVER_USTAR_NEUTRAL * ustar,
        SIGMA_W_OVER_USTAR_NEUTRAL * ustar,
    )


def _phi_m_local(zeta: np.ndarray) -> np.ndarray:
    """Local (un-integrated) momentum stability function φ_m(ζ).

    Businger-Dyer: (1 − γ_m ζ)^(−1/4) unstable; 1 + β_m ζ stable.
    """
    # np.where evaluates both branches; clip the unstable-branch base so the
    # (discarded) stable-ζ values can't form a negative-base power (NaN warning).
    # For ζ ≤ 0 the base is ≥ 1, so the clip never changes a selected value.
    unstable_base = np.maximum(1.0 - BUSINGER_GAMMA_M * zeta, 1e-12)
    return np.where(
        zeta <= 0.0,
        unstable_base ** (-0.25),
        1.0 + BUSINGER_BETA_M * zeta,
    )


def _epsilon_sl(z: np.ndarray, ustar: float, L: float) -> np.ndarray:
    """Surface-layer TKE dissipation rate ε(z) [m² s-³].

    From the surface-layer TKE budget (Wyngaard & Coté 1971; Businger et al. 1971):
        ε = (u*³ / κz) × φ_ε(ζ),   φ_ε(ζ) = φ_m(ζ) − ζ

    φ_ε is clamped to 1e-6 to ensure ε > 0 for all stabilities.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zeta = np.where(L == 0.0, 0.0, z / L)
    phi_eps = np.maximum(_phi_m_local(zeta) - zeta, 1e-6)
    return (ustar**3 / (VON_KARMAN * np.maximum(z, 1e-6))) * phi_eps


def sigma_profiles(
    z: float | np.ndarray,
    ustar: float,
    L: float,
    h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Height-resolved velocity standard deviations (σ_u, σ_v, σ_w)(z) [m s-1].

    Blends mechanical (∝ u*) and convective (∝ w*) contributions following
    Kljun et al. (2004) and Rannik et al. (2003):

    **Mechanical** (surface-layer MOST; Panofsky et al. 1977):
        σ_w,mech(z) = 1.25 u* (1 − 3ζ)^(1/3) √(1 − z/h)   [unstable, ζ = z/L ≤ 0]
        σ_w,mech(z) = 1.25 u* √(1 − z/h)                   [neutral / stable]
    For σ_u, σ_v the MOST coefficient is (1 − 5ζ)^(1/3).

    **Convective** (Lenschow et al. 1980):
        σ_w,conv(z) = 0.96 w* (z/h)^(1/3) (1 − z/h)
        where w* = u* (h / |κL|)^(1/3) for L < 0; zero otherwise.

    **Combined**: σ_i = √(σ_i,mech² + σ_i,conv²)

    In the neutral limit (L → ±∞, w* → 0), ratios σ_i/u* recover the constants
    in :mod:`lagranged.constants` to within < 1 % for z/h ≤ 0.1.

    Parameters
    ----------
    z:
        Height(s) above ground [m].
    ustar:
        Friction velocity [m s-1].
    L:
        Obukhov length [m]. ``L = 0`` is treated as neutral.
    h:
        Boundary-layer mixing height [m].

    Returns
    -------
    sigma_u, sigma_v, sigma_w : np.ndarray
        Velocity standard deviations at each level [m s-1].
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zeta = np.where(L == 0.0, 0.0, z / L)

    # Convective velocity scale w* (zero for neutral/stable)
    wstar: float = ustar * (h / (abs(L) * VON_KARMAN)) ** (1.0 / 3.0) if L < 0.0 else 0.0

    zr = np.clip(z / h, 0.0, 1.0)
    decay = np.maximum(1.0 - zr, 0.0)

    # σ_w: Panofsky et al. (1977) MOST correction × height decay + Lenschow convective
    # (np.where evaluates both branches; clip the unstable base so discarded
    # stable-ζ values stay finite. The base is ≥ 1 wherever ζ ≤ 0 is selected.)
    phi_w = np.where(zeta <= 0.0, np.maximum(1.0 - 3.0 * zeta, 1e-12) ** (1.0 / 3.0), 1.0)
    sw_mech = SIGMA_W_OVER_USTAR_NEUTRAL * ustar * phi_w * np.sqrt(decay)
    # (z/h)^(1/3) regularised at z=0 with a tiny offset to avoid 0^(1/3) branch issues
    sw_conv = 0.96 * wstar * (zr + 1e-10) ** (1.0 / 3.0) * decay
    sigma_w = np.sqrt(sw_mech**2 + sw_conv**2)

    # σ_u, σ_v: horizontal MOST coefficient + bulk convective contribution
    # (Stull 1988; Moeng & Wyngaard 1989: σ_h ≈ 0.6 w* in the mid-BL)
    phi_h = np.where(zeta <= 0.0, np.maximum(1.0 - 5.0 * zeta, 1e-12) ** (1.0 / 3.0), 1.0)
    su_mech = SIGMA_U_OVER_USTAR_NEUTRAL * ustar * phi_h * np.sqrt(decay)
    su_conv = 0.60 * wstar * decay
    sigma_u = np.sqrt(su_mech**2 + su_conv**2)

    sv_mech = SIGMA_V_OVER_USTAR_NEUTRAL * ustar * phi_h * np.sqrt(decay)
    sv_conv = 0.50 * wstar * decay
    sigma_v = np.sqrt(sv_mech**2 + sv_conv**2)

    return sigma_u, sigma_v, sigma_w


def lagrangian_timescales(
    z: float | np.ndarray,
    sigma_w: float | np.ndarray,
    ustar: float,
    L: float,
    C0: float = C0_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lagrangian integral timescales T_Lu, T_Lv, T_Lw(z) [s].

    Uses the Lagrangian fluctuation–dissipation relation (Thomson 1987; Rodean 1996):
        T_Lw = 2 σ_w² / (C0 ε)
    where ε is the surface-layer TKE dissipation from :func:`_epsilon_sl`.

    Horizontal timescales are scaled by the neutral-limit variance ratios:
        T_Li = (σ_i / σ_w)²_neutral × T_Lw

    Parameters
    ----------
    z:
        Height(s) [m].
    sigma_w:
        Vertical velocity std dev [m s-1] (same shape as *z*).
    ustar:
        Friction velocity [m s-1].
    L:
        Obukhov length [m].
    C0:
        Kolmogorov/Langevin constant (default :data:`~lagranged.constants.C0_DEFAULT`).

    Returns
    -------
    T_Lu, T_Lv, T_Lw : np.ndarray
        Lagrangian timescales [s].
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    sigma_w = np.atleast_1d(np.asarray(sigma_w, dtype=float))

    eps = _epsilon_sl(z, ustar, L)
    T_Lw = 2.0 * sigma_w**2 / (C0 * eps)

    # Scale horizontal timescales using neutral-limit (σ_i / σ_w)² ratios
    ratio_u = (SIGMA_U_OVER_USTAR_NEUTRAL / SIGMA_W_OVER_USTAR_NEUTRAL) ** 2
    ratio_v = (SIGMA_V_OVER_USTAR_NEUTRAL / SIGMA_W_OVER_USTAR_NEUTRAL) ** 2
    return ratio_u * T_Lw, ratio_v * T_Lw, T_Lw


def sigma_squared_gradients(
    z: float | np.ndarray,
    ustar: float,
    L: float,
    h: float,
    dz: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vertical gradients d(σ_i²)/dz [m² s-² m-1] via central finite differences.

    Required by the Thomson (1987) well-mixed drift correction:
        a_i(z) ⊃ ½ d(σ_i²)/dz

    Parameters
    ----------
    z:
        Height(s) [m].
    ustar:
        Friction velocity [m s-1].
    L:
        Obukhov length [m].
    h:
        Boundary-layer height [m].
    dz:
        Finite-difference step [m] (default 0.1 m).

    Returns
    -------
    d_su2_dz, d_sv2_dz, d_sw2_dz : np.ndarray
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    z_plus = z + dz
    z_minus = np.maximum(z - dz, 1e-3)
    step = z_plus - z_minus

    su_p, sv_p, sw_p = sigma_profiles(z_plus, ustar, L, h)
    su_m, sv_m, sw_m = sigma_profiles(z_minus, ustar, L, h)

    return (
        (su_p**2 - su_m**2) / step,
        (sv_p**2 - sv_m**2) / step,
        (sw_p**2 - sw_m**2) / step,
    )


def dissipation(
    sigma_w: float | np.ndarray,
    T_L: float | np.ndarray,
    C0: float,
) -> np.ndarray:
    """Dissipation rate ε = 2 σ_w² / (C0 T_L) [m² s-³]."""
    return 2.0 * np.asarray(sigma_w) ** 2 / (C0 * np.asarray(T_L))


def get_turbulence(
    z: float | np.ndarray,
    ustar: float,
    L: float,
    h: float,
    tower: TowerTurbulence | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (σ_u, σ_v, σ_w)(z), preferring measured *tower* values.

    When *tower* (:class:`~lagranged.inputs.TowerTurbulence`) carries measured
    sigma_u, sigma_v, and sigma_w, those values are broadcast to the shape of *z*
    and returned unchanged.  If any of the three are absent, the MOST-based
    :func:`sigma_profiles` is called and a :class:`UserWarning` is emitted.

    Parameters
    ----------
    z:
        Height(s) [m].
    ustar, L, h:
        Passed to :func:`sigma_profiles` when tower measurements are absent.
    tower:
        Optional :class:`~lagranged.inputs.TowerTurbulence` instance.

    Returns
    -------
    sigma_u, sigma_v, sigma_w : np.ndarray
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))

    if tower is not None:
        su_m = getattr(tower, "sigma_u", None)
        sv_m = getattr(tower, "sigma_v", None)
        sw_m = getattr(tower, "sigma_w", None)
        if su_m is not None and sv_m is not None and sw_m is not None:
            bc = z_arr.shape
            return (
                np.broadcast_to(np.asarray(su_m, dtype=float), bc).copy(),
                np.broadcast_to(np.asarray(sv_m, dtype=float), bc).copy(),
                np.broadcast_to(np.asarray(sw_m, dtype=float), bc).copy(),
            )

    warnings.warn(
        "Measured TowerTurbulence sigma values not available; "
        "falling back to parameterized MOST-based turbulence profiles.",
        UserWarning,
        stacklevel=2,
    )
    return sigma_profiles(z, ustar, L, h)
