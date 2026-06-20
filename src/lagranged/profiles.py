"""Monin–Obukhov similarity (MOST) mean-wind profile and stability functions.

Implements the integrated stability correction ψ_m (and ψ_h) using the
Businger–Dyer / Paulson formulations for the unstable branch and a linear
(Beljaars–Holtslag style) form for the stable branch.

These are surface-layer relations; warnings elsewhere flag use outside the
surface layer.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    BUSINGER_BETA_H,
    BUSINGER_BETA_M,
    BUSINGER_GAMMA_H,
    BUSINGER_GAMMA_M,
    VON_KARMAN,
)


def psi_m(zeta: float | np.ndarray) -> np.ndarray:
    """Integrated stability function for momentum, ψ_m(ζ), ζ = z/L."""
    zeta = np.asarray(zeta, dtype=float)
    out = np.empty_like(zeta)

    unstable = zeta < 0
    x = (1.0 - BUSINGER_GAMMA_M * np.clip(zeta, None, 0.0)) ** 0.25
    out_unstable = (
        2.0 * np.log((1.0 + x) / 2.0)
        + np.log((1.0 + x**2) / 2.0)
        - 2.0 * np.arctan(x)
        + np.pi / 2.0
    )
    out_stable = -BUSINGER_BETA_M * zeta

    out = np.where(unstable, out_unstable, out_stable)
    return out


def psi_h(zeta: float | np.ndarray) -> np.ndarray:
    """Integrated stability function for heat, ψ_h(ζ), ζ = z/L."""
    zeta = np.asarray(zeta, dtype=float)
    unstable = zeta < 0
    x = (1.0 - BUSINGER_GAMMA_H * np.clip(zeta, None, 0.0)) ** 0.25
    out_unstable = 2.0 * np.log((1.0 + x**2) / 2.0)
    out_stable = -BUSINGER_BETA_H * zeta
    return np.where(unstable, out_unstable, out_stable)


def mean_wind(
    z: float | np.ndarray,
    ustar: float,
    z0: float,
    L: float,
    d: float = 0.0,
) -> np.ndarray:
    """MOST mean horizontal wind speed U(z) [m s-1].

    .. math::
        U(z) = \\frac{u_*}{\\kappa}\\left[\\ln\\frac{z-d}{z_0}
               - \\psi_m\\!\\left(\\frac{z-d}{L}\\right)
               + \\psi_m\\!\\left(\\frac{z_0}{L}\\right)\\right]

    Reduces to the neutral log law as ``L → ±∞``.
    """
    z = np.asarray(z, dtype=float)
    zr = z - d
    if np.any(zr <= z0):
        raise ValueError("z - d must exceed z0 for the log-law profile.")
    # Guard against L == 0 (treat as neutral).
    inv_L = 0.0 if L == 0 else 1.0 / L
    corr = psi_m(zr * inv_L) - psi_m(z0 * inv_L)
    return (ustar / VON_KARMAN) * (np.log(zr / z0) - corr)
