"""Touchdown (ground-contact) weighting for the backward LS estimator.

Each surface contact carries the bLS source weight ∝ ``1/|w|`` (Flesch, Wilson &
Yee 1995, 2004), where ``w`` is the vertical velocity at contact.  The weight is
*capped* by flooring ``|w|`` so grazing contacts (``|w| → 0``) cannot produce an
unbounded weight.  Landing ``(x, y)`` positions and weights are returned for
accumulation in :mod:`lagranged.gridding`.

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

import numpy as np

__all__ = ["detect_touchdowns"]

# Absolute |w| floor [m s-1] used when no σ_w reference is supplied.
_W_FLOOR_ABS: float = 1e-3


def detect_touchdowns(
    x_td: np.ndarray,
    y_td: np.ndarray,
    w_contact: np.ndarray,
    *,
    sigma_w0: float | None = None,
    cap_frac: float = 1e-2,
    w_floor: float = _W_FLOOR_ABS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Assign each surface contact its capped bLS weight ``∝ 1/|w|``.

    Parameters
    ----------
    x_td, y_td:
        Touchdown coordinates in the upwind model frame [m].
    w_contact:
        Vertical velocity at each contact [m s-1]; only its magnitude is used.
    sigma_w0:
        Near-surface σ_w [m s-1].  When given, the magnitude floor is
        ``cap_frac · sigma_w0`` (a physically scaled cap); otherwise the absolute
        ``w_floor`` is used.
    cap_frac:
        Fraction of ``sigma_w0`` used as the |w| floor (Flesch-style cap on the
        contribution of grazing contacts).
    w_floor:
        Absolute |w| floor [m s-1] used when ``sigma_w0`` is None.

    Returns
    -------
    x_td, y_td, weight : np.ndarray
        The (unchanged) coordinates and the per-contact weights — all finite and
        strictly positive.
    """
    x = np.asarray(x_td, dtype=float)
    y = np.asarray(y_td, dtype=float)
    w_abs = np.abs(np.asarray(w_contact, dtype=float))
    if not x.shape == y.shape == w_abs.shape:
        raise ValueError(
            f"x_td, y_td and w_contact must share a shape; "
            f"got {x.shape}, {y.shape}, {w_abs.shape}."
        )

    floor = cap_frac * sigma_w0 if sigma_w0 is not None else w_floor
    floor = max(float(floor), 1e-12)
    weight = 1.0 / np.maximum(w_abs, floor)
    return x, y, weight
