"""Physical constants and tunable model defaults, gathered in one place.

All values are SI. Defaults that are genuinely tunable (``C0``, particle counts,
etc.) live on :class:`lagranged.config.ModelConfig`; this module holds the
true physical constants plus a few well-established empirical coefficients.
"""

from __future__ import annotations

# --- Physical constants ---
VON_KARMAN: float = 0.40
"""von Kármán constant κ (dimensionless). Commonly 0.40 (range 0.38–0.41)."""

GRAVITY: float = 9.81
"""Gravitational acceleration g [m s-2]."""

# --- Langevin / turbulence coefficients ---
C0_DEFAULT: float = 4.0
"""Kolmogorov/Langevin universal constant C0 (dimensionless). Literature 3–6."""

# Neutral-limit similarity ratios σ_i / u* (Stull 1988; Garratt 1992).
SIGMA_U_OVER_USTAR_NEUTRAL: float = 2.5
SIGMA_V_OVER_USTAR_NEUTRAL: float = 2.0
SIGMA_W_OVER_USTAR_NEUTRAL: float = 1.25

# Businger–Dyer stability-function coefficients (unstable branch).
BUSINGER_GAMMA_M: float = 16.0
BUSINGER_GAMMA_H: float = 16.0
# Stable branch (Paulson / Beljaars-Holtslag style linear coefficient).
BUSINGER_BETA_M: float = 5.0
BUSINGER_BETA_H: float = 5.0
