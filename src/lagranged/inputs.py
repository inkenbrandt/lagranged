"""Immutable input objects describing one EC averaging period.

The physics inputs are deliberately split from numerical/geometry choices
(see :mod:`lagranged.config`), so a single met record can be re-run at
different resolutions or particle counts without touching the physics.

Units are SI and documented on every field.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FootprintInputs:
    """FFP-style minimum inputs for one averaging period.

    Sign convention: Obukhov length ``L < 0`` is unstable, ``L > 0`` stable.
    """

    zm: float
    """Measurement height above ground [m]."""
    L: float
    """Obukhov length [m]."""
    ustar: float
    """Friction velocity u* [m s-1]."""
    wind_dir: float
    """Mean wind direction [deg from N, meteorological convention]."""
    h: float
    """Boundary-layer (mixing) height z_i [m]."""
    sigma_v: float
    """Lateral velocity standard deviation σ_v [m s-1]."""
    z0: float | None = None
    """Aerodynamic roughness length [m]. ``None`` → derive from U and u*."""
    d: float = 0.0
    """Displacement height [m]."""
    umean: float | None = None
    """Mean wind speed at ``zm`` [m s-1] (optional; used for consistency checks)."""

    @property
    def zm_eff(self) -> float:
        """Effective measurement height above the displacement plane [m]."""
        return self.zm - self.d


@dataclass(frozen=True)
class ReynoldsStress:
    """Symmetric 3×3 velocity covariance ⟨u_i' u_j'⟩ [m² s-2].

    Drives the full 3-D correlated drift term (Thomson 1987 generalized form).
    The matrix is validated to be symmetric and positive semi-definite.
    """

    matrix: np.ndarray

    def __post_init__(self) -> None:
        m = np.asarray(self.matrix, dtype=float)
        if m.shape != (3, 3):
            raise ValueError(f"Reynolds-stress matrix must be 3x3, got {m.shape}.")
        if not np.allclose(m, m.T, atol=1e-9):
            raise ValueError("Reynolds-stress matrix must be symmetric.")
        eigvals = np.linalg.eigvalsh(m)
        if np.min(eigvals) < -1e-9:
            raise ValueError(
                "Reynolds-stress matrix is not positive semi-definite "
                f"(min eigenvalue {np.min(eigvals):.3e})."
            )
        object.__setattr__(self, "matrix", m)

    @classmethod
    def from_components(
        cls,
        su: float,
        sv: float,
        sw: float,
        cuv: float = 0.0,
        cuw: float = 0.0,
        cvw: float = 0.0,
    ) -> ReynoldsStress:
        """Build from standard deviations σ_u, σ_v, σ_w and covariances."""
        m = np.array(
            [
                [su**2, cuv, cuw],
                [cuv, sv**2, cvw],
                [cuw, cvw, sw**2],
            ],
            dtype=float,
        )
        return cls(matrix=m)

    @property
    def inverse(self) -> np.ndarray:
        """Inverse covariance matrix (used by the 3-D drift term)."""
        return np.linalg.inv(self.matrix)


@dataclass(frozen=True)
class TowerTurbulence:
    """Optional measured turbulence + scalar fluxes.

    When fields are provided they override the MOST parameterizations in
    :mod:`lagranged.turbulence`; when ``None`` the parameterized fallbacks are
    used (with a ``warnings`` notice).
    """

    sigma_u: float | None = None
    sigma_v: float | None = None
    sigma_w: float | None = None
    cov_uv: float | None = None
    cov_uw: float | None = None  # ≈ -u*² in aligned coordinates
    cov_vw: float | None = None
    tke: float | None = None
    epsilon: float | None = None  # dissipation rate [m² s-3]
    H: float | None = None  # sensible heat flux [W m-2]
    LE: float | None = None  # latent heat flux [W m-2]
    co2_flux: float | None = None  # [µmol m-2 s-1]
    qc_flag: int | None = None  # EC quality flag (0 = best)

    def reynolds_stress(self) -> ReynoldsStress | None:
        """Assemble a :class:`ReynoldsStress` if the variances are present.

        Returns ``None`` if σ_u, σ_v or σ_w is missing. Missing covariances
        default to zero (uncorrelated assumption).
        """
        if self.sigma_u is None or self.sigma_v is None or self.sigma_w is None:
            return None
        return ReynoldsStress.from_components(
            su=self.sigma_u,
            sv=self.sigma_v,
            sw=self.sigma_w,
            cuv=self.cov_uv or 0.0,
            cuw=self.cov_uw or 0.0,
            cvw=self.cov_vw or 0.0,
        )
