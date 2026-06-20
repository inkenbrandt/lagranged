"""Input range checks and physical-consistency warnings.

Pure functions: raise :class:`ValueError` on impossible values, emit
:func:`warnings.warn` for approximate or edge-of-validity regimes.
"""

from __future__ import annotations

import warnings

from .inputs import FootprintInputs

# Threshold above which |z/L| is treated as effectively neutral.
NEUTRAL_ZL_THRESHOLD = 1e-3


def validate_inputs(inp: FootprintInputs) -> None:
    """Validate a :class:`FootprintInputs` record.

    Raises on physically impossible configurations; warns on approximate regimes.
    """
    if inp.zm <= 0:
        raise ValueError(f"zm must be > 0, got {inp.zm}.")
    if inp.ustar <= 0:
        raise ValueError(f"ustar must be > 0, got {inp.ustar}.")
    if inp.h <= 0:
        raise ValueError(f"h must be > 0, got {inp.h}.")
    if inp.sigma_v <= 0:
        raise ValueError(f"sigma_v must be > 0, got {inp.sigma_v}.")

    if inp.z0 is not None and inp.zm_eff <= inp.z0:
        raise ValueError(f"Effective height zm_eff={inp.zm_eff} must exceed z0={inp.z0}.")
    if inp.zm_eff >= inp.h:
        raise ValueError(f"Effective height zm_eff={inp.zm_eff} must be below BL height h={inp.h}.")

    if inp.zm_eff > 0.8 * inp.h:
        warnings.warn(
            "zm is in the upper boundary layer (zm_eff > 0.8 h); surface-layer "
            "similarity assumptions are weak here.",
            stacklevel=2,
        )
    if not (0.0 <= inp.wind_dir <= 360.0):
        warnings.warn(
            f"wind_dir={inp.wind_dir} is outside [0, 360]; it will be wrapped.",
            stacklevel=2,
        )
