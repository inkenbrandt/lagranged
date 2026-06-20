"""Tests for the MOST mean-wind profile and stability functions."""

from __future__ import annotations

import numpy as np
import pytest

from lagranged.constants import VON_KARMAN
from lagranged.profiles import mean_wind, psi_m


def test_psi_m_neutral_limit_is_zero():
    # As |L| -> inf, zeta -> 0 and psi_m -> 0.
    assert psi_m(0.0) == pytest.approx(0.0, abs=1e-9)


def test_log_law_recovered_in_neutral():
    ustar, z0 = 0.35, 0.03
    z = 3.0
    u = mean_wind(z, ustar=ustar, z0=z0, L=0.0)
    expected = (ustar / VON_KARMAN) * np.log(z / z0)
    assert u == pytest.approx(expected, rel=1e-10)


def test_wind_increases_with_height():
    z = np.array([1.0, 2.0, 4.0, 8.0])
    u = mean_wind(z, ustar=0.35, z0=0.03, L=-50.0)
    assert np.all(np.diff(u) > 0)


def test_below_z0_raises():
    with pytest.raises(ValueError):
        mean_wind(0.01, ustar=0.35, z0=0.03, L=0.0)
