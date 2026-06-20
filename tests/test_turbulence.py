"""Tests for the turbulence parameterizations (turbulence.py)."""

from __future__ import annotations

import numpy as np
import pytest

from lagranged.constants import (
    C0_DEFAULT,
    SIGMA_U_OVER_USTAR_NEUTRAL,
    SIGMA_V_OVER_USTAR_NEUTRAL,
    SIGMA_W_OVER_USTAR_NEUTRAL,
)
from lagranged.inputs import TowerTurbulence
from lagranged.turbulence import (
    dissipation,
    get_turbulence,
    lagrangian_timescales,
    sigma_profiles,
    sigma_squared_gradients,
)

USTAR = 0.35  # friction velocity [m s-1]
H = 1000.0  # BL height [m]
ZM = 3.0  # measurement height [m]

# Effectively neutral: L > 0 but |L| >> h so w* = 0 and ζ ≈ 0
L_NEUTRAL = 1.0e6
L_UNSTABLE = -50.0
L_STABLE = 200.0


# ---------------------------------------------------------------------------
# σ_w/u* ≈ 1.25 in the neutral limit
# ---------------------------------------------------------------------------


def test_sigma_w_neutral_ratio():
    """σ_w/u* recovers SIGMA_W_OVER_USTAR_NEUTRAL (1.25) in the neutral limit."""
    _, _, sw = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    assert sw.item() / USTAR == pytest.approx(SIGMA_W_OVER_USTAR_NEUTRAL, rel=0.02)


def test_sigma_u_neutral_ratio():
    su, _, _ = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    assert su.item() / USTAR == pytest.approx(SIGMA_U_OVER_USTAR_NEUTRAL, rel=0.02)


def test_sigma_v_neutral_ratio():
    _, sv, _ = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    assert sv.item() / USTAR == pytest.approx(SIGMA_V_OVER_USTAR_NEUTRAL, rel=0.02)


# ---------------------------------------------------------------------------
# σ_i grow with instability
# ---------------------------------------------------------------------------


def test_sigma_w_larger_unstable_than_neutral():
    """Unstable σ_w must exceed the neutral-limit value at the same height."""
    _, _, sw_n = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    _, _, sw_u = sigma_profiles(ZM, USTAR, L_UNSTABLE, H)
    assert sw_u.item() > sw_n.item()


def test_sigma_u_larger_unstable_than_neutral():
    su_n, _, _ = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    su_u, _, _ = sigma_profiles(ZM, USTAR, L_UNSTABLE, H)
    assert su_u.item() > su_n.item()


def test_sigma_v_larger_unstable_than_neutral():
    _, sv_n, _ = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    _, sv_u, _ = sigma_profiles(ZM, USTAR, L_UNSTABLE, H)
    assert sv_u.item() > sv_n.item()


def test_sigma_w_increases_with_increasing_instability():
    """More negative L → larger σ_w (stronger convection)."""
    _, _, sw_weak = sigma_profiles(ZM, USTAR, -200.0, H)
    _, _, sw_strong = sigma_profiles(ZM, USTAR, -20.0, H)
    assert sw_strong.item() > sw_weak.item()


# ---------------------------------------------------------------------------
# ε > 0
# ---------------------------------------------------------------------------


def test_dissipation_positive_unstable():
    z = np.array([0.5, 1.0, 2.0, 5.0, 10.0])
    _, _, sw = sigma_profiles(z, USTAR, L_UNSTABLE, H)
    _, _, T_Lw = lagrangian_timescales(z, sw, USTAR, L_UNSTABLE)
    eps = dissipation(sw, T_Lw, C0_DEFAULT)
    assert np.all(eps > 0.0)


def test_dissipation_positive_neutral():
    z = np.linspace(0.5, 100.0, 20)
    _, _, sw = sigma_profiles(z, USTAR, L_NEUTRAL, H)
    _, _, T_Lw = lagrangian_timescales(z, sw, USTAR, L_NEUTRAL)
    eps = dissipation(sw, T_Lw, C0_DEFAULT)
    assert np.all(eps > 0.0)


def test_dissipation_positive_stable():
    z = np.array([1.0, 3.0, 10.0])
    _, _, sw = sigma_profiles(z, USTAR, L_STABLE, H)
    _, _, T_Lw = lagrangian_timescales(z, sw, USTAR, L_STABLE)
    eps = dissipation(sw, T_Lw, C0_DEFAULT)
    assert np.all(eps > 0.0)


def test_dissipation_consistent_with_timescales():
    """ε = 2 σ_w² / (C0 T_Lw) must hold for all returned timescales."""
    z = np.array([2.0, 5.0])
    _, _, sw = sigma_profiles(z, USTAR, L_UNSTABLE, H)
    _, _, T_Lw = lagrangian_timescales(z, sw, USTAR, L_UNSTABLE)
    eps = dissipation(sw, T_Lw, C0_DEFAULT)
    eps_expected = 2.0 * sw**2 / (C0_DEFAULT * T_Lw)
    np.testing.assert_allclose(eps, eps_expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# Lagrangian timescales: positive and ordered
# ---------------------------------------------------------------------------


def test_timescales_all_positive():
    z = np.array([1.0, 2.0, 5.0, 10.0])
    _, _, sw = sigma_profiles(z, USTAR, L_NEUTRAL, H)
    T_Lu, T_Lv, T_Lw = lagrangian_timescales(z, sw, USTAR, L_NEUTRAL)
    assert np.all(T_Lu > 0)
    assert np.all(T_Lv > 0)
    assert np.all(T_Lw > 0)


def test_timescales_ordering():
    """T_Lu > T_Lv > T_Lw (larger variance → longer decorrelation)."""
    z = np.array([3.0])
    _, _, sw = sigma_profiles(z, USTAR, L_NEUTRAL, H)
    T_Lu, T_Lv, T_Lw = lagrangian_timescales(z, sw, USTAR, L_NEUTRAL)
    assert np.all(T_Lu >= T_Lv)
    assert np.all(T_Lv >= T_Lw)


# ---------------------------------------------------------------------------
# Measured TowerTurbulence values pass through unchanged
# ---------------------------------------------------------------------------


def test_measured_values_pass_through():
    """All three measured σ values must be returned identically."""
    tower = TowerTurbulence(sigma_u=0.87, sigma_v=0.65, sigma_w=0.42)
    su, sv, sw = get_turbulence(ZM, USTAR, L_NEUTRAL, H, tower=tower)
    assert su.item() == pytest.approx(0.87)
    assert sv.item() == pytest.approx(0.65)
    assert sw.item() == pytest.approx(0.42)


def test_measured_values_broadcast_to_z_array():
    """Scalar tower values must broadcast correctly to array z."""
    tower = TowerTurbulence(sigma_u=0.87, sigma_v=0.65, sigma_w=0.42)
    z = np.array([1.0, 2.0, 3.0])
    su, sv, sw = get_turbulence(z, USTAR, L_NEUTRAL, H, tower=tower)
    assert su.shape == (3,)
    np.testing.assert_array_equal(su, 0.87)
    np.testing.assert_array_equal(sw, 0.42)


def test_no_tower_emits_warning():
    """Falling back to parameterized profiles emits a UserWarning."""
    with pytest.warns(UserWarning, match="parameterized"):
        get_turbulence(ZM, USTAR, L_NEUTRAL, H, tower=None)


def test_partial_tower_falls_back_with_warning():
    """If any of the three σ values is missing, parameterization is used."""
    tower = TowerTurbulence(sigma_w=0.42)  # sigma_u and sigma_v absent
    with pytest.warns(UserWarning):
        su, sv, sw = get_turbulence(ZM, USTAR, L_NEUTRAL, H, tower=tower)
    # result must differ from 0.42 (it's parameterized, not passthrough)
    _, _, sw_param = sigma_profiles(ZM, USTAR, L_NEUTRAL, H)
    assert sw.item() == pytest.approx(sw_param.item(), rel=1e-9)


# ---------------------------------------------------------------------------
# Gradients match finite differences
# ---------------------------------------------------------------------------


def test_sigma_squared_gradients_match_independent_finite_diff():
    """sigma_squared_gradients must agree with a coarser independent FD."""
    z = np.array([2.0, 5.0, 10.0, 20.0])

    d_su2, d_sv2, d_sw2 = sigma_squared_gradients(z, USTAR, L_UNSTABLE, H, dz=0.05)

    # Independent reference: central difference with a larger step
    z_plus = z + 1.0
    z_minus = z - 1.0
    su_p, sv_p, sw_p = sigma_profiles(z_plus, USTAR, L_UNSTABLE, H)
    su_m, sv_m, sw_m = sigma_profiles(z_minus, USTAR, L_UNSTABLE, H)
    d_sw2_ref = (sw_p**2 - sw_m**2) / 2.0
    d_su2_ref = (su_p**2 - su_m**2) / 2.0
    d_sv2_ref = (sv_p**2 - sv_m**2) / 2.0

    np.testing.assert_allclose(d_sw2, d_sw2_ref, rtol=0.05, atol=1e-8)
    np.testing.assert_allclose(d_su2, d_su2_ref, rtol=0.05, atol=1e-8)
    np.testing.assert_allclose(d_sv2, d_sv2_ref, rtol=0.05, atol=1e-8)


def test_sigma_squared_gradient_sign_neutral():
    """In neutral conditions σ_w² must decrease with height (dσ²/dz < 0)."""
    z = np.array([5.0, 10.0, 20.0])
    _, _, d_sw2 = sigma_squared_gradients(z, USTAR, L_NEUTRAL, H)
    assert np.all(d_sw2 < 0.0)
