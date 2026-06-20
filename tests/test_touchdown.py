"""Focused tests for touchdown weighting (touchdown.py)."""

from __future__ import annotations

import numpy as np
import pytest

from lagranged.touchdown import detect_touchdowns


def test_weights_finite_and_positive():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.1, -0.2, 0.3, -0.4])
    # Include grazing / zero / both-sign contact velocities.
    w = np.array([-0.5, -0.01, 0.0, 0.3])
    xo, yo, weight = detect_touchdowns(x, y, w)
    assert np.all(np.isfinite(weight))
    assert np.all(weight > 0.0)
    np.testing.assert_array_equal(xo, x)
    np.testing.assert_array_equal(yo, y)


def test_weight_is_inverse_abs_w_above_floor():
    w = np.array([-0.5, 0.25])
    _, _, weight = detect_touchdowns(np.zeros(2), np.zeros(2), w, w_floor=1e-3)
    np.testing.assert_allclose(weight, 1.0 / np.abs(w))


def test_floor_caps_grazing_contacts():
    """A near-zero |w| is capped by the floor, not allowed to blow up."""
    w = np.array([0.0, 1e-9])
    _, _, weight = detect_touchdowns(np.zeros(2), np.zeros(2), w, w_floor=1e-2)
    assert np.all(np.isfinite(weight))
    np.testing.assert_allclose(weight, 1.0 / 1e-2)


def test_sigma_w0_scales_the_floor():
    w = np.array([1e-6])
    _, _, weight = detect_touchdowns(np.zeros(1), np.zeros(1), w, sigma_w0=0.4, cap_frac=0.05)
    # floor = 0.05 * 0.4 = 0.02 -> weight = 1/0.02 = 50
    np.testing.assert_allclose(weight, [50.0])


def test_empty_input_yields_empty_weights():
    xo, yo, weight = detect_touchdowns(np.empty(0), np.empty(0), np.empty(0))
    assert xo.shape == yo.shape == weight.shape == (0,)


def test_shape_mismatch_raises():
    with pytest.raises(ValueError, match="same shape|share a shape"):
        detect_touchdowns(np.zeros(3), np.zeros(2), np.zeros(3))


def test_weight_sign_independent_of_w_sign():
    """Only |w| matters: up- and down-moving contacts of equal speed match."""
    _, _, w_down = detect_touchdowns(np.zeros(1), np.zeros(1), np.array([-0.3]))
    _, _, w_up = detect_touchdowns(np.zeros(1), np.zeros(1), np.array([0.3]))
    np.testing.assert_allclose(w_down, w_up)
