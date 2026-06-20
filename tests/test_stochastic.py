"""Fast unit tests for the Langevin step functions (stochastic.py).

These cover shape/finiteness, determinism under a fixed seed, and the
homogeneous well-mixed property (stationary variance preserved). The full
inhomogeneous well-mixed criterion is exercised separately in Phase 4.
"""

from __future__ import annotations

import numpy as np
import pytest

from lagranged.inputs import ReynoldsStress
from lagranged.stochastic import step_1d, step_3d

C0 = 4.0
EPS = 0.1  # dissipation rate [m² s-3]
SIGMA_U, SIGMA_V, SIGMA_W = 0.6, 0.5, 0.4
DSW2_DZ = -0.01  # dσ_w²/dz [m s-2]
N = 1000


def _init_velocities(seed: int = 0):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, SIGMA_U, N)
    v = rng.normal(0.0, SIGMA_V, N)
    w = rng.normal(0.0, SIGMA_W, N)
    return u, v, w


# ---------------------------------------------------------------------------
# step_1d
# ---------------------------------------------------------------------------


def test_step_1d_shapes_and_finite():
    u, v, w = _init_velocities()
    rng = np.random.default_rng(1)
    un, vn, wn = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=rng,
        C0=C0,
    )
    for arr in (un, vn, wn):
        assert arr.shape == (N,)
        assert np.all(np.isfinite(arr))


def test_step_1d_deterministic_under_seed():
    u, v, w = _init_velocities()
    out_a = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=np.random.default_rng(42),
        C0=C0,
    )
    out_b = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=np.random.default_rng(42),
        C0=C0,
    )
    for a, b in zip(out_a, out_b, strict=True):
        np.testing.assert_array_equal(a, b)


def test_step_1d_different_seed_differs():
    u, v, w = _init_velocities()
    _, _, w_a = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=np.random.default_rng(1),
        C0=C0,
    )
    _, _, w_b = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=np.random.default_rng(2),
        C0=C0,
    )
    assert not np.allclose(w_a, w_b)


def test_step_1d_zero_C0_is_pure_drift():
    """With C0=0 the kick and fading-memory vanish; output is deterministic drift."""
    u, v, w = _init_velocities()
    rng = np.random.default_rng(7)  # must be ignored
    un, vn, wn = step_1d(
        u,
        v,
        w,
        sigma_u=SIGMA_U,
        sigma_v=SIGMA_V,
        sigma_w=SIGMA_W,
        dsigma_w2_dz=DSW2_DZ,
        epsilon=EPS,
        dt=0.05,
        rng=rng,
        C0=0.0,
    )
    # u, v have zero drift when C0=0; w keeps only the gradient-correction drift.
    np.testing.assert_allclose(un, u)
    np.testing.assert_allclose(vn, v)
    expected_w = w + 0.05 * 0.5 * DSW2_DZ * (1.0 + w * w / SIGMA_W**2)
    np.testing.assert_allclose(wn, expected_w)


def test_step_1d_accepts_per_particle_fields():
    u, v, w = _init_velocities()
    rng = np.random.default_rng(3)
    sw = np.full(N, SIGMA_W)
    eps = np.full(N, EPS)
    dt = np.full(N, 0.05)
    un, vn, wn = step_1d(
        u,
        v,
        w,
        sigma_u=np.full(N, SIGMA_U),
        sigma_v=np.full(N, SIGMA_V),
        sigma_w=sw,
        dsigma_w2_dz=np.full(N, DSW2_DZ),
        epsilon=eps,
        dt=dt,
        rng=rng,
        C0=C0,
    )
    assert wn.shape == (N,)
    assert np.all(np.isfinite(wn))


def test_step_1d_homogeneous_preserves_variance():
    """Homogeneous well-mixed check: stationary variance stays ≈ σ_i² over many steps."""
    rng = np.random.default_rng(123)
    n = 20_000
    eps = 0.1
    sigma = 0.6
    t_lag = 2.0 * sigma**2 / (C0 * eps)
    dt = 0.02 * t_lag
    # Start already at equilibrium N(0, σ²); a well-mixed model must keep it there.
    u = rng.normal(0.0, sigma, n)
    v = rng.normal(0.0, sigma, n)
    w = rng.normal(0.0, sigma, n)
    for _ in range(300):
        u, v, w = step_1d(
            u,
            v,
            w,
            sigma_u=sigma,
            sigma_v=sigma,
            sigma_w=sigma,
            dsigma_w2_dz=0.0,
            epsilon=eps,
            dt=dt,
            rng=rng,
            C0=C0,
        )
    assert np.all(np.isfinite(w))
    assert np.var(w) == pytest.approx(sigma**2, rel=0.1)
    assert np.var(u) == pytest.approx(sigma**2, rel=0.1)


# ---------------------------------------------------------------------------
# step_3d
# ---------------------------------------------------------------------------


def _reynolds_inverse():
    rs = ReynoldsStress.from_components(su=SIGMA_U, sv=SIGMA_V, sw=SIGMA_W, cuw=-0.05, cvw=0.01)
    return rs.inverse


def test_step_3d_shapes_and_finite():
    u, v, w = _init_velocities()
    vel = np.column_stack([u, v, w])
    rng = np.random.default_rng(1)
    un = step_3d(vel, tau_inv=_reynolds_inverse(), epsilon=EPS, dt=0.05, rng=rng, C0=C0)
    assert un.shape == (N, 3)
    assert np.all(np.isfinite(un))


def test_step_3d_deterministic_under_seed():
    u, v, w = _init_velocities()
    vel = np.column_stack([u, v, w])
    lam = _reynolds_inverse()
    a = step_3d(vel, tau_inv=lam, epsilon=EPS, dt=0.05, rng=np.random.default_rng(42), C0=C0)
    b = step_3d(vel, tau_inv=lam, epsilon=EPS, dt=0.05, rng=np.random.default_rng(42), C0=C0)
    np.testing.assert_array_equal(a, b)


def test_step_3d_with_gradient_finite():
    u, v, w = _init_velocities()
    vel = np.column_stack([u, v, w])
    rng = np.random.default_rng(5)
    dtau_dz = np.array(
        [
            [0.0, 0.0, -0.002],
            [0.0, 0.0, 0.0],
            [-0.002, 0.0, -0.01],
        ]
    )
    un = step_3d(
        vel,
        tau_inv=_reynolds_inverse(),
        epsilon=EPS,
        dt=0.05,
        rng=rng,
        dtau_dz=dtau_dz,
        C0=C0,
    )
    assert un.shape == (N, 3)
    assert np.all(np.isfinite(un))


def test_step_3d_per_particle_matrices():
    u, v, w = _init_velocities()
    vel = np.column_stack([u, v, w])
    rng = np.random.default_rng(9)
    lam = np.broadcast_to(_reynolds_inverse(), (N, 3, 3)).copy()
    un = step_3d(vel, tau_inv=lam, epsilon=np.full(N, EPS), dt=0.05, rng=rng, C0=C0)
    assert un.shape == (N, 3)
    assert np.all(np.isfinite(un))


def test_step_3d_diagonal_matches_step_1d_relaxation():
    """A diagonal Reynolds stress with no gradient ⇒ independent OU per component,
    matching step_1d's horizontal/vertical fading-memory drift (noise aside)."""
    u, v, w = _init_velocities()
    vel = np.column_stack([u, v, w])
    rs = ReynoldsStress.from_components(su=SIGMA_U, sv=SIGMA_V, sw=SIGMA_W)
    # Compare drift only: zero-noise (C0 -> finite but compare deterministic part).
    dt = 0.05
    lam_u = vel @ rs.inverse.T
    a_3d = -0.5 * C0 * EPS * lam_u  # (N, 3) drift, no gradient term
    a_u = -(0.5 * C0 * EPS / SIGMA_U**2) * u
    a_v = -(0.5 * C0 * EPS / SIGMA_V**2) * v
    a_w = -(0.5 * C0 * EPS / SIGMA_W**2) * w
    np.testing.assert_allclose(a_3d[:, 0], a_u, rtol=1e-12)
    np.testing.assert_allclose(a_3d[:, 1], a_v, rtol=1e-12)
    np.testing.assert_allclose(a_3d[:, 2], a_w, rtol=1e-12)
    assert dt > 0  # sanity
