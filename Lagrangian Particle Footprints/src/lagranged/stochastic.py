"""The Langevin (well-mixed) stochastic integrator.

Particle velocity evolves by a Langevin equation whose increment combines a
deterministic drift :math:`a_i` (the *well-mixed* correction term of
Thomson 1987) with an isotropic random kick scaled by :math:`(C_0\\varepsilon)^{1/2}`:

.. math::
    du_i = a_i(\\mathbf{x}, \\mathbf{u})\\,dt + (C_0 \\varepsilon)^{1/2}\\,dW_i ,
    \\qquad dx_i = u_i\\,dt .

**Well-mixed condition (Thomson 1987).** A stochastic model is admissible only
if an ensemble of particles initially distributed according to the Eulerian
velocity PDF :math:`P_E(\\mathbf{u};\\mathbf{x})` *stays* so for all time — i.e.
an initially well-mixed tracer remains well-mixed. For Gaussian turbulence with
covariance (Reynolds stress) :math:`\\tau_{ij}=\\langle u_i'u_j'\\rangle` and its
inverse :math:`\\lambda=\\tau^{-1}`, the unique "simplest" drift satisfying that
condition (uniform mean density) is

.. math::
    a_i = -\\tfrac12 C_0\\varepsilon\\,\\lambda_{ij}u_j
          + \\tfrac12\\,\\frac{\\partial\\tau_{il}}{\\partial x_l}
          + \\tfrac12\\,\\frac{\\partial\\tau_{il}}{\\partial x_k}\\,\\lambda_{lj}\\,u_k u_j .

The first term is the fading-memory (dissipation) drift; the gradient terms are
the inhomogeneity correction that keeps the model well-mixed.

Two modes are provided:

* :func:`step_1d` — ``"param"`` mode: the full 1-D inhomogeneous well-mixed
  equation for vertical ``w`` (including the ``dσ_w²/dz`` correction) plus
  parameterized, decoupled Ornstein–Uhlenbeck horizontal velocities.
* :func:`step_3d` — ``"reynolds"`` mode: the full 3-D correlated Gaussian drift
  above, driven by the inverse Reynolds-stress matrix
  (:attr:`lagranged.inputs.ReynoldsStress.inverse`).

All randomness is drawn from a caller-supplied :class:`numpy.random.Generator`;
there is no global RNG state, so a fixed seed fully determines a step. Every
function is vectorized over particles, operating on ``(N,)`` / ``(N, 3)`` arrays.

References
----------
Thomson DJ (1987) Criteria for the selection of stochastic models of particle
    trajectories in turbulent flows. J Fluid Mech 180:529–556.
    https://doi.org/10.1017/S0022112087001940
Rodean HC (1996) Stochastic Lagrangian Models of Turbulent Diffusion.
    Meteorological Monographs 26. AMS, Boston.
Wilson JD, Sawford BL (1996) Review of Lagrangian stochastic models for
    trajectories in the turbulent atmosphere. Boundary-Layer Meteorol 78:191–210.
"""

from __future__ import annotations

import numpy as np

from .constants import C0_DEFAULT

__all__ = ["step_1d", "step_3d"]

# Floor applied to velocity variances so the fading-memory drift stays finite
# where a component variance collapses toward zero (e.g. at the surface).
_VAR_FLOOR: float = 1e-12


def step_1d(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    *,
    sigma_u: np.ndarray | float,
    sigma_v: np.ndarray | float,
    sigma_w: np.ndarray | float,
    dsigma_w2_dz: np.ndarray | float,
    epsilon: np.ndarray | float,
    dt: np.ndarray | float,
    rng: np.random.Generator,
    C0: float = C0_DEFAULT,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Advance ``(u, v, w)`` one Langevin step in ``"param"`` mode.

    The vertical component uses the **full 1-D inhomogeneous well-mixed**
    equation (Thomson 1987; the 1-D reduction of the 3-D drift in the module
    docstring):

    .. math::
        a_w = -\frac{C_0\varepsilon}{2\sigma_w^2}\,w
              + \frac12\,\frac{\partial\sigma_w^2}{\partial z}
                \left(1 + \frac{w^2}{\sigma_w^2}\right),

    where :math:`-C_0\varepsilon/(2\sigma_w^2) = -1/T_{Lw}` is the fading-memory
    rate and the second term is the drift correction that keeps the vertical
    distribution well-mixed in a height-varying :math:`\sigma_w^2(z)`.

    The horizontal components are evolved as **decoupled Ornstein–Uhlenbeck**
    processes relaxing toward their local variance — the parameterized
    approximation of ``"param"`` mode (horizontal homogeneity assumed; the
    weaker vertical inhomogeneity of :math:`\sigma_{u,v}` is neglected):

    .. math::
        a_{u} = -\frac{C_0\varepsilon}{2\sigma_u^2}\,u, \qquad
        a_{v} = -\frac{C_0\varepsilon}{2\sigma_v^2}\,v .

    All three share the isotropic kick :math:`(C_0\varepsilon\,dt)^{1/2}\,\xi`,
    :math:`\xi\sim\mathcal N(0,1)`. With these timescales the stationary
    variance of each component equals its local :math:`\sigma_i^2`.

    Parameters
    ----------
    u, v, w:
        Current velocity components, each shape ``(N,)`` [m s-1].
    sigma_u, sigma_v, sigma_w:
        Local velocity standard deviations [m s-1]; scalar or broadcastable to
        ``(N,)``.
    dsigma_w2_dz:
        Local vertical gradient :math:`\partial\sigma_w^2/\partial z`
        [m s-2]; scalar or ``(N,)``.
    epsilon:
        Local TKE dissipation rate :math:`\varepsilon` [m² s-3]; scalar or
        ``(N,)``.
    dt:
        Time step [s]; scalar or per-particle ``(N,)``.
    rng:
        Source of randomness (:class:`numpy.random.Generator`). No global state.
    C0:
        Kolmogorov/Langevin constant (default
        :data:`~lagranged.constants.C0_DEFAULT`).

    Returns
    -------
    u_new, v_new, w_new : np.ndarray
        Updated velocity components, each shape ``(N,)``.
    """
    u = np.atleast_1d(np.asarray(u, dtype=float))
    v = np.atleast_1d(np.asarray(v, dtype=float))
    w = np.atleast_1d(np.asarray(w, dtype=float))
    shape = w.shape
    n = w.shape[0]

    su2 = np.maximum(np.broadcast_to(np.asarray(sigma_u, dtype=float) ** 2, shape), _VAR_FLOOR)
    sv2 = np.maximum(np.broadcast_to(np.asarray(sigma_v, dtype=float) ** 2, shape), _VAR_FLOOR)
    sw2 = np.maximum(np.broadcast_to(np.asarray(sigma_w, dtype=float) ** 2, shape), _VAR_FLOOR)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), shape)
    dsw2_dz = np.broadcast_to(np.asarray(dsigma_w2_dz, dtype=float), shape)
    dt_arr = np.asarray(dt, dtype=float)

    half_c0_eps = 0.5 * C0 * eps

    # Vertical: fading-memory + Thomson well-mixed drift correction.
    a_w = -(half_c0_eps / sw2) * w + 0.5 * dsw2_dz * (1.0 + w * w / sw2)
    # Horizontal: decoupled OU relaxation toward the local variance.
    a_u = -(half_c0_eps / su2) * u
    a_v = -(half_c0_eps / sv2) * v

    noise_std = np.sqrt(C0 * eps * dt_arr)  # (N,)
    xi = rng.standard_normal((n, 3))

    u_new = u + a_u * dt_arr + noise_std * xi[:, 0]
    v_new = v + a_v * dt_arr + noise_std * xi[:, 1]
    w_new = w + a_w * dt_arr + noise_std * xi[:, 2]
    return u_new, v_new, w_new


def step_3d(
    u: np.ndarray,
    *,
    tau_inv: np.ndarray,
    epsilon: np.ndarray | float,
    dt: np.ndarray | float,
    rng: np.random.Generator,
    dtau_dz: np.ndarray | None = None,
    C0: float = C0_DEFAULT,
) -> np.ndarray:
    r"""Advance the 3-D correlated velocity one Langevin step (``"reynolds"`` mode).

    Implements the Thomson (1987) generalized Gaussian drift driven by the
    inverse Reynolds-stress matrix :math:`\lambda = \tau^{-1}`
    (:attr:`lagranged.inputs.ReynoldsStress.inverse`). For horizontally
    homogeneous turbulence only the vertical gradient
    :math:`G_{il} \equiv \partial\tau_{il}/\partial z` survives, giving (with
    :math:`u_w \equiv u_3`)

    .. math::
        a_i = -\frac{C_0\varepsilon}{2}\,\lambda_{ij}u_j
              + \frac12\,G_{i3}
              + \frac12\,u_w\,G_{il}\,\lambda_{lj}\,u_j .

    When ``dtau_dz`` is ``None`` the turbulence is treated as locally
    homogeneous: the gradient terms vanish and the step is the exactly
    well-mixed Ornstein–Uhlenbeck process for a Gaussian field. This is the
    natural choice when a single, height-constant measured Reynolds stress is
    supplied. The kick is isotropic: :math:`(C_0\varepsilon\,dt)^{1/2}\,\xi_i`.

    Parameters
    ----------
    u:
        Current velocity vectors, shape ``(N, 3)`` ordered ``(u, v, w)`` [m s-1].
    tau_inv:
        Inverse covariance :math:`\lambda=\tau^{-1}` [s² m-2]; either a single
        ``(3, 3)`` matrix (broadcast to all particles) or a per-particle
        ``(N, 3, 3)`` stack.
    epsilon:
        Local TKE dissipation rate [m² s-3]; scalar or ``(N,)``.
    dt:
        Time step [s]; scalar or per-particle ``(N,)``.
    rng:
        Source of randomness (:class:`numpy.random.Generator`). No global state.
    dtau_dz:
        Optional vertical gradient of the covariance matrix
        :math:`\partial\tau_{il}/\partial z` [m s-2], ``(3, 3)`` or
        ``(N, 3, 3)``. ``None`` → homogeneous (gradient terms dropped).
    C0:
        Kolmogorov/Langevin constant (default
        :data:`~lagranged.constants.C0_DEFAULT`).

    Returns
    -------
    u_new : np.ndarray
        Updated velocity vectors, shape ``(N, 3)``.
    """
    u = np.atleast_2d(np.asarray(u, dtype=float))
    if u.shape[-1] != 3:
        raise ValueError(f"u must have shape (N, 3); got {u.shape}.")
    n = u.shape[0]

    lam = _as_matrix_stack(tau_inv, n)  # (N, 3, 3)
    eps = np.broadcast_to(np.asarray(epsilon, dtype=float), (n,))
    dt_arr = np.asarray(dt, dtype=float)
    dt_col = dt_arr[:, None] if dt_arr.ndim == 1 else dt_arr

    # Fading-memory drift: -1/2 C0 eps (lambda . u).
    lam_u = np.einsum("nij,nj->ni", lam, u)
    a = -0.5 * C0 * eps[:, None] * lam_u

    if dtau_dz is not None:
        g = _as_matrix_stack(dtau_dz, n)  # (N, 3, 3)
        # Term II: 1/2 G_{i3}  (vertical gradient of the i–w covariance).
        a = a + 0.5 * g[:, :, 2]
        # Term III: 1/2 u_w (G . lambda . u).
        g_lam_u = np.einsum("nil,nlj,nj->ni", g, lam, u)
        a = a + 0.5 * u[:, 2:3] * g_lam_u

    noise_std = np.sqrt(C0 * eps * dt_arr)  # (N,)
    xi = rng.standard_normal((n, 3))
    return u + a * dt_col + noise_std[:, None] * xi


def _as_matrix_stack(m: np.ndarray, n: int) -> np.ndarray:
    """Broadcast a ``(3, 3)`` matrix or validate a ``(N, 3, 3)`` stack to ``(N, 3, 3)``."""
    m = np.asarray(m, dtype=float)
    if m.shape == (3, 3):
        return np.broadcast_to(m, (n, 3, 3))
    if m.shape == (n, 3, 3):
        return m
    raise ValueError(f"matrix must be (3, 3) or ({n}, 3, 3); got {m.shape}.")
