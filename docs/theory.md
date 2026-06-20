# Theory & assumptions

This page sets out the equations `lagranged` actually integrates. The
implementation lives in [`turbulence`][lagranged.turbulence],
[`profiles`][lagranged.profiles], [`stochastic`][lagranged.stochastic],
[`particles`][lagranged.particles] and [`touchdown`][lagranged.touchdown]; the
formulas below match those modules.

Throughout, $\kappa = 0.4$ is the von Kármán constant, $u_*$ the friction
velocity, $L$ the Obukhov length (with the convention $L<0$ unstable, $L>0$
stable), $\zeta = z/L$ the stability parameter, $h$ the boundary-layer (mixing)
height, $C_0$ the Kolmogorov/Langevin constant (default $4$), and
$\varepsilon$ the turbulent kinetic energy (TKE) dissipation rate.

## 1. The footprint and the backward estimator

The **flux footprint** $f$ is the source-weight function linking an upwind
surface flux density $S(x,y)$ to the flux $F$ measured at the receptor
$\mathbf{x}_r$:

$$
F(\mathbf{x}_r) = \iint_{\text{surface}} S(x, y)\,
                  f\!\left(x - x_r,\, y - y_r\right)\, dx\, dy ,
\qquad
\iint f \, dx\, dy = 1 .
$$

`lagranged` estimates $f$ with the **backward Lagrangian stochastic (bLS)**
method of Flesch, Wilson & Yee (1995, 2004). An ensemble of $N$ marked fluid
particles is released at the receptor and tracked *backward* in time through a
turbulent flow field. Wherever a particle touches the ground it has had contact
with the surface source; the relationship between a unit surface source $Q$ and
the receptor signal is

$$
\frac{\overline{c}}{Q} \;=\; \frac{1}{N} \sum_{\text{touchdowns}}
                              \frac{2}{\lvert w_{\mathrm{td}} \rvert} ,
$$

where $w_{\mathrm{td}}$ is the particle's vertical velocity at the touchdown and
the factor $2$ accounts for reflection (only downward-moving contacts are
counted). The footprint $f(x,y)$ is the spatial density of these
$1/\lvert w_{\mathrm{td}} \rvert$-weighted touchdowns, accumulated onto the model
grid ([`gridding.accumulate`][lagranged.gridding]) and normalized to unit
integral.

## 2. Langevin (well-mixed) trajectory model

Particle velocity $u_i$ and position $x_i$ evolve by a Langevin equation: a
deterministic drift $a_i$ plus an isotropic random kick scaled by
$(C_0\varepsilon)^{1/2}$,

$$
du_i = a_i(\mathbf{x}, \mathbf{u})\, dt + (C_0\varepsilon)^{1/2}\, dW_i ,
\qquad
dx_i = u_i\, dt ,
$$

with $dW_i$ independent Wiener increments. The drift is **not** free: Thomson's
(1987) *well-mixed criterion* requires that an ensemble drawn from the Eulerian
velocity PDF stay distributed according to it for all time. For Gaussian
turbulence with covariance (Reynolds stress) $\tau_{ij} = \langle u_i' u_j'
\rangle$ and inverse $\lambda = \tau^{-1}$, the unique "simplest" admissible
drift is

$$
a_i = -\tfrac12 C_0 \varepsilon\, \lambda_{ij} u_j
      + \tfrac12\, \frac{\partial \tau_{il}}{\partial x_l}
      + \tfrac12\, \frac{\partial \tau_{il}}{\partial x_k}\, \lambda_{lj}\, u_k u_j .
$$

The first term is the fading-memory (dissipation) drift; the gradient terms are
the inhomogeneity correction that keeps the ensemble well-mixed. The package
provides two reductions of this drift.

### 3-D correlated mode (`mode="reynolds"`)

For horizontally homogeneous turbulence only the vertical gradient
$G_{il} \equiv \partial \tau_{il}/\partial z$ survives, giving (with $u_w \equiv
u_3$, implemented in [`stochastic.step_3d`][lagranged.stochastic.step_3d]):

$$
a_i = -\tfrac12 C_0\varepsilon\, \lambda_{ij} u_j
      + \tfrac12\, G_{i3}
      + \tfrac12\, u_w\, G_{il}\, \lambda_{lj}\, u_j .
$$

When a single height-constant measured Reynolds stress is supplied the gradient
terms vanish and the step is the exactly well-mixed Ornstein–Uhlenbeck process
for a Gaussian field. The covariance comes from
[`ReynoldsStress`][lagranged.inputs.ReynoldsStress] (validated symmetric and
positive semi-definite).

### 1-D parameterized mode (`mode="param"`)

The vertical component uses the full 1-D inhomogeneous well-mixed equation,
including the $\partial\sigma_w^2/\partial z$ correction
([`stochastic.step_1d`][lagranged.stochastic.step_1d]):

$$
a_w = -\frac{C_0\varepsilon}{2\sigma_w^2}\, w
      + \frac12\, \frac{\partial \sigma_w^2}{\partial z}
        \left(1 + \frac{w^2}{\sigma_w^2}\right) ,
$$

where $-C_0\varepsilon/(2\sigma_w^2) = -1/T_{Lw}$ is the fading-memory rate. The
horizontal components relax as decoupled Ornstein–Uhlenbeck processes toward
their local variance,

$$
a_u = -\frac{C_0\varepsilon}{2\sigma_u^2}\, u ,
\qquad
a_v = -\frac{C_0\varepsilon}{2\sigma_v^2}\, v ,
$$

so each component's stationary variance equals its local $\sigma_i^2$. All
randomness is drawn from a caller-supplied `numpy.random.Generator`; a fixed seed
fully determines a run.

## 3. Mean wind (Monin–Obukhov similarity)

The mean horizontal wind follows MOST
([`profiles.mean_wind`][lagranged.profiles.mean_wind]):

$$
U(z) = \frac{u_*}{\kappa}\left[\ln\frac{z-d}{z_0}
       - \psi_m\!\left(\frac{z-d}{L}\right)
       + \psi_m\!\left(\frac{z_0}{L}\right)\right] ,
$$

with $d$ the displacement height and $z_0$ the roughness length. The integrated
stability correction uses the Businger–Dyer / Paulson form on the unstable
branch and a linear form on the stable branch. With $x = (1 - \gamma_m\zeta)^{1/4}$
and $\gamma_m = 16$, $\beta_m = 5$:

$$
\psi_m(\zeta) =
\begin{cases}
2\ln\dfrac{1+x}{2} + \ln\dfrac{1+x^2}{2} - 2\arctan x + \dfrac{\pi}{2}
   & \zeta < 0 \quad(\text{unstable}),\\[1.2ex]
-\beta_m\,\zeta & \zeta \ge 0 \quad(\text{stable}).
\end{cases}
$$

As $L \to \pm\infty$ the profile reduces to the neutral log law.

## 4. Turbulence statistics

When measured values are supplied via
[`TowerTurbulence`][lagranged.inputs.TowerTurbulence] they are passed through
unchanged; otherwise the MOST-based parameterizations of
[`turbulence.sigma_profiles`][lagranged.turbulence.sigma_profiles] are used (with
a `warnings.warn`).

### Velocity standard deviations

The profiles blend a **mechanical** part ($\propto u_*$) with a **convective**
part ($\propto w_*$), following Kljun et al. (2004) and Rannik et al. (2003). For
the vertical component (Panofsky et al. 1977; Lenschow et al. 1980):

$$
\sigma_{w,\text{mech}}(z) = 1.25\, u_* \,(1 - 3\zeta)^{1/3}\, \sqrt{1 - z/h}
\quad (\zeta \le 0),
\qquad
\sigma_{w,\text{conv}}(z) = 0.96\, w_*\, (z/h)^{1/3}\,(1 - z/h),
$$

with the convective velocity scale $w_* = u_*\,(h / \kappa\lvert L\rvert)^{1/3}$
for $L<0$ (and $0$ otherwise), combined in quadrature
$\sigma_w = \sqrt{\sigma_{w,\text{mech}}^2 + \sigma_{w,\text{conv}}^2}$. The
horizontal components use the MOST coefficient $(1 - 5\zeta)^{1/3}$ with bulk
convective additions. In the neutral limit the ratios recover the surface-layer
constants

$$
\frac{\sigma_u}{u_*} \to 2.5, \qquad
\frac{\sigma_v}{u_*} \to 2.0, \qquad
\frac{\sigma_w}{u_*} \to 1.25
$$

(Stull 1988; Garratt 1992).

### Dissipation and Lagrangian timescales

The surface-layer TKE dissipation rate
([`turbulence`][lagranged.turbulence], internal `_epsilon_sl`) comes from the
surface-layer TKE budget:

$$
\varepsilon(z) = \frac{u_*^3}{\kappa z}\,\phi_\varepsilon(\zeta),
\qquad
\phi_\varepsilon(\zeta) = \phi_m(\zeta) - \zeta,
\qquad
\phi_m(\zeta) =
\begin{cases}
(1 - 16\zeta)^{-1/4} & \zeta \le 0,\\
1 + 5\zeta & \zeta > 0.
\end{cases}
$$

The vertical Lagrangian integral timescale follows the
fluctuation–dissipation relation (Thomson 1987; Rodean 1996), with horizontal
timescales scaled by the neutral variance ratios
([`turbulence.lagrangian_timescales`][lagranged.turbulence.lagrangian_timescales]):

$$
T_{Lw} = \frac{2\sigma_w^2}{C_0\varepsilon},
\qquad
T_{Li} = \left(\frac{\sigma_i}{\sigma_w}\right)^{\!2}_{\text{neutral}} T_{Lw}.
$$

The well-mixed drift correction needs the vertical gradients
$\partial\sigma_i^2/\partial z$, computed by central finite differences in
[`turbulence.sigma_squared_gradients`][lagranged.turbulence.sigma_squared_gradients]
(the term $\tfrac12\,\partial\sigma_i^2/\partial z$ in the drift).

## 5. Numerical integration

[`particles.simulate_trajectories`][lagranged.particles] releases
`config.n_particles` backward from the effective measurement height $z_m - d$,
with initial velocities sampled from the local $\sigma_i$ (or the Reynolds
stress). The time step is **adaptive**,

$$
dt = \texttt{dt\_factor} \cdot \min(\tau_L),
$$

so the step resolves the shortest local Lagrangian timescale. Each particle is
integrated until it touches down, leaves the domain, or reaches `t_max`.
Perfect reflection is applied at the surface contact height ($\approx z_0$, or
`config.rebound_height`) and optionally at $z = h$ when `config.bl_reflection`
is set. Particles are processed in chunks to bound memory.

## 6. Touchdown weighting

Each surface contact is assigned the bLS estimator weight $\propto
1/\lvert w_{\mathrm{td}} \rvert$ in
[`touchdown.detect_touchdowns`][lagranged.touchdown]. Because this weight
diverges for grazing contacts ($w_{\mathrm{td}} \to 0$), it is **capped** using
the surface vertical-velocity scale $\sigma_{w,0}$, following Flesch, Wilson &
Yee (1995, 2004). The cap slightly biases near-field contributions (see
[Limitations](limitations.md)).

## 7. Monte-Carlo convergence

The estimate carries sampling noise that falls as $N^{-1/2}$. `lagranged`
reports a noise estimate from the Kish *effective sample size* of the touchdown
weights ([`FootprintModel.run`][lagranged.model.FootprintModel.run]):

$$
N_{\text{eff}} = \frac{\left(\sum_k w_k\right)^2}{\sum_k w_k^2},
\qquad
\texttt{mc\_noise} = \frac{1}{\sqrt{N_{\text{eff}}}} .
$$

Check `result.mc_noise` and increase `n_particles` until results are stable.

## 8. Assumptions at a glance

- Horizontally homogeneous, stationary, flat-terrain **surface layer**; MOST
  profiles and $\sigma_i$ parameterizations apply.
- **Gaussian** velocity statistics (required by the Thomson well-mixed drift).
- The well-mixed drift is consistent with the package's turbulence fields only
  for those idealized fields; complex/inhomogeneous terrain violates it.
- Reflection at $z = h$ is a simplification of entrainment-zone physics.
- The $1/\lvert w \rvert$ touchdown weight is capped for grazing contacts.

These are revisited, with their consequences, in
[Limitations & caveats](limitations.md).

## Key references

- Thomson, D. J. (1987). *Criteria for the selection of stochastic models of
  particle trajectories in turbulent flows.* J. Fluid Mech. **180**, 529–556.
  <https://doi.org/10.1017/S0022112087001940>
- Flesch, T. K., Wilson, J. D., & Yee, E. (1995). *Backward-time Lagrangian
  stochastic dispersion models and their application to estimate gaseous
  emissions.* J. Appl. Meteorol. **34**, 1320–1332.
- Flesch, T. K., Wilson, J. D., Harper, L. A., Crenna, B. P., & Sharpe, R. R.
  (2004). *Deducing ground-to-air emissions from observed trace gas
  concentrations: a field trial.* J. Appl. Meteorol. **43**, 487–502.
- Wilson, J. D., & Sawford, B. L. (1996). *Review of Lagrangian stochastic
  models for trajectories in the turbulent atmosphere.* Bound.-Layer Meteorol.
  **78**, 191–210.
- Kljun, N., Calanca, P., Rotach, M. W., & Schmid, H. P. (2004). *A simple
  parameterisation for flux footprint predictions.* Bound.-Layer Meteorol.
  **112**, 503–523. <https://doi.org/10.1023/B:BOUN.0000030653.71031.96>
- Rannik, Ü., Markkanen, T., Raittila, J., Hari, P., & Vesala, T. (2003).
  *Turbulence statistics above and within two Scots pine forests.* Agric. For.
  Meteorol. **114**, 231–252.
- Rodean, H. C. (1996). *Stochastic Lagrangian Models of Turbulent Diffusion.*
  Meteorological Monographs **26**, AMS, Boston.
- Stull, R. B. (1988). *An Introduction to Boundary Layer Meteorology.* Kluwer,
  Dordrecht.

Primary-source PDFs are staged in the repository root.
