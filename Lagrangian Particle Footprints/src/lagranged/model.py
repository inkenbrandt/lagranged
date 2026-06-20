"""The :class:`FootprintModel` orchestrator and convenience functions.

Everything composes through :meth:`FootprintModel.run`; :func:`compute_footprint`
and the CLI are thin wrappers over it (a single seam for reasoning about
correctness).

The pipeline is::

    validate → build profiles/turbulence → simulate trajectories →
    detect touchdowns → grid → contour → assemble FootprintResult
"""

from __future__ import annotations

import math
import warnings
from dataclasses import fields as _dataclass_fields

import numpy as np

from ._rng import get_rng
from .config import DomainGrid, ModelConfig
from .contours import cumulative_levels
from .gridding import accumulate, grid_cell_centers
from .inputs import FootprintInputs, TowerTurbulence
from .io import iter_records
from .particles import simulate_trajectories
from .results import FootprintResult
from .touchdown import detect_touchdowns
from .validation import validate_inputs

# Field names accepted by FootprintInputs, and the subset that must be present
# (and finite) for a tabular record to be runnable. The optional remainder
# (z0, d, umean) falls back to the dataclass defaults when absent or NaN.
_INPUT_FIELDS: frozenset[str] = frozenset(f.name for f in _dataclass_fields(FootprintInputs))
_REQUIRED_FIELDS: frozenset[str] = frozenset({"zm", "L", "ustar", "wind_dir", "h", "sigma_v"})

# Cumulative source-area fractions reported as contour levels.
_CONTOUR_FRACTIONS: tuple[float, ...] = (0.5, 0.8, 0.9)


class FootprintModel:
    """Compose inputs + turbulence + config + grid into a footprint computation."""

    def __init__(
        self,
        inputs: FootprintInputs,
        grid: DomainGrid,
        turbulence: TowerTurbulence | None = None,
        config: ModelConfig | None = None,
    ) -> None:
        self.inputs = inputs
        self.grid = grid
        self.turbulence = turbulence
        self.config = config or ModelConfig()

    def run(self) -> FootprintResult:
        """Execute the pipeline: validate → simulate → touchdown → grid → contour.

        Returns
        -------
        FootprintResult
            With ``density`` (normalized so ``Σ density·dx·dy == 1``), the
            upwind-frame cell-center coordinates ``x``/``y``, cumulative
            source-area ``contours``, the along-wind peak ``x_peak`` of the
            crosswind-integrated density, the touchdown count ``n_touchdowns``
            and a Monte-Carlo noise estimate ``mc_noise``.
        """
        validate_inputs(self.inputs)

        if self.config.mode == "reynolds" and (
            self.turbulence is None or self.turbulence.reynolds_stress() is None
        ):
            raise ValueError("mode='reynolds' requires a TowerTurbulence with σ_u, σ_v, σ_w.")
        if self.turbulence is None:
            warnings.warn(
                "No measured TowerTurbulence supplied; using parameterized "
                "turbulence (research-grade approximation).",
                stacklevel=2,
            )

        # --- Simulate the backward ensemble and weight surface touchdowns ---
        rng = get_rng(self.config.seed)
        traj = simulate_trajectories(self.inputs, self.turbulence, self.config, rng)
        x_td, y_td, weight = detect_touchdowns(
            traj.x, traj.y, traj.w_contact, sigma_w0=traj.sigma_w_surface
        )

        # --- Grid → normalized density (integrates to 1) and contour levels ---
        x, y = grid_cell_centers(self.grid)
        density = accumulate(x_td, y_td, weight, self.grid)
        cell_area = self.grid.dx * self.grid.dy
        contours = cumulative_levels(density, _CONTOUR_FRACTIONS, cell_area=cell_area)

        # --- Along-wind peak of the crosswind-integrated density f(x) = ∫ f dy ---
        fx = density.sum(axis=0) * self.grid.dy
        x_peak = float(x[int(np.argmax(fx))]) if np.any(fx > 0.0) else float("nan")

        # --- Monte-Carlo noise from the Kish effective sample size ------------
        # N_eff = (Σw)² / Σw²; the relative MC standard error of the weighted
        # estimate is 1/√N_eff. Falls ∝ N^(−1/2) as more particles are released.
        total_weight = float(weight.sum())
        sum_w2 = float(np.square(weight).sum())
        if total_weight > 0.0 and sum_w2 > 0.0:
            n_eff = total_weight**2 / sum_w2
            mc_noise = 1.0 / np.sqrt(n_eff)
        else:
            n_eff = 0.0
            mc_noise = float("nan")

        return FootprintResult(
            density=density,
            x=x,
            y=y,
            contours=contours,
            x_peak=x_peak,
            n_touchdowns=traj.n_touchdown,
            mc_noise=mc_noise,
            inputs=self.inputs,
            config=self.config,
            meta={
                "mode": self.config.mode,
                "dt": traj.dt,
                "n_released": traj.n_released,
                "n_touchdown": traj.n_touchdown,
                "total_weight": total_weight,
                "n_eff": n_eff,
                "sigma_w_surface": traj.sigma_w_surface,
            },
        )


def compute_footprint(
    *,
    zm: float,
    L: float,
    ustar: float,
    wind_dir: float,
    h: float,
    sigma_v: float,
    z0: float | None = None,
    d: float = 0.0,
    umean: float | None = None,
    grid: DomainGrid,
    turbulence: TowerTurbulence | None = None,
    n_particles: int = 50_000,
    seed: int | None = None,
    **config_kwargs,
) -> FootprintResult:
    """One-call convenience wrapper around :class:`FootprintModel`."""
    inputs = FootprintInputs(
        zm=zm,
        L=L,
        ustar=ustar,
        wind_dir=wind_dir,
        h=h,
        sigma_v=sigma_v,
        z0=z0,
        d=d,
        umean=umean,
    )
    config = ModelConfig(n_particles=n_particles, seed=seed, **config_kwargs)
    return FootprintModel(inputs, grid=grid, turbulence=turbulence, config=config).run()


def _inputs_from_row(row: dict) -> FootprintInputs:
    """Build :class:`FootprintInputs` from one tabular record.

    Only recognized columns are used; values that are ``None`` or NaN are
    dropped (so optional fields fall back to their defaults). Raises
    ``ValueError`` if any required physics field is missing or non-finite.
    """
    kwargs: dict[str, float] = {}
    for name in _INPUT_FIELDS:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        try:
            fvalue = float(value)
        except (TypeError, ValueError):
            continue
        if math.isnan(fvalue):
            continue
        kwargs[name] = fvalue

    missing = sorted(_REQUIRED_FIELDS - kwargs.keys())
    if missing:
        raise ValueError(f"missing or non-finite required field(s): {missing}")
    return FootprintInputs(**kwargs)


def _row_failed_qc(row: dict) -> bool:
    """True if a ``qc_flag`` column marks the record as poor quality.

    Follows the common 0/1/2 eddy-covariance scheme (0 = best, 1 = acceptable,
    2 = discard); records flagged ``>= 2`` are skipped. No flag → not failed.
    """
    flag = row.get("qc_flag")
    if flag is None:
        return False
    try:
        return float(flag) >= 2
    except (TypeError, ValueError):
        return False


def run_batch(
    df,
    *,
    grid: DomainGrid,
    config: ModelConfig | None = None,
    progress: bool = False,
) -> dict:
    """Run a footprint for each row of a tabular EC dataset.

    Parameters
    ----------
    df:
        A pandas DataFrame whose columns map to :class:`FootprintInputs` fields
        (see :func:`lagranged.io.read_ec_csv`). An optional ``qc_flag`` column
        gates poor-quality records.
    grid, config:
        Shared geometry and numerics applied to every record. ``config=None``
        uses the :class:`ModelConfig` defaults.
    progress:
        Show a ``tqdm`` progress bar if the package is installed (a no-op
        warning otherwise).

    Returns
    -------
    dict
        ``{record_index: FootprintResult}`` for every row that passed QC. Rows
        failing quality control or input validation are skipped with a
        ``warnings.warn`` and omitted from the result.
    """
    records = list(iter_records(df))

    iterator = records
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(records, total=len(records), desc="footprints")
        except ImportError:
            warnings.warn(
                "progress=True but tqdm is not installed; run `pip install tqdm`. "
                "Continuing without a progress bar.",
                stacklevel=2,
            )

    results: dict = {}
    for idx, row in iterator:
        if _row_failed_qc(row):
            warnings.warn(f"Skipping record {idx!r}: qc_flag marks it poor quality.", stacklevel=2)
            continue
        try:
            inputs = _inputs_from_row(row)
        except (ValueError, TypeError) as exc:
            warnings.warn(f"Skipping record {idx!r}: {exc}", stacklevel=2)
            continue
        try:
            results[idx] = FootprintModel(inputs, grid=grid, config=config).run()
        except ValueError as exc:  # input validation failed inside run()
            warnings.warn(f"Skipping record {idx!r} (failed validation): {exc}", stacklevel=2)
            continue

    return results
