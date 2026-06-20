"""Regression (golden) + property tests for the footprint model (Phase 9).

Two complementary guards against silent changes in the numerics/physics:

1. **Golden regression.** Three fixed ``(stability, seed)`` footprints are stored
   as ``.npy`` density arrays under ``tests/data/golden/``. Re-running the model
   and asserting :func:`numpy.allclose` against the stored array flags any
   unintended change in model output. A fixed seed reproduces a run *bit-for-bit*
   on the generating platform (see ``tests/test_model_integration.py``), so the
   only way a golden drifts is a genuine change to the model — exactly what we
   want a regression test to catch.

2. **Property tests (Hypothesis).** Over a wide band of physically plausible
   inputs, a successful run must never emit ``NaN``/``inf`` in its density, and
   inputs *outside* the physical domain must be rejected by
   :func:`lagranged.validation.validate_inputs`.

Regenerating the goldens
------------------------
The stored arrays are *expected* to change whenever the physics changes on
purpose. Regenerate them intentionally (never by hand) from the project root,
with the dev environment active::

    python scripts/regenerate_goldens.py        # or:  nox -s regenerate_goldens

then review the ``git diff`` under ``tests/data/golden/`` before committing: a
non-empty diff there means the change altered model output. See
``tests/data/README.md`` for the full procedure.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

import lagranged as lg
from lagranged.validation import validate_inputs

# ===========================================================================
# Golden regression
# ===========================================================================

# Directory holding the stored golden arrays (one .npy per case).
GOLDEN_DIR = Path(__file__).parent / "data" / "golden"

# Shared base state; only the Obukhov length L (stability) and the RNG seed vary
# between cases. Mirrors the near-neutral record used elsewhere in the suite.
_BASE_INPUTS: dict[str, float] = dict(
    zm=3.0,
    z0=0.03,
    d=0.2,
    ustar=0.35,
    umean=2.4,
    wind_dir=210.0,
    h=1000.0,
    sigma_v=0.6,
)

# Small, short grid + ensemble: a fixed (ny, nx) density that every case fills
# with a few hundred touchdowns, kept cheap so the regression stays fast.
_GOLDEN_GRID = lg.DomainGrid(nx=40, ny=40, dx=5.0, dy=5.0, x0=-30.0, y0=-100.0)

# name -> (Obukhov length L [m], seed). Fixed seeds make the goldens reproducible.
GOLDEN_CASES: dict[str, tuple[float, int]] = {
    "neutral": (-10_000.0, 0),  # |z/L| ~ 0
    "unstable": (-15.0, 1),  # strongly convective
    "stable": (30.0, 2),  # stably stratified
}


def _golden_config(seed: int) -> lg.ModelConfig:
    """Cheap-but-representative numerics shared by every golden case.

    A raised ``rebound_height`` keeps the adaptive step off the stiff surface
    layer (matching ``tests/test_particles.py`` and the integration tests); the
    short ``t_max`` truncates the far upwind tail so the run stays fast.
    """
    return lg.ModelConfig(
        n_particles=600,
        dt_factor=0.05,
        t_max=120.0,
        rebound_height=0.5,
        seed=seed,
    )


def compute_golden_density(name: str) -> np.ndarray:
    """Deterministically compute the ``(ny, nx)`` density for a named case."""
    L, seed = GOLDEN_CASES[name]
    inputs = lg.FootprintInputs(L=L, **_BASE_INPUTS)
    config = _golden_config(seed)
    with warnings.catch_warnings():
        # The "parameterized turbulence" notice is expected here (no tower data).
        warnings.simplefilter("ignore")
        result = lg.FootprintModel(inputs, grid=_GOLDEN_GRID, config=config).run()
    return result.density


def golden_path(name: str) -> Path:
    """Path to the stored golden array for a case."""
    return GOLDEN_DIR / f"footprint_{name}.npy"


def regenerate(verbose: bool = True) -> list[Path]:
    """(Re)write every golden array; used by ``scripts/regenerate_goldens.py``.

    Returns the list of paths written. This is the single source of truth for
    both the regression test and the regeneration script — they share the same
    case definitions and numerics, so a golden can never silently diverge from
    what the test checks.
    """
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name in GOLDEN_CASES:
        density = compute_golden_density(name)
        path = golden_path(name)
        np.save(path, density)
        written.append(path)
        if verbose:
            captured = float(density.sum()) * _GOLDEN_GRID.dx * _GOLDEN_GRID.dy
            print(f"wrote {path.name}: shape={density.shape} integral={captured:.6f}")
    return written


@pytest.mark.parametrize("name", sorted(GOLDEN_CASES))
def test_footprint_matches_golden(name: str) -> None:
    """A re-run reproduces the stored golden density for each stability class."""
    path = golden_path(name)
    if not path.exists():
        pytest.fail(
            f"missing golden array {path.name!r}; generate it with "
            "`python scripts/regenerate_goldens.py` (see tests/data/README.md)."
        )
    expected = np.load(path)
    actual = compute_golden_density(name)

    assert actual.shape == expected.shape, (
        f"golden shape changed for {name!r}: {expected.shape} -> {actual.shape}; "
        "regenerate the goldens if the grid changed on purpose."
    )
    # A fixed seed reproduces the run bit-for-bit on the generating platform; the
    # tolerance only absorbs harmless floating-point reassociation across numpy
    # patch versions. A real physics/numerics change blows well past it, failing
    # the test — at which point regenerate the goldens deliberately.
    assert np.allclose(actual, expected, rtol=1e-6, atol=1e-12), (
        f"footprint {name!r} drifted from its golden (max abs diff "
        f"{np.max(np.abs(actual - expected)):.3e}). If this change is intentional, "
        "regenerate with `python scripts/regenerate_goldens.py` and review the diff."
    )


# ===========================================================================
# Property tests (Hypothesis)
# ===========================================================================

# Catch-all property grid: wide enough to land touchdowns across stability
# classes. The no-NaN guarantee holds regardless of how many fall off-grid.
_PROPERTY_GRID = lg.DomainGrid(nx=40, ny=40, dx=6.0, dy=6.0, x0=-50.0, y0=-120.0)


@st.composite
def _valid_inputs(draw: st.DrawFn) -> lg.FootprintInputs:
    """Draw a :class:`~lagranged.FootprintInputs` guaranteed to pass validation.

    Ranges are deliberately *plausible* surface-layer micromet values (not the
    full mathematical domain), and the constraints below keep every draw inside
    the physical envelope ``validate_inputs`` accepts:

    * ``zm`` (with ``d=0`` so ``zm_eff = zm``) always exceeds ``z0`` and stays
      well below the boundary-layer height ``h``;
    * ``|L| >= 10`` keeps stability away from the (unphysical) ``L = 0``
      singularity while still spanning strongly unstable → strongly stable.
    """
    f = dict(allow_nan=False, allow_infinity=False)
    zm = draw(st.floats(1.5, 8.0, **f))
    l_mag = draw(st.floats(10.0, 3000.0, **f))
    sign = draw(st.sampled_from((-1.0, 1.0)))
    return lg.FootprintInputs(
        zm=zm,
        d=0.0,
        z0=draw(st.floats(0.001, 0.3, **f)),
        L=sign * l_mag,
        ustar=draw(st.floats(0.1, 1.0, **f)),
        umean=draw(st.floats(0.5, 8.0, **f)),
        wind_dir=draw(st.floats(0.0, 360.0, **f)),
        h=draw(st.floats(300.0, 2000.0, **f)),
        sigma_v=draw(st.floats(0.1, 1.5, **f)),
    )


@given(inputs=_valid_inputs())
@settings(
    max_examples=20,
    deadline=None,  # per-example runtime varies with stability; no fixed deadline
    suppress_health_check=[HealthCheck.too_slow],
)
def test_no_naninf_escapes_for_plausible_inputs(inputs: lg.FootprintInputs) -> None:
    """Across plausible inputs, a run never leaks NaN/inf into its outputs."""
    config = lg.ModelConfig(
        n_particles=120,
        dt_factor=0.05,
        t_max=45.0,
        rebound_height=0.5,
        seed=20240601,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # parameterized-turbulence notice is expected
        result = lg.FootprintModel(inputs, grid=_PROPERTY_GRID, config=config).run()

    density = result.density
    assert np.all(np.isfinite(density)), "density must contain no NaN/inf"
    assert np.all(density >= 0.0), "density must be non-negative"
    # Scalar metrics may be NaN when no particle reaches the surface (a valid
    # "no touchdowns" signal), but must never be +/-inf.
    assert not np.isinf(result.x_peak)
    assert not np.isinf(result.mc_noise)
    if result.n_touchdowns > 0:
        assert np.isfinite(result.x_peak)
        assert np.isfinite(result.mc_noise) and result.mc_noise > 0.0
        integral = float(density.sum()) * _PROPERTY_GRID.dx * _PROPERTY_GRID.dy
        assert integral == pytest.approx(1.0, abs=1e-9)


# Each label corrupts exactly one field of an otherwise-valid record so that
# validate_inputs must raise. The comment records which guard should fire.
_VIOLATIONS = (
    "zm_nonpositive",  # zm <= 0
    "ustar_nonpositive",  # ustar <= 0
    "h_nonpositive",  # h <= 0
    "sigma_v_nonpositive",  # sigma_v <= 0
    "above_bl_top",  # zm_eff >= h
    "below_roughness",  # zm_eff <= z0
)


@st.composite
def _invalid_inputs(draw: st.DrawFn) -> lg.FootprintInputs:
    """Draw inputs that violate exactly one physical constraint."""
    f = dict(allow_nan=False, allow_infinity=False)
    base: dict[str, float] = dict(
        zm=3.0,
        z0=0.03,
        d=0.0,
        L=-50.0,
        ustar=0.3,
        umean=2.0,
        wind_dir=200.0,
        h=1000.0,
        sigma_v=0.5,
    )
    kind = draw(st.sampled_from(_VIOLATIONS))
    if kind == "zm_nonpositive":
        base["zm"] = draw(st.floats(-50.0, 0.0, **f))
    elif kind == "ustar_nonpositive":
        base["ustar"] = draw(st.floats(-2.0, 0.0, **f))
    elif kind == "h_nonpositive":
        base["h"] = draw(st.floats(-100.0, 0.0, **f))
    elif kind == "sigma_v_nonpositive":
        base["sigma_v"] = draw(st.floats(-2.0, 0.0, **f))
    elif kind == "above_bl_top":
        base["h"] = draw(st.floats(0.1, 2.0, **f))  # below zm_eff = 3.0
    elif kind == "below_roughness":
        base["z0"] = draw(st.floats(3.0, 10.0, **f))  # at/above zm_eff = 3.0
    return lg.FootprintInputs(**base)


@given(inputs=_invalid_inputs())
@settings(max_examples=60, deadline=None)
def test_validation_rejects_unphysical_inputs(inputs: lg.FootprintInputs) -> None:
    """Every single-constraint violation is caught by ``validate_inputs``."""
    with pytest.raises(ValueError):
        validate_inputs(inputs)


if __name__ == "__main__":  # pragma: no cover - manual regeneration entry point
    regenerate()
