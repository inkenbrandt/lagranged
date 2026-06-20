"""Smoke tests: the package imports, exposes its version, and core objects work."""

from __future__ import annotations

import re

import numpy as np
import pytest

import lagranged as lg


def test_version_string():
    assert isinstance(lg.__version__, str)
    assert re.match(r"^\d+\.\d+\.\d+", lg.__version__)


def test_public_api_present():
    for name in (
        "FootprintInputs",
        "TowerTurbulence",
        "ReynoldsStress",
        "ModelConfig",
        "DomainGrid",
        "FootprintModel",
        "FootprintResult",
        "compute_footprint",
        "run_batch",
        "plot_footprint",
        "plot_contours",
    ):
        assert hasattr(lg, name), f"missing public export: {name}"


def test_lazy_submodules():
    assert lg.io is not None
    # geo import should succeed even without the optional stack installed
    assert lg.geo is not None


def test_inputs_zm_eff(neutral_inputs):
    assert neutral_inputs.zm_eff == pytest.approx(3.0 - 0.2)


def test_reynolds_stress_from_components_is_psd():
    rs = lg.ReynoldsStress.from_components(su=0.7, sv=0.6, sw=0.4, cuw=-0.12)
    assert rs.matrix.shape == (3, 3)
    assert np.all(np.linalg.eigvalsh(rs.matrix) >= -1e-9)


def test_reynolds_stress_rejects_asymmetric():
    bad = np.array([[1.0, 0.5, 0.0], [0.1, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError):
        lg.ReynoldsStress(matrix=bad)


def test_model_runs_end_to_end(neutral_inputs, small_grid):
    cfg = lg.ModelConfig(n_particles=200, dt_factor=0.05, t_max=90.0, rebound_height=0.5, seed=0)
    model = lg.FootprintModel(neutral_inputs, grid=small_grid, config=cfg)
    result = model.run()
    assert isinstance(result, lg.FootprintResult)
    assert result.density.shape == (small_grid.ny, small_grid.nx)
    assert np.all(np.isfinite(result.density))
