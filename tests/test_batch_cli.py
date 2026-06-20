"""Tests for batch ingestion (:func:`lagranged.model.run_batch`) and the CLI.

Kept fast: small particle counts, a raised surface contact height and a short
``t_max`` (mirroring tests/test_model_integration.py) so the full
validate → simulate → touchdown → grid → contour chain runs per record quickly.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

import lagranged as lg
from lagranged import cli
from lagranged.model import run_batch


def _fast_config(**overrides) -> lg.ModelConfig:
    base = dict(n_particles=300, dt_factor=0.05, t_max=90.0, rebound_height=0.5, seed=0)
    base.update(overrides)
    return lg.ModelConfig(**base)


def _synthetic_df(n: int = 3) -> pd.DataFrame:
    """A tiny EC-style table whose columns map onto FootprintInputs fields."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "zm": np.full(n, 3.0),
            "z0": np.full(n, 0.03),
            "d": np.full(n, 0.2),
            "L": np.full(n, -10_000.0),
            "ustar": 0.30 + 0.05 * rng.random(n),
            "umean": np.full(n, 2.4),
            "wind_dir": 180.0 + 30.0 * rng.random(n),
            "h": np.full(n, 1000.0),
            "sigma_v": np.full(n, 0.6),
        }
    )


# --------------------------------------------------------------------------- #
# run_batch
# --------------------------------------------------------------------------- #
def test_run_batch_returns_n_results(small_grid):
    """A clean N-row DataFrame yields N FootprintResults, each a valid footprint."""
    df = _synthetic_df(4)
    results = run_batch(df, grid=small_grid, config=_fast_config())

    assert len(results) == len(df)
    for idx, result in results.items():
        assert isinstance(result, lg.FootprintResult)
        assert result.n_touchdowns > 0
        integral = float(result.density.sum()) * small_grid.dx * small_grid.dy
        assert integral == pytest.approx(1.0, abs=1e-9)
        assert idx in df.index


def test_run_batch_is_keyed_by_record_index(small_grid):
    """Results are keyed by the source DataFrame index (here a custom label)."""
    df = _synthetic_df(2)
    df.index = ["2020-06-01T12:00", "2020-06-01T12:30"]
    results = run_batch(df, grid=small_grid, config=_fast_config())
    assert set(results) == set(df.index)


def test_run_batch_skips_bad_and_flagged_rows(small_grid):
    """Rows that fail QC (NaN required field, bad qc_flag) are dropped with a warning."""
    df = _synthetic_df(4)
    df.loc[df.index[1], "ustar"] = np.nan  # non-finite required field -> skip
    df["qc_flag"] = 0
    df.loc[df.index[2], "qc_flag"] = 2  # flagged poor quality -> skip

    with pytest.warns(UserWarning):
        results = run_batch(df, grid=small_grid, config=_fast_config())

    assert len(results) == 2
    assert df.index[1] not in results
    assert df.index[2] not in results


def test_run_batch_empty_frame(small_grid):
    """An empty table yields an empty result mapping (no crash)."""
    df = _synthetic_df(0)
    assert run_batch(df, grid=small_grid, config=_fast_config()) == {}


# --------------------------------------------------------------------------- #
# Grid-spec parsing
# --------------------------------------------------------------------------- #
def test_parse_grid_spec_basic():
    grid = cli.parse_grid_spec("400x400@2m")
    assert (grid.nx, grid.ny) == (400, 400)
    assert grid.dx == pytest.approx(2.0)
    assert grid.dy == pytest.approx(2.0)
    # Centered on the receptor at the origin.
    assert grid.x0 == pytest.approx(-400.0)
    assert grid.y0 == pytest.approx(-400.0)


def test_parse_grid_spec_variants():
    g = cli.parse_grid_spec("100x200@5")  # no trailing 'm', non-square extent
    assert (g.nx, g.ny) == (100, 200)
    assert g.dx == pytest.approx(5.0) and g.dy == pytest.approx(5.0)

    g2 = cli.parse_grid_spec("256X256@1.5m")  # capital X, fractional resolution
    assert (g2.nx, g2.ny) == (256, 256)
    assert g2.dx == pytest.approx(1.5)


@pytest.mark.parametrize("bad", ["400@2m", "400x400", "axb@2m", "400x400@", ""])
def test_parse_grid_spec_rejects_garbage(bad):
    with pytest.raises(ValueError):
        cli.parse_grid_spec(bad)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def test_cli_version(capsys):
    """``lagranged --version`` prints the version and exits 0."""
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert lg.__version__ in capsys.readouterr().out


def test_cli_no_command_prints_help(capsys):
    assert cli.main([]) == 0
    assert "usage" in capsys.readouterr().out.lower()


def _sample_run_config() -> dict:
    return {
        "inputs": {
            "zm": 3.0,
            "z0": 0.03,
            "d": 0.2,
            "L": -10_000.0,
            "ustar": 0.35,
            "umean": 2.4,
            "wind_dir": 210.0,
            "h": 1000.0,
            "sigma_v": 0.6,
        },
        "grid": {"nx": 50, "ny": 50, "dx": 4.0, "dy": 4.0, "x0": -100.0, "y0": -100.0},
        "config": {
            "n_particles": 300,
            "dt_factor": 0.05,
            "t_max": 90.0,
            "rebound_height": 0.5,
            "seed": 0,
        },
    }


def test_cli_run_produces_output_file(tmp_path):
    """``lagranged run`` reads a config and writes a footprint file."""
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps(_sample_run_config()), encoding="utf-8")
    out_path = tmp_path / "footprint.npz"

    rc = cli.main(["run", "--config", str(cfg_path), "--out", str(out_path)])

    assert rc == 0
    assert out_path.exists()
    with np.load(out_path) as data:
        assert data["density"].shape == (50, 50)
        # Density is normalized to integrate to 1 over the grid.
        assert float(data["density"].sum()) * 4.0 * 4.0 == pytest.approx(1.0, abs=1e-9)


def test_cli_run_accepts_grid_spec_string(tmp_path):
    """The 'grid' block may be a spec string instead of a DomainGrid mapping."""
    cfg = _sample_run_config()
    cfg["grid"] = "40x40@5m"
    cfg_path = tmp_path / "run.json"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    out_path = tmp_path / "footprint.npy"

    assert cli.main(["run", "--config", str(cfg_path), "--out", str(out_path)]) == 0
    assert np.load(out_path).shape == (40, 40)
