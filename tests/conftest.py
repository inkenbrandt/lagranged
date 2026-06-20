"""Shared pytest fixtures."""

from __future__ import annotations

import pytest

import lagranged as lg


@pytest.fixture
def neutral_inputs() -> lg.FootprintInputs:
    """A simple near-neutral FFP-style input record."""
    return lg.FootprintInputs(
        zm=3.0,
        z0=0.03,
        d=0.2,
        L=-10_000.0,  # effectively neutral
        ustar=0.35,
        umean=2.4,
        wind_dir=210.0,
        h=1000.0,
        sigma_v=0.6,
    )


@pytest.fixture
def small_grid() -> lg.DomainGrid:
    return lg.DomainGrid(nx=50, ny=50, dx=4.0, dy=4.0, x0=-100.0, y0=-100.0)
