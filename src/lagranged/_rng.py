"""Centralized, seedable random-number generation for reproducibility.

All stochastic components route through :func:`get_rng` so that a single
``seed`` on :class:`lagranged.config.ModelConfig` fully determines a run.
"""

from __future__ import annotations

import numpy as np


def get_rng(seed: int | None = None) -> np.random.Generator:
    """Return a NumPy :class:`~numpy.random.Generator`.

    Parameters
    ----------
    seed:
        Seed for reproducibility. ``None`` draws fresh entropy from the OS.
    """
    return np.random.default_rng(seed)
