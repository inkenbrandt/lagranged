"""lagranged — eddy-covariance flux footprints via a backward Lagrangian
stochastic (bLS) particle model.

Particles are released at the receptor (tower measurement height) and integrated
backward in time; their contacts with the ground ("touchdowns") build a 2-D
source-weight / probability-density field that, once normalized, is the flux
footprint.

This is a research-grade prototype, **not** a validated replacement for published
footprint models (Kljun FFP, Kormann–Meixner). Approximate parameterizations emit
``warnings``; see ``docs/limitations.md``.

Quick start
-----------
>>> import lagranged as lg
>>> print(lg.__version__)
0.2.0
"""

from __future__ import annotations

__version__ = "0.2.0"

# --- Core public objects (lightweight imports only; no heavy geo deps here) ---
from .config import DomainGrid, ModelConfig
from .inputs import FootprintInputs, ReynoldsStress, TowerTurbulence
from .model import FootprintModel, compute_footprint, run_batch
from .plotting import plot_contours, plot_footprint
from .results import FootprintResult

# ``geo`` and ``io`` are imported lazily via __getattr__ so that the heavy,
# optional geospatial stack is only required when actually used.

__all__ = [
    "__version__",
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
    "io",
    "geo",
]


def __getattr__(name: str):  # PEP 562 lazy submodule loading
    if name in ("io", "geo"):
        import importlib

        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
