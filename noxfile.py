"""Nox sessions for the lagranged package.

These are convenience wrappers; CI invokes the underlying tools directly. Run a
session with, e.g., ``nox -s tests`` or ``nox -s regenerate_goldens`` (install
nox first: ``pip install nox``).
"""

from __future__ import annotations

import nox

nox.options.sessions = ["tests"]


@nox.session(python=["3.10", "3.11", "3.12"])
def tests(session: nox.Session) -> None:
    """Run the test suite (extra args pass through, e.g. ``nox -s tests -- -m slow``)."""
    session.install("-e", ".[dev]")
    session.run("pytest", "-q", *session.posargs)


@nox.session(name="regenerate_goldens", python=False)
def regenerate_goldens(session: nox.Session) -> None:
    """Intentionally regenerate the golden footprint arrays (Phase 9 regression).

    Use only after a deliberate physics/numerics change. Review the resulting
    ``git diff -- tests/data/golden`` before committing.
    """
    session.run("python", "scripts/regenerate_goldens.py")
