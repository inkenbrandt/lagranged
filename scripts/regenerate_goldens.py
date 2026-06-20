#!/usr/bin/env python
"""Regenerate the stored golden footprint arrays for ``tests/test_regression.py``.

Run this **intentionally** — only when a deliberate change to the physics or
numerics makes the regression goldens out of date. The goldens are the recorded
output of :class:`lagranged.FootprintModel` for a few fixed ``(stability, seed)``
cases; overwriting them silently would defeat the regression test, so this is a
manual, reviewed step.

Procedure
---------
1. Make and commit the deliberate physics/numerics change.
2. From the project root, with the dev environment active, run::

       python scripts/regenerate_goldens.py

3. Inspect ``git diff -- tests/data/golden``. A non-empty diff is expected and
   confirms the change altered model output; an *empty* diff means nothing
   actually changed. Commit the updated ``.npy`` files alongside the change,
   noting in the commit message why the footprints moved.

The case definitions and numerics live in ``tests/test_regression.py`` (a single
source of truth shared with the test), so what is regenerated is exactly what the
regression test re-checks.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_TEST_MODULE = _ROOT / "tests" / "test_regression.py"


def _load_regression_module():
    """Import ``tests/test_regression.py`` by path (tests/ is not a package)."""
    spec = importlib.util.spec_from_file_location("_lagranged_regression", _TEST_MODULE)
    if spec is None or spec.loader is None:  # pragma: no cover - defensive
        raise RuntimeError(f"could not load {_TEST_MODULE}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    module = _load_regression_module()
    written = module.regenerate(verbose=True)
    print(f"\nRegenerated {len(written)} golden array(s) under {module.GOLDEN_DIR}.")
    print(
        "Review `git diff -- tests/data/golden` before committing: a non-empty "
        "diff means model output changed (expected after a deliberate physics edit)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
