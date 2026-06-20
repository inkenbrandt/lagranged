# `tests/data/`

Fixtures and recorded outputs used by the test suite.

## `eddypro_sample.csv` — synthetic EC table

A tiny EddyPro/AmeriFlux-style CSV (6 half-hourly records) used by
`examples/03_tower_csv_ingest.ipynb`. Columns follow EddyPro "full output" names
(`u*`, `boundary_layer_height`, `v_var`, `wind_speed`, …) so `lagranged.io.EDDYPRO_MAP`
maps them onto `FootprintInputs` fields; `v_var` is a **variance** (square-root it to
get `sigma_v`). Two rows carry `qc_flag = 2` to exercise quality-control filtering.

## `golden/` — regression footprints

`golden/footprint_<case>.npy` holds the recorded `(ny, nx)` footprint **density**
array produced by `lagranged.FootprintModel` for a fixed `(stability, seed)` case.
`tests/test_regression.py::test_footprint_matches_golden` re-runs the model and
asserts `np.allclose` against each stored array, so any unintended change in the
numerics or physics shows up as a failing regression test.

The three cases (defined in `tests/test_regression.py`):

| case       | Obukhov length `L` | seed | regime              |
|------------|--------------------|------|---------------------|
| `neutral`  | −10 000 m          | 0    | effectively neutral |
| `unstable` | −15 m              | 1    | strongly convective |
| `stable`   | +30 m              | 2    | stably stratified   |

All three share the base inputs and grid in `tests/test_regression.py`
(`_BASE_INPUTS`, `_GOLDEN_GRID`) and a small, short ensemble (`_golden_config`) so
the regression stays fast.

### Regenerating the goldens

The goldens are expected to change **only** when the physics/numerics change on
purpose. Regenerate them deliberately — never edit the `.npy` files by hand:

```bash
# from the project root, with the dev environment active:
python scripts/regenerate_goldens.py
# or, if you use nox:
nox -s regenerate_goldens
```

Then inspect the diff before committing:

```bash
git diff -- tests/data/golden
```

A non-empty diff is expected after a deliberate change and confirms the model
output moved; an empty diff means nothing actually changed. Commit the updated
`.npy` files alongside the code change and note in the commit message why the
footprints moved.

> A fixed seed reproduces a run bit-for-bit on the generating platform, so a
> golden only drifts when the model genuinely changes. The regression test uses a
> small `rtol`/`atol` that absorbs harmless floating-point reassociation across
> numpy patch versions but fails on any real change.
