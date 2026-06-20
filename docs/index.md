# lagranged

Eddy-covariance flux footprints via a **backward Lagrangian stochastic (bLS)**
particle model.

Particles are released at the receptor (tower measurement height) and integrated
backward in time; their contacts with the ground ("touchdowns") build a 2-D
source-weight / probability-density field that, once normalized, is the flux
footprint.

> ⚠️ **Research prototype.** This is not a validated replacement for published
> footprint models (Kljun FFP, Kormann–Meixner). Approximate parameterizations
> emit `warnings`. See [limitations](limitations.md).

## Contents

- [Quickstart](quickstart.md)
- [Theory & assumptions](theory.md)
- [API reference](api.md)
- [Limitations & caveats](limitations.md)

## Install

```bash
pip install -e .            # core
pip install -e ".[geo]"     # + geospatial export
pip install -e ".[dev]"     # + test/lint tooling
```
