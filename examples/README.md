# Examples

Worked examples for `lagranged`. Every notebook is executed in CI via
[`nbmake`](https://github.com/treebeardtech/nbmake) so the examples can't silently
rot as the API evolves. Particle counts are kept small so they run quickly.

| File | What it shows |
|------|---------------|
| `scripts/run_single.py` | Minimal end-to-end API call for one neutral case (headless smoke test). |
| `01_synthetic_neutral.ipynb` | "Hello world": run + plot density/contours, read off `x_peak`. |
| `02_stability_sweep.ipynb` | Footprint contracting/expanding across stability; small-multiples. |
| `03_tower_csv_ingest.ipynb` | Load an EddyPro/AmeriFlux CSV with `lg.io`, map columns, filter QC. |
| `04_batch_geotiff_export.ipynb` | `run_batch` → climatological footprint → GeoTIFF + contour shapefile. |
| `05_measured_turbulence_reynolds.ipynb` | Measured σ + full covariance, `mode="reynolds"`. |

## Running

Headless script:

```bash
python examples/scripts/run_single.py
```

Execute and check every notebook (notebooks 04–05 need the geospatial extra):

```bash
pip install -e ".[dev,geo]"
pytest --nbmake examples/
```

`pytest --nbmake examples/*.ipynb` runs each notebook top-to-bottom and fails if any
cell errors. Notebook 03 reads the sample CSV at
[`tests/data/eddypro_sample.csv`](../tests/data/eddypro_sample.csv).
