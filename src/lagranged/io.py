"""Tabular eddy-covariance ingestion and batch helpers.

Reads EC processor output (EddyPro / AmeriFlux-style CSV) into records that map
onto :class:`lagranged.inputs.FootprintInputs`, via a column-mapping dict.
"""

from __future__ import annotations

# Default column mapping for EddyPro "full output" style files.
# Maps FootprintInputs field -> source column name.
EDDYPRO_MAP: dict[str, str] = {
    "zm": "zm",
    "L": "L",
    "ustar": "u*",
    "wind_dir": "wind_dir",
    "h": "boundary_layer_height",
    "sigma_v": "v_var",  # note: variance vs. std — see read_ec_csv docstring
    "umean": "wind_speed",
}


def read_ec_csv(path: str, mapping: dict[str, str] | None = None, **read_csv_kwargs):
    """Read an EC CSV into a DataFrame with columns renamed to input fields.

    Parameters
    ----------
    path:
        Path to the CSV file.
    mapping:
        ``{input_field: source_column}``. Defaults to :data:`EDDYPRO_MAP`.

    Notes
    -----
    Some processors report variances rather than standard deviations; convert
    (e.g. ``sigma_v = sqrt(v_var)``) before passing rows to the model.
    """
    import pandas as pd

    mapping = mapping or EDDYPRO_MAP
    df = pd.read_csv(path, **read_csv_kwargs)
    rename = {src: field for field, src in mapping.items() if src in df.columns}
    return df.rename(columns=rename)


def iter_records(df):  # pragma: no cover - thin helper
    """Yield ``(index, row-dict)`` pairs suitable for FootprintInputs(**row)."""
    for idx, row in df.iterrows():
        yield idx, row.to_dict()
