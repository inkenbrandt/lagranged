"""Command-line interface: ``lagranged ...``.

Thin wrapper over the public API. Uses the stdlib ``argparse`` so the base
install has no CLI dependency; a richer ``typer`` front-end is planned behind
the ``lagranged[cli]`` extra.

Two subcommands::

    lagranged run   --config run.yaml --out footprint.tif
    lagranged batch data.csv --grid 400x400@2m --out results/

The ``run`` config is a YAML or JSON document with an ``inputs`` block, a
``grid`` block (a mapping of :class:`~lagranged.config.DomainGrid` fields *or* a
grid spec string such as ``"400x400@2m"``), and optional ``config`` (numerics)
and ``turbulence`` (measured σ) blocks. Output format is inferred from the
``--out`` extension (``.tif``/``.tiff`` GeoTIFF, ``.nc`` netCDF, ``.npz``/
``.npy`` numpy).
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np

from . import __version__
from .config import DomainGrid, ModelConfig
from .inputs import FootprintInputs, TowerTurbulence
from .model import FootprintModel, run_batch
from .results import FootprintResult

# e.g. "400x400@2m", "100 x 200 @ 5", "256X256@1.5m" — <nx> x <ny> @ <res>[m].
_GRID_SPEC_RE = re.compile(
    r"^\s*(\d+)\s*[xX]\s*(\d+)\s*@\s*([0-9]*\.?[0-9]+)\s*m?\s*$",
)


def parse_grid_spec(spec: str) -> DomainGrid:
    """Parse a ``"<nx>x<ny>@<res>[m]"`` grid spec into a centered :class:`DomainGrid`.

    The resolution applies to both axes (square cells). The grid is centered on
    the receptor, i.e. ``x0 = -nx*res/2`` and ``y0 = -ny*res/2``, so the tower at
    the model origin sits at the middle of the domain.

    Examples
    --------
    >>> parse_grid_spec("400x400@2m")
    DomainGrid(nx=400, ny=400, dx=2.0, dy=2.0, x0=-400.0, y0=-400.0, ...)
    """
    match = _GRID_SPEC_RE.match(spec)
    if match is None:
        raise ValueError(
            f"invalid grid spec {spec!r}; expected '<nx>x<ny>@<res>[m]', e.g. '400x400@2m'."
        )
    nx, ny = int(match.group(1)), int(match.group(2))
    res = float(match.group(3))
    return DomainGrid(nx=nx, ny=ny, dx=res, dy=res, x0=-nx * res / 2.0, y0=-ny * res / 2.0)


def load_run_config(path: str | os.PathLike) -> dict:
    """Load a run config from a YAML or JSON file.

    ``.json`` is parsed with the stdlib; ``.yaml``/``.yml`` (and anything else)
    needs PyYAML — a clear :class:`ImportError` is raised if it is missing.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as exc:  # pragma: no cover - depends on install
            raise ImportError(
                f"Reading {path.suffix or 'YAML'} configs needs PyYAML "
                "(`pip install pyyaml`), or use a .json config."
            ) from exc
        data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise ValueError(f"config {str(path)!r} must be a mapping, got {type(data).__name__}.")
    return data


def _grid_from_config(grid_cfg) -> DomainGrid:
    """Build a grid from either a spec string or a mapping of DomainGrid fields."""
    if isinstance(grid_cfg, str):
        return parse_grid_spec(grid_cfg)
    if isinstance(grid_cfg, dict):
        return DomainGrid(**grid_cfg)
    raise ValueError("'grid' must be a grid-spec string or a mapping of DomainGrid fields.")


def _build_run_objects(
    data: dict,
) -> tuple[FootprintInputs, DomainGrid, ModelConfig, TowerTurbulence | None]:
    """Assemble the model objects described by a run-config mapping."""
    if "inputs" not in data:
        raise ValueError("config is missing the required 'inputs' block.")
    if "grid" not in data:
        raise ValueError("config is missing the required 'grid' block.")

    inputs = FootprintInputs(**data["inputs"])
    grid = _grid_from_config(data["grid"])
    config = ModelConfig(**data.get("config", {}))
    turbulence_cfg = data.get("turbulence")
    turbulence = TowerTurbulence(**turbulence_cfg) if turbulence_cfg else None
    return inputs, grid, config, turbulence


def export_result(result: FootprintResult, grid: DomainGrid, path: str | os.PathLike) -> None:
    """Write a :class:`FootprintResult` to ``path``, dispatching on extension.

    ``.tif``/``.tiff`` → georeferenced GeoTIFF (needs the ``geo`` extra);
    ``.nc`` → netCDF (via xarray); ``.npz`` → density + coordinates; ``.npy`` →
    density only.
    """
    path = str(path)
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        from . import geo

        geo.write_geotiff(result, grid, path)
    elif ext == ".nc":
        result.to_xarray().to_netcdf(path)
    elif ext == ".npz":
        np.savez(path, density=result.density, x=result.x, y=result.y)
    elif ext == ".npy":
        np.save(path, result.density)
    else:
        raise ValueError(
            f"unsupported output extension {ext!r}; use .tif/.tiff, .nc, .npz or .npy."
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lagranged",
        description="Backward Lagrangian stochastic flux-footprint model.",
    )
    parser.add_argument("--version", action="version", version=f"lagranged {__version__}")
    sub = parser.add_subparsers(dest="command")

    run = sub.add_parser("run", help="Run a single footprint from a config file.")
    run.add_argument("--config", required=True, help="Path to a YAML/JSON run config.")
    run.add_argument("--out", default="footprint.tif", help="Output raster path.")

    batch = sub.add_parser("batch", help="Run a footprint per row of an EC CSV.")
    batch.add_argument("csv", help="Path to the EC CSV file.")
    batch.add_argument("--out", default="results/", help="Output directory.")
    batch.add_argument("--grid", default="400x400@2m", help="Grid spec, e.g. 400x400@2m.")
    batch.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON file of ModelConfig numerics applied to every record.",
    )
    batch.add_argument(
        "--ext",
        default=".tif",
        help="Per-record output extension (.tif/.nc/.npz/.npy). Default: .tif",
    )

    return parser


def _cmd_run(args: argparse.Namespace) -> int:
    data = load_run_config(args.config)
    inputs, grid, config, turbulence = _build_run_objects(data)
    result = FootprintModel(inputs, grid=grid, turbulence=turbulence, config=config).run()
    export_result(result, grid, args.out)
    print(
        f"Wrote footprint -> {args.out} "
        f"(n_touchdowns={result.n_touchdowns}, x_peak={result.x_peak:.1f} m)"
    )
    return 0


def _cmd_batch(args: argparse.Namespace) -> int:
    from . import io as _io

    df = _io.read_ec_csv(args.csv)
    grid = parse_grid_spec(args.grid)
    config = ModelConfig(**load_run_config(args.config).get("config", {})) if args.config else None

    results = run_batch(df, grid=grid, config=config, progress=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ext = args.ext if args.ext.startswith(".") else f".{args.ext}"
    for idx, result in results.items():
        export_result(result, grid, out_dir / f"footprint_{idx}{ext}")

    print(f"Wrote {len(results)} footprint(s) -> {out_dir}{os.sep}")
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return 0
    if args.command == "run":
        return _cmd_run(args)
    if args.command == "batch":
        return _cmd_batch(args)

    parser.error(f"unknown command {args.command!r}")  # pragma: no cover
    return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
