"""Headless example: compute a single footprint and save a figure.

Releases a backward Lagrangian stochastic ensemble for a near-neutral surface
layer, accumulates the touchdowns into a normalized footprint density, overlays
the 50/80/90 % cumulative source-area contours, and writes
``footprint_neutral.png`` next to the working directory.

Run with::

    python examples/scripts/run_single.py
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; no display required

import matplotlib.pyplot as plt  # noqa: E402

import lagranged as lg  # noqa: E402

OUTPUT = "footprint_neutral.png"


def main() -> None:
    print(f"lagranged version: {lg.__version__}")

    grid = lg.DomainGrid(nx=200, ny=200, dx=2.0, dy=2.0, x0=-200.0, y0=-200.0)

    result = lg.compute_footprint(
        zm=3.0,
        z0=0.03,
        d=0.2,
        L=-10_000.0,  # effectively neutral
        ustar=0.35,
        umean=2.4,
        wind_dir=210.0,
        h=1000.0,
        sigma_v=0.6,
        grid=grid,
        n_particles=6_000,
        seed=42,
        # A modestly raised surface-contact height keeps this demo's adaptive
        # time step out of the (expensive) near-z0 stiff regime so it runs in
        # ~1 min; lower it toward z0 for a production-quality footprint.
        rebound_height=0.2,
        t_max=600.0,
    )

    print(
        f"touchdowns = {result.n_touchdowns}, "
        f"x_peak = {result.x_peak:.1f} m, "
        f"mc_noise = {result.mc_noise:.3f}"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    lg.plot_footprint(result, ax=ax)
    lg.plot_contours(result, ax=ax, colors="white", linewidths=0.8)
    ax.set_title("Near-neutral flux footprint (upwind frame)")
    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=150)
    print(f"Saved {OUTPUT}")


if __name__ == "__main__":
    main()
