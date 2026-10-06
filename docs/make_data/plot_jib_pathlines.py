"""
Plot pathlines over the observed speed at Jakobshavn Isbræ.

Usage::

    python docs/make_data/plot_jib_pathlines.py flowlines.gpkg jib_pathlines.png

"""

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from importlib.resources import files
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from glacier_flow_tools.utils import register_colormaps

DATA = files("glacier_flow_tools.data")


def plot(pathlines_file: Path, outfile: Path, velocity_file: Path, gates_file: Path) -> None:
    """
    Plot the pathlines and the flux gate over the speed and save the figure.

    Parameters
    ----------
    pathlines_file : Path
        The pathlines computed with ``compute_pathlines``.
    outfile : Path
        The figure to write.
    velocity_file : Path
        The velocities, with the variables ``vx`` and ``vy``.
    gates_file : Path
        The flux gates the pathlines start from.
    """
    # The colormaps that ship with the package, among them "speed_colorblind".
    register_colormaps()

    ds = xr.open_dataset(velocity_file)
    speed = np.hypot(ds["vx"], ds["vy"])
    pathlines = gp.read_file(pathlines_file)
    gates = gp.read_file(gates_file)

    # The projection of the data, NSIDC Sea Ice Polar Stereographic North (EPSG:3413).
    crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70)

    fig = plt.figure(figsize=(6.4, 5.2))
    ax = fig.add_subplot(111, projection=crs)
    im = ax.pcolormesh(
        ds["x"],
        ds["y"],
        speed,
        cmap="speed_colorblind",
        vmin=10,
        vmax=1500,
        shading="auto",
        transform=crs,
        rasterized=True,
    )
    pathlines.plot(ax=ax, color="k", markersize=0.2, transform=crs)
    gates.plot(ax=ax, color="w", lw=2, transform=crs)

    ax.set_extent([ds["x"].min(), ds["x"].max(), ds["y"].min(), ds["y"].max()], crs=crs)
    gl = ax.gridlines(
        draw_labels=True, x_inline=False, y_inline=False, rotate_labels=False, color="k", linestyle=":", linewidth=0.75
    )
    gl.right_labels = False
    gl.bottom_labels = False
    fig.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.8, extend="both", ticks=[10, 250, 500, 750, 1000, 1500]
    ).set_label("Speed (m/yr)")

    fig.savefig(outfile, dpi=200, bbox_inches="tight")


def main() -> None:
    """
    Command line interface.
    """
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Plot pathlines over the observed speed at Jakobshavn Isbræ."
    parser.add_argument("pathlines_file", help="Pathlines computed with compute_pathlines.", type=Path)
    parser.add_argument("outfile", help="Figure to write.", type=Path)
    parser.add_argument("--velocity_file", help="Velocities.", type=Path, default=DATA.joinpath("jib_velocities.nc"))
    parser.add_argument("--gates_file", help="Flux gates.", type=Path, default=DATA.joinpath("jib-flux-gates.gpkg"))
    options = parser.parse_args()
    plot(options.pathlines_file, options.outfile, options.velocity_file, options.gates_file)


if __name__ == "__main__":
    main()
