# Copyright (C) 2025 Andy Aschwanden
#
# This file is part of glacier-flow-tools.
#
# GLACIER-FLOW-TOOLS is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# GLACIER-FLOW-TOOLS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Extract along profiles and compute statistics.
"""

from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path

import cartopy.crs as ccrs
import cf_xarray.units  # pylint: disable=unused-import

import matplotlib
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from matplotlib.colors import ListedColormap
import matplotlib.pylab as plt
from shapely import get_coordinates

from glacier_flow_tools.utils import (
    blend_multiply,
    get_dataarray_extent,
    register_colormaps,
)

import numpy as np
import xarray as xr
from dask.distributed import Client, progress

xr.set_options(keep_attrs=True)


def plot_glacier(
    surface: xr.DataArray,
    overlay: xr.DataArray,
    vert_exag: float = 0.015,
    sealevel: float | None = None,
    cmap: str = "viridis",
    interactive: bool = False,
    title: str | None = None,
    vmin: float = 10,
    vmax: float = 1500,
    fontsize: float = 6,
    figwidth: float = 3.2,
    sealevel_color="#bdd7e7",
) -> plt.figure:
    """
    Plot a surface over a hillshade, add profile and correlation coefficient.

    This function plots a surface over a hillshade, adds a profile and correlation coefficient.
    The plot is saved as a PDF file in the specified result directory.

    Parameters
    ----------
    surface : xr.DataArray
        The surface to be plotted over the hillshade.
    overlay : xr.DataArray
        The overlay to be added to the plot.
    interactive : bool
        If False (default), use non-interactive matplotlib backend for plotting.
        Needed for distributed plottinging.
    cmap : str, optional
        The colormap to be used for the plot, by default "viridis".
    vmin : float, optional
        The minimum value for the colormap, by default 10.
    vmax : float, optional
        The maximum value for the colormap, by default 1500.
    ticks : Union[List[float], np.ndarray], optional
        The ticks to be used for the colorbar, by default [10, 100, 250, 500, 750, 1500].
    fontsize : float, optional
        The font size to be used for the plot, by default 6.
    figwidth : float, optional
        The width of the figure in inches, by default 3.2.

    Examples
    --------
    >>> plot_glacier(profile_series, surface, overlay, '/path/to/result_dir')
    """

    if interactive:
        # The standard backend is not thread-safe, but 'agg' works with the dask client.
        matplotlib.use("agg")
    else:
        matplotlib.use("module://matplotlib_inline.backend_inline")

    try:
        register_colormaps()
    except:
        pass

    extent = None
    plt.rcParams["font.size"] = fontsize
    cartopy_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)
    # Shade from the northwest, with the sun 45 degrees from horizontal
    light_source = LightSource(azdeg=315, altdeg=45)
    glacier_overlay = overlay
    glacier_surface = surface.interp_like(glacier_overlay)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    v = mapper.to_rgba(glacier_overlay.to_numpy())
    z = glacier_surface.to_numpy()

    ar = 1.0  # initial aspect ratio for first trial
    wi = figwidth  # width in inches
    hi = wi * ar  # height in inches

    fig = plt.figure(figsize=(wi, hi))
    ax = fig.add_subplot(111, projection=cartopy_crs)
    if sealevel is not None:
        if isinstance(sealevel, float):
            sealevel_da = xr.zeros_like(glacier_overlay) + sealevel
        else:
            sealevel_da = sealevel.interp_like(glacier_overlay)
        sl = ax.pcolormesh(sealevel_da, cmap=ListedColormap([sealevel_color]))
        sl.set_zorder(-1)
    rgb = light_source.shade_rgb(v, elevation=z, vert_exag=vert_exag, blend_mode=blend_multiply)
    # Use a proxy artist for the colorbar...
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
    im.remove()
    ax.imshow(rgb, extent=extent, origin="lower", transform=cartopy_crs)
    ax.set_title(title)

    # ax.set_extent(extent, crs=cartopy_crs)
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.025, label=overlay.units, extend="both")
    plt.draw()

    # Get proper ratio here
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    y2x_ratio = (ymax - ymin) / (xmax - xmin)
    fig.set_figheight(wi * y2x_ratio)
    fig.tight_layout()
    return fig


def plot_mapplane(surface, overlay, k: int = 0, p: str | Path = "result", **kwargs):
    title = surface.time.values
    fig = plot_glacier(surface, overlay, title=title, **kwargs)
    p = Path(p)
    p.mkdir(parents=True, exist_ok=True)
    fname = p / Path(f"frame_{k:04}")
    fig.savefig(fname, dpi=600)
    del fig


if __name__ == "__main__":
    __spec__ = None  # type: ignore

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Make a mapplane animation."
    parser.add_argument(
        "--result_dir",
        help="""Result directory.""",
        type=str,
        default="./results",
    )
    parser.add_argument(
        "FILE",
        help="""Input netCDF file""",
        nargs=1,
    )

    options, unknown = parser.parse_known_args()
    infile = options.FILE

    # x_bnds = [-212900, 291100]
    # y_bnds = [-2340650, -2016650]
    x_bnds = [None, None]
    y_bnds = [None, None]
    client = Client()
    ds = xr.open_dataset(infile[0]).sel({"x": slice(*x_bnds), "y": slice(*y_bnds)})
    print(f"Open client in browser: {client.dashboard_link}")
    ds["velsurf_mag"] = ds["velsurf_mag"].where(ds["thk"] > 10, other=np.nan)
    ds["usurf"] = ds["usurf"].where(ds["usurf"] > 0, other=np.nan)
    surfaces = [ds.isel({"time": k})["usurf"].load() for k, _ in enumerate(ds.time)]
    overlays = [ds.isel({"time": k})["velsurf_mag"].load() for k, _ in enumerate(ds.time)]

    futures = client.map(
        plot_mapplane,
        surfaces,
        overlays,
        range(ds.time.size),
        sealevel=0.0,
        p=options.result_dir,
        cmap="speed_colorblind",
    )
    progress(futures)
    client.gather(futures)
