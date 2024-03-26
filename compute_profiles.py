# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Calculate profiles
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import List, Union

import cartopy.crs as ccrs
import dask_geopandas
import fsspec
import geopandas as gp
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import pylab as plt
import xarray as xr
from dask import dataframe as dd
from joblib import Parallel, delayed
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from tqdm.auto import tqdm

from pypism import profiles
from pypism.hillshade import hillshade
from pypism.profiles import process_profile, process_profile_xr
from pypism.utils import blend_multiply, preprocess_nc, qgis2cmap, tqdm_joblib


def merge_on_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
    """
    Merge two pd.DataFrame on interection keys
    """
    intersection_keys = list(set(df1.columns) & set(df2.columns))
    return dd.merge(df1, df2, on=intersection_keys)


def round(x: float, mult: int = 1000):
    return np.round(x / mult) * mult


def figure_extent(x_c: float, y_c: float, x_e: float = 50_000, y_e: float = 50_000):

    return {"x": slice(x_c - x_e / 2, x_c + x_e / 2), "y": slice(y_c + y_e / 2, y_c - y_e / 2)}


def plot_profile(ds: xr.Dataset, result_dir: Path, alpha: float = 0.0, sigma: float = 1.0):

    from pypism.profiles import CustomDatasetMethods

    fig = ds.profiles.plot(palette="Greens", sigma=sigma, alpha=alpha)
    profile_name = ds["profile_name"].values[0]
    fig.savefig(result_dir / f"{profile_name}_profile.pdf")
    plt.close()
    del fig


def plot_glacier(
    profile: gp.GeoDataFrame,
    surface: xr.DataArray,
    overlay: xr.DataArray,
    result_dir: Union[str, Path],
    cmap="viridis",
    vmin: float = 10,
    vmax: float = 1500,
    ticks: Union[List[float], np.ndarray] = [10, 100, 250, 500, 750, 1500],
):

    from pypism.profiles import CustomDatasetMethods

    def get_extent(ds: xr.DataArray):
        return [ds["x"].values[0], ds["x"].values[-1], ds["y"].values[-1], ds["y"].values[0]]

    profile_centroid = gp.GeoDataFrame(profile, geometry=profile.geometry.centroid)
    glacier_name = profile.iloc[0]["profile_name"]
    exp_id = profile.iloc[0]["exp_id"]
    x_c = round(profile_centroid.geometry.x.values[0])
    y_c = round(profile_centroid.geometry.y.values[0])
    extent_slice = figure_extent(x_c, y_c)
    crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)
    # Shade from the northwest, with the sun 45 degrees from horizontal
    light_source = LightSource(azdeg=315, altdeg=45)
    glacier_overlay = overlay.sel(**extent_slice)
    glacier_surface = surface.interp_like(glacier_overlay)

    extent = get_extent(glacier_overlay)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    v = mapper.to_rgba(glacier_overlay.to_numpy())
    z = glacier_surface.to_numpy()
    fig = plt.figure(figsize=(6.2, 6.2))
    ax = fig.add_subplot(111, projection=crs)
    rgb = light_source.shade_rgb(v, elevation=z, vert_exag=0.01, blend_mode=blend_multiply)
    # Use a proxy artist for the colorbar...
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
    im.remove()
    corr = ax.imshow(
        v,
        vmin=0,
        vmax=1,
        cmap="RdYlGn_r",
    )
    corr.remove()
    ax.imshow(rgb, extent=extent, origin="upper", transform=crs)
    profile.plot(ax=ax, color="k", lw=1)
    profile_centroid.plot(
        column="pearson_r", vmin=0, vmax=1, cmap="RdYlGn_r", markersize=50, legend=False, missing_kwds={}, ax=ax
    )
    ax.annotate(glacier_name, (x_c, y_c), (10, 10), xycoords="data", textcoords="offset points")
    ax.coastlines(linewidth=0.25, resolution="10m")
    ax.gridlines(
        draw_labels={"top": "x", "right": "y"},
        dms=True,
        xlocs=np.arange(-50, 0, 1),
        ylocs=np.arange(50, 88, 1),
        x_inline=False,
        y_inline=False,
        rotate_labels=20,
        ls="dotted",
        color="k",
    )

    ax.set_extent(extent, crs=crs)
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.025, label=overlay.units, extend="max", ticks=ticks)
    fig.colorbar(
        corr, ax=ax, shrink=0.5, pad=0.025, label="Pearson $r$ (1)", orientation="horizontal", location="bottom"
    )
    fig.savefig(result_dir / Path(f"{glacier_name}_{exp_id}_speed.pdf"))
    plt.close()
    del fig


if __name__ == "__main__":

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    parser.add_argument(
        "--result_dir",
        help="""Path to where output is saved. Directory will be created if needed.""",
        default=Path("./results"),
    )
    parser.add_argument(
        "--obs_url",
        help="""Path to velocity dataset.""",
        default=None,
    )
    parser.add_argument(
        "--thickness_url",
        help="""Path to thickness dataset.""",
        default=None,
    )
    parser.add_argument("--alpha", help="""Scale observational error. Default=0.""", default=0.0, type=float)
    parser.add_argument(
        "--sigma",
        help="""Sigma multiplier observational error. Default=1. (i.e. error is 1 standard deviation""",
        default=1.0,
        type=float,
    )
    parser.add_argument("--segmentize", help="""Profile resolution in meters Default=200m.""", default=200, type=float)
    parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    parser.add_argument("INFILES", nargs="*", help="PISM experiment files", default=None)
    parser.add_argument("--profiles_url", help="""Path to profiles.""", default=None, type=str)

    options = parser.parse_args()
    result_dir = options.result_dir
    result_dir.mkdir(parents=True, exist_ok=True)
    alpha = options.alpha
    sigma = options.sigma
    profile_resolution = options.segmentize

    profiles_path = Path(options.profiles_url)
    profiles_gp = gp.read_file(profiles_path).rename(columns={"id": "profile_id", "name": "profile_name"})
    geom = profiles_gp.segmentize(profile_resolution)
    profiles_gp = gp.GeoDataFrame(profiles_gp, geometry=geom)

    crs = "EPSG:3413"
    fs = fsspec.filesystem("https")
    if options.obs_url:
        is_url = options.obs_url.split(":")[0] == ("https" or "http")
        if is_url:
            obs_file = fs.open(options.obs_url)
        else:
            obs_file = Path(options.obs_url)

        obs_ds = xr.open_dataset(obs_file, chunks="auto")

    if options.thickness_url:
        thickness_file = Path(options.thickness_url)
        thickness_ds = xr.open_dataset(thickness_file, chunks="auto")

    exp_files = [Path(x) for x in options.INFILES]
    exp_ds = xr.open_mfdataset(
        exp_files, preprocess=preprocess_nc, concat_dim="exp_id", combine="nested", chunks="auto", parallel=True
    )

    qgis_colormap = Path("data/speed-colorblind.txt")
    cmap = qgis2cmap(qgis_colormap, name="speeds")

    stats: List[str] = ["rmsd", "pearson_r"]
    start = time.process_time()

    with tqdm_joblib(tqdm(desc="Extracting profiles", total=len(profiles_gp))) as progress_bar:
        obs_sims_profiles = Parallel(n_jobs=options.n_jobs, timeout=1_000)(
            delayed(process_profile_xr)(profile, obs_ds, exp_ds, stats=stats) for _, profile in profiles_gp.iterrows()
        )

    time_elapsed = time.process_time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    n_partitions = 2
    profiles_l = dask_geopandas.from_geopandas(
        gp.GeoDataFrame(profiles_gp, geometry=profiles_gp.geometry), npartitions=n_partitions
    )
    profiles_c = dask_geopandas.from_geopandas(
        gp.GeoDataFrame(profiles_gp, geometry=profiles_gp.geometry.centroid), npartitions=n_partitions
    )

    stats_profiles = dd.concat(
        [merge_on_intersection(profiles_l, p.mean(["profile_axis"]).to_dask_dataframe()) for p in obs_sims_profiles]
    )

    gris_ds = xr.open_dataset(Path("/Users/andy/Google Drive/My Drive/data/MCdataset/BedMachineGreenland-v5.nc"))

    # all_profiles = [xr.merge([obs_profile.squeeze(), sims_profile.squeeze()]) for obs_profile, sims_profile in zip(obs_profiles, sims_profiles)]

    start = time.process_time()
    #    stats_profiles_computed = stats_profiles.compute()
    with tqdm_joblib(tqdm(desc="Plotting glaciers", total=len(stats_profiles))) as progress_bar:
        Parallel(n_jobs=options.n_jobs)(
            delayed(plot_glacier)(
                gp.GeoDataFrame(pd.DataFrame(glacier).T, crs=crs),
                gris_ds["surface"],
                obs_ds["v"],
                result_dir,
                cmap=cmap,
            )
            for _, glacier in stats_profiles.iterrows()
        )
    time_elapsed = time.process_time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    # with tqdm_joblib(tqdm(desc="Plotting profiles", total=len(profiles_gp))) as progress_bar:
    #     Parallel(n_jobs=options.n_jobs)(
    #         delayed(plot_profile)(ds, result_dir, alpha=alpha, sigma=sigma)
    #         for ds in obs_sims_profiles
    #         )
