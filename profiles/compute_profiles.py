# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
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
Calculate proifles and compute statistics along profiles.
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from pathlib import Path
from typing import List

import dask_geopandas
import geopandas as gp
import xarray as xr
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster, progress
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from glacier_flow_tools.profiles import plot_glacier, plot_profile, process_profile
from glacier_flow_tools.utils import (
    merge_on_intersection,
    preprocess_nc,
    qgis2cmap,
    tqdm_joblib,
)

if __name__ == "__main__":

    # set up the option parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    parser.add_argument(
        "--crs", help="""Coordinate reference system. Default is EPSG:3413.""", type=str, default="EPSG:3413"
    )
    parser.add_argument(
        "--result_dir",
        help="""Path to where output is saved. Directory will be created if needed.""",
        default=Path("./results"),
    )
    parser.add_argument(
        "--velocity_url",
        help="""Path to velocity dataset.""",
        default=None,
    )
    parser.add_argument(
        "--thickness_url",
        help="""Path to thickness dataset.""",
        default=None,
    )
    parser.add_argument(
        "--alpha",
        help="""Scale observational error. Use 0.05 to reproduce 'Commplex Outlet Glacier Flow Captured'. Default=0.""",
        default=0.0,
        type=float,
    )
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
    profile_result_dir = Path(options.result_dir)
    profile_result_dir.mkdir(parents=True, exist_ok=True)
    obs_scale_alpha = options.alpha
    crs = options.crs
    obs_sigma = options.sigma
    profile_resolution = options.segmentize

    profiles_path = Path(options.profiles_url)
    profiles_gp = gp.read_file(profiles_path).rename(columns={"id": "profile_id", "name": "profile_name"})
    geom = profiles_gp.segmentize(profile_resolution)
    profiles_gp = gp.GeoDataFrame(profiles_gp, geometry=geom)
    profiles_gp = profiles_gp[["profile_id", "profile_name", "geometry"]]

    velocity_file = Path(options.velocity_url)
    velocity_ds = xr.open_dataset(velocity_file, chunks="auto", decode_times=False)

    if options.thickness_url:
        thickness_file = Path(options.thickness_url)
        thickness_ds = xr.open_dataset(thickness_file, chunks="auto")

    print("Opening experiments")
    exp_files = [Path(x) for x in options.INFILES]
    start = time.time()
    exp_ds = xr.open_mfdataset(
        exp_files,
        preprocess=partial(preprocess_nc, drop_dims=["z", "zb"]),
        concat_dim="exp_id",
        combine="nested",
        chunks="auto",
        engine="h5netcdf",
        parallel=True,
        decode_times=False,
    )
    time_elapsed = time.time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    qgis_colormap = Path("../data/speed-colorblind.txt")
    overlay_cmap = qgis2cmap(qgis_colormap, name="speeds")

    stats: List[str] = ["rmsd", "pearson_r"]

    cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
    client = Client(cluster)
    n_jobs = len(client.ncores())
    print(f"Open client in browser: {client.dashboard_link}")

    with client:
        start = time.time()
        velocity_ds_scattered = client.scatter(velocity_ds)
        exp_ds_scattered = client.scatter(exp_ds)
        futures = []
        for _, p in profiles_gp.iterrows():
            future = client.submit(process_profile, p, velocity_ds_scattered, exp_ds_scattered, stats=stats)
            futures.append(future)

        futures_computed = client.compute(futures)
        progress(futures_computed)
        obs_sims_profiles = client.gather(futures_computed)

        time_elapsed = time.time() - start
        print(f"Time elapsed {time_elapsed:.0f}s")

        n_partitions = 1
        profiles = dask_geopandas.from_geopandas(
            gp.GeoDataFrame(profiles_gp, geometry=profiles_gp.geometry), npartitions=n_partitions
        )

        def concat(profiles_df, profiles_ds):
            """
            Concatenate a merged profiles
            """
            return dd.concat(
                [
                    merge_on_intersection(profiles_df, p.mean(["profile_axis"], skipna=True).to_dask_dataframe())
                    for p in profiles_ds
                ]
            )

        print("Merging dataframes")
        start = time.time()

        profiles_scattered = client.scatter(profiles)
        obs_sims_profiles_scattered = client.scatter(obs_sims_profiles)
        futures = client.submit(concat, profiles_scattered, obs_sims_profiles_scattered)
        progress(futures)
        stats_profiles = client.gather(futures).compute().reset_index(drop=True)

        time_elapsed = time.time() - start
        print(f"Time elapsed {time_elapsed:.0f}s")

    gris_ds = xr.open_dataset(Path("/Users/andy/Google Drive/My Drive/data/MCdataset/BedMachineGreenland-v5.nc"))
    surface_da = gris_ds["surface"]
    overlay_da = velocity_ds["v"].where(velocity_ds["ice"])

    start = time.time()
    print("Plotting glaciers")
    with tqdm_joblib(tqdm(desc="Plotting glaciers", total=len(stats_profiles))) as progress_bar:
        Parallel(n_jobs=n_jobs)(
            delayed(plot_glacier)(
                p,
                surface_da,
                overlay_da,
                profile_result_dir,
                cmap=overlay_cmap,
            )
            for _, p in stats_profiles.iterrows()
        )
    time_elapsed = time.time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    with tqdm_joblib(tqdm(desc="Plotting profiles", total=len(profiles_gp))) as progress_bar:
        Parallel(n_jobs=n_jobs)(
            delayed(plot_profile)(ds, profile_result_dir, alpha=obs_scale_alpha, sigma=obs_sigma)
            for ds in obs_sims_profiles
        )
