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
    parser.add_argument("--segmentize", help="""Profile resolution in meters Default=100m.""", default=100, type=float)
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

    its_live_units_dict = {
        "vx": "m/yr",
        "vy": "m/yr",
        "v": "m/yr",
        "vx_err": "m/yr",
        "vy_err": "m/yr",
        "v_err": "m/yr",
        "rock": "1",
        "count": "1",
        "ocean": "1",
        "ice": "1",
    }

    for k, v in its_live_units_dict.items():
        velocity_ds[k].attrs["units"] = v

    # def add_fluxes(velocity_ds: xr.Dataset, thickness_ds: Optional[xr.Dataset] = None,
    #                flux_vars: Dict = {"x": "ice_mass_flux_x", "y": "ice_mass_flux_y",
    #                                   "xe": "ice_mass_flux_err_x", "ye": "ice_mass_flux_err_y"}
    #                ) -> xr.Dataset:
    #     """
    #     Add ice mass flux and its error to the velocity dataset.

    #     This function calculates the ice mass flux and its error in x and y directions and adds them to the velocity dataset.
    #     The flux is calculated as the product of velocity, ice thickness, and grid resolution, multiplied by the ice density.
    #     The error is calculated using the error propagation formula.

    #     Parameters
    #     ----------
    #     velocity_ds : xr.Dataset
    #         A Dataset containing the ice velocity data.
    #     thickness_ds : xr.Dataset, optional
    #         A Dataset containing the ice thickness data. If not provided, only the velocity data is returned.
    #     flux_vars : dict, optional
    #         A dictionary mapping the direction to the variable name for the flux and its error. The default is
    #         {"x": "ice_mass_flux_x", "y": "ice_mass_flux_y", "xe": "ice_mass_flux_err_x", "ye": "ice_mass_flux_err_y"}.

    #     Returns
    #     -------
    #     xr.Dataset
    #         The velocity dataset with the added flux and its error.

    #     Examples
    #     --------
    #     >>> velocity_ds = xr.Dataset(data_vars={"vx": ("x", [1, 2, 3]), "vy": ("y", [4, 5, 6])})
    #     >>> thickness_ds = xr.Dataset(data_vars={"thickness": ("x", [7, 8, 9])})
    #     >>> add_fluxes(velocity_ds, thickness_ds)
    #     <xarray.Dataset>
    #     Dimensions:            (x: 3, y: 3)
    #     Dimensions without coordinates: x, y
    #     Data variables:
    #         vx                 (x) int64 1 2 3
    #         vy                 (y) int64 4 5 6
    #         ice_mass_flux_x    (x) float64 6.917e+03 1.383e+04 2.075e+04
    #         ice_mass_flux_y    (y) float64 3.668e+04 4.585e+04 5.502e+04
    #         ice_mass_flux_err_x (x) float64 0.0 0.0 0.0
    #         ice_mass_flux_err_y (y) float64 0.0 0.0 0.0
    #     """
    #     # Extract units
    #     vx_units, vy_units = velocity_ds["vx"].attrs["units"], velocity_ds["vy"].attrs["units"]
    #     vx_err_units, vy_err_units = velocity_ds["vx_err"].attrs["units"], velocity_ds["vy_err"].attrs["units"]
    #     resolution_units = velocity_ds["x"].attrs["units"]

    #     # Check if all elements in dx and dy are equal
    #     dx, dy = velocity_ds["x"].diff(dim="x"), velocity_ds["y"].diff(dim="y")
    #     assert np.all(dx == dx[0]) and np.all(dy == dy[0])

    #     # Quantify datasets and constants
    #     velocity_ds = velocity_ds.pint.quantify()
    #     ice_density = xr.DataArray(917.0).pint.quantify("kg m-3").pint.to("Gt m-3")
    #     resolution = xr.DataArray(dx[0]).pint.quantify(resolution_units)
    #     vx_e_norm, vy_e_norm = xr.DataArray(1).pint.quantify(vx_err_units), xr.DataArray(1).pint.quantify(vy_err_units)

    #     das = {}
    #     if thickness_ds:
    #         thickness_units = thickness_ds["thickness"].attrs["units"]
    #         thickness_ds = thickness_ds.pint.quantify()
    #         thickness_norm = xr.DataArray(1).pint.quantify(thickness_units)

    #         # Calculate flux and its error
    #         for direction in ["x", "y"]:
    #             flux_da = velocity_ds[f"v{direction}"] * thickness_ds["thickness"] * resolution * ice_density
    #             das[flux_vars[direction]] = flux_da
    #             flux_err_da = flux_da * np.sqrt((velocity_ds[f"v{direction}_err"]**2 / vx_e_norm**2)  * (thickness_ds["errbed"]**2 / thickness_norm**2))
    #             das[flux_vars[f"{direction}e"]] = flux_err_da

    #     return velocity_ds.assign(das).pint.dequantify()

    if options.thickness_url:
        thickness_file = Path(options.thickness_url)
        thickness_ds = xr.open_dataset(thickness_file, chunks="auto").interp_like(velocity_ds)
        velocity_ds = velocity_ds.fluxes.add_fluxes(thickness_ds)

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

    stats: List[str] = ["rmsd", "pearson_r"]
    stats_kwargs = {"obs_var": "v_normal", "sim_var": "velsurf_normal"}

    qgis_colormap = Path("../data/speed-colorblind.txt")
    overlay_cmap = qgis2cmap(qgis_colormap, name="speeds")

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
            future = client.submit(
                process_profile,
                p,
                velocity_ds_scattered,
                exp_ds_scattered,
                stats=stats,
                compute_profile_normal=True,
                stats_kwargs=stats_kwargs,
            )
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
