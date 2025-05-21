# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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
b  # along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Extract along profiles and compute statistics.
"""

import logging
import time
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from importlib.resources import files
from pathlib import Path

import dask_geopandas
import geopandas as gp
import numpy as np
import toml
import xarray as xr
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster, progress
from distributed.utils import silence_logging_cmgr

from glacier_flow_tools.profiles import extract_profile, plot_obs_sims_profile
from glacier_flow_tools.utils import preprocess_nc

default_project_file_url = files("glacier_flow_tools.data").joinpath("default.toml")

# pylint: disable=redefined-outer-name


class ParseKwargs(Action):
    """
    Custom action for parsing keyword arguments from the command line.

    This class is used as an action within argparse to parse keyword arguments from the command line and store them in a dictionary.

    Methods
    -------
    __call__(parser, namespace, values, option_string=None):
        The method called when the action is triggered.

    Examples
    --------
    >>> parser = argparse.ArgumentParser()
    >>> parser.add_argument('--kwargs', nargs='*', action=ParseKwargs)
    >>> args = parser.parse_args('--kwargs key1=value1 key2=value2'.split())
    >>> print(args.kwargs)
    {'key1': 'value1', 'key2': 'value2'}
    """

    def __call__(self, parser, namespace, values, option_string=None):
        """
        Called when the action is triggered.

        This method is called when the action is triggered. It parses the keyword arguments from the command line and stores them in a dictionary.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The argument parser object.
        namespace : argparse.Namespace
            The namespace object that will be updated with the parsed values.
        values : list
            The command-line arguments to be parsed.
        option_string : str, optional
            The option string that was used to invoke this action.
        """
        setattr(namespace, self.dest, {})
        for value in values:
            key, value = value.split("=")
            getattr(namespace, self.dest)[key] = value


if __name__ == "__main__":
    __spec__ = None

    # set up the option parser
    profiles_parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    profiles_parser.description = "Compute pathlines (forward/backward) given a velocity field (xr.Dataset) and starting points (geopandas.GeoDataFrame)."
    profiles_parser.add_argument(
        "--alpha",
        help="""Scale observational error. Use 0.05 to reproduce 'Commplex Outlet Glacier Flow Captured'. Default=0.""",
        default=0.0,
        type=float,
    )
    profiles_parser.add_argument(
        "--crs", help="""Coordinate reference system. Default is EPSG:3413.""", type=str, default="EPSG:3413"
    )
    profiles_parser.add_argument("--n_jobs", help="""Number of parallel jobs.""", type=int, default=4)
    profiles_parser.add_argument("--profiles_url", help="""Path to profiles.""", default=None, type=str)
    profiles_parser.add_argument(
        "--result_dir",
        help="""Path to where output is saved. Directory will be created if needed.""",
        default=Path("./results"),
    )
    profiles_parser.add_argument(
        "--project_file",
        nargs=1,
        help=f"Project files in toml. Default={default_project_file_url}",
        default=default_project_file_url,
    )
    profiles_parser.add_argument(
        "--sigma",
        help="""Sigma multiplier observational error. Default=1. (i.e. error is 1 standard deviation""",
        default=1.0,
        type=float,
    )
    profiles_parser.add_argument(
        "--segmentize", help="""Profile resolution in meters Default=250m.""", default=250, type=float
    )
    profiles_parser.add_argument(
        "--thickness_url",
        help="""Path to thickness dataset.""",
        default=None,
    )
    profiles_parser.add_argument(
        "--velocity_url",
        help="""Path to velocity dataset.""",
        default=None,
    )
    profiles_parser.add_argument(
        "--velocity_cmap",
        help="""Matplotlib colormap used for overlay. Default: 'speeds-colorblind', a custome colormaps.""",
        type=str,
        default="speed_colorblind",
    )
    profiles_parser.add_argument("INFILES", nargs="*", help="PISM experiment files", default=None)

    options = profiles_parser.parse_args()
    project = toml.load(options.project_file)

    profile_result_dir = Path(options.result_dir)
    profile_result_dir.mkdir(parents=True, exist_ok=True)
    profile_figure_dir = profile_result_dir / Path("figures")
    profile_figure_dir.mkdir(parents=True, exist_ok=True)
    profile_output_dir = profile_result_dir / Path("files")
    profile_output_dir.mkdir(parents=True, exist_ok=True)

    crs = options.crs
    obs_sigma = options.sigma
    obs_scale_alpha = options.alpha
    profile_resolution = options.segmentize
    profiles_path = Path(options.profiles_url)
    velocity_cmap = options.velocity_cmap
    profiles_gp = gp.read_file(profiles_path).rename(columns={"id": "profile_id", "name": "profile_name"})
    geom = profiles_gp.segmentize(profile_resolution)
    profiles_gp = gp.GeoDataFrame(profiles_gp, geometry=geom)
    profiles_gp = profiles_gp[["profile_id", "profile_name", "geometry"]]

    velocity_file = Path(options.velocity_url)
    velocity_ds = xr.open_dataset(velocity_file, chunks="auto")

    for k, v in project["ITS_LIVE"]["units"].items():
        velocity_ds[k].attrs["units"] = v

    if project["Profiles"]["compute_profile_normal"]:
        sim_vars_to_keep = list(project["Simulations"]["normal_component_vars"].values())
        if project["Profiles"]["compute_flux"]:
            sim_vars_to_keep += [project["Simulations"]["thickness_var"]]
    else:
        sim_vars_to_keep = [project["Simulations"]["profile_var"]] + ["time_bnds"]

    if project["Profiles"]["compute_flux"]:
        sim_vars_to_keep += ["thk"]

    print("Opening experiments")
    exp_files = [Path(x) for x in options.INFILES]
    start = time.time()
    exp_ds = xr.open_mfdataset(
        exp_files,
        preprocess=partial(
            preprocess_nc,
            regexp=project["Preprocess"]["regexp"],
            drop_dims=["z", "zb", "nv4"],
            drop_vars=["timestamp", "shelfbtemp", "effective_ice_surface_temp", "ice_surface_temp", "hardav", "nv4"],
        ),
        concat_dim="exp_id",
        combine="nested",
        chunks="auto",
        engine="h5netcdf",
        parallel=True,
    )[sim_vars_to_keep]

    time_elapsed = time.time() - start
    print(f"Time elapsed {time_elapsed:.0f}s")

    if options.thickness_url:
        thickness_file = Path(options.thickness_url)
        thickness_ds = xr.open_dataset(thickness_file, chunks="auto").interp_like(velocity_ds)
        velocity_ds = velocity_ds.fluxes.add_fluxes(
            thickness_ds=thickness_ds,
            thickness_var=project["Observations"]["thickness_var"],
            velocity_vars=project["Observations"]["normal_component_vars"],
            error_vars=project["Observations"]["normal_component_error_vars"],
            flux_vars={
                "x": "obs_ice_mass_flux_x",
                "y": "obs_ice_mass_flux_y",
                "x_err": "obs_ice_mass_flux_err_x",
                "y_err": "obs_ice_mass_flux_err_y",
                "magnitude": "obs_ice_mass_flux_normal",
                "magnitude_err": "obs_ice_mass_flux_normal_err",
            },
        )
        exp_ds = exp_ds.fluxes.add_fluxes(
            thickness_var=project["Simulations"]["thickness_var"],
            velocity_vars=project["Simulations"]["normal_component_vars"],
            flux_vars={"x": "sim_ice_mass_flux_x", "y": "sim_ice_mass_flux_y", "magnitude": "sim_ice_mass_flux_normal"},
        )

    profile_stats = project["Statistics"]["metrics"]
    profile_stats_kwargs = project["Statistics"]["metrics_vars"]

    npartitions = 1
    profiles = dask_geopandas.from_geopandas(
        gp.GeoDataFrame(profiles_gp, geometry=profiles_gp.geometry), npartitions=npartitions
    )

    if project["Profiles"]["compute_flux"]:
        obs_normal_var = project["Observations"]["profile_var"]
        obs_normal_error_var = project["Observations"]["profile_error_var"]
        obs_normal_component_vars = project["Observations"]["normal_component_flux_vars"]
        obs_normal_component_error_vars = project["Observations"]["normal_component_flux_error_vars"]
        sim_normal_var = project["Simulations"]["profile_var"]
        sim_normal_component_vars = project["Simulations"]["normal_component_flux_vars"]
    else:
        obs_normal_var = project["Observations"]["profile_var"]
        obs_normal_error_var = project["Observations"]["profile_error_var"]
        obs_normal_component_vars = project["Observations"]["normal_component_vars"]
        obs_normal_component_error_vars = project["Observations"]["normal_component_error_vars"]
        sim_normal_var = project["Simulations"]["profile_var"]
        sim_normal_component_vars = project["Simulations"]["normal_component_vars"]

    # Add alpha to errors. Alpha is a tuning factor that can be used, e.g., to adjust for annual vs winter velocities.
    velocity_ds[obs_normal_component_error_vars["x"]].values = np.abs(
        velocity_ds[obs_normal_component_vars["x"]]
    ) * np.sqrt(
        velocity_ds[obs_normal_component_error_vars["x"]] ** 2 / velocity_ds[obs_normal_component_vars["x"]] ** 2
        + obs_scale_alpha**2
    )
    velocity_ds[obs_normal_component_error_vars["y"]].values = np.abs(
        velocity_ds[obs_normal_component_vars["y"]]
    ) * np.sqrt(
        velocity_ds[obs_normal_component_error_vars["y"]] ** 2 / velocity_ds[obs_normal_component_vars["y"]] ** 2
        + obs_scale_alpha**2
    )

    with silence_logging_cmgr(logging.CRITICAL):

        cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=2)
        with Client(cluster, asynchronous=True) as client:
            n_jobs = len(client.ncores())
            print(f"Open client in browser: {client.dashboard_link}")

            start = time.time()
            velocity_ds_scattered = client.scatter(velocity_ds)
            exp_ds_scattered = client.scatter(exp_ds)
            profiles_scattered = client.scatter([p for _, p in profiles.iterrows()])
            profiles_df_scattered = client.scatter(profiles)

            futures = client.map(
                extract_profile,
                profiles_scattered,
                obs_ds=velocity_ds_scattered,
                sim_ds=exp_ds_scattered,
                compute_profile_normal=project["Profiles"]["compute_profile_normal"],
                obs_normal_var=obs_normal_var,
                obs_normal_error_var=obs_normal_error_var,
                obs_normal_component_vars=obs_normal_component_vars,
                obs_normal_component_error_vars=obs_normal_component_error_vars,
                sim_normal_var=sim_normal_var,
                sim_normal_component_vars=sim_normal_component_vars,
                stats=profile_stats,
                stats_kwargs=profile_stats_kwargs,
            )
            futures_computed = client.compute(futures)
            progress(futures_computed)
            obs_sims_profiles = [p.compute() for p in client.gather(futures_computed)]

            futures = []
            for p in obs_sims_profiles:
                future = client.submit(
                    dd.merge,
                    p[["profile_id", "rmsd", "pearson_r", "obs_flux", "sim_flux"]].to_dataframe().reset_index(),
                    profiles_gp,
                    on="profile_id",
                )
                futures.append(future)

            futures_computed = client.compute(futures)
            progress(futures_computed)
            stats_profiles = dd.concat(client.gather(futures_computed)).compute().reset_index(drop=True)
            stats_profiles = gp.GeoDataFrame(stats_profiles, geometry=stats_profiles.geometry, crs=crs)
            stats_file = profile_output_dir / "stats.gpkg"
            stats_profiles.to_file(stats_file)

            obs_sims_profiles_scattered = client.scatter(obs_sims_profiles)

            print("Plotting profiles")
            futures = client.map(
                plot_obs_sims_profile,
                obs_sims_profiles_scattered,
                obs_var=project["Observations"]["profile_var"],
                obs_error_var=project["Observations"]["profile_error_var"],
                sim_var=project["Simulations"]["profile_var"],
                result_dir=profile_figure_dir,
                sigma=obs_sigma,
                plot_kwargs=project["Plotting"],
            )
            futures_computed = client.compute(futures)
            progress(futures_computed)
            futures_gathered = client.gather(futures_computed)

            time_elapsed = time.time() - start
            print(f"Time elapsed {time_elapsed:.0f}s")
