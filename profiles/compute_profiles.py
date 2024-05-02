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

import logging
import time
from argparse import Action, ArgumentDefaultsHelpFormatter, ArgumentParser
from functools import partial
from importlib.resources import files
from pathlib import Path
from typing import Dict, List, Union

import dask_geopandas
import geopandas as gp
import toml
import xarray as xr
from dask import dataframe as dd
from dask.distributed import Client, LocalCluster, progress
from distributed.utils import silence_logging_cmgr

from glacier_flow_tools.profiles import plot_glacier, plot_profile, process_profile
from glacier_flow_tools.utils import preprocess_nc

default_project_file_url = files("glacier_flow_tools.data").joinpath("default.toml")


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


def extract_profile(
    profile,
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    result_dir: Union[str, Path, None] = None,
    obs_normal_var: str = "obs_v_normal",
    obs_normal_error_var: str = "obs_v_err_normal",
    obs_normal_component_vars: dict = {"x": "vx", "y": "vy"},
    obs_normal_component_error_vars: dict = {"x": "vx_err", "y": "vy_err"},
    sim_normal_var: str = "sim_v_normal",
    sim_normal_component_vars: dict = {"x": "uvelsurf", "y": "vvelsurf"},
    compute_profile_normal: bool = True,
    stats: List[str] = ["rmsd", "pearson_r"],
    stats_kwargs: Dict = {},
) -> xr.Dataset:
    """
    Extract and process profiles from observation and simulation datasets.

    This function extracts profiles from the given observation and simulation datasets, processes them, and saves the processed profile as a netCDF file. It also merges the processed profile with a given GeoDataFrame on intersection.

    Parameters
    ----------
    profile : dict
        The profile to be processed.
    obs_ds : xr.Dataset
        The observation dataset.
    sim_ds : xr.Dataset
        The simulation dataset.
    result_dir : str or Path, optional
        The directory where the result netCDF file will be saved, by default None, no file is saved.
    obs_normal_var : str, optional
        The variable name for the normal component of the observations, by default 'obs_v_normal'.
    obs_normal_error_var : str, optional
        The variable name for the error of the normal component of the observations, by default 'obs_v_err_normal'.
    obs_normal_component_vars : dict, optional
        The variable names for the components of the normal component of the observations, by default {"x": "vx", "y": "vy"}.
    obs_normal_component_error_vars : dict, optional
        The variable names for the errors of the components of the normal component of the observations, by default {"x": "vx_err", "y": "vy_err"}.
    sim_normal_var : str, optional
        The variable name for the normal component of the simulations, by default 'sim_v_normal'.
    sim_normal_component_vars : dict, optional
        The variable names for the components of the normal component of the simulations, by default {"x": "uvelsurf", "y": "vvelsurf"}.
    compute_profile_normal : bool, optional
        Whether to compute the normal component of the profile, by default True.
    stats : list of str, optional
        The statistics to be computed for the profile, by default ["rmsd", "pearson_r"].
    stats_kwargs : dict, optional
        Additional keyword arguments to pass to the function for computing the statistics, by default {}.

    Returns
    -------
    tuple of xr.Dataset and gp.GeoDataFrame
        The processed profile as an xr.Dataset and the merged GeoDataFrame.

    Examples
    --------
    >>> extract_profiles(profile, profiles_df, obs_ds, sim_ds, stats=["rmsd", "pearson_r"], result_dir='.', obs_normal_var='obs_v_normal', obs_normal_error_var='obs_v_err_normal', obs_normal_component_vars={"x": "vx", "y": "vy"}, obs_normal_component_error_vars={"x": "vx_err", "y": "vy_err"}, sim_normal_var='sim_v_normal', sim_normal_component_vars={"x": "uvelsurf", "y": "vvelsurf"}, compute_profile_normal=True, stats_kwargs={})
    """
    os_profile = process_profile(
        profile,
        obs_ds=obs_ds,
        sim_ds=sim_ds,
        compute_profile_normal=compute_profile_normal,
        obs_normal_var=obs_normal_var,
        obs_normal_error_var=obs_normal_error_var,
        obs_normal_component_vars=obs_normal_component_vars,
        obs_normal_component_error_vars=obs_normal_component_error_vars,
        sim_normal_var=sim_normal_var,
        sim_normal_component_vars=sim_normal_component_vars,
        stats=stats,
        stats_kwargs=stats_kwargs,
    )

    if result_dir is not None:
        os_file = Path(result_dir) / f"""{profile["profile_name"]}_profile.nc"""
        os_profile.to_netcdf(os_file, engine="h5netcdf")
    return os_profile


if __name__ == "__main__":

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

    gris_ds = xr.open_dataset(Path("/Users/andy/Google Drive/My Drive/data/MCdataset/BedMachineGreenland-v5.nc"))
    surface_da = gris_ds["surface"]
    overlay_da = velocity_ds["v"].where(velocity_ds["ice"])

    for k, v in project["ITS_LIVE"]["units"].items():
        velocity_ds[k].attrs["units"] = v

    sim_vars_to_keep = list(project["Simulations"]["normal_component_vars"].values())
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
            velocity_var=project["Observations"]["normal_component_vars"],
            error_vars=project["Observations"]["normal_component_error_vars"],
        )
        exp_ds = exp_ds.fluxes.add_fluxes(
            thickness_var=project["Simulations"]["thickness_var"],
            velocity_var=project["Simulations"]["normal_component_vars"],
            flux_vars={"x": "sim_ice_mass_flux_x", "y": "sim_ice_mass_flux_y"},
        )

    profile_stats = project["Statistics"]["metrics"]
    profile_stats_kwargs = project["Statistics"]["metrics_vars"]

    npartitions = 1
    profiles = dask_geopandas.from_geopandas(
        gp.GeoDataFrame(profiles_gp, geometry=profiles_gp.geometry), npartitions=npartitions
    )

    with silence_logging_cmgr(logging.CRITICAL):

        cluster = LocalCluster(n_workers=options.n_jobs, threads_per_worker=1)
        client = Client(cluster)
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
            obs_normal_var=project["Observations"]["profile_var"],
            obs_normal_error_var=project["Observations"]["profile_error_var"],
            obs_normal_component_vars=project["Observations"]["normal_component_vars"],
            obs_normal_component_error_vars=project["Observations"]["normal_component_error_vars"],
            sim_normal_var=project["Simulations"]["profile_var"],
            sim_normal_component_vars=project["Simulations"]["normal_component_vars"],
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
                p[["profile_id", "rmsd", "pearson_r"]].to_dataframe().reset_index(),
                profiles_gp,
                on="profile_id",
            )
            futures.append(future)

        futures_computed = client.compute(futures)
        progress(futures_computed)
        stats_profiles = dd.concat(client.gather(futures_computed)).reset_index(drop=True)

        # futures = client.map(
        #     extract_profiles,
        #     profiles_scattered,
        #     profiles_df=profiles_df_scattered,
        #     obs_ds=velocity_ds_scattered,
        #     sim_ds=exp_ds_scattered,
        #     result_dir=profile_output_dir,
        #     compute_profile_normal=project["Profiles"]["compute_profile_normal"],
        #     obs_normal_var=project["Observations"]["profile_var"],
        #     obs_normal_error_var=project["Observations"]["profile_error_var"],
        #     obs_normal_component_vars=project["Observations"]["normal_component_vars"],
        #     obs_normal_component_error_vars=project["Observations"]["normal_component_error_vars"],
        #     sim_normal_var=project["Simulations"]["profile_var"],
        #     sim_normal_component_vars=project["Simulations"]["normal_component_vars"],
        #     stats=profile_stats,
        #     stats_kwargs=profile_stats_kwargs,
        # )

        # futures_computed = client.compute(futures)
        # progress(futures_computed)
        # futures_gathered = client.gather(futures_computed)
        # obs_sims_profiles, stats_profiles = [p[0] for p in futures_gathered], dd.concat(
        #     [p[1] for p in futures_gathered]
        # ).reset_index(drop=True)

        obs_sims_profiles_scattered = client.scatter(obs_sims_profiles)
        overlay_da_scattered = client.scatter(overlay_da)
        surface_da_scattered = client.scatter(surface_da)

        print("Plotting profiles")
        futures = client.map(
            plot_profile,
            obs_sims_profiles_scattered,
            obs_var=project["Observations"]["profile_var"],
            obs_error_var=project["Observations"]["profile_error_var"],
            sim_var=project["Simulations"]["profile_var"],
            result_dir=profile_figure_dir,
            alpha=obs_scale_alpha,
            sigma=obs_sigma,
        )
        futures_computed = client.compute(futures)
        progress(futures_computed)
        futures_gathered = client.gather(futures_computed)

        print("Plotting glaciers")
        futures = client.map(
            plot_glacier,
            [p for _, p in stats_profiles.iterrows()],
            surface=surface_da_scattered,
            overlay=overlay_da_scattered,
            result_dir=profile_figure_dir,
            cmap=velocity_cmap,
        )
        futures_computed = client.compute(futures)
        progress(futures_computed)
        futures_gathered = client.gather(futures_computed)

        time_elapsed = time.time() - start
        print(f"Time elapsed {time_elapsed:.0f}s")

        client.shutdown()
        del client
