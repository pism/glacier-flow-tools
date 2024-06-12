# Copyright (C) 2015, 2016, 2018, 2021, 2023 Constantine Khroulev and Andy Aschwanden
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
Module provides profile functions.
"""

# pylint: disable=too-many-lines

from pathlib import Path
from typing import Dict, List, Optional, Union

import cartopy.crs as ccrs
import cf_xarray.units  # pylint: disable=unused-import
import geopandas as gp
import matplotlib
import numpy as np
import pint_xarray  # pylint: disable=unused-import
import pylab as plt
import seaborn as sns
import xarray as xr
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from shapely import get_coordinates

from glacier_flow_tools.utils import (
    blend_multiply,
    figure_extent,
    get_dataarray_extent,
    register_colormaps,
)

register_colormaps()


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


def plot_profile(
    ds: xr.Dataset,
    result_dir: Path,
    interactive: bool = False,
    sigma: float = 1.0,
    obs_var: str = "v",
    obs_error_var: str = "v_err",
    sim_var: str = "velsurf_mag",
    palette: str = "Greens",
    obs_kwargs: dict = {"color": "0", "lw": 0.75, "marker": "o", "ms": 1.5},
    obs_error_kwargs: dict = {"color": "0.75"},
    sim_kwargs: dict = {"lw": 0.5, "marker": "o", "ms": 1.5},
    figsize=[3.2, 3.2],
    fontsize: float = 6,
    plot_kwargs: dict = {
        "x_axis_label": "Distance (m)",
        "y_axis_label": "Speed (m/yr)",
        "rmsd_units": "m/yr",
        "rmsd_formatting": ".0f",
    },
):
    """
    Plot a profile dataset created with ds.profiles.extract_profile and save it as a PDF.

    This function plots a profile dataset that was created with the `extract_profile` method of the `profiles`
    attribute of an `xr.Dataset` object. The plot is saved as a PDF file in the specified result directory.

    Parameters
    ----------
    ds : xr.Dataset
        The profile dataset to be plotted.
    result_dir : Path
        The directory where the result PDF file will be saved.
    interactive : bool
        If False (default), use non-interactive matplotlib backend for plotting.
        Needed for distributed plottinging.
    sigma : float, optional
        The sigma value to be used for the plot, which determines the width of the Gaussian kernel, by default 1.0.
    obs_var : str, optional
        The variable name for the observations, by default 'v'.
    obs_error_var : str, optional
        The variable name for the observation errors, by default 'v_err'.
    sim_var : str, optional
        The variable name for the simulations, by default 'velsurf_mag'.
    palette : str, optional
        The color palette to use for the plot, by default 'Paired'.
    obs_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function for the observations, by default {"color": "0", "lw": 0.75, "marker": "o", "ms": 1.5}.
    obs_error_kwargs : dict, optional
        Additional keyword arguments to pass to the fill_between function for the observation errors, by default {"color": "0.75"}.
    sim_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function for the simulations, by default {"lw": 0.5, "marker": "o", "ms": 1.5}.
    figsize : list, optional
        The size of the figure in inches, by default [3.2, 3.2].
    fontsize : float, optional
        The font size to be used for the plot, by default 6.
    plot_kwargs : dict, optional
        Additional keyword arguments to pass to the plot function for the figure and axis, by default {"x_axis_label": "Distance (m)",
          "y_axis_label": "Speed (m/yr)"}.

    Examples
    --------
    >>> plot_profile(ds, result_dir, sigma=1.0, obs_var='v', obs_error_var='v_err', sim_var='velsurf_mag', palette='Paired')
    """

    if interactive:
        # The standard backend is not thread-safe, but 'agg' works with the dask client.
        matplotlib.use("agg")
    else:
        matplotlib.use("module://matplotlib_inline.backend_inline")

    fig = ds.profiles.plot(
        sigma=sigma,
        obs_var=obs_var,
        obs_error_var=obs_error_var,
        sim_var=sim_var,
        sim_kwargs=sim_kwargs,
        palette=palette,
        obs_kwargs=obs_kwargs,
        obs_error_kwargs=obs_error_kwargs,
        figsize=figsize,
        fontsize=fontsize,
        plot_kwargs=plot_kwargs,
    )
    profile_name = ds["profile_name"].values[0]
    fig.savefig(result_dir / f"{profile_name}_profile.pdf")
    plt.close()
    del fig


def plot_glacier(
    profile_series: gp.GeoSeries,
    surface: xr.DataArray,
    overlay: xr.DataArray,
    result_dir: Union[str, Path],
    interactive: bool = False,
    cmap: str = "viridis",
    vmin: float = 10,
    vmax: float = 1500,
    ticks: Union[List[float], np.ndarray] = [10, 100, 250, 500, 750, 1500],
    fontsize: float = 6,
    figwidth: float = 3.2,
):
    """
    Plot a surface over a hillshade, add profile and correlation coefficient.

    This function plots a surface over a hillshade, adds a profile and correlation coefficient.
    The plot is saved as a PDF file in the specified result directory.

    Parameters
    ----------
    profile_series : gp.GeoSeries
        The profile to be plotted.
    surface : xr.DataArray
        The surface to be plotted over the hillshade.
    overlay : xr.DataArray
        The overlay to be added to the plot.
    result_dir : Union[str, Path]
        The directory where the result PDF file will be saved.
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

    plt.rcParams["font.size"] = fontsize
    geom = getattr(profile_series, "geometry")
    geom_centroid = geom.centroid
    profile_centroid = gp.GeoDataFrame([profile_series], geometry=[geom_centroid])
    profile = gp.GeoDataFrame([profile_series], geometry=[geom])
    glacier_name = getattr(profile, "profile_name").values[0]
    exp_id = getattr(profile, "exp_id").values[0]
    geom = getattr(profile_centroid, "geometry")
    x, y = get_coordinates(geom).T
    x_c, y_c = np.round(x[0]), np.round(y[0])
    extent_slice = figure_extent(x_c, y_c)
    cartopy_crs = ccrs.NorthPolarStereo(central_longitude=-45, true_scale_latitude=70, globe=None)
    # Shade from the northwest, with the sun 45 degrees from horizontal
    light_source = LightSource(azdeg=315, altdeg=45)
    glacier_overlay = overlay.sel(extent_slice)
    glacier_surface = surface.interp_like(glacier_overlay)

    extent = get_dataarray_extent(glacier_overlay)
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap=cmap)

    v = mapper.to_rgba(glacier_overlay.to_numpy())
    z = glacier_surface.to_numpy()

    ar = 1.0  # initial aspect ratio for first trial
    wi = figwidth  # width in inches
    hi = wi * ar  # height in inches

    fig = plt.figure(figsize=(wi, hi))
    ax = fig.add_subplot(111, projection=cartopy_crs)
    rgb = light_source.shade_rgb(v, elevation=z, vert_exag=0.01, blend_mode=blend_multiply)
    # Use a proxy artist for the colorbar...
    im = ax.imshow(v, cmap=cmap, vmin=vmin, vmax=vmax)
    im.remove()
    corr = ax.imshow(
        v,
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
    )
    corr.remove()
    ax.imshow(rgb, extent=extent, origin="upper", transform=cartopy_crs)
    profile.plot(ax=ax, color="k", lw=1)
    profile_centroid.plot(
        column="pearson_r",
        vmin=0,
        vmax=1,
        cmap="RdYlGn",
        markersize=50,
        legend=False,
        missing_kwds={},
        ax=ax,
        edgecolors="k",
        linewidths=0.2,
    )
    ax.set_title(glacier_name)
    ax.gridlines(
        draw_labels={"top": "x", "left": "y"},
        dms=True,
        xlocs=np.arange(-50, 0, 1),
        ylocs=np.arange(50, 88, 1),
        x_inline=False,
        y_inline=False,
        rotate_labels=20,
        ls="dotted",
        color="k",
    )

    ax.set_extent(extent, crs=cartopy_crs)
    fig.colorbar(im, ax=ax, shrink=0.5, pad=0.025, label=overlay.units, extend="both", ticks=ticks)
    fig.colorbar(
        corr, ax=ax, shrink=0.5, pad=0.025, label="Pearson $r$ (1)", orientation="horizontal", location="bottom"
    )
    plt.draw()

    # Get proper ratio here
    xmin, xmax = ax.get_xbound()
    ymin, ymax = ax.get_ybound()
    y2x_ratio = (ymax - ymin) / (xmax - xmin)
    fig.set_figheight(wi * y2x_ratio)
    fig.tight_layout()
    fig.savefig(result_dir / Path(f"{glacier_name}_{exp_id}_speed.pdf"))
    plt.close()
    del fig


def normal(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """
    Compute the unit normal vector orthogonal to (point1-point0), pointing 'to the right' of (point1-point0).

    Parameters
    ----------
    point0 : np.ndarray
        The starting point of the vector.
    point1 : np.ndarray
        The ending point of the vector.

    Returns
    -------
    np.ndarray
        The unit normal vector orthogonal to the vector from point0 to point1.

    Notes
    -----
    This function computes the unit normal vector orthogonal to the vector from point0 to point1.
    The normal vector points to the right of the vector from point0 to point1.
    If the dot product of the vector from point0 to point1 and the normal vector is negative, the direction of the normal vector is flipped.
    """

    a = point0 - point1
    n = np.array([-a[1], a[0]])  # compute the normal vector
    n = n / np.linalg.norm(n)  # normalize

    # flip direction if needed:
    if np.cross(a, n) < 0:
        n = -1.0 * n
    return n


def tangential(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """
    Compute the unit tangential vector to (point1-point0), pointing 'to the right' of (point1-point0).

    Parameters
    ----------
    point0 : np.ndarray
        The starting point of the vector.
    point1 : np.ndarray
        The ending point of the vector.

    Returns
    -------
    np.ndarray
        The unit tangential vector pointing from point0 to point1.

    Notes
    -----
    This function computes the unit tangential vector from point0 to point1.
    If the norm of the vector from point0 to point1 is zero, the function returns the zero vector.
    """
    a = point1 - point0
    norm = np.linalg.norm(a)

    # protect from division by zero
    return a if norm == 0 else a / norm


def compute_normals(px: Union[np.ndarray, xr.DataArray, list], py: Union[np.ndarray, xr.DataArray, list]):
    """
    Compute normals to a profile described by 'p'. Normals point 'to the right' of the path.

    Parameters
    ----------
    px : Union[np.ndarray, xr.DataArray, list]
        The x-coordinates of the points describing the profile.
    py : Union[np.ndarray, xr.DataArray, list]
        The y-coordinates of the points describing the profile.

    Returns
    -------
    tuple of np.ndarray
        The x and y components of the normal vectors to the profile.

    Notes
    -----
    This function computes the normal vectors to a profile described by the points (px, py).
    The normal vector at each point is computed as the vector from the previous point to the next point, rotated 90 degrees clockwise.
    For the first and last points, the normal vector is computed as the vector from the first point to the second point and from the second last point to the last point, respectively, also rotated 90 degrees clockwise.
    """
    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ns = np.zeros_like(p)
    ns[0] = normal(p[0], p[1])
    for j in range(1, len(p) - 1):
        ns[j] = normal(p[j - 1], p[j + 1])

    ns[-1] = normal(p[-2], p[-1])

    return ns[:, 0], ns[:, 1]


def compute_tangentials(px: Union[np.ndarray, list], py: Union[np.ndarray, list]):
    """
    Compute tangentials to a profile described by 'p'.

    Parameters
    ----------
    px : Union[np.ndarray, list]
        The x-coordinates of the points describing the profile.
    py : Union[np.ndarray, list]
        The y-coordinates of the points describing the profile.

    Returns
    -------
    tuple of np.ndarray
        The x and y components of the tangential vectors to the profile.

    Notes
    -----
    This function computes the tangential vectors to a profile described by the points (px, py).
    The tangential vector at each point is computed as the vector from the previous point to the next point.
    For the first and last points, the tangential vector is computed as the vector from the first point to the second point and from the second last point to the last point, respectively.
    """
    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ts = np.zeros_like(p)
    ts[0] = tangential(p[0], p[1])
    for j in range(1, len(p) - 1):
        ts[j] = tangential(p[j - 1], p[j + 1])

    ts[-1] = tangential(p[-2], p[-1])

    return ts[:, 0], ts[:, 1]


def process_profile(
    profile,
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    stats: List[str] = ["rmsd", "pearson_r"],
    obs_normal_var: str = "v_normal",
    obs_normal_error_var: str = "v_err_normal",
    obs_normal_component_vars: dict = {"x": "vx", "y": "vy"},
    obs_normal_component_error_vars: dict = {"x": "vx_err", "y": "vy_err"},
    sim_normal_var: str = "velsurf_normal",
    sim_normal_component_vars: dict = {"x": "uvelsurf", "y": "vvelsurf"},
    compute_profile_normal: bool = False,
    stats_kwargs: Dict = {},
) -> xr.Dataset:
    """
    Compute a profile from observed and simulated datasets.

    Parameters
    ----------
    profile : GeoDataFrame
        The profile to process.
    obs_ds : xr.Dataset
        The observed dataset.
    sim_ds : xr.Dataset
        The simulated dataset.
    stats : List[str], optional
        The list of statistics to calculate, by default ["rmsd", "pearson_r"].
    obs_normal_var : str, optional
        The name of the normal variable in the observed dataset, by default "v_normal".
    obs_normal_error_var : str, optional
        The name of the normal error variable in the observed dataset, by default "v_err_normal".
    obs_normal_component_vars : dict, optional
        The dictionary of normal component variables in the observed dataset, by default {"x": "vx", "y": "vy"}.
    obs_normal_component_error_vars : dict, optional
        The dictionary of normal component error variables in the observed dataset, by default {"x": "vx_err", "y": "vy_err"}.
    sim_normal_var : str, optional
        The name of the normal variable in the simulated dataset, by default "velsurf_normal".
    sim_normal_component_vars : dict, optional
        The dictionary of normal component variables in the simulated dataset, by default {"x": "uvelsurf", "y": "vvelsurf"}.
    compute_profile_normal : bool, optional
        Whether to compute the profile normal, by default False.
    stats_kwargs : Dict
        A dictionary passed on to `calculate_stats`.

    Returns
    -------
    xr.Dataset
        The processed profile as an xarray Dataset.

    Notes
    -----
    This function extracts profiles from the observed and simulated datasets along the given profile, merges them, and calculates the specified statistics.
    """

    coords = get_coordinates(profile["geometry"])
    x, y = coords[:, 0], coords[:, 1]
    profile_name = profile["profile_name"]
    profile_id = profile["profile_id"]
    stats_kwargs_copy = stats_kwargs.copy()
    stats_obs_var = stats_kwargs_copy["obs_var"]
    del stats_kwargs_copy["obs_var"]
    stats_sim_var = stats_kwargs_copy["sim_var"]
    del stats_kwargs_copy["sim_var"]

    def extract_and_prepare(
        ds: xr.Dataset,
        profile_name: str = profile_name,
        profile_id: int = profile_id,
        normal_var: str = "v_normal",
        normal_error_var: str = "v_normal_err",
        normal_component_vars: Dict[str, str] = {"x": "vx", "y": "vy"},
        normal_component_error_vars: Dict[str, str] = {"x": "vx_err", "y": "vy_err"},
        compute_profile_normal: bool = False,
    ) -> xr.Dataset:
        """
        Extract from xr.Dataset along (x,y) profile and prepare it for further processing.

        Parameters
        ----------
        ds : xr.Dataset
            The input dataset from which to extract the profile.
        profile_name : str, optional
            The name of the profile to extract, by default the value of the global variable 'profile_name'.
        profile_id : int, optional
            The id of the profile to extract, by default the value of the global variable 'profile_id'.
        normal_var : str, optional
            The name of the variable representing the normal velocity, by default 'v_normal'.
        normal_error_var : str, optional
            The name of the variable representing the error in the normal velocity, by default 'v_normal_err'.
        normal_component_vars : dict, optional
            A dictionary mapping the x and y components to their variable names, by default {"x": "vx", "y": "vy"}.
        normal_component_error_vars : dict, optional
            A dictionary mapping the x and y components to their error variable names, by default {"x": "vx_err", "y": "vy_err"}.
        compute_profile_normal : bool, optional
            Whether to compute the normal of the profile, by default False.

        Returns
        -------
        xr.Dataset
            The extracted profile as an xarray Dataset.

        Notes
        -----
        This function uses the 'extract_profile' method of the 'profiles' accessor of the input dataset.
        """
        ds_profile = ds.profiles.extract_profile(
            x,
            y,
            profile_name=profile_name,
            profile_id=profile_id,
            normal_var=normal_var,
            normal_error_var=normal_error_var,
            normal_component_vars=normal_component_vars,
            normal_component_error_vars=normal_component_error_vars,
            compute_profile_normal=compute_profile_normal,
        )
        return ds_profile

    obs_profile = extract_and_prepare(
        obs_ds,
        profile_name=profile_name,
        profile_id=profile_id,
        normal_var=obs_normal_var,
        normal_error_var=obs_normal_error_var,
        normal_component_vars=obs_normal_component_vars,
        normal_component_error_vars=obs_normal_component_error_vars,
        compute_profile_normal=compute_profile_normal,
    )
    sims_profile = extract_and_prepare(
        sim_ds,
        profile_name=profile_name,
        profile_id=profile_id,
        normal_var=sim_normal_var,
        normal_component_vars=sim_normal_component_vars,
        compute_profile_normal=compute_profile_normal,
    )
    merged_profile = xr.merge([obs_profile, sims_profile])
    merged_profile.profiles.calculate_stats(
        obs_var=stats_obs_var, sim_var=stats_sim_var, stats=stats, **stats_kwargs_copy
    )

    return merged_profile


@xr.register_dataset_accessor("fluxes")
class FluxMethods:
    """
    Fluxes methods for xarray Dataset.

    This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'fluxes' attribute.

    Parameters
    ----------

    xarray_obj : xr.Dataset
      The xarray Dataset to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the FluxesMethods class.

        Parameters
        ----------

        xarray_obj : xr.Dataset
            The xarray Dataset to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def __repr__(self):
        """
        Flux methods.
        """
        return """
Fluxes methods for xarray Dataset.

This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'fluxes' attribute.

Parameters
----------

xarray_obj : xr.Dataset
  The xarray Dataset to which to add the custom methods.
      """

    def add_fluxes(
        self,
        thickness_var: str = "thickness",
        thickness_err_var: str = "errbed",
        velocity_vars: Dict = {
            "x": "vx",
            "y": "vy",
        },
        flux_vars: Dict = {
            "x": "ice_mass_flux_x",
            "y": "ice_mass_flux_y",
            "x_err": "ice_mass_flux_err_x",
            "y_err": "ice_mass_flux_err_y",
            "magnitude": "ice_mass_flux_normal",
            "magnitude_err": "ice_mass_flux_normal_err",
        },
        error_vars: Optional[Dict] = None,
        thickness_ds: Optional[xr.Dataset] = None,
    ):
        """
        Add ice mass flux and its error to the velocity dataset.

        This function calculates the ice mass flux and its error in x and y directions and adds them to the velocity dataset.
        The flux is calculated as the product of velocity, ice thickness, and grid resolution, multiplied by the ice density.
        The error is calculated using the error propagation formula.

        Parameters
        ----------
        thickness_var : str, optional
            The variable name for the ice thickness data. The default is "thickness".
        thickness_err_var : str, optional
            The variable name for the ice thickness error data. The default is "errbed".
        velocity_vars : dict, optional
            A dictionary mapping the direction to the variable name for the velocity. The default is {"x": "vx", "y": "vy"}.
        flux_vars : dict, optional
            A dictionary mapping the direction to the variable name for the flux and its error. The default is
            {"x": "ice_mass_flux_x", "y": "ice_mass_flux_y", "xe": "ice_mass_flux_err_x", "ye": "ice_mass_flux_err_y"}.
        error_vars : dict, optional
            A dictionary mapping the direction to the variable name for the velocity error. If not provided, no error is calculated.
        thickness_ds : xr.Dataset, optional
            A Dataset containing the ice thickness data. If not provided, only the velocity data is returned.

        Returns
        -------
        xr.Dataset
            The xarray Dataset with the fluxes added.

        Examples
        --------
        >>> velocity_ds = xr.Dataset(data_vars={"vx": ("x", [1, 2, 3]), "vy": ("y", [4, 5, 6])})
        >>> thickness_ds = xr.Dataset(data_vars={"thickness": ("x", [7, 8, 9])})
        >>> add_fluxes(thickness_ds)
        <xarray.Dataset>
        Dimensions:            (x: 3, y: 3)
        Dimensions without coordinates: x, y
        Data variables:
            vx                 (x) int64 1 2 3
            vy                 (y) int64 4 5 6
            ice_mass_flux_x    (x) float64 6.917e+03 1.383e+04 2.075e+04
            ice_mass_flux_y    (y) float64 3.668e+04 4.585e+04 5.502e+04
            ice_mass_flux_err_x (x) float64 0.0 0.0 0.0
            ice_mass_flux_err_y (y) float64 0.0 0.0 0.0
        """
        # Extract units
        resolution_units = self._obj["x"].attrs["units"]

        # Check if all elements in dx and dy are equal
        dx, dy = self._obj["x"].diff(dim="x"), self._obj["y"].diff(dim="y")
        assert np.all(dx == dx[0]) and np.all(dy == dy[0])

        # Quantify datasets and constants
        ds = self._obj.pint.quantify()
        ice_density = xr.DataArray(917.0).pint.quantify("kg m-3").pint.to("Gt m-3")
        resolution = xr.DataArray(dx[0]).pint.quantify(resolution_units)
        das = {}
        if thickness_ds is not None:
            das[thickness_var] = thickness_ds[thickness_var]
            if error_vars:
                das[thickness_err_var] = thickness_ds[thickness_err_var]
        else:
            thickness_ds = self._obj
        thickness_ds_pint = thickness_ds.pint.quantify()

        # Calculate flux and its error
        for direction in ["x", "y"]:
            flux_da = ds[velocity_vars[direction]] * thickness_ds_pint[thickness_var] * resolution * ice_density
            das[flux_vars[direction]] = flux_da
            if error_vars:
                flux_err_da = np.abs(flux_da) * np.sqrt(
                    (ds[error_vars[direction]] ** 2 / ds[velocity_vars[direction]] ** 2)
                    + (thickness_ds_pint[thickness_err_var] ** 2 / thickness_ds_pint[thickness_var] ** 2)
                )
                das[flux_vars[f"{direction}_err"]] = flux_err_da
        das[flux_vars["magnitude"]] = (das[flux_vars["x"]] ** 2 + das[flux_vars["y"]] ** 2) ** (1.0 / 2)
        if error_vars:
            ds[flux_vars["magnitude_err"]] = (das[flux_vars["x_err"]] ** 2 + das[flux_vars["y_err"]] ** 2) ** (1.0 / 2)
        self._obj = ds.assign(das).pint.dequantify()
        return self._obj


@xr.register_dataset_accessor("profiles")
class ProfilesMethods:
    """
    Profiles methods for xarray Dataset.

    This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'profiles' attribute.

    Parameters
    ----------

    xarray_obj : xr.Dataset
      The xarray Dataset to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the ProfilesMethods class.

        Parameters
        ----------

        xarray_obj : xr.Dataset
            The xarray Dataset to which to add the custom methods.
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method.

        This method is needed to work with joblib Parallel.
        """

    def add_normal_component(
        self,
        x_component: str = "vx",
        y_component: str = "vy",
        normal_name: str = "v_normal",
    ):
        """
        Add a normal component to the xarray Dataset.

        This method computes the normal component of the vectors defined by the x and y components, and adds it to the Dataset.

        Parameters
        ----------
        x_component : str, optional
            The name of the x component variable in the Dataset, by default "vx".
        y_component : str, optional
            The name of the y component variable in the Dataset, by default "vy".
        normal_name : str, optional
            The name of the normal component variable to add to the Dataset, by default "v_normal".

        Returns
        -------
        xr.Dataset
            The xarray Dataset with the normal variables added.
        """
        assert (x_component and y_component) in self._obj.data_vars

        def func(x, x_n, y, y_n):
            """
            Calculate the normal component of a vector.

            This function computes the normal component of a vector by performing a dot product operation.
            The inputs are the x and y components of the vector and their corresponding normal components.

            Parameters
            ----------
            x : float
                The x-component of the vector.
            x_n : float
                The x-component of the normal.
            y : float
                The y-component of the vector.
            y_n : float
                The y-component of the normal.

            Returns
            -------
            float
                The normal component of the vector.

            Examples
            --------
            >>> func(1, 2, 3, 4)
            14
            """
            return x * x_n + y * y_n

        self._obj[normal_name] = xr.apply_ufunc(
            func, self._obj[x_component], self._obj["nx"], self._obj[y_component], self._obj["ny"], dask="allowed"
        )
        return self._obj

    def calculate_stats(
        self,
        obs_var: str = "obs_v_normal",
        sim_var: str = "sim_v_normal",
        dim: str = "profile_axis",
        stats: List[str] = ["rmsd", "pearson_r"],
    ):
        """
        Calculate statistical metrics between observed and simulated data.

        This function calculates the Root Mean Square Deviation (RMSD) and Pearson correlation coefficient between
        observed and simulated data along a specified dimension.

        Parameters
        ----------
        obs_var : str, optional
            The observed data variable name in the xarray Dataset, by default "v".
        sim_var : str, optional
            The simulated data variable name in the xarray Dataset, by default "velsurf_mag".
        dim : str, optional
            The dimension along which to calculate the statistics, by default "profile_axis".
        stats : List[str], optional
            The list of statistical metrics to calculate, by default ["rmsd", "pearson_r"].

        Returns
        -------
        xr.Dataset
            The xarray Dataset with the calculated statistical metrics added as new data variables.
        """
        assert (
            obs_var and sim_var
        ) in self._obj.data_vars, f"{obs_var} or {sim_var} not in {list(self._obj.data_vars)}."

        def rmsd(sim: xr.DataArray, obs: xr.DataArray) -> float:
            """
            Compute the Root Mean Square Deviation (RMSD) between simulated and observed data.

            This function computes the RMSD between two xarray DataArrays. The RMSD is calculated as the square root
            of the mean of the squared differences between the simulated and observed data.

            Parameters
            ----------
            sim : xr.DataArray
                The simulated data as an xarray DataArray.
            obs : xr.DataArray
                The observed data as an xarray DataArray.

            Returns
            -------

            float
              RMSD of sim and obs.
            """
            diff = sim - obs
            return np.sqrt(np.nanmean(diff**2, axis=-1))

        def pearson_r(sim: xr.DataArray, obs: xr.DataArray) -> xr.DataArray:
            """
            Compute the Pearson correlation coefficient between simulated and observed data.

            This function computes the Pearson correlation coefficient (r) between two xarray DataArrays along the "profile_axis" dimension.

            Parameters
            ----------
            sim : xr.DataArray
                The simulated data as an xarray DataArray.
            obs : xr.DataArray
                The observed data as an xarray DataArray.

            Returns
            -------
            xr.DataArray
                The computed Pearson correlation coefficient as an xarray DataArray.
            """
            return xr.corr(sim, obs, dim="profile_axis")

        stats_func = {"rmsd": {"func": rmsd, "ufunc": True}, "pearson_r": {"func": pearson_r, "ufunc": False}}
        fluxes = {obs_var: "obs_flux", sim_var: "sim_flux"}

        for k, v in fluxes.items():
            self._obj[v] = self._obj[k].integrate(coord="profile_axis")

        for stat in stats:
            if stats_func[stat]["ufunc"]:
                self._obj[stat] = xr.apply_ufunc(
                    stats_func[stat]["func"],  # type: ignore[arg-type]
                    self._obj[obs_var],
                    self._obj[sim_var],
                    dask="allowed",
                    input_core_dims=[[dim], [dim]],
                    output_core_dims=[[]],
                )
            else:
                self._obj[stat] = pearson_r(self._obj[obs_var], self._obj[sim_var])

        return self._obj

    def extract_profile(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        profile_name: str = "Glacier X",
        profile_id: int = 0,
        data_vars: Union[None, List[str]] = None,
        normal_var: str = "v_normal",
        normal_error_var: str = "v_err_normal",
        normal_component_vars: dict = {"x": "vx", "y": "vy"},
        normal_component_error_vars: dict = {"x": "vx_err", "y": "vy_err"},
        compute_profile_normal: bool = False,
    ) -> xr.Dataset:
        """
        Extract a profile from a dataset given x and y coordinates.

        Parameters
        ----------
        xs : np.ndarray
            The x-coordinates of the profile.
        ys : np.ndarray
            The y-coordinates of the profile.
        profile_name : str, optional
            The name of the profile, by default "Glacier X".
        profile_id : int, optional
            The id of the profile, by default 0.
        data_vars : Union[None, List[str]], optional
            The list of data variables to include in the profile. If None, all data variables are included, by default None.
        normal_var : str, optional
            The name of the normal variable, by default "v_normal".
        normal_error_var : str, optional
            The name of the normal error variable, by default "v_err_normal".
        normal_component_vars : dict, optional
            The dictionary of normal component variables, by default {"x": "vx", "y": "vy"}.
        normal_component_error_vars : dict, optional
            The dictionary of normal component error variables, by default {"x": "vx_err", "y": "vy_err"}.
        compute_profile_normal : bool, optional
            Whether to compute the profile normal, by default False.

        Returns
        -------
        xr.Dataset
            A new xarray Dataset containing the extracted profile.
        """
        profile_axis = np.sqrt((xs - xs[0]) ** 2 + (ys - ys[0]) ** 2)

        x: xr.DataArray
        y: xr.DataArray
        x = xr.DataArray(
            xs,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["x"].attrs,
            name="x",
        )
        y = xr.DataArray(
            ys,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["y"].attrs,
            name="y",
        )

        nx, ny = compute_normals(x, y)
        normals = {"nx": nx, "ny": ny}

        das = [
            xr.DataArray(
                val,
                dims="profile_axis",
                coords={"profile_axis": profile_axis},
                name=key,
            )
            for key, val in normals.items()
        ]

        name = xr.DataArray(
            [profile_name],
            dims="profile_id",
            attrs={"units": "m", "long_name": "distance along profile"},
            name="profile_name",
        )
        das.append(name)

        if data_vars is None:
            data_vars = list(self._obj.data_vars)

        for m_var in data_vars:
            da = self._obj[m_var]
            try:
                das.append(da.interp(x=x, y=y, kwargs={"fill_value": np.nan}))
            except:
                pass

        ds = xr.merge(das)
        ds["profile_id"] = [profile_id]
        if compute_profile_normal:
            a = [(v in ds.data_vars) for v in normal_component_vars.values()]
            if np.all(np.array(a)):
                ds.profiles.add_normal_component(
                    x_component=normal_component_vars["x"],
                    y_component=normal_component_vars["y"],
                    normal_name=normal_var,
                )
                assert ds[normal_component_vars["x"]].units == ds[normal_component_vars["y"]].units
                profile_units = ds[normal_component_vars["x"]].units
                ds[normal_var]["units"] = profile_units

            a = [(v in ds.data_vars) for v in normal_component_error_vars.values()]
            if np.all(np.array(a)):
                ds.profiles.add_normal_component(
                    x_component=normal_component_error_vars["x"],
                    y_component=normal_component_error_vars["y"],
                    normal_name=normal_error_var,
                )
                if (normal_component_error_vars["x"] and normal_component_error_vars["y"]) in ds.data_vars:
                    assert ds[normal_component_error_vars["x"]].units == ds[normal_component_error_vars["y"]].units
                    profile_error_units = ds[normal_component_error_vars["x"]].units
                    ds[normal_error_var]["units"] = profile_error_units

        return ds

    def plot(
        self,
        interactive: bool = False,
        sigma: float = 1,
        title: Union[str, None] = None,
        obs_var: str = "v",
        obs_error_var: Union[str, None] = "v_err",
        sim_var: str = "velsurf_mag",
        palette: str = "Paired",
        obs_kwargs: dict = {"color": "0", "lw": 0.75, "marker": "o", "ms": 1.5},
        obs_error_kwargs: dict = {"color": "0.75"},
        sim_kwargs: dict = {"lw": 0.5, "marker": "o", "ms": 1.5},
        figsize=[3.2, 3.2],
        fontsize: float = 6,
        plot_kwargs: dict = {
            "x_axis_label": "Distance (m)",
            "y_axis_label": "Speed (m/yr)",
            "rmsd_units": "m/yr",
            "rmsd_formatting": ".0f",
        },
    ) -> plt.Figure:
        """
        Plot observations and simulations along a profile.

        This function plots observations and simulations along a profile. The observations are plotted with error bars,
        and the simulations are plotted with different colors for each experiment.

        Parameters
        ----------
        interactive : bool
            If False (default), use non-interactive matplotlib backend for plotting.
            Needed for distributed plottinging.
        sigma : float, optional
            The standard deviation of the observations, by default 1.
        title : str or None, optional
            The title of the plot, by default None.
        obs_var : str, optional
            The variable name for the observations, by default 'v'.
        obs_error_var : str, optional
            The variable name for the observation errors, by default 'v_err'.
        sim_var : str, optional
            The variable name for the simulations, by default 'velsurf_mag'.
        palette : str, optional
            The color palette to use for the plot, by default 'Paired'.
        obs_kwargs : dict, optional
            Additional keyword arguments to pass to the plot function for the observations, by default {"color": "0", "lw": 0.75, "marker": "o", "ms": 1.5}.
        obs_error_kwargs : dict, optional
            Additional keyword arguments to pass to the fill_between function for the observation errors, by default {"color": "0.75"}.
        sim_kwargs : dict, optional
            Additional keyword arguments to pass to the plot function for the simulations, by default {"lw": 0.5, "marker": "o", "ms": 1.5}.
        figsize : list, optional
            The size of the figure in inches, by default [3.2, 3.2].
        fontsize : float, optional
            The font size to be used for the plot, by default 6.
        plot_kwargs : dict, optional
            Additional keyword arguments to pass to the plot function for the figure and axis, by default {"x_axis_label": "Distance (m)",
              "y_axis_label": "Speed (m/yr)", "rmsd_units": "m/yr", "rmsd_formatting": ".0f"}.

        Returns
        -------
        plt.Figure
            The created matplotlib Figure object.

        Examples
        --------
        >>> plot(sigma=1, title='My Plot', obs_var='v', obs_error_var='v_err', sim_var='velsurf_mag', palette='Paired')
        """

        if interactive:
            # The standard backend is not thread-safe, but 'agg' works with the dask client.
            matplotlib.use("agg")
        else:
            matplotlib.use("module://matplotlib_inline.backend_inline")

        plt.rcParams["font.size"] = fontsize
        n_exps = self._obj["exp_id"].size
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        if obs_error_var is not None:
            ax.fill_between(
                self._obj["profile_axis"],
                self._obj[obs_var] - sigma * self._obj[obs_error_var],
                self._obj[obs_var] + sigma * self._obj[obs_error_var],
                **obs_error_kwargs,
            )
        ax.plot(
            self._obj["profile_axis"],
            self._obj[obs_var],
            label="Observed",
            **obs_kwargs,
        )
        palette = sns.color_palette(palette, n_colors=n_exps)
        # Loop over the data and plot each line with a different color
        if n_exps > 1:
            for i in range(n_exps):
                exp_label = f"""{self._obj["exp_id"].values[i].item()} $r$={self._obj["pearson_r"].values[i][0]:.2f} rmsd={self._obj["rmsd"].values[i][0]:{plot_kwargs["rmsd_formatting"]}}{plot_kwargs["rmsd_units"]}"""
                ax.plot(
                    self._obj["profile_axis"],
                    np.squeeze(self._obj[sim_var].isel(exp_id=i).T),
                    color=palette[i],
                    label=exp_label,
                    **sim_kwargs,
                )
        else:
            exp_label = f"""{self._obj["exp_id"].values.item()} $r$={self._obj["pearson_r"].values.item():.2f} rmsd={self._obj["rmsd"].values.item():{plot_kwargs["rmsd_formatting"]}}{plot_kwargs["rmsd_units"]}"""
            ax.plot(
                self._obj["profile_axis"],
                np.squeeze(self._obj[sim_var].T),
                color=palette[0],
                label=exp_label,
                **sim_kwargs,
            )
        ax.set_xlabel("Distance along profile (m)")
        ax.set_ylabel(plot_kwargs["y_axis_label"])
        legend = ax.legend(loc="upper left")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        if title is None:
            title = self._obj["profile_name"].values.item()
        plt.title(title)
        return fig
