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
from pathlib import Path
from typing import List, Union

import cartopy.crs as ccrs
import geopandas as gp
import numpy as np
import pylab as plt
import seaborn as sns
import xarray as xr
from matplotlib import cm, colors
from matplotlib.colors import LightSource
from shapely import get_coordinates

from glacier_flow_tools.utils import blend_multiply, figure_extent, get_dataarray_extent


def plot_profile(ds: xr.Dataset, result_dir: Path, alpha: float = 0.0, sigma: float = 1.0):
    """
    Plot a profile dataset created with ds.profiles.extract_profile.

    This function plots a profile dataset that was created with the `extract_profile` method of the `profiles`
    attribute of an `xr.Dataset` object. The plot is saved as a PDF file in the specified result directory.

    Parameters
    ----------
    ds : xr.Dataset
        The profile dataset to be plotted.
    result_dir : Path
        The directory where the result PDF file will be saved.
    alpha : float, optional
        The alpha value to be used for the plot, which determines the transparency of the plot, by default 0.0.
    sigma : float, optional
        The sigma value to be used for the plot, which determines the width of the Gaussian kernel, by default 1.0.
    """

    fig = ds.profiles.plot(palette="Greens", sigma=sigma, alpha=alpha)
    profile_name = ds["profile_name"].values[0]
    fig.savefig(result_dir / f"{profile_name}_profile.pdf")
    plt.close()
    del fig


def plot_glacier(
    profile_series: gp.GeoSeries,
    surface: xr.DataArray,
    overlay: xr.DataArray,
    result_dir: Union[str, Path],
    cmap="viridis",
    vmin: float = 10,
    vmax: float = 1500,
    ticks: Union[List[float], np.ndarray] = [10, 100, 250, 500, 750, 1500],
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
    cmap : str, optional
        The colormap to be used for the plot, by default "viridis".
    vmin : float, optional
        The minimum value for the colormap, by default 10.
    vmax : float, optional
        The maximum value for the colormap, by default 1500.
    ticks : Union[List[float], np.ndarray], optional
        The ticks to be used for the colorbar, by default [10, 100, 250, 500, 750, 1500].
    """

    geom = getattr(profile_series, "geometry")
    geom_centroid = geom.centroid
    profile_centroid = gp.GeoDataFrame([profile_series], geometry=[geom_centroid])
    profile = gp.GeoDataFrame([profile_series], geometry=[geom])
    glacier_name = getattr(profile, "profile_name").values[0]
    exp_id = getattr(profile, "exp_id").values[0]
    geom = getattr(profile, "geometry")
    x, y = get_coordinates(geom).T
    x_c, y_c = round(x), round(y)
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
    fig = plt.figure(figsize=(6.2, 6.2))
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
        column="pearson_r", vmin=0, vmax=1, cmap="RdYlGn", markersize=50, legend=False, missing_kwds={}, ax=ax
    )
    ax.annotate(f"{glacier_name}", (x_c, y_c), (10, 10), xycoords="data", textcoords="offset points")
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
    if np.dot(a, n) < 0:
        n = -n

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

    Returns
    -------
    xr.Dataset
        The processed profile as an xarray Dataset.

    Notes
    -----
    This function extracts profiles from the observed and simulated datasets along the given profile, merges them, and calculates the specified statistics.
    """

    x, y = map(np.asarray, profile["geometry"].xy)
    profile_name = profile["profile_name"]
    profile_id = profile["profile_id"]

    def extract_and_prepare(
        ds: xr.Dataset, profile_name: str = profile_name, profile_id: int = profile_id, **kwargs
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
        **kwargs
            Additional keyword arguments to pass to the 'extract_profile' method.

        Returns
        -------
        xr.Dataset
            The extracted profile as an xarray Dataset.

        Notes
        -----
        This function uses the 'extract_profile' method of the 'profiles' accessor of the input dataset.
        """
        ds_profile = ds.profiles.extract_profile(x, y, profile_name=profile_name, profile_id=profile_id, **kwargs)
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
    merged_profile.profiles.calculate_stats(stats=stats)

    return merged_profile


@xr.register_dataset_accessor("profiles")
class CustomDatasetMethods:
    """
    Custom methods for xarray Dataset.

    This class is used to add custom methods to xarray Dataset objects. The methods can be accessed via the 'profiles' attribute.

    Parameters
    ----------

    xarray_obj : xr.Dataset
      The xarray Dataset to which to add the custom methods.
    """

    def __init__(self, xarray_obj: xr.Dataset):
        """
        Initialize the CustomDatasetMethods class.

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
            func,
            self._obj[x_component],
            self._obj["nx"],
            self._obj[y_component],
            self._obj["ny"],
        )
        return self._obj

    def calculate_stats(
        self,
        obs_var: str = "v",
        sim_var: str = "velsurf_mag",
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
        assert (obs_var and sim_var) in self._obj.data_vars

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

        func = {"rmsd": {"func": rmsd, "ufunc": True}, "pearson_r": {"func": pearson_r, "ufunc": False}}

        for stat in stats:
            if func[stat]["ufunc"]:
                self._obj[stat] = xr.apply_ufunc(
                    func[stat]["func"],  # type: ignore[arg-type]
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

            a = [(v in ds.data_vars) for v in normal_component_error_vars.values()]
            if np.all(np.array(a)):
                ds.profiles.add_normal_component(
                    x_component=normal_component_error_vars["x"],
                    y_component=normal_component_error_vars["y"],
                    normal_name=normal_error_var,
                )
        return ds

    def plot(
        self,
        sigma: float = 1,
        alpha: float = 0.0,
        title: Union[str, None] = None,
        obs_var: str = "v",
        obs_error_var: str = "v_err",
        sim_var: str = "velsurf_mag",
        palette: str = "Paired",
        obs_kwargs: dict = {"color": "0", "lw": 1, "marker": "o", "ms": 2},
        obs_error_kwargs: dict = {"color": "0.75"},
        sim_kwargs: dict = {"lw": 1, "marker": "o", "ms": 2},
    ) -> plt.Figure:
        """
        Plot observations and simulations along profile.

        Parameters
        ----------
        sigma : float, optional
            The standard deviation of the observations, by default 1.
        alpha : float, optional
            The alpha level for the error bars, by default 0.0.
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
            Additional keyword arguments to pass to the plot function for the observations, by default {"color": "0", "lw": 1, "marker": "o", "ms": 2}.
        obs_error_kwargs : dict, optional
            Additional keyword arguments to pass to the fill_between function for the observation errors, by default {"color": "0.75"}.
        sim_kwargs : dict, optional
            Additional keyword arguments to pass to the plot function for the simulations, by default {"lw": 1, "marker": "o", "ms": 2}.

        Returns
        -------
        plt.Figure
            The created matplotlib Figure object.
        """
        n_exps = self._obj["exp_id"].size

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.fill_between(
            self._obj["profile_axis"],
            self._obj[obs_var]
            - self._obj[obs_var] * np.sqrt((sigma * self._obj[obs_error_var] / self._obj[obs_var]) ** 2 + alpha**2),
            self._obj[obs_var]
            + self._obj[obs_var] * np.sqrt((sigma * self._obj[obs_error_var] / self._obj[obs_var]) ** 2 + alpha**2),
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
                exp_label = f"""{self._obj["exp_id"].values[i].item()} $r$={self._obj["pearson_r"].values[i][0]:.2f} rmsd={self._obj["rmsd"].values[i][0]:.0f}m/yr"""
                ax.plot(
                    self._obj["profile_axis"],
                    np.squeeze(self._obj[sim_var].isel(exp_id=i).T),
                    color=palette[i],
                    label=exp_label,
                    **sim_kwargs,
                )
        else:
            exp_label = f"""{self._obj["exp_id"].values.item()} $r$={self._obj["pearson_r"].values.item():.2f} rmsd={self._obj["rmsd"].values.item():.0f}m/yr"""
            ax.plot(
                self._obj["profile_axis"],
                np.squeeze(self._obj[sim_var].T),
                color=palette[0],
                label=exp_label,
                **sim_kwargs,
            )
        ax.set_xlabel("Distance along profile (m)")
        ax.set_ylabel("Speed (m/yr)")
        legend = ax.legend(loc="upper left")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        if title is None:
            title = self._obj["profile_name"].values.item()
        plt.title(title)
        return fig
