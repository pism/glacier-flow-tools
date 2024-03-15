# Copyright (C) 2015, 2016, 2018, 2021, 2023 Constantine Khroulev and Andy Aschwanden
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PYPISM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA
"""
Module provides profile functions
"""
from typing import List, Tuple, Union

import geopandas as gp
import numpy as np
import pandas as pd
import pylab as plt
import seaborn as sns
import xarray as xr


def normal(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit normal vector orthogonal to (point1-point0),
    pointing 'to the right' of (point1-point0).
    """

    a = point0 - point1
    n = np.array([-a[1], a[0]])  # compute the normal vector
    n = n / np.linalg.norm(n)  # normalize

    # flip direction if needed:
    if np.dot(a, n) < 0:
        n = -n

    return n


def tangential(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit tangential vector to (point1-point0),
    pointing 'to the right' of (point1-point0).
    """

    a = point1 - point0
    norm = np.linalg.norm(a)

    # protect from division by zero
    return a if norm == 0 else a / norm


def compute_normals(px: Union[np.ndarray, list], py: Union[np.ndarray, list]):
    """
    Compute normals to a profile described by 'p'. Normals point 'to
    the right' of the path.
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
    Compute tangetials to a profile described by 'p'.
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


def distance_from_start(px, py):
    "Initialize the distance along a profile."
    result = np.zeros_like(px)
    result[1::] = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
    return result.cumsum()


def calculate_stats(df: pd.DataFrame, col1: str, col2: str) -> pd.DataFrame:
    """
    Calculate Pearson correlation and root mean square difference between two DataFrame columns.
    """
    pearson_r = df[col2].corr(df[col1])
    rmsd = np.sqrt(np.nanmean((df[col1] - df[col2]) ** 2))
    return pd.DataFrame(data=[[pearson_r, rmsd]], columns=["pearson_r", "rmsd"])


def process_profile(
    profile,
    p: int,
    obs_ds: xr.Dataset,
    sim_ds: xr.Dataset,
    obs_var: str = "v",
    obs_error_var: str = "v_err",
    sim_var: str = "velsurf_mag",
    crs: str = "epsg:3413",
) -> Tuple[xr.Dataset, xr.Dataset, pd.DataFrame]:
    """
    Process a profile from observed and simulated datasets.
    """
    x, y = map(np.asarray, profile["geometry"].xy)
    profile_name = profile["name"]
    profile_id = profile["profile_id"]

    def extract_and_prepare(
        ds: xr.Dataset,
        profile_name: str = profile_name,
        profile_id: int = profile_id,
        obs_var: str = "v",
        obs_error_var: str = "v_err",
        sim_var: str = "velsurf_mag",
    ) -> xr.Dataset:
        ds_profile = ds.profiles.extract_profile(
            x, y, profile_name=profile_name, profile_id=profile_id
        )
        return ds_profile

    obs_profile = extract_and_prepare(obs_ds, profile_name=profile_name)
    sims_profile = extract_and_prepare(sim_ds, profile_name=profile_name)

    def merge_on_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        intersection_keys = list(set(df1.columns) & set(df2.columns))
        return pd.merge(df1, df2, on=intersection_keys)

    obs_df = obs_profile.to_dataframe().reset_index()
    sims_df = sims_profile.to_dataframe().reset_index()
    obs_sims_df = merge_on_intersection(obs_df, sims_df)

    profile_gp = gp.GeoDataFrame([profile], geometry=[profile.geometry], crs=crs)
    stats = obs_sims_df.groupby(by=["exp_id", "profile_id"]).apply(
        calculate_stats, col1=sim_var, col2=obs_var, include_groups=False
    )
    stats_profile = merge_on_intersection(
        profile_gp, stats.reset_index().assign(**profile_gp.iloc[0])
    )
    for s in ["rmsd", "pearson_r"]:
        d = xr.DataArray(
            stats.groupby(by="exp_id")[s].agg(lambda x: x).values, dims="exp_id", name=s
        )
        sims_profile[s] = d

    stats_profile = gp.GeoDataFrame(
        stats_profile, geometry=stats_profile["geometry"], crs=crs
    )
    return obs_profile, sims_profile, stats_profile


@xr.register_dataset_accessor("profiles")
class CustomDatasetMethods:
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def add_normal_component(
        self,
        x_component: str = "vx",
        y_component: str = "vy",
        normal_name: str = "v_normal",
    ) -> xr.Dataset:
        assert (x_component and y_component) in self._obj.data_vars
        func = lambda x, x_n, y, y_n: x * x_n + y * y_n
        self._obj[normal_name] = xr.apply_ufunc(
            func,
            self._obj[x_component],
            self._obj["nx"],
            self._obj[y_component],
            self._obj["ny"],
        )
        return self._obj

    def extract_profile(
        self,
        x: np.ndarray,
        y: np.ndarray,
        profile_name: str = "Glacier X",
        profile_id: int = 0,
        data_vars: List[str] = None,
    ) -> xr.Dataset:
        """
        Extract a profile from a dataset given x and y coordinates.

        Parameters:
        x: x-coordinates of the profile
        y: y-coordinates of the profile
        profile_name: name of the profile
        profile_id: id of the profile
        data_vars: list of data variables to include in the profile. If None, all data variables are included.

        Returns:
        A new xarray Dataset containing the extracted profile.
        """
        profile_axis = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)

        x: xr.DataArray
        y: xr.DataArray
        x = xr.DataArray(
            x,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["x"].attrs,
            name="x",
        )
        y = xr.DataArray(
            y,
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            attrs=self._obj["y"].attrs,
            name="y",
        )
        pid = xr.DataArray([profile_id], dims="profile_id", name="profile_id")

        nx, ny = compute_normals(x, y)
        normals = {"nx": nx, "ny": ny}

        das = [
            xr.DataArray(
                normals[var],
                dims="profile_axis",
                coords={"profile_axis": profile_axis},
                name=var,
            )
            for var in normals.keys()
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

        return xr.merge(das)

    def plot(
        self,
        sigma: float = 1,
        title: Union[str, None] = None,
        obs_var: str = "v",
        obs_error_var: str = "v_err",
        sim_var: str = "velsurf_mag",
        palette: str = "Paired",
        obs_kwargs: dict = {"color": "0", "lw": 1, "marker": "o", "ms": 2},
        obs_error_kwargs: dict = {"color": "0.75"},
        sim_kwargs: dict = {"lw": 1, "marker": "o", "ms": 2},
    ) -> plt.Figure:

        n_exps = len(self._obj["exp_id"])

        label = [
            f"{exp}: rmsd={rmsd:.0f}"
            for exp, rmsd in zip(
                self._obj["exp_id"].to_numpy(), self._obj["rmsd"].to_numpy()
            )
        ]
        fig = plt.figure()
        ax = fig.add_subplot(111)
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
        for i in range(n_exps):
            exp_label = f"""{self._obj["exp_id"][i].to_numpy()} rmsd={self._obj["rmsd"][i].to_numpy():.0f}m/yr"""
            ax.plot(
                self._obj["profile_axis"],
                self._obj[sim_var].isel(exp_id=i).T,
                color=palette[i],
                label=exp_label,
                **sim_kwargs,
            )
        ax.set_xlabel("Distance along profile (m)")
        ax.set_ylabel("Speed (m/yr)")
        legend = ax.legend(loc="upper left")
        legend.get_frame().set_linewidth(0.0)
        legend.get_frame().set_alpha(0.0)

        if title is None:
            title = self._obj["profile_name"].to_numpy()
        plt.title(title)
        return fig
