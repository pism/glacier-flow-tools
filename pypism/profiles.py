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
    profile, p: int, obs_ds: xr.Dataset, sim_ds: xr.Dataset, crs: str = "epsg:3413"
) -> Tuple[xr.Dataset, xr.Dataset, pd.DataFrame]:
    """
    Process a profile from observed and simulated datasets.
    """
    x, y = map(np.asarray, profile["geometry"].xy)

    def extract_and_prepare(ds: xr.Dataset) -> xr.Dataset:
        ds_profile = ds.profiles.extract_profile(x, y)
        ds_profile = ds_profile.expand_dims(dim="profile_id")
        ds_profile["profile_id"] = [p]
        return ds_profile

    obs_profile = extract_and_prepare(obs_ds)
    sims_profile = extract_and_prepare(sim_ds)

    def merge_on_intersection(df1: pd.DataFrame, df2: pd.DataFrame) -> pd.DataFrame:
        intersection_keys = list(set(df1.columns) & set(df2.columns))
        return pd.merge(df1, df2, on=intersection_keys)

    obs_df = obs_profile.to_dataframe().reset_index()
    sims_df = sims_profile.to_dataframe().reset_index()
    obs_sims_df = merge_on_intersection(obs_df, sims_df)

    profile_gp = gp.GeoDataFrame([profile], geometry=[profile.geometry], crs=crs)
    stats = obs_sims_df.groupby(by=["exp_id", "profile_id"]).apply(
        calculate_stats, col1="velsurf_mag", col2="v", include_groups=False
    )
    stats_profile = merge_on_intersection(
        profile_gp, stats.reset_index().assign(**profile_gp.iloc[0])
    )

    stats_profile = gp.GeoDataFrame(
        stats_profile, geometry=stats_profile["geometry"], crs=crs
    )
    return obs_profile, sims_profile, stats_profile


def extract_profile(
    ds: xr.Dataset,
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    data_vars: List[str] = None,
) -> xr.Dataset:
    """
    Extract a profile from a dataset given x and y coordinates.
    """
    profile_axis = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)

    x = xr.DataArray(x, dims="profile_axis", coords={"profile_axis": profile_axis})
    y = xr.DataArray(y, dims="profile_axis", coords={"profile_axis": profile_axis})

    nx, ny = compute_normals(x, y)
    normals = {"nx": nx, "ny": ny}

    das = [
        xr.DataArray(
            normals[aux_var],
            dims="profile_axis",
            coords={"profile_axis": profile_axis},
            name=aux_var,
        )
        for aux_var in ["nx", "ny"]
    ]
    if data_vars is None:
        data_vars = ds.data_vars

    for m_var in data_vars:
        da = ds[m_var]
        try:
            das.append(da.interp(x=x, y=y, kwargs={"fill_value": np.nan}))
        except:
            pass

    return xr.merge(das)


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
        x: Union[List[float], np.ndarray],
        y: Union[List[float], np.ndarray],
        data_vars: List[str] = None,
    ) -> xr.Dataset:
        """
        Extract a profile from a dataset given x and y coordinates.
        """
        profile_axis = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)

        x = xr.DataArray(x, dims="profile_axis", coords={"profile_axis": profile_axis})
        y = xr.DataArray(y, dims="profile_axis", coords={"profile_axis": profile_axis})

        nx, ny = compute_normals(x, y)
        normals = {"nx": nx, "ny": ny}

        das = [
            xr.DataArray(
                normals[aux_var],
                dims="profile_axis",
                coords={"profile_axis": profile_axis},
                name=aux_var,
            )
            for aux_var in ["nx", "ny"]
        ]
        if data_vars is None:
            data_vars = self._obj.data_vars

        for m_var in data_vars:
            da = self._obj[m_var]
            try:
                das.append(da.interp(x=x, y=y, kwargs={"fill_value": np.nan}))
            except:
                pass

        return xr.merge(das)
