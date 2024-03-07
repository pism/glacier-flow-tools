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
from typing import List, Union

import numpy as np
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


# def extract_profile(ds: xr.Dataset, x: Union[list, np.ndarray], y: Union[list, np.ndarray]) -> xr.Dataset:
#     """
#     Extract a profile from a dataset given x and y coordinates.
#     """
#     profile_axis = np.sqrt((x - x[0]) ** 2 + (y - y[0]) ** 2)

#     x = xr.DataArray(x, dims="profile_axis", coords={"profile_axis": profile_axis})
#     y = xr.DataArray(y, dims="profile_axis", coords={"profile_axis": profile_axis})

#     aux_vars = ["nx", "ny"]
#     nx, ny = compute_normals(x, y)
#     n ={"nx": nx, "ny": ny}
#     das = [xr.DataArray(n[aux_var], dims="profile_axis", coords={"profile_axis": profile_axis}, name=aux_var) for aux_var in aux_vars]
#     for m_var in ds.data_vars:
#         da = ds[m_var]
#         try:
#             das.append(da.interp(x=x, y=y, kwargs={"fill_value": np.nan}))
#         except:
#             pass
#     return xr.merge(das)


def extract_profile(
    ds: xr.Dataset, x: Union[List[float], np.ndarray], y: Union[List[float], np.ndarray]
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

    for m_var in ds.data_vars:
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
