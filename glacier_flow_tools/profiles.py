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
Module provides profile functions
"""
from typing import List, Union

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


def compute_normals(px: Union[np.ndarray, xr.DataArray, list], py: Union[np.ndarray, xr.DataArray, list]):
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

    with np.errstate(divide="ignore", invalid="ignore"):

        diff = df[col1] - df[col2]
        if np.isnan(diff).all() or (len(diff) <= 2):
            rmsd = np.nan
            pearson_r = np.nan
        else:
            pearson_r = df[col2].corr(df[col1])
            rmsd = np.sqrt(np.nanmean(diff**2))
    return pd.DataFrame(data=[[pearson_r, rmsd]], columns=["pearson_r", "rmsd"])


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
    Process a profile from observed and simulated datasets.
    """
    x, y = map(np.asarray, profile["geometry"].xy)
    profile_name = profile["profile_name"]
    profile_id = profile["profile_id"]

    def extract_and_prepare(
        ds: xr.Dataset, profile_name: str = profile_name, profile_id: int = profile_id, **kwargs
    ) -> xr.Dataset:
        """
        Extract from xr.Dataset along (x,y) profile.
        """
        ds_profile = ds.profiles.extract_profile(
            x, y, profile_name=profile_name, profile_id=profile_id, normal_var=obs_normal_var, **kwargs
        )
        return ds_profile

    obs_profile = extract_and_prepare(
        obs_ds,
        profile_name=profile_name,
        profile_id=profile_id,
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
    Custom Dataset Methods
    """

    def __init__(self, xarray_obj):
        """
        Init
        """
        self._obj = xarray_obj

    def init(self):
        """
        Do-nothing method

        Needed to work with joblib Parallel
        """

    def add_normal_component(
        self,
        x_component: str = "vx",
        y_component: str = "vy",
        normal_name: str = "v_normal",
    ) -> xr.Dataset:
        """
        Add normal component
        """
        assert (x_component and y_component) in self._obj.data_vars

        def func(x, x_n, y, y_n):
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
    ) -> xr.Dataset:
        """
        Add rmsd
        """
        assert (obs_var and sim_var) in self._obj.data_vars

        def rmsd(sim, obs):
            diff = sim - obs

            return np.sqrt(np.nanmean(diff**2, axis=-1))

        def pearson_r(sim, obs):

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

        Parameters:
        x: x-coordinates of the profile
        y: y-coordinates of the profile
        profile_name: name of the profile
        profile_id: id of the profile
        data_vars: list of data variables to include in the profile. If None, all data variables are included.

        Returns:
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

            a = [(v in v.data_vars) for v in normal_component_vars.values()]
            assert np.alltrue(np.array(a))  # type: ignore[attr-defined]
            a = [(v in v.data_vars) for v in normal_component_error_vars.values()]
            assert np.alltrue(np.array(a))  # type: ignore[attr-defined]

            ds.profiles.add_normal_component(
                x_component=normal_component_vars["x"],
                y_component=normal_component_vars["y"],
                normal_name=normal_var,
            )
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
