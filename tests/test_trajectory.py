# Copyright (C) 2023 Andy Aschwanden
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-RAGIS is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for procesing module
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal
from shapely import Point

from pypism.trajectory import compute_trajectory

np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture(name="create_linear_flow")
def fixture_create_linear_flow() -> xr.Dataset:
    """
    Create xr.Dataset with linear velocity field
    """
    time = pd.date_range("2000-01-01", periods=1)

    nx = 201
    ny = 201
    x_min = -1
    x_max = 1
    y_min = -1
    y_max = 1
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    # Directional vectors
    vx = X
    vy = -Y
    v = np.sqrt(vx**2 + vy**2)

    vx = vx.reshape(1, ny, nx)
    vy = vy.reshape(1, ny, nx)
    v = v.reshape(1, ny, nx)

    v_err = v / 10
    vx_err = np.abs(vx / 20)
    vy_err = np.abs(vy / 20)

    coords = {
        "x": (
            ["x"],
            x,
            {
                "units": "m",
                "axis": "X",
                "standard_name": "projection_x_coordinate",
                "long_name": "x-coordinate in projected coordinate system",
            },
        ),
        "y": (
            ["y"],
            y,
            {
                "units": "m",
                "axis": "Y",
                "standard_name": "projection_y_coordinate",
                "long_name": "y-coordinate in projected coordinate system",
            },
        ),
        "time": (["time"], time, {}),
    }

    ds = xr.Dataset(
        {
            "vx": xr.DataArray(
                data=vx,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in x-direction", "units": "m/yr"},
            ),
            "vy": xr.DataArray(
                data=vy,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in y-direction", "units": "m/yr"},
            ),
            "v": xr.DataArray(
                data=v,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={
                    "standard_name": "magnitude",
                    "units": "m/yr",
                    "grid_mapping": "polar_stereographic",
                },
            ),
            "vx_err": xr.DataArray(
                data=vx_err,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in x-direction", "units": "m/yr"},
            ),
            "vy_err": xr.DataArray(
                data=vy_err,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in y-direction", "units": "m/yr"},
            ),
            "v_err": xr.DataArray(
                data=v_err,
                dims=["time", "y", "x"],
                coords=coords,
                attrs={
                    "standard_name": "magnitude",
                    "units": "m/yr",
                    "grid_mapping": "polar_stereographic",
                },
            ),
        },
        attrs={"Conventions": "CF-1.7"},
    )
    ds["Polar_Stereographic"] = int()
    ds.Polar_Stereographic.attrs["grid_mapping_name"] = "polar_stereographic"
    ds.Polar_Stereographic.attrs["false_easting"] = 0.0
    ds.Polar_Stereographic.attrs["false_northing"] = 0.0
    ds.Polar_Stereographic.attrs["latitude_of_projection_origin"] = 90.0
    ds.Polar_Stereographic.attrs["scale_factor_at_projection_origin"] = 1.0
    ds.Polar_Stereographic.attrs["standard_parallel"] = 70.0
    ds.Polar_Stereographic.attrs["straight_vertical_longitude_from_pole"] = -45
    ds.Polar_Stereographic.attrs["proj_params"] = "epsg:3413"

    return ds


def test_linear_flow(create_linear_flow):
    """
    Test linear flow

    """
    ds = create_linear_flow

    def exact_solution(x0, t):
        x = x0.x * np.exp(t)
        y = x0.y * np.exp(-t)
        return Point(x, y)

    Vx = np.squeeze(ds["vx"].to_numpy())
    Vy = np.squeeze(ds["vy"].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()
    total_time = 1
    starting_point = Point(0.05, 0.95)

    r_exact = exact_solution(starting_point, total_time)

    dt = 0.0001
    pts, _ = compute_trajectory(
        starting_point, Vx, Vy, x, y, total_time=total_time, dt=dt
    )

    assert_array_almost_equal([pts[-1].x, pts[-1].y], [r_exact.x, r_exact.y], decimal=4)
