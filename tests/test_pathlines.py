# Copyright (C) 2023-24 Andy Aschwanden
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
Tests for pathlines.
"""

import geopandas as gp
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal
from pandas.testing import assert_frame_equal
from shapely.geometry import Point

from glacier_flow_tools.interpolation import velocity
from glacier_flow_tools.pathlines import (
    compute_pathline,
    compute_pathline_rkf,
    pathline_to_geopandas_dataframe,
)

np.seterr(divide="ignore", invalid="ignore")


@pytest.fixture(name="create_linear_flow")
def fixture_create_linear_flow() -> xr.Dataset:
    """
    Create xr.Dataset with linear velocity field.

    Returns
    -------
    xr.Dataset
        Dataset describing linear flow.

    Examples
    --------
    FIXME: Add docs.
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


def test_linear_flow_np(create_linear_flow: xr.Dataset):
    """
    Test linear flow.

    Parameters
    ----------
    create_linear_flow : xr.Dataset
        A test dataset describing linear flow.

    Examples
    --------
    FIXME: Add docs.
    """

    ds = create_linear_flow

    def exact_solution(x0: np.ndarray, t: float) -> np.ndarray:
        """
        Exact solution for linear flow.

        Parameters
        ----------
        x0 : np.ndarray
            Initial position.
        t : float
            Time.

        Returns
        -------

        np.ndarray
          Exact position after time t.

        Examples
        --------
        FIXME: Add docs.
        """
        return x0 * np.exp([t, -t])

    Vx = np.squeeze(ds["vx"].to_numpy())
    Vy = np.squeeze(ds["vy"].to_numpy())
    x = ds["x"].to_numpy()
    y = ds["y"].to_numpy()
    total_time = 1.0
    starting_point = np.array([0.05, 0.95])

    r_exact = exact_solution(starting_point, total_time)

    dt = 0.0001
    pts, _, _ = compute_pathline(starting_point, Vx, Vy, x, y, total_time=total_time, dt=dt)
    assert_array_almost_equal(pts[-1, :], *r_exact)

    pts, _, _, _ = compute_pathline_rkf(
        starting_point, velocity, f_args=(Vx, Vy, x, y), start_time=0, end_time=total_time, hmin=0.1, hmax=0.1, tol=1e-4
    )
    assert_array_almost_equal(pts[-1, :], *r_exact)


def test_pathline_to_geopandas():
    """
    Test converting pathline points to geopandas.GeoDataFrame.
    """

    true_dict = {
        "geometry": {
            0: Point(0, -50000),
            1: Point(249.994, -49750.006),
            2: Point(499.975, -49500.025),
            3: Point(749.943, -49250.057),
            4: Point(999.897, -49000.103),
            5: Point(1249.825, -48750.175),
            6: Point(1499.713, -48500.287),
            7: Point(1749.56, -48250.44),
            8: Point(1999.364, -48000.636),
            9: Point(2249.113, -47750.887),
            10: Point(2498.79, -47501.21),
        },
        "pathline_id": {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0, 10: 0},
        "vx": {
            0: 250.0,
            1: 249.98737724611084,
            2: 249.97450152124216,
            3: 249.9613611799404,
            4: 249.94796017540068,
            5: 249.90805496694236,
            6: 249.86733950159993,
            7: 249.8257702467272,
            8: 249.7833771945812,
            9: 249.71298833558737,
            10: 249.64113140550904,
        },
        "vy": {
            0: 250.0,
            1: 249.98737724611084,
            2: 249.97450152124216,
            3: 249.9613611799404,
            4: 249.94796017540068,
            5: 249.90805496694236,
            6: 249.86733950159993,
            7: 249.8257702467272,
            8: 249.7833771945812,
            9: 249.71298833558737,
            10: 249.64113140550904,
        },
    }
    gp_true = gp.GeoDataFrame.from_dict(true_dict)

    pts_shapely_points = [
        Point(0, -50000),
        Point(249.994, -49750.006),
        Point(499.975, -49500.025),
        Point(749.943, -49250.057),
        Point(999.897, -49000.103),
        Point(1249.825, -48750.175),
        Point(1499.713, -48500.287),
        Point(1749.56, -48250.44),
        Point(1999.364, -48000.636),
        Point(2249.113, -47750.887),
        Point(2498.79, -47501.21),
    ]

    v = np.array(
        [
            [250.0, 250.0],
            [249.98737725, 249.98737725],
            [249.97450152, 249.97450152],
            [249.96136118, 249.96136118],
            [249.94796018, 249.94796018],
            [249.90805497, 249.90805497],
            [249.8673395, 249.8673395],
            [249.82577025, 249.82577025],
            [249.78337719, 249.78337719],
            [249.71298834, 249.71298834],
            [249.64113141, 249.64113141],
        ]
    )

    pts_np_points = np.array([p.xy for p in pts_shapely_points]).reshape(-1, 2)

    attributes = {"pathline_id": 0, "vx": [vx[0] for vx in v], "vy": [vy[1] for vy in v]}

    pathline_gp_np = pathline_to_geopandas_dataframe(pts_np_points, attributes)
    assert_frame_equal(pathline_gp_np, gp_true, check_exact=False, atol=0.01)

    pathline_gp_shapely = pathline_to_geopandas_dataframe(pts_shapely_points, attributes)
    assert_frame_equal(pathline_gp_shapely, gp_true, check_exact=False, atol=0.01)
