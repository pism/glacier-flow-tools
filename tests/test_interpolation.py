# Copyright (C) 2023 Andy Aschwanden
#
# This file is part of pism-ragis.
#
# PISM-RAGIS is free software; you can redistribute it and/or modify it under the
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
import xarray as xr

from trajectory.interpolation import InterpolationMatrix

np.seterr(divide="ignore", invalid="ignore")


def test_masked_interpolation():
    """Test matrix adjustment."""

    # 2x2 grid of ones
    x = [0, 1, 2]
    y = [0, 1]
    z = np.ones((len(y), len(x)))
    # set the [0,0] element to a nan and mark that value
    # as "missing" by turning it into a masked array
    z[0, 0] = np.nan
    z = np.ma.array(z, mask=[[True, False, False], [False, False, False]])
    # sample in the middle
    px = [0.5]
    py = [0.5]

    A = InterpolationMatrix(x, y, px, py)

    # We should get the average of the three remaining ones, i.e. 1.0.
    # (We would get a nan without adjusting the matrix.)
    assert A.apply(z)[0] == 1.0


def test_masked_missing_interpolation():
    """Test interpolation from a masked array that produces missing values
    in the output."""

    x = [-1, 0, 1, 2]
    y = [-1, 0, 1]
    z = np.ones((len(y), len(x)))

    # set the four elements in the corner to nan and mark them as
    # missing
    z[0:2, 0:2] = np.nan
    mask = np.zeros_like(z, dtype=np.bool_)
    mask[0:2, 0:2] = np.nan

    z = np.ma.array(z, mask=mask)

    px = [-0.5, 0.5]
    py = [-0.5, 0.5]

    A = InterpolationMatrix(x, y, px, py)

    z_interpolated = A.apply(z)

    assert z_interpolated.mask[0] == True  # noqa: E712  # pylint: disable=C0121
    assert z_interpolated[1] == 1.0


def test_interpolation():
    """Test interpolation by recovering values of a linear function."""

    Lx = 10.0  # size of the box in the x direction
    Ly = 20.0  # size of the box in the y direction
    P = 1000  # number of test points

    # grid size (note: it should not be a square)
    Mx = 101
    My = 201
    x = np.linspace(0, Lx, Mx)
    y = np.linspace(0, Ly, My)

    # test points
    np.random.seed([100])
    px = np.random.rand(P) * Lx
    py = np.random.rand(P) * Ly

    try:
        A = InterpolationMatrix(x, y, px, py, bilinear=False)
        raise RuntimeError(
            "Update this test if you implemented nearest neighbor interpolation."
        )  # pragma: nocover
    except NotImplementedError:
        pass

    # initialize the interpolation matrix
    A = InterpolationMatrix(x, y, px, py)

    # a linear function (perfectly recovered using bilinear
    # interpolation)
    def Z(x, y):
        "A linear function for testing."
        return 0.3 * x + 0.2 * y + 0.1

    # compute values of Z on the grid
    xx, yy = np.meshgrid(x, y)
    z = Z(xx, yy)

    # interpolate
    z_interpolated = A.apply(z)

    assert np.max(np.fabs(z_interpolated - Z(px, py))) < 1e-12


def test_circular():
    """
    Test circular velocity field
    """
    time = pd.date_range("2000-01-01", periods=1)

    nx = 201
    ny = 401
    x = np.linspace(-100e3, 100e3, nx)
    y = np.linspace(-200e3, 200e3, ny)
    X, Y = np.meshgrid(x, y)

    # Directional vectors
    vx = -Y / np.sqrt(X**2 + Y**2) * 250
    vy = X / np.sqrt(X**2 + Y**2) * 250
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
