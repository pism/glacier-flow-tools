# Copyright (C) 2024 Andy Aschwanden
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
Module provides profile test functions.
"""

from typing import Dict

import geopandas as gp
import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal
from shapely import LineString, get_coordinates

from glacier_flow_tools.profiles import normal, tangential


@pytest.fixture(name="quadratic_flow")
def fixture_create_quadratic_flow() -> xr.Dataset:
    """
    Create an xarray Dataset with a quadratic velocity field.

    This function generates a 2D velocity field with a quadratic distribution in the y-direction.
    The velocity in the x-direction is zero. The velocity field is defined on a grid with
    dimensions `x` and `y` ranging from -10 to 10. The velocity field is perturbed with random noise.

    Returns
    -------
    xr.Dataset
        An xarray Dataset containing the following data variables:
        - 'vx': velocity in the x-direction (m/yr), zero everywhere.
        - 'vy': velocity in the y-direction (m/yr), quadratic distribution with added random noise.
        The Dataset also contains coordinate variables 'x', 'y', and follows the CF-1.7 conventions.
        The Dataset also includes a 'Polar_Stereographic' variable with associated projection parameters.

    Examples
    --------
    >>> quadratic_flow = fixture_create_quadratic_flow()
    >>> print(quadratic_flow)
    <xarray.Dataset>
    Dimensions:  (x: 201, y: 201)
    Coordinates:
      * x        (x) float64 -10.0 -9.9 -9.8 -9.7 -9.6 ... 9.7 9.8 9.9 10.0
      * y        (y) float64 -10.0 -9.9 -9.8 -9.7 -9.6 ... 9.7 9.8 9.9 10.0
    Data variables:
        vx       (y, x) float64 0.0 0.0 0.0 0.0 0.0 ... 0.0 0.0 0.0 0.0
        vy       (y, x) float64 ...
        Polar_Stereographic  int64 ...
    """
    nx = 201
    ny = 201
    x_min = -10
    x_max = 10
    y_min = -10
    y_max = 10
    x = np.linspace(x_min, x_max, nx)
    y = np.linspace(y_min, y_max, ny)
    X, Y = np.meshgrid(x, y)

    rng = np.random.default_rng(seed=42)
    # Directional vectors
    vy = np.zeros_like(X)
    vx = (y_max**2 - Y**2) + rng.random(size=Y.shape)
    thickness = (y_max**2 - Y**2) + rng.random(size=Y.shape)

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
    }

    ds = xr.Dataset(
        {
            "thickness": xr.DataArray(
                data=thickness,
                dims=["y", "x"],
                coords=coords,
                attrs={"standard_name": "land_ice_thickness", "units": "m"},
            ),
            "vx": xr.DataArray(
                data=vx,
                dims=["y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in x-direction", "units": "m/yr"},
            ),
            "vy": xr.DataArray(
                data=vy,
                dims=["y", "x"],
                coords=coords,
                attrs={"standard_name": "velocity in y-direction", "units": "m/yr"},
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


def test_extract_profiles(quadratic_flow):
    """
    Test profile extraction.

    Parameters
    ----------
    quadratic_flow : xr.Dataset
        A test dataset describing quadratic flow.
    """

    profiles_dict = {
        "profile_id": {0: 0, 1: 1},
        "profile_name": {0: "Horizontal Gletscher", 1: "Vertical"},
        "geometry": {0: LineString([[-10, 0], [10, 0]]), 1: LineString([[0, -10], [0, 10]])},
    }
    profiles_gp = gp.GeoDataFrame.from_dict(profiles_dict)
    geom = profiles_gp.segmentize(1.0)
    profiles_gp = gp.GeoDataFrame(profiles_gp, geometry=geom)
    profiles_gp = profiles_gp[["profile_id", "profile_name", "geometry"]]

    profile = profiles_gp.loc[[0]]
    geom = getattr(profile, "geometry")
    x_p, y_p = get_coordinates(geom).T
    profile_name = getattr(profile, "profile_name").values[0]
    profile_id = getattr(profile, "profile_id").values[0]

    kwargs: Dict = {}
    x_profile = quadratic_flow.profiles.extract_profile(
        x_p, y_p, profile_name=profile_name, profile_id=profile_id, **kwargs
    )

    profile = profiles_gp.loc[[1]]
    geom = getattr(profile, "geometry")
    x_p, y_p = get_coordinates(geom).T
    profile_name = getattr(profile, "profile_name").values[0]
    profile_id = getattr(profile, "profile_id").values[0]
    x_vy_true = np.array(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    assert_array_almost_equal(x_profile.vy, x_vy_true)

    y_profile = quadratic_flow.profiles.extract_profile(
        x_p, y_p, profile_name=profile_name, profile_id=profile_id, **kwargs
    )
    y_vx_true = np.array(
        [
            0.90858069,
            19.51916723,
            36.95104485,
            51.28749629,
            64.28259608,
            75.47610046,
            84.02253044,
            91.83760948,
            96.92277666,
            99.93217176,
            100.29915991,
            99.22948628,
            96.05096098,
            91.20227129,
            84.15173917,
            75.92502223,
            64.51036844,
            51.41748735,
            36.1520003,
            19.53233673,
            0.23981163,
        ]
    )
    assert_array_almost_equal(y_profile.vx, y_vx_true)

    kwargs.update({"compute_profile_normal": True})
    y_profile_normal = quadratic_flow.profiles.extract_profile(
        x_p, y_p, profile_name=profile_name, profile_id=profile_id, **kwargs
    )
    y_vx_normal_true = np.array(
        [
            0.90858069,
            19.51916723,
            36.95104485,
            51.28749629,
            64.28259608,
            75.47610046,
            84.02253044,
            91.83760948,
            96.92277666,
            99.93217176,
            100.29915991,
            99.22948628,
            96.05096098,
            91.20227129,
            84.15173917,
            75.92502223,
            64.51036844,
            51.41748735,
            36.1520003,
            19.53233673,
            0.23981163,
        ]
    )
    assert_array_almost_equal(y_profile_normal.vx, y_vx_normal_true)

    quadratic_flow_missing_y_component = quadratic_flow.drop_vars("vy")
    y_profile_no_normal = quadratic_flow_missing_y_component.profiles.extract_profile(
        x_p, y_p, profile_name=profile_name, profile_id=profile_id, **kwargs
    )
    assert "v_normal" not in y_profile_no_normal


def test_normal():
    """
    Test normal vector code.
    """
    point0 = np.array([0, 0])
    point1 = np.array([1, 0])

    expected_output = np.array([0, -1])
    np.testing.assert_array_equal(normal(point0, point1), expected_output)

    point0 = np.array([0, 0])
    point1 = np.array([0, 1])
    expected_output = np.array([1, 0])
    np.testing.assert_array_equal(normal(point0, point1), expected_output)

    point0 = np.array([1, 1])
    point1 = np.array([2, 2])
    expected_output = np.array([0.70710678, -0.70710678])
    np.testing.assert_array_almost_equal(normal(point0, point1), expected_output, decimal=8)


def test_tangential():
    """
    Test tangential vector code.
    """
    point0 = np.array([0, 0])
    point1 = np.array([1, 0])
    expected_output = np.array([1, 0])
    np.testing.assert_array_equal(tangential(point0, point1), expected_output)

    point0 = np.array([0, 0])
    point1 = np.array([0, 1])
    expected_output = np.array([0, 1])
    np.testing.assert_array_equal(tangential(point0, point1), expected_output)

    point0 = np.array([1, 1])
    point1 = np.array([2, 2])
    expected_output = np.array([0.70710678, 0.70710678])
    np.testing.assert_array_almost_equal(tangential(point0, point1), expected_output, decimal=8)

    point0 = np.array([0, 0])
    point1 = np.array([0, 0])
    expected_output = np.array([0, 0])
    np.testing.assert_array_equal(tangential(point0, point1), expected_output)
