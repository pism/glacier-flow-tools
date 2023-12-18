# Copyright (C) 2015, 2016, 2018, 2021, 2023 Constantine Khroulev and Andy Aschwanden
#
# This file is part of pypism.
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
Test extract_profile
"""

from itertools import permutations
from pathlib import Path
from typing import Any

import numpy as np
import pyproj
import pytest
import xarray as xr
from numpy.testing import assert_array_almost_equal

from pypism.extract_profile import Profile, extract_profile, read_shapefile


def linear_function(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """A function linear in x, y, and z. Used to test our interpolation
    scheme."""
    return 10.0 + 0.01 * x + 0.02 * y + 0.03 + 0.04 * z


def create_dummy_input_dataset(F) -> xr.Dataset:
    """Create an input file for testing. Does not use unlimited
    dimensions, creates one time record only."""

    Mx = 88
    My = 152
    Mz = 11
    Mt = 1

    # use X and Y ranges corresponding to a grid covering Greenland
    x = np.linspace(-669650.0, 896350.0, Mx)
    y = np.linspace(-3362600.0, -644600.0, My)
    z = np.linspace(0, 4000.0, Mz)

    xx, yy = np.meshgrid(x, y)

    def write(dimensions: list):
        "Write test data to the file using given storage order."

        slices: dict[str, Any] = {
            "x": slice(0, Mx),
            "y": slice(0, My),
            "time": 0,
            "z": None,
        }
        dim_map = {"x": Mx, "y": My, "z": Mz, "time": Mt}

        # set indexes for all dimensions (z index will be re-set below)
        indexes: list[Any] = [Ellipsis] * len(dimensions)
        for k, d in enumerate(dimensions):
            indexes[k] = slices[d]

        # transpose 2D array if needed
        if dimensions.index("y") < dimensions.index("x"):

            def T(x):
                return x

        else:
            T = np.transpose

        dims = [dim_map[d] for d in dimensions]
        variable = np.zeros(dims)
        if "z" in dimensions:
            for k in range(Mz):
                indexes[dimensions.index("z")] = k
                variable[*indexes] = T(F(xx, yy, z[k]))
        else:
            variable[*indexes] = T(F(xx, yy, 0))

        return (dimensions, variable, {"long_name": name + " (make it long!)"})

    def P(x):
        return list(permutations(x))

    data_vars = {}
    for d in sorted(P(["x", "y"]) + P(["time", "x", "y"])):
        prefix = "test_2D_"
        name = prefix + "_".join(d)
        data_vars[name] = write(d)

    for d in sorted(P(["x", "y", "z"]) + P(["time", "x", "y", "z"])):
        prefix = "test_3D_"
        name = prefix + "_".join(d)
        data_vars[name] = write(d)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "time": (["time"], [0], {}),
            "z": (["z"], z, {"_FillValue": False, "units": "m"}),
            "y": (
                ["y"],
                y,
                {
                    "_FillValue": False,
                    "units": "m",
                    "axis": "Y",
                    "standard_name": "projection_y_coordinate",
                },
            ),
            "x": (
                ["x"],
                x,
                {
                    "_FillValue": False,
                    "units": "m",
                    "axis": "X",
                    "standard_name": "projection_x_coordinate",
                },
            ),
        },
        attrs={"description": "Test data.", "proj": "epsg:3413", "proj4": "epsg:3413"},
    )
    return ds


@pytest.fixture(name="dummy_input_dataset")
def fixture_create_dummy_input_dataset_xyz():
    """
    Return a dummy dataset.
    """
    return create_dummy_input_dataset(linear_function)


@pytest.fixture(name="dummy_profile")
def fixture_create_dummy_profile(dummy_input_dataset):
    """
    Return a dummy profile.

    """

    x = dummy_input_dataset["x"]
    y = dummy_input_dataset["y"]
    proj = dummy_input_dataset.attrs["proj"]
    projection = pyproj.Proj(str(proj))

    n_points = 4
    # move points slightly to make sure we can interpolate
    epsilon = 0.1
    x_profile = np.linspace(x[0] + epsilon, x[-1] - epsilon, n_points)
    y_profile = np.linspace(y[0] + epsilon, y[-1] - epsilon, n_points)
    x_center = 0.5 * (x_profile[0] + x_profile[-1])
    y_center = 0.5 * (y_profile[0] + y_profile[-1])

    lon, lat = projection(x_profile, y_profile, inverse=True)  # pylint: disable=E0633
    clon, clat = projection(x_center, y_center, inverse=True)  # pylint: disable=E0633

    flightline = 2
    glaciertype = 4
    flowtype = 2

    profile = Profile(
        0,
        "test profile",
        lat,
        lon,
        clat,
        clon,
        flightline,
        glaciertype,
        flowtype,
        projection,
    )

    return profile


def test_create_dummy_profile(dummy_profile):
    "Test dummy profile creation."

    Mx = 88
    My = 152

    # use X and Y ranges corresponding to a grid covering Greenland
    x = np.linspace(-669650.0, 896350.0, Mx)
    y = np.linspace(-3362600.0, -644600.0, My)

    n_points = 4
    # move points slightly to make sure we can interpolate
    epsilon = 0.1
    x_profile = np.linspace(x[0] + epsilon, x[-1] - epsilon, n_points)
    y_profile = np.linspace(y[0] + epsilon, y[-1] - epsilon, n_points)
    x_center = 0.5 * (x_profile[0] + x_profile[-1])
    y_center = 0.5 * (y_profile[0] + y_profile[-1])

    projection = pyproj.Proj(str("epsg:3413"))

    lon, lat = projection(x_profile, y_profile, inverse=True)  # pylint: disable=E0633
    clon, clat = projection(x_center, y_center, inverse=True)  # pylint: disable=E0633

    flightline = 2
    glaciertype = 4
    flowtype = 2

    assert dummy_profile.name == "test profile"
    assert dummy_profile.flightline == flightline
    assert dummy_profile.glaciertype == glaciertype
    assert dummy_profile.flowtype == flowtype
    assert_array_almost_equal(dummy_profile.lon, lon)
    assert_array_almost_equal(dummy_profile.lat, lat)
    assert_array_almost_equal(dummy_profile.center_lon, clon)
    assert_array_almost_equal(dummy_profile.center_lat, clat)


# def file_handling_test():
#     """Test functions that copy variable metadata, define variables, etc."""

#     in_filename = "metadata_test_file_1.nc"
#     out_filename = "metadata_test_file_2.nc"

#     global attributes_not_copied
#     attributes_not_copied = []

#     create_dummy_input_file(in_filename, lambda x, y, z: 0)

#     try:
#         import netCDF4

#         input_file = netCDF4.Dataset(in_filename, "r")
#         output_file = netCDF4.Dataset(out_filename, "w")

#         define_profile_variables(output_file, special_vars=True)

#         copy_global_attributes(input_file, output_file)

#         copy_dimensions(input_file, output_file, ["time"])
#         copy_dimensions(input_file, output_file, ["x", "y", "z"])

#         copy_time_dimension(input_file, output_file, "time")

#         create_variable_like(input_file, "test_2D_x_y", output_file)
#         create_variable_like(input_file, "test_2D_x_y_time", output_file)
#         create_variable_like(input_file, "test_3D_x_y_z", output_file)
#         create_variable_like(input_file, "test_3D_x_y_z_time", output_file)

#         create_variable_like(
#             input_file,
#             "test_2D_y_x",
#             output_file,
#             output_dimensions(input_file.variables["test_2D_y_x"].dimensions),
#         )

#         print(output_dimensions(("x", "y")))
#         print(output_dimensions(("x", "y", "time")))
#         print(output_dimensions(("x", "y", "z")))
#         print(output_dimensions(("x", "y", "z", "time")))

#         write_profile(output_file, 0, create_dummy_profile(in_filename))

#         input_file.close()
#         output_file.close()
#     finally:
#         import os

#         os.remove(in_filename)
#         os.remove(out_filename)


def test_profile_extraction(dummy_input_dataset, dummy_profile):
    """
    Test extract_profile by using an input file with fake data
    """

    n_points = len(dummy_profile.x)
    z = dummy_input_dataset["z"]

    desired_result = linear_function(dummy_profile.x, dummy_profile.y, 0.0)

    desired_3d_result = np.zeros((n_points, len(z)))
    for k, level in enumerate(z):
        desired_3d_result[:, k] = linear_function(
            dummy_profile.x, dummy_profile.y, level.to_numpy()
        )

    def P(x):
        return list(permutations(x))

    # 2D variables
    for d in P(["x", "y"]) + P(["time", "x", "y"]):
        variable_name = "test_2D_" + "_".join(d)
        variable = dummy_input_dataset[variable_name]

        result, _ = extract_profile(variable, dummy_profile)

        assert_array_almost_equal(np.squeeze(result), desired_result)

    # 3D variables
    for d in P(["x", "y", "z"]) + P(["time", "x", "y", "z"]):
        variable_name = "test_3D_" + "_".join(d)
        variable = dummy_input_dataset[variable_name]

        result, _ = extract_profile(variable, dummy_profile)

        assert_array_almost_equal(np.squeeze(result), desired_3d_result)


# def profile_extraction_test():
#     """Test extract_profile() by using an input file with fake data."""

#     def F(x, y, z):
#         """A function linear in x, y, and z. Used to test our interpolation
#         scheme."""
#         return 10.0 + 0.01 * x + 0.02 * y + 0.03 + 0.04 * z

#     # create a test file
#     import os
#     import tempfile

#     fd, filename = tempfile.mkstemp(suffix=".nc", prefix="extract_profile_test_")
#     os.close(fd)

#     create_dummy_input_file(filename, F)

#     import netCDF4

#     nc = netCDF4.Dataset(filename)

#     profile = create_dummy_profile(filename)
#     n_points = len(profile.x)
#     z = nc.variables["z"][:]

#     desired_result = F(profile.x, profile.y, 0.0)

#     desired_3d_result = np.zeros((n_points, len(z)))
#     for k, level in enumerate(z):
#         desired_3d_result[:, k] = F(profile.x, profile.y, level)

#     from itertools import permutations

#     def P(x):
#         return list(permutations(x))

#     try:
#         # 2D variables
#         for d in P(["x", "y"]) + P(["time", "x", "y"]):
#             print("Trying %s..." % str(d))
#             variable_name = "test_2D_" + "_".join(d)
#             variable = nc.variables[variable_name]

#             result, _ = extract_profile(variable, profile)

#             assert np.max(np.fabs(np.squeeze(result) - desired_result)) < 1e-9
#         # 3D variables
#         for d in P(["x", "y", "z"]) + P(["time", "x", "y", "z"]):
#             print("Trying %s..." % str(d))
#             variable_name = "test_3D_" + "_".join(d)
#             variable = nc.variables[variable_name]

#             result, _ = extract_profile(variable, profile)

#             assert np.max(np.fabs(np.squeeze(result) - desired_3d_result)) < 1e-9
#     finally:
#         os.remove(filename)
#         nc.close()


# def profile_test():
#     """Test Profile constructor."""
#     import pyproj

#     x = np.linspace(-1.0, 1.0, 101)
#     y = np.linspace(1.0, -1.0, 101)

#     projection = pyproj.Proj("+proj=latlon")

#     lon, lat = projection(x, y, inverse=True)

#     center_lat, center_lon = projection(0.0, 0.0, inverse=True)

#     flightline = None
#     glaciertype = None
#     flowtype = None

#     profile = Profile(
#         0,
#         "test_profile",
#         lat,
#         lon,
#         center_lat,
#         center_lon,
#         flightline,
#         glaciertype,
#         flowtype,
#         projection,
#     )

#     assert profile.nx[0] == -1.0 / np.sqrt(2.0)
#     assert profile.ny[0] == -1.0 / np.sqrt(2.0)

#     assert np.fabs(profile.distance_from_start[1] - 0.02 * np.sqrt(2.0)) < 1e-12

#     x = -1.0 * x
#     lon, lat = projection(x, y, inverse=True)

#     profile = Profile(
#         0,
#         "flipped_profile",
#         lat,
#         lon,
#         center_lat,
#         center_lon,
#         flightline,
#         glaciertype,
#         flowtype,
#         projection,
#         flip=True,
#     )

#     assert profile.nx[0] == -1.0 / np.sqrt(2.0)
#     assert profile.ny[0] == 1.0 / np.sqrt(2.0)

#     x = np.linspace(-1.0, 1.0, 101)
#     y = np.zeros_like(x)
#     lon, lat = projection(x, y, inverse=True)

#     profile = Profile(
#         0,
#         "test_profile",
#         lat,
#         lon,
#         center_lat,
#         center_lon,
#         flightline,
#         glaciertype,
#         flowtype,
#         projection,
#     )

#     assert profile.nx[0] == 0.0
#     assert profile.ny[0] == -1.0


def test_read_shapefile():
    """
    Test reading a shapefile
    """
    filenames = [
        Path("tests/data/greenland-flux-gates-29_500m.shp"),
        Path("tests/data/greenland-flux-gates-29_500m.gpkg"),
    ]

    for filename in filenames:
        profiles = read_shapefile(filename)

        assert len(profiles) == 28
