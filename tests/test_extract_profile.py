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

import numpy as np
import pytest
import xarray as xr


def F(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """A function linear in x, y, and z. Used to test our interpolation
    scheme."""
    return 10.0 + 0.01 * x + 0.02 * y + 0.03 + 0.04 * z


def create_dummy_input_file(F) -> xr.Dataset:
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

    def write(prefix, dimensions):
        "Write test data to the file using given storage order."

        slices = {"x": slice(0, Mx), "y": slice(0, My), "time": 0, "z": None}
        dim_map = {"x": Mx, "y": My, "z": Mz, "time": Mt}

        # set indexes for all dimensions (z index will be re-set below)
        indexes = [Ellipsis] * len(dimensions)
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

    from itertools import permutations

    def P(x):
        return list(permutations(x))

    data_vars = dict()
    for d in sorted(P(["x", "y"]) + P(["time", "x", "y"])):
        prefix = "test_2D_"
        name = prefix + "_".join(d)
        data_vars[name] = write(prefix, d)

    for d in sorted(P(["x", "y", "z"]) + P(["time", "x", "y", "z"])):
        prefix = "test_3D_"
        name = prefix + "_".join(d)
        data_vars[name] = write(prefix, d)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            time=(["time"], [0], {}),
            z=(["z"], z, {"_FillValue": False, "units": "m"}),
            y=(
                ["y"],
                y,
                {
                    "_FillValue": False,
                    "units": "m",
                    "axis": "Y",
                    "standard_name": "projection_y_coordinate",
                },
            ),
            x=(
                ["x"],
                x,
                {
                    "_FillValue": False,
                    "units": "m",
                    "axis": "X",
                    "standard_name": "projection_x_coordinate",
                },
            ),
        ),
        attrs=dict(description="Test data."),
    )
    return ds


@pytest.fixture(name="create_dummpy_input_file")
def fixture_create_dummy_input_file_xyz():
    return create_dummy_input_file(F)
