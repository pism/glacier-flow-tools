# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
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
# along with glacier-flow-tools; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for the utility functions.
"""

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_allclose

from glacier_flow_tools.utils import to_numpy_in_units

SECONDS_PER_YEAR = 365.25 * 86400.0


@pytest.mark.parametrize(
    "data_units, units, factor",
    [
        ("m/yr", "m/yr", 1.0),
        ("m year-1", "m/yr", 1.0),
        ("meter/year", "m/yr", 1.0),
        ("km/yr", "m/yr", 1000.0),
        ("m/s", "m/yr", SECONDS_PER_YEAR),
        ("m s-1", "m/yr", SECONDS_PER_YEAR),
        ("m day-1", "m/yr", 365.25),
        ("m", "m", 1.0),
        ("meters", "m", 1.0),
        ("km", "m", 1000.0),
    ],
)
def test_to_numpy_in_units(data_units, units, factor):
    """
    Convert values from the units of the data to the requested units.

    Parameters
    ----------
    data_units : str
        The units attribute of the data.
    units : str
        The units to convert to.
    factor : float
        The expected conversion factor.
    """
    da = xr.DataArray(np.array([[1.0, 2.0], [3.0, np.nan]]), dims=("y", "x"), attrs={"units": data_units}, name="v")
    result = to_numpy_in_units(da, units)
    assert isinstance(result, np.ndarray)
    assert_allclose(result, da.to_numpy() * factor)


def test_to_numpy_in_units_converts_a_dimension_coordinate():
    """
    Convert a coordinate that indexes its own dimension.
    """
    ds = xr.Dataset(coords={"x": ("x", [0.0, 1.0, 2.0], {"units": "km"})})
    assert_allclose(to_numpy_in_units(ds["x"], "m"), [0.0, 1000.0, 2000.0])


def test_to_numpy_in_units_without_units():
    """
    Raise an error for data without units, unless units to assume are given.
    """
    da = xr.DataArray([1.0, 2.0], name="vx")
    with pytest.raises(ValueError, match="'vx' has no units"):
        to_numpy_in_units(da, "m/yr")
    with pytest.warns(UserWarning, match="'vx' has no units, assuming 'm/yr'"):
        result = to_numpy_in_units(da, "m/yr", assume="m/yr")
    assert_allclose(result, [1.0, 2.0])


def test_to_numpy_in_units_override():
    """
    Use the override in place of the units attribute.
    """
    da = xr.DataArray([1.0], attrs={"units": "m/y"}, name="vx")
    assert_allclose(to_numpy_in_units(da, "m/yr", override="m/d"), [365.25])


@pytest.mark.parametrize(
    "data_units, units, message",
    [
        ("m/y", "m/yr", "are not understood"),
        ("m", "m/yr", "cannot be converted to 'm/yr'"),
        ("degrees_north", "m", "cannot be converted to 'm'"),
    ],
)
def test_to_numpy_in_units_rejects(data_units, units, message):
    """
    Raise an error for units that are not understood or do not fit.

    Parameters
    ----------
    data_units : str
        The units attribute of the data.
    units : str
        The units to convert to.
    message : str
        Part of the expected error message.
    """
    da = xr.DataArray([1.0], attrs={"units": data_units}, name="vx")
    with pytest.raises(ValueError, match=message):
        to_numpy_in_units(da, units)
