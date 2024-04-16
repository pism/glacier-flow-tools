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
Module provides profile functions
"""

import numpy as np

from glacier_flow_tools.profiles import normal, tangential


def test_normal():
    """
    Test normal vector code
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
    Test tangential vector code
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
