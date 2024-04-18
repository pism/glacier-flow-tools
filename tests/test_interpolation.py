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
Tests for interpolation module.
"""

import numpy as np
from numpy.testing import assert_array_almost_equal

from glacier_flow_tools.interpolation import InterpolationMatrix

np.seterr(divide="ignore", invalid="ignore")


def test_masked_interpolation():
    """
    Test matrix adjustment.
    """

    # 2x2 grid of ones
    x = [0, 1, 2]
    y = [0, 1]
    z = np.ones((len(y), len(x)))
    # set the [0,0] element to a nan and mark that value
    # as "missing" by turning it into a masked array
    z[0, 0] = np.nan
    z = np.ma.array(z, mask=[[True, False, False], [False, False, False]])
    # sample in the middle
    px = 0.5
    py = 0.5

    A = InterpolationMatrix(x, y, px, py)

    # We should get the average of the three remaining ones, i.e. 1.0.
    # (We would get a nan without adjusting the matrix.)
    assert A.apply(z)[0] == 1.0


def test_masked_missing_interpolation():
    """
    Test interpolation from a masked array that produces missing values in the output.
    """

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


def test_flipped_y_interpolation():
    """
    Test interpolation from a grid with decreasing y coordinates.
    """

    x = [-2, -1, 0, 1]
    y = [1, 0, -1]

    # a linear function (perfectly recovered using bilinear
    # interpolation)
    def Z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        A linear function for testing.

        Parameters
        ----------
        x : np.ndarray
          Y is a np.ndarray.
        y : np.ndarray
          Y is a np.ndarray.

        Returns
        -------
        np.ndarray
          Returns a linear function of x and y.

        Examples
        --------
        FIXME: Add docs.
        """
        return 0.3 * x + 0.2 * y + 0.1

    xx, yy = np.meshgrid(x, y)

    z = Z(xx, yy)

    px = np.array([-1.75, -0.5, 0.75])
    py = np.array([-0.25, 0.0, 0.25])

    A = InterpolationMatrix(x, y, px, py)

    z_interpolated = A.apply(z)

    assert_array_almost_equal(z_interpolated, Z(px, py), decimal=12)


def test_interpolation():
    """
    Test interpolation by recovering values of a linear function.
    """

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
        raise RuntimeError("Update this test if you implemented nearest neighbor interpolation.")  # pragma: nocover
    except NotImplementedError:
        pass

    # initialize the interpolation matrix
    A = InterpolationMatrix(x, y, px, py)

    # a linear function (perfectly recovered using bilinear
    # interpolation)
    def Z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        A linear function for testing.

        Parameters
        ----------
        x : np.ndarray
          Y is a np.ndarray.
        y : np.ndarray
          Y is a np.ndarray.

        Returns
        -------
        np.ndarray
          Returns a linear function of x and y.

        Examples
        --------
        FIXME: Add docs.
        """
        return 0.3 * x + 0.2 * y + 0.1

    # compute values of Z on the grid
    xx, yy = np.meshgrid(x, y)
    z = Z(xx, yy)

    # interpolate
    z_interpolated = A.apply(z)

    assert_array_almost_equal(z_interpolated, Z(px, py), decimal=12)
