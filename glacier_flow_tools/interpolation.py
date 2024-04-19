# Copyright (C) 2023 Andy Aschwanden, Constantine Khroulev
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
Module provides functions for interpolation.
"""

import numbers
from typing import Tuple, Union

import numpy as np
import scipy
from numpy import ndarray


class InterpolationMatrix:
    """
    Stores bilinear and nearest neighbor interpolation weights used to
    extract profiles.
    """

    def __init__(
        self,
        x: Union[list, ndarray],
        y: Union[list, ndarray],
        px: Union[float, list, ndarray],
        py: Union[float, list, ndarray],
        bilinear: bool = True,
    ):
        """
        Interpolate values of z to points (px,py) assuming that z is on a
        regular grid defined by x and y.
        """
        super().__init__()

        def to_array(data):
            """
            Convert data to an array.
            """
            if isinstance(data, numbers.Number):
                data = np.array([data])
            elif isinstance(data, list):
                data = np.array(data)
            return data

        px_arr: ndarray = to_array(px)
        py_arr: ndarray = to_array(py)

        assert px_arr.size == py_arr.size

        # The grid has to be equally spaced.
        assert np.fabs(np.diff(x).max() - np.diff(x).min()) < 1e-9
        assert np.fabs(np.diff(y).max() - np.diff(y).min()) < 1e-9

        dx = x[1] - x[0]
        dy = y[1] - y[0]

        assert dx != 0
        assert dy != 0

        cs = [self.grid_column(x, dx, p_x) for p_x in px_arr]
        rs = [self.grid_column(y, dy, p_y) for p_y in py_arr]

        self.c_min = np.min(cs)
        self.c_max = min(np.max(cs) + 1, len(x) - 1)

        self.r_min = np.min(rs)
        self.r_max = min(np.max(rs) + 1, len(y) - 1)

        # compute the size of the subset needed for interpolation
        self.n_rows = self.r_max - self.r_min + 1
        self.n_cols = self.c_max - self.c_min + 1

        n_points = len(px_arr)
        self.A = scipy.sparse.lil_matrix((n_points, self.n_rows * self.n_cols))

        if bilinear:
            self._compute_bilinear_matrix(x, y, dx, dy, px_arr, py_arr)
        else:
            raise NotImplementedError

    def column(self, r, c):
        """
        Interpolation matrix column number corresponding to r,c of the
        array *subset*. This is the same as the linear index within
        the subset needed for interpolation.
        """
        return self.n_cols * min(r, self.n_rows - 1) + min(c, self.n_cols - 1)

    @staticmethod
    def find(grid, delta, point):
        """
        Find the point to the left of point on the grid with spacing
        delta.
        """
        if delta > 0:
            # grid points are stored in the increasing order
            if point <= grid[0]:  # pylint: disable=R1705
                return 0
            elif point >= grid[-1]:
                return len(grid) - 1  # pylint: disable=R1705
            else:
                return int(np.floor((point - grid[0]) / delta))
        else:
            # grid points are stored in the decreasing order
            if point >= grid[0]:  # pylint: disable=R1705
                return 0
            elif point <= grid[-1]:
                return len(grid) - 1
            else:
                return int(np.floor((point - grid[0]) / delta))

    def grid_column(self, x, dx, X):
        "Input grid column number corresponding to X."
        return self.find(x, dx, X)

    def grid_row(self, y, dy, Y):
        "Input grid row number corresponding to Y."
        return self.find(y, dy, Y)

    def _compute_bilinear_matrix(self, x, y, dx, dy, px, py):
        """Initialize a bilinear interpolation matrix."""
        for k in range(self.A.shape[0]):
            x_k = px[k]
            y_k = py[k]

            x_min = np.min(x)
            x_max = np.max(x)

            y_min = np.min(y)
            y_max = np.max(y)

            # make sure we are in the bounding box defined by the grid
            x_k = max(x_k, x_min)
            x_k = min(x_k, x_max)
            y_k = max(y_k, y_min)
            y_k = min(y_k, y_max)

            C = self.grid_column(x, dx, x_k)
            R = self.grid_row(y, dy, y_k)

            alpha = (x_k - x[C]) / dx
            beta = (y_k - y[R]) / dy

            if alpha < 0.0:
                alpha = 0.0
            elif alpha > 1.0:
                alpha = 1.0

            if beta < 0.0:
                beta = 0.0
            elif beta > 1.0:
                beta = 1.0

            # indexes within the subset needed for interpolation
            c = C - self.c_min
            r = R - self.r_min

            self.A[k, self.column(r, c)] += (1.0 - alpha) * (1.0 - beta)
            self.A[k, self.column(r + 1, c)] += (1.0 - alpha) * beta
            self.A[k, self.column(r, c + 1)] += alpha * (1.0 - beta)
            self.A[k, self.column(r + 1, c + 1)] += alpha * beta

    def adjusted_matrix(self, mask):
        """Return adjusted interpolation matrix that ignores missing (masked)
        values."""

        A = self.A.tocsr()
        n_points = A.shape[0]

        output_mask = np.zeros(n_points, dtype=np.bool_)

        for r in range(n_points):
            # for each row, i.e. each point along the profile
            row = np.s_[A.indptr[r] : A.indptr[r + 1]]
            # get the locations and values
            indexes = A.indices[row]
            values = A.data[row]

            # if a particular location is masked, set the
            # interpolation weight to zero
            for k, index in enumerate(indexes):
                if np.ravel(mask)[index]:
                    values[k] = 0.0

            # normalize so that we still have an interpolation matrix
            if values.sum() > 0:
                values = values / values.sum()
            else:
                output_mask[r] = True

            A.data[row] = values

        A.eliminate_zeros()

        return A, output_mask

    def apply(self, array):
        """Apply the interpolation to an array. Returns values at points along
        the profile."""
        subset = array[self.r_min : self.r_max + 1, self.c_min : self.c_max + 1]
        return self.apply_to_subset(subset)

    def apply_to_subset(self, subset):
        """Apply interpolation to an array subset."""

        if np.ma.is_masked(subset):
            A, mask = self.adjusted_matrix(subset.mask)
            data = A * np.ravel(subset)
            return np.ma.array(data, mask=mask)

        return self.A.tocsr() * np.ravel(subset)


# def interpolate_rkf(
#     Vx: ndarray,
#     Vy: ndarray,
#     x: ndarray,
#     y: ndarray,
#     start_pt: Union[list, ndarray],
#     delta_time: float = 0.1,
# ) -> Tuple[Union[list, ndarray], Union[list, ndarray], float]:

#     def check_nan(val):
#         if np.any(np.isnan(val)):
#             return True
#         return False

#     def interpolate_and_check(Vx, Vy, x, y, pt):
#         v = interpolate_at_point(Vx, Vy, x, y, *pt)
#         if check_nan(v):
#             return [np.nan, np.nan]
#         return v

#     def calculate_k(p, k_values):
#         factors = [
#             (0.25, 0),
#             (3.0 / 32.0, 9.0 / 32.0),
#             (1932.0 / 2197.0, -7200.0 / 2197.0, 7296.0 / 2197.0),
#             (439.0 / 216.0, -8.0, 3680.0 / 513.0, -845.0 / 4104.0),
#             (-8.0 / 27.0, 2.0, -3544.0 / 2565.0, 1859.0 / 4104.0, -11.0 / 40.0),
#         ]
#         for i, factor in enumerate(factors):
#             p += delta_time * sum(f * v for f, v in zip(factor, k_values[: len(factor)]))
#             v = interpolate_and_check(Vx, Vy, x, y, p)
#             if check_nan(v):
#                 return [np.nan, np.nan]
#             k_values.append(v)
#         return k_values

#     if check_nan(start_pt):
#         return start_pt, [np.nan, np.nan], np.nan

#     k_values = [interpolate_and_check(Vx, Vy, x, y, start_pt)]
#     if check_nan(k_values[0]):
#         return [np.nan, np.nan], k_values[0], np.nan

#     k_values = calculate_k(start_pt, k_values)

#     rkf_4o_pt = start_pt + delta_time * (
#         (25.0 / 216.0) * k_values[0]
#         + (1408.0 / 2565.0) * k_values[2]
#         + (2197.0 / 4104.0) * k_values[3]
#         - (1.0 / 5.0) * k_values[4]
#     )

#     interp_pt = start_pt + delta_time * (
#         (16.0 / 135.0) * k_values[0]
#         + (6656.0 / 12825.0) * k_values[2]
#         + (28561.0 / 56430.0) * k_values[3]
#         - (9.0 / 50.0) * k_values[4]
#         + (2.0 / 55.0) * k_values[5]
#     )

#     interp_pt_error_estim = distance(interp_pt, rkf_4o_pt)

#     interp_v = interpolate_and_check(Vx, Vy, x, y, interp_pt)

#     return interp_pt, interp_v, interp_pt_error_estim


def interpolate_rkf_np(
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    start_pt: Union[list, ndarray],
    delta_time: float = 0.1,
) -> Tuple[Union[list, ndarray], Union[list, ndarray], float]:
    """
    Interpolate point-like object position according to the Runge-Kutta-Fehlberg method.

    This function computes the new position of a point-like object moving in a velocity field (Vx, Vy) over a grid (x, y) using the Runge-Kutta-Fehlberg method. The function also estimates the error of the interpolation.

    Parameters
    ----------
    Vx : ndarray
        The x-component of the velocity field.
    Vy : ndarray
        The y-component of the velocity field.
    x : ndarray
        The coordinates in the x direction.
    y : ndarray
        The coordinates in the y direction.
    start_pt : Union[list, ndarray]
        The initial position of the point-like object. It should be a list or 1D array of length 2, where the first element is the x-coordinate and the second element is the y-coordinate.
    delta_time : float, optional
        The time step for the interpolation. The default is 0.1.

    Returns
    -------
    interp_pt : Union[list, ndarray]
        The interpolated position of the point-like object at the incremented time.
    interp_v : Union[list, ndarray]
        The interpolated velocity of the point-like object at the incremented time.
    interp_pt_error_estim : float
        The estimated error of the interpolated position.

    Examples
    --------
    >>> Vx = np.array([[0, 1], [0, 1]])
    >>> Vy = np.array([[0, 0], [1, 1]])
    >>> x = np.array([0, 1])
    >>> y = np.array([0, 1])
    >>> start_pt = [0, 0]
    >>> delta_time = 0.1
    >>> interpolate_rkf_np(Vx, Vy, x, y, start_pt, delta_time)
    ([0.1, 0.0], [0.1, 0.0], 0.0)
    """

    def k2(p, k1_v):
        return p + (0.25) * delta_time * k1_v

    def k3(p, k1_v, k2_v):
        return p + (3.0 / 32.0) * delta_time * k1_v + (9.0 / 32.0) * delta_time * k2_v

    def k4(p, k1_v, k2_v, k3_v):
        return (
            p
            + (1932.0 / 2197.0) * delta_time * k1_v
            - (7200.0 / 2197.0) * delta_time * k2_v
            + (7296.0 / 2197.0) * delta_time * k3_v
        )

    def k5(p, k1_v, k2_v, k3_v, k4_v):
        return (
            p
            + (439.0 / 216.0) * delta_time * k1_v
            - (8.0) * delta_time * k2_v
            + (3680.0 / 513.0) * delta_time * k3_v
            - (845.0 / 4104.0) * delta_time * k4_v
        )

    def k6(p, k1_v, k2_v, k3_v, k4_v, k5_v):
        return (
            p
            - (8.0 / 27.0) * delta_time * k1_v
            + (2.0) * delta_time * k2_v
            - (3544.0 / 2565.0) * delta_time * k3_v
            + (1859.0 / 4104.0) * delta_time * k4_v
            - (11.0 / 40.0) * delta_time * k5_v
        )

    def rkf_4o(p, k1_v, k3_v, k4_v, k5_v):
        return p + delta_time * (
            (25.0 / 216.0) * k1_v + (1408.0 / 2565.0) * k3_v + (2197.0 / 4104.0) * k4_v - (1.0 / 5.0) * k5_v
        )

    def interp(p, k1_v, k3_v, k4_v, k5_v, k6_v):
        return p + delta_time * (
            (16.0 / 135.0) * k1_v
            + (6656.0 / 12825.0) * k3_v
            + (28561.0 / 56430.0) * k4_v
            - (9.0 / 50.0) * k5_v
            + (2.0 / 55.0) * k6_v
        )

    if np.any(np.isnan(start_pt)):
        return start_pt, [np.nan, np.nan], np.nan

    k1_v = interpolate_at_point(Vx, Vy, x, y, *start_pt)

    if np.any(np.isnan(k1_v)):
        return [np.nan, np.nan], k1_v, np.nan

    k2_pt = k2(start_pt, k1_v)

    if np.any(np.isnan(k2_pt)):
        return k2_pt, [np.nan, np.nan], np.nan

    k2_v = interpolate_at_point(Vx, Vy, x, y, *k2_pt)

    if np.any(np.isnan(k2_v)):
        return [np.nan, np.nan], k2_v, np.nan

    k3_pt = k3(start_pt, k1_v, k2_v)

    if np.any(np.isnan(k3_pt)):
        return k3_pt, [np.nan, np.nan], np.nan

    k3_v = interpolate_at_point(Vx, Vy, x, y, *k3_pt)

    if np.any(np.isnan(k3_v)):
        return [np.nan, np.nan], k3_v, np.nan

    k4_pt = k4(start_pt, k1_v, k2_v, k3_v)

    if np.any(np.isnan(k4_pt)):
        return k4_pt, [np.nan, np.nan], np.nan

    k4_v = interpolate_at_point(Vx, Vy, x, y, *k4_pt)

    if np.any(np.isnan(k4_v)):
        return [np.nan, np.nan], k4_v, np.nan

    k5_pt = k5(start_pt, k1_v, k2_v, k3_v, k4_v)

    if np.any(np.isnan(k5_pt)):
        return k5_pt, [np.nan, np.nan], np.nan

    k5_v = interpolate_at_point(Vx, Vy, x, y, *k5_pt)

    if np.any(np.isnan(k5_v)):
        return [np.nan, np.nan], k5_v, np.nan

    k6_pt = k6(start_pt, k1_v, k2_v, k3_v, k4_v, k5_v)

    if np.any(np.isnan(k6_pt)):
        return k6_pt, [np.nan, np.nan], np.nan

    k6_v = interpolate_at_point(Vx, Vy, x, y, *k6_pt)

    if np.any(np.isnan(k6_v)):
        return [np.nan, np.nan], k6_v, np.nan

    rkf_4o_pt = rkf_4o(start_pt, k1_v, k3_v, k4_v, k5_v)

    interp_pt = interp(start_pt, k1_v, k3_v, k4_v, k5_v, k6_v)

    interp_pt_error_estim = distance(interp_pt, rkf_4o_pt)

    interp_v = interpolate_at_point(Vx, Vy, x, y, *interp_pt)

    return interp_pt, interp_v, interp_pt_error_estim


def distance(p: ndarray, other: ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    This function computes the Euclidean distance between point `p` and another point `other`.

    Parameters
    ----------
    p : ndarray
        The coordinates of the first point. It should be a 1D array of length 2, where the first element is the x-coordinate and the second element is the y-coordinate.
    other : ndarray
        The coordinates of the second point. It should be a 1D array of length 2, where the first element is the x-coordinate and the second element is the y-coordinate.

    Returns
    -------
    float
        The Euclidean distance between point `p` and point `other`.

    Examples
    --------
    >>> p1 = np.array([0, 0])
    >>> p2 = np.array([1, 1])
    >>> distance(p1, p2)
    1.4142135623730951
    """
    return np.sqrt((p[0] - other[0]) ** 2 + (p[1] - other[1]) ** 2)


def interpolate_at_point(
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    px: ndarray,
    py: ndarray,
) -> ndarray:
    """
    Return velocity at a given point using bilinear interpolation.

    This function computes the velocity at a given point (px, py) by performing bilinear interpolation on the velocity field (Vx, Vy) over the grid defined by (x, y).

    Parameters
    ----------
    Vx : ndarray
        The x-component of the velocity field.
    Vy : ndarray
        The y-component of the velocity field.
    x : ndarray
        The coordinates in the x direction.
    y : ndarray
        The coordinates in the y direction.
    px : ndarray
        The x-coordinate of the point.
    py : ndarray
        The y-coordinate of the point.

    Returns
    -------
    ndarray
        The interpolated velocity at the point (px, py). The first element is the x-component of the velocity, and the second element is the y-component of the velocity.

    Notes
    -----
    The function `InterpolationMatrix` is used to generate a matrix that performs bilinear interpolation when applied to a field. The `apply` method of this matrix is then used to compute the interpolated velocity.
    """
    A = InterpolationMatrix(x, y, px, py)
    vx = A.apply(Vx)
    vy = A.apply(Vy)
    return np.array([vx.item(), vy.item()])


# pylint: disable=unused-argument
def velocity(
    p: ndarray,
    t: float,
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
) -> ndarray:
    """
    Return velocity at a given point using bilinear interpolation.

    This function computes the velocity at a given point (px, py) by performing bilinear interpolation on the velocity field (Vx, Vy) over the grid defined by (x, y).

    Parameters
    ----------
    p : ndarray
        The point to interpolate at.
    t : float
        The time to interpolate at. This is currently not being used.
    Vx : ndarray
        The x-component of the velocity field.
    Vy : ndarray
        The y-component of the velocity field.
    x : ndarray
        The coordinates in the x direction.
    y : ndarray
        The coordinates in the y direction.

    Returns
    -------
    ndarray
        The interpolated velocity at the point (px, py). The first element is the x-component of the velocity, and the second element is the y-component of the velocity.

    Notes
    -----
    The function `InterpolationMatrix` is used to generate a matrix that performs bilinear interpolation when applied to a field. The `apply` method of this matrix is then used to compute the interpolated velocity.
    """
    A = InterpolationMatrix(x, y, p[0], p[1])
    vx = A.apply(Vx)
    vy = A.apply(Vy)
    return np.array([vx.item(), vy.item()])
