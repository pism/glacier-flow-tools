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
Module provides functions for interpolation
"""

import numbers
from typing import Optional, Tuple, Union

import numpy as np
import scipy
from numpy import ndarray
from shapely import Point


class InterpolationMatrix:
    """Stores bilinear and nearest neighbor interpolation weights used to
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
        """Interpolate values of z to points (px,py) assuming that z is on a
        regular grid defined by x and y."""
        super().__init__()

        def to_array(data):
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
        """Interpolation matrix column number corresponding to r,c of the
        array *subset*. This is the same as the linear index within
        the subset needed for interpolation.

        """
        return self.n_cols * min(r, self.n_rows - 1) + min(c, self.n_cols - 1)

    @staticmethod
    def find(grid, delta, point):
        """Find the point to the left of point on the grid with spacing
        delta."""
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


def interpolate_rkf(
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    start_pt: Point,
    delta_time: float = 0.1,
) -> Tuple[Optional[ndarray], Optional[float]]:
    """
    Interpolate point-like object position according to the Runge-Kutta-Fehlberg method.

    :param geoarray: the flow field expressed as a GeoArray.
    :type geoarray: GeoArray.
    :param delta_time: the flow field expressed as a GeoArray.
    :type delta_time: GeoArray.
    :param start_pt: the initial point.
    :type start_pt: Point.
    :return: the estimated point-like object position at the incremented time, with the estimation error.
    :rtype: tuple of optional point and optional float.

    Examples:
    """

    if start_pt.is_empty:
        return None, None

    k1_vx, k1_vy = velocity_at_point(Vx, Vy, x, y, start_pt)

    if k1_vx is None or k1_vy is None:
        return None, None

    k2_pt = Point(
        start_pt.x + (0.25) * delta_time * k1_vx,
        start_pt.y + (0.25) * delta_time * k1_vy,
    )

    if k2_pt.is_empty:
        return None, None

    k2_vx, k2_vy = velocity_at_point(Vx, Vy, x, y, k2_pt)

    if k2_vx is None or k2_vy is None:
        return None, None

    k3_pt = Point(
        start_pt.x + (3.0 / 32.0) * delta_time * k1_vx + (9.0 / 32.0) * delta_time * k2_vx,
        start_pt.y + (3.0 / 32.0) * delta_time * k1_vy + (9.0 / 32.0) * delta_time * k2_vy,
    )

    if k3_pt.is_empty:
        return None, None

    k3_vx, k3_vy = velocity_at_point(Vx, Vy, x, y, k3_pt)

    if k3_vx is None or k3_vy is None:
        return None, None

    k4_pt = Point(
        start_pt.x
        + (1932.0 / 2197.0) * delta_time * k1_vx
        - (7200.0 / 2197.0) * delta_time * k2_vx
        + (7296.0 / 2197.0) * delta_time * k3_vx,
        start_pt.y
        + (1932.0 / 2197.0) * delta_time * k1_vy
        - (7200.0 / 2197.0) * delta_time * k2_vy
        + (7296.0 / 2197.0) * delta_time * k3_vy,
    )

    if k4_pt.is_empty:
        return None, None

    k4_vx, k4_vy = velocity_at_point(Vx, Vy, x, y, k4_pt)

    if k4_vx is None or k4_vy is None:
        return None, None

    k5_pt = Point(
        start_pt.x
        + (439.0 / 216.0) * delta_time * k1_vx
        - (8.0) * delta_time * k2_vx
        + (3680.0 / 513.0) * delta_time * k3_vx
        - (845.0 / 4104.0) * delta_time * k4_vx,
        start_pt.y
        + (439.0 / 216.0) * delta_time * k1_vy
        - (8.0) * delta_time * k2_vy
        + (3680.0 / 513.0) * delta_time * k3_vy
        - (845.0 / 4104.0) * delta_time * k4_vy,
    )

    if k5_pt.is_empty:
        return None, None

    k5_vx, k5_vy = velocity_at_point(Vx, Vy, x, y, k5_pt)

    if k5_vx is None or k5_vy is None:
        return None, None

    k6_pt = Point(
        start_pt.x
        - (8.0 / 27.0) * delta_time * k1_vx
        + (2.0) * delta_time * k2_vx
        - (3544.0 / 2565.0) * delta_time * k3_vx
        + (1859.0 / 4104.0) * delta_time * k4_vx
        - (11.0 / 40.0) * delta_time * k5_vx,
        start_pt.y
        - (8.0 / 27.0) * delta_time * k1_vy
        + (2.0) * delta_time * k2_vy
        - (3544.0 / 2565.0) * delta_time * k3_vy
        + (1859.0 / 4104.0) * delta_time * k4_vy
        - (11.0 / 40.0) * delta_time * k5_vy,
    )

    if k6_pt.is_empty:
        return None, None

    k6_vx, k6_vy = velocity_at_point(Vx, Vy, x, y, k6_pt)

    if k6_vx is None or k6_vy is None:
        return None, None

    rkf_4o_x = start_pt.x + delta_time * (
        (25.0 / 216.0) * k1_vx + (1408.0 / 2565.0) * k3_vx + (2197.0 / 4104.0) * k4_vx - (1.0 / 5.0) * k5_vx
    )
    rkf_4o_y = start_pt.y + delta_time * (
        (25.0 / 216.0) * k1_vy + (1408.0 / 2565.0) * k3_vy + (2197.0 / 4104.0) * k4_vy - (1.0 / 5.0) * k5_vy
    )
    temp_pt = Point(rkf_4o_x, rkf_4o_y)

    interp_x = start_pt.x + delta_time * (
        (16.0 / 135.0) * k1_vx
        + (6656.0 / 12825.0) * k3_vx
        + (28561.0 / 56430.0) * k4_vx
        - (9.0 / 50.0) * k5_vx
        + (2.0 / 55.0) * k6_vx
    )
    interp_y = start_pt.y + delta_time * (
        (16.0 / 135.0) * k1_vy
        + (6656.0 / 12825.0) * k3_vy
        + (28561.0 / 56430.0) * k4_vy
        - (9.0 / 50.0) * k5_vy
        + (2.0 / 55.0) * k6_vy
    )
    interp_pt = Point(interp_x, interp_y)

    interp_pt_error_estim = interp_pt.distance(temp_pt)

    return interp_pt, interp_pt_error_estim


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

    :param geoarray: the flow field expressed as a GeoArray.
    :type geoarray: GeoArray.
    :param delta_time: the flow field expressed as a GeoArray.
    :type delta_time: GeoArray.
    :param start_pt: the initial point.
    :type start_pt: Point.
    :return: the estimated point-like object position at the incremented time, with the estimation error.
    :rtype: tuple of optional point and optional float.

    Examples:
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


def distance(p, other):
    """
    Return distance to other point
    """
    return np.sqrt((p[0] - other[0]) ** 2 + (p[1] - other[1]) ** 2)


def velocity_at_point(
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    p: Union[list[Point], Point, ndarray],
) -> Tuple:
    """
    Return velocity at Point p using bilinear interpolation
    """

    if isinstance(p, Point):
        if p.is_empty:
            return None, None
        px = np.array([p.x])
        py = np.array([p.y])
    else:
        px = np.array([pt.x for pt in p])
        py = np.array([pt.y for pt in p])
    A = InterpolationMatrix(x, y, px, py)
    vx = A.apply(Vx)
    vy = A.apply(Vy)
    if isinstance(p, Point):
        vx = vx[0]
        vy = vy[0]
    return vx, vy


def interpolate_at_point(
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
    px: ndarray,
    py: ndarray,
) -> ndarray:
    """
    Return velocity at Point px,py using bilinear interpolation
    """

    A = InterpolationMatrix(x, y, px, py)
    vx = A.apply(Vx)
    vy = A.apply(Vy)
    return np.array([vx.item(), vy.item()])
