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

from typing import List, Tuple, Union

import numpy as np
import scipy
from numpy import ndarray


class InterpolationMatrix:
    """
    Stores bilinear and nearest neighbor interpolation weights used to extract profiles.

    Parameters
    ----------
    x : Union[list, ndarray]
        The x-coordinates of the grid points.
    y : Union[list, ndarray]
        The y-coordinates of the grid points.
    px : Union[float, list, ndarray]
        The x-coordinates of the points where the interpolation is to be computed.
    py : Union[float, list, ndarray]
        The y-coordinates of the points where the interpolation is to be computed.
    bilinear : bool, optional
        If True, use bilinear interpolation. If False, raise a NotImplementedError. Default is True.

    Raises
    ------
    AssertionError
        If the grid is not equally spaced, or if the spacing between grid points in the x or y direction is zero.
    NotImplementedError
        If bilinear is False.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 3.0])
    >>> y = np.array([4.0, 5.0, 6.0])
    >>> px = np.array([1.5, 2.5])
    >>> py = np.array([4.5, 5.5])
    >>> interp_matrix = InterpolationMatrix(x, y, px, py)
    """

    def __init__(
        self,
        x: Union[list, ndarray],
        y: Union[list, ndarray],
        px: Union[float, list[float], ndarray],
        py: Union[float, list[float], ndarray],
        bilinear: bool = True,
    ):
        """
        Initialize the InterpolationMatrix object.

        Interpolate values of z to points (px,py) assuming that z is on a regular grid defined by x and y.

        Parameters
        ----------
        x : Union[list, ndarray]
            The x-coordinates of the grid points.
        y : Union[list, ndarray]
            The y-coordinates of the grid points.
        px : Union[float, list, ndarray]
            The x-coordinates of the points where the interpolation is to be computed.
        py : Union[float, list, ndarray]
            The y-coordinates of the points where the interpolation is to be computed.
        bilinear : bool, optional
            If True, use bilinear interpolation. If False, raise a NotImplementedError. Default is True.

        Raises
        ------
        AssertionError
            If the grid is not equally spaced, or if the spacing between grid points in the x or y direction is zero.
        NotImplementedError
            If bilinear is False.

        Examples
        --------
        >>> x = np.array([1.0, 2.0, 3.0])
        >>> y = np.array([4.0, 5.0, 6.0])
        >>> px = np.array([1.5, 2.5])
        >>> py = np.array([4.5, 5.5])
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        """
        super().__init__()

        px_arr: ndarray = self._to_array(px)
        py_arr: ndarray = self._to_array(py)

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

    def _to_array(self, data: Union[float, list[float], np.ndarray]) -> np.ndarray:
        """
        Convert data to an array.

        Parameters
        ----------
        data : Union[float, List[float], np.ndarray]
            The data to be converted to an array. It can be a single number, a list of numbers, or a numpy array.

        Returns
        -------
        np.ndarray
            The data converted to a numpy array.

        Examples
        --------
        >>> _to_array(5)
        array([5])
        >>> _to_array([1, 2, 3])
        array([1, 2, 3])
        >>> _to_array(np.array([4, 5, 6]))
        array([4, 5, 6])
        """
        if isinstance(data, float):
            data = np.array([data])
        elif isinstance(data, List):
            data = np.array(data)
        return data

    def column(self, r: int, c: int) -> int:
        """
        Interpolate matrix column.

        Interpolation matrix column number corresponding to r,c of the
        array *subset*. This is the same as the linear index within
        the subset needed for interpolation.

        Parameters
        ----------
        r : int
            The row number in the grid.
        c : int
            The column number in the grid.

        Returns
        -------
        int
            The column number in the interpolation matrix corresponding to r, c.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> interp_matrix.column(1, 2)
        4
        """
        return self.n_cols * min(r, self.n_rows - 1) + min(c, self.n_cols - 1)

    @staticmethod
    def find(grid: Union[np.ndarray, List[float]], delta: float, point: float) -> int:
        """
        Find the point to the left of point on the grid with spacing delta.

        Parameters
        ----------
        grid : np.ndarray
            The coordinates of the grid points.
        delta : float
            The spacing between grid points.
        point : float
            The coordinate of the point where the interpolation is to be computed.

        Returns
        -------
        int
            The index of the grid point to the left of the point.

        Examples
        --------
        >>> InterpolationMatrix.find(np.array([1.0, 2.0, 3.0]), 1.0, 2.5)
        1
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

    def grid_column(self, x: Union[np.ndarray, List[float]], dx: float, X: float) -> int:
        """
        Input grid column number corresponding to X.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the grid points.
        dx : float
            The spacing between grid points in the x-direction.
        X : float
            The x-coordinate of the point where the interpolation is to be computed.

        Returns
        -------
        int
            The column number in the grid corresponding to X.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> interp_matrix.grid_column(x, 1.0, 2.0)
        2
        """
        return self.find(x, dx, X)

    def grid_row(self, y: Union[np.ndarray, List[float]], dy: float, Y: float) -> int:
        """
        Input grid row number corresponding to Y.

        Parameters
        ----------
        y : np.ndarray
            The y-coordinates of the grid points.
        dy : float
            The spacing between grid points in the y-direction.
        Y : float
            The y-coordinate of the point where the interpolation is to be computed.

        Returns
        -------
        int
            The row number in the grid corresponding to Y.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> interp_matrix.grid_row(y, 1.0, 2.0)
        2
        """
        return self.find(y, dy, Y)

    def _compute_bilinear_matrix(
        self,
        x: Union[np.ndarray, List[float]],
        y: Union[np.ndarray, List[float]],
        dx: float,
        dy: float,
        px: np.ndarray,
        py: np.ndarray,
    ) -> None:
        """
        Initialize a bilinear interpolation matrix.

        Parameters
        ----------
        x : np.ndarray
            The x-coordinates of the grid points.
        y : np.ndarray
            The y-coordinates of the grid points.
        dx : float
            The spacing between grid points in the x-direction.
        dy : float
            The spacing between grid points in the y-direction.
        px : np.ndarray
            The x-coordinates of the points where the interpolation is to be computed.
        py : np.ndarray
            The y-coordinates of the points where the interpolation is to be computed.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> interp_matrix._compute_bilinear_matrix(x, y, 1.0, 1.0, px, py)
        """
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

    def adjusted_matrix(self, mask: np.ndarray) -> Tuple[scipy.sparse.csr_matrix, np.ndarray]:
        """
        Return adjusted interpolation matrix that ignores missing (masked) values.

        Parameters
        ----------
        mask : np.ndarray
            A boolean array indicating the masked (missing) values in the original array.

        Returns
        -------
        A : scipy.sparse.csr_matrix
            The adjusted interpolation matrix.
        output_mask : np.ndarray
            A boolean array indicating the masked (missing) values in the output.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> mask = np.array([[False, True], [False, False]])
        >>> A, output_mask = interp_matrix.adjusted_matrix(mask)
        """
        A = self.A.tocsr()
        n_points = A.shape[0]

        output_mask = np.zeros(n_points, dtype=bool)

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
        """
        Apply the interpolation to an array. Returns values at points along the profile.

        Parameters
        ----------
        array : np.ndarray or np.ma.MaskedArray
            The array to which the interpolation should be applied.

        Returns
        -------
        np.ndarray or np.ma.MaskedArray
            The result of applying the interpolation to the array. If the input array is a masked array, the output will also be a masked array.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        >>> interp_matrix.apply(array)
        array([5.])
        """
        subset = array[self.r_min : self.r_max + 1, self.c_min : self.c_max + 1]
        return self.apply_to_subset(subset)

    def apply_to_subset(self, subset):
        """
        Apply interpolation to an array subset.

        Parameters
        ----------
        subset : np.ndarray or np.ma.MaskedArray
            The subset of the array to which the interpolation should be applied.

        Returns
        -------
        np.ndarray or np.ma.MaskedArray
            The result of applying the interpolation to the subset. If the input subset is a masked array, the output will also be a masked array.

        Examples
        --------
        >>> interp_matrix = InterpolationMatrix(x, y, px, py)
        >>> subset = np.array([[1, 2], [3, 4]])
        >>> interp_matrix.apply_to_subset(subset)
        array([2.5])
        """
        if np.ma.is_masked(subset):
            A, mask = self.adjusted_matrix(subset.mask)
            data = A * np.ravel(subset)
            return np.ma.array(data, mask=mask)

        return self.A.tocsr() * np.ravel(subset)


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
def velocity_steady(
    p: ndarray,
    Vx: ndarray,
    Vy: ndarray,
    x: ndarray,
    y: ndarray,
) -> ndarray:
    """
    Return velocity at a given point using bilinear interpolation for steady flow.

    This function computes the velocity at a given point (px, py) by performing bilinear interpolation on the velocity field (Vx, Vy) over the grid defined by (x, y).

    Parameters
    ----------
    p : ndarray
        The point to interpolate at.
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

    Examples
    --------

    >>> from glacier_flow_tools.interpolation import velocity_steady

    >>> x = np.linspace(-1, 1, 26)
    >>> y = np.linspace(-1, 1, 26)
    >>> X, Y = np.meshgrid(x, y)

    >>> # Directional vectors
    >>> Vx = X
    >>> Vy = -Y

    >>> point = np.array([0.0, 0.0])
    >>> velocity_steady(point, Vx, Vy, x, y)
    array([ 5.26662047e-17, -5.44009282e-17])
    """

    return velocity(p, 0, Vx, Vy, x, y)


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
