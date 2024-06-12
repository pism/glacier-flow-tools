# cython: language_level=3, cdivision=True, boundscheck=False, wraparound=False
# distutils: language = c

import numpy as np
cimport cython
from cython.parallel import prange

from libc.math cimport exp, pi, sqrt, fmin, lround

cdef double kernel_gaussian(double sigma, double r_squared) nogil:
    return 1.0 / (2.0 * pi * sigma*sigma) * exp(-r_squared / (2 * sigma*sigma))

cdef double kernel_triangular(double sigma, double r_squared) nogil:
    return max(1.0 - sqrt(r_squared) / sigma, 0.0)

cdef smooth(double[:,::1] surface_elevation, double[:,::1] thickness,
            double (*kernel)(double, double) nogil,
            double sigma_k, double w_k, int w_max, double dx, double dy,
            double[:,::1] output = None):

    assert sigma_k > 0
    assert w_k > 0
    assert w_max > 0
    assert dx > 0
    assert dy > 0

    if output is None:
        output_ = np.zeros_like(surface_elevation, dtype=np.float64)
        output = output_

    cdef int nrows = surface_elevation.shape[0]
    cdef int ncols = surface_elevation.shape[1]

    cdef double dot, weight_sum
    cdef double x, y, weight, H, sigma
    cdef int width, row, col, row_offset, col_offset, r, c

    for row in prange(nrows, nogil=True, schedule='dynamic'):
        for col in range(ncols):

            H = thickness[row, col]

            if H == 0.0:
                output[row, col] = surface_elevation[row, col]
                continue

            # sigma is proportional to ice thickness with the
            # proportionality constant sigma_k
            sigma = sigma_k * H
            # width is the half-width of the filter (units: number of grid
            # points). It is proportional to sigma with the
            # proportionality constant w_k. width cannot exceed w_max.
            width = min(lround(w_k * sigma / fmin(dx, dy)), w_max)

            dot = 0.0
            weight_sum = 0.0

            for row_offset in range(-width, width + 1):
                for col_offset in range(-width, width + 1):
                    # row and column indexes in the input array:
                    r = row + row_offset
                    c = col + col_offset

                    if r < 0 or r >= nrows or c < 0 or c > ncols:
                        # don't use points outside the domain
                        continue

                    if thickness[r, c] == 0:
                        # exclude points with zero thickness
                        continue

                    # coordinates of the current point in the
                    # coordinate system centered at (row,col):
                    x = dx * col_offset
                    y = dy * row_offset
                    weight = kernel(sigma, x*x + y*y)

                    # avoid += so that Cython does not interpret this as a parallel reduction
                    dot = dot + surface_elevation[r, c] * weight

                    # avoid += so that Cython does not interpret this as a parallel reduction
                    weight_sum = weight_sum + weight

            output[row, col] = dot / weight_sum if weight_sum > 0.0 else 0.0

    return output_

def gaussian(sigma, x, y):
    """Kernel of a Gaussian filter with the spatial width `sigma`."""
    return kernel_gaussian(sigma, x*x + y*y)

def triangular(sigma, x, y):
    """Kernel of a triangular filter with the spatial width `sigma`."""
    return kernel_triangular(sigma, x*x + y*y)

def smooth_gaussian(double[:,::1] surface_elevation, double[:,::1] ice_thickness, double sigma_k, double w_k, int w_max, double dx, double dy):
    """Ice surface DEM smoothing for quantifying ice flow.

    Smooth the surface elevation DEM using a Gaussian filter with the
    spatial scale `sigma` proportional to ice thickness.

    Surface elevations at grid points where ice thickness is zero are
    *not* used.

    McCormack et al suggest using `sigma_k` between 2 and 4.

    Note
    ----
    This code uses OpenMP to take advantage of all available CPU
    cores.

    Parameters
    ----------
    surface_elevation : array_like
        surface elevation DEM (2D array of double)
    ice_thickness : array_like
        ice thickness, in meters
    sigma_k : double
        coefficient in `sigma = sigma_k * H`, where `H` is the ice thickness
    w_k : double
        filter half-width, in units of `sigma`
    w_max : int
        maximum allowed filter half-width
    dx : double
        grid resolution in the `x` direction (columns)
    dy : double
        grid resolution in the `y` direction (rows)

    Reference
    ---------
    F. S. McCormack, J. L. Roberts, L. M. Jong, D. A. Young, and L. H.
    Beem, “A note on digital elevation model smoothing and driving
    stresses,” Polar Research, vol. 38, no. 0, Mar. 2019, doi:
    10.33265/polar.v38.3498.

    """
    return smooth(surface_elevation, ice_thickness, kernel_gaussian, sigma_k, w_k, w_max, dx, dy)

def smooth_triangular(double[:,::1] surface_elevation, double[:,::1] ice_thickness, double sigma_k, int w_max, double dx, double dy):
    """Ice surface DEM smoothing for quantifying ice flow.

    Smooth the surface elevation DEM using a "triangular" filter with
    the spatial scale `sigma` proportional to ice thickness.

    Surface elevations at grid points where ice thickness is zero are
    *not* used.

    McCormack et al suggest using `sigma_k` between 8 and 10.

    Note
    ----
    This code uses OpenMP to take advantage of all available CPU
    cores.

    Parameters
    ----------
    surface_elevation : array_like
        surface elevation DEM (2D array of double)
    ice_thickness : array_like
        ice thickness, in meters
    sigma_k : double
        coefficient in `sigma = sigma_k * H`, where `H` is the ice thickness
    w_max : int
        maximum allowed filter half-width
    dx : double
        grid resolution in the `x` direction (columns)
    dy : double
        grid resolution in the `y` direction (rows)

    Reference
    ---------
    F. S. McCormack, J. L. Roberts, L. M. Jong, D. A. Young, and L. H.
    Beem, “A note on digital elevation model smoothing and driving
    stresses,” Polar Research, vol. 38, no. 0, Mar. 2019, doi:
    10.33265/polar.v38.3498.

    """
    w_k = 1.0
    return smooth(surface_elevation, ice_thickness, kernel_triangular, sigma_k, w_k, w_max, dx, dy)
