#!/usr/bin/env python
# Copyright (C) 2015, 2016, 2018, 2021, 2023 Constantine Khroulev and Andy Aschwanden
#

# nosetests --with-coverage --cover-branches --cover-html
# --cover-package=extract_profiles scripts/extract_profiles.py

# pylint -d C0301,C0103,C0325,W0621
# --msg-template="{path}:{line}:[{msg_id}({symbol}), {obj}] {msg}"
# extract_profiles.py > lint.txt

"""This script containts tools for extracting 'profiles', that is
sampling 2D and 3D fields on a regular grid at points along a flux
gate or a any kind of profile.
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from typing import Union

import numpy as np
from osgeo import gdal, ogr, osr

from .interpolation import InterpolationMatrix

gdal.UseExceptions()

# from pyproj import Proj


profiledim = "profile"
stationdim = "station"


def normal(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit normal vector orthogonal to (point1-point0),
    pointing 'to the right' of (point1-point0).

    """

    a = point0 - point1
    if a[1] != 0.0:
        n = np.array([1.0, -a[0] / a[1]])
        n = n / np.linalg.norm(n)  # normalize
    else:
        n = np.array([0, 1])

    # flip direction if needed:
    if np.cross(a, n) < 0:
        n = -1.0 * n

    return n


def tangential(point0: np.ndarray, point1: np.ndarray) -> np.ndarray:
    """Compute the unit tangential vector to (point1-point0),
    pointing 'to the right' of (point1-point0).

    """

    a = point1 - point0
    norm = np.linalg.norm(a)
    # protect from division by zero
    if norm > 0.0:
        return a / norm
    else:
        return a


def compute_normals(px: Union[np.ndarray, list], py: Union[np.ndarray, list]):
    """
    Compute normals to a profile described by 'p'. Normals point 'to
    the right' of the path.
    """

    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ns = np.zeros_like(p)
    ns[0] = normal(p[0], p[1])
    for j in range(1, len(p) - 1):
        ns[j] = normal(p[j - 1], p[j + 1])

    ns[-1] = normal(p[-2], p[-1])

    return ns[:, 0], ns[:, 1]


def compute_tangentials(px: Union[np.ndarray, list], py: Union[np.ndarray, list]):
    """
    Compute tangetials to a profile described by 'p'.
    """

    p = np.vstack((px, py)).T

    if len(p) < 2:
        return [0], [0]

    ts = np.zeros_like(p)
    ts[0] = tangential(p[0], p[1])
    for j in range(1, len(p) - 1):
        ts[j] = tangential(p[j - 1], p[j + 1])

    ts[-1] = tangential(p[-2], p[-1])

    return ts[:, 0], ts[:, 1]


def distance_from_start(px, py):
    "Initialize the distance along a profile."
    result = np.zeros_like(px)
    result[1::] = np.sqrt(np.diff(px) ** 2 + np.diff(py) ** 2)
    return result.cumsum()


def add_function(f, geometry):
    px, py = geometry.xy
    return f(px, py)


def extract_profile(
    variable,
    profile,
    xdim: str = "x",
    ydim: str = "y",
    zdim: str = "z",
    tdim: str = "time",
):
    """Extract values of a variable along a profile."""
    x = variable.coords[xdim].to_numpy()
    y = variable.coords[ydim].to_numpy()

    px, py = profile["geometry"].xy
    dim_length = dict(list(zip(variable.dims, variable.shape)))

    def init_interpolation():
        """Initialize interpolation weights. Takes care of the transpose."""
        if variable.dims.index(ydim) < variable.dims.index(xdim):
            A = InterpolationMatrix(x, y, px, py)
            return A, slice(A.c_min, A.c_max + 1), slice(A.r_min, A.r_max + 1)
        else:
            A = InterpolationMatrix(y, x, py, px)
            return A, slice(A.r_min, A.r_max + 1), slice(A.c_min, A.c_max + 1)

    # try to get the matrix we (possibly) pre-computed earlier:
    try:
        # Check if we are extracting from the grid of the same shape
        # as before. This will make sure that we re-compute weights if
        # one variable is stored as (x,y) and a different as (y,x),
        # but will not catch grids that are of the same shape, but
        # with different extents and spacings. We'll worry about this
        # case later -- if we have to.
        if profile.grid_shape == variable.shape:
            A = profile.A
            x_slice = profile.x_slice
            y_slice = profile.y_slice
        else:
            A, x_slice, y_slice = init_interpolation()
    except AttributeError:
        A, x_slice, y_slice = init_interpolation()
        profile.A = A
        profile.x_slice = x_slice
        profile.y_slice = y_slice
        profile.grid_shape = variable.shape

    def return_indexes(indexes):
        return (*indexes,)

    def read_subset(t=0, z=0):
        """Assemble the indexing tuple and get a subset from a variable."""
        index = []
        indexes = {xdim: x_slice, ydim: y_slice, zdim: z, tdim: t}
        for dim in variable.dims:
            try:
                index.append(indexes[dim])
            except KeyError:
                index.append(Ellipsis)
            starred_index = return_indexes(index)
        return variable[starred_index]

    n_points = len(px)

    if tdim in variable.coords and zdim in variable.coords:
        dim_names = ["time", "profile", "z"]
        result = np.zeros((dim_length[tdim], n_points, dim_length[zdim]))
        for j in range(dim_length[tdim]):
            for k in range(dim_length[zdim]):
                result[j, :, k] = A.apply_to_subset(read_subset(t=j, z=k))
    elif tdim in variable.coords:
        dim_names = ["time", "profile"]
        result = np.zeros((dim_length[tdim], n_points))
        for j in range(dim_length[tdim]):
            result[j, :] = A.apply_to_subset(read_subset(t=j))
    elif zdim in variable.coords:
        dim_names = ["profile", "z"]
        result = np.zeros((n_points, dim_length[zdim]))
        for k in range(dim_length[zdim]):
            result[:, k] = A.apply_to_subset(read_subset(z=k))
    else:
        dim_names = ["profile"]
        result = A.apply_to_subset(read_subset())

    return result, dim_names
