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
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module geometry functions.
"""

from typing import List

import geopandas as gp
import numpy as np
import pandas as pd
from numpy import ndarray
from shapely.geometry import LineString, MultiLineString, Point


def multilinestring_to_points(multilinestring: MultiLineString) -> List[Point]:
    """
    Convert a MultiLineString to a list of Points.

    Parameters
    ----------

    multilinestring : MultiLineString
      The MultiLineString to convert.

    Returns
    -------

    List[Point]
      The list of Points.
    """
    return [Point(x, y) for linestring in multilinestring.geoms for x, y in zip(*linestring.xy)]


def linestring_to_points(linestring: LineString) -> List[Point]:
    """
    Convert a LineString to a list of Points.

    Parameters
    ----------

    linestring : LineString
      The LineString to convert.

    Returns
    -------

    List[Point]
      The list of Points.
    """
    return [Point(x, y) for x, y in zip(*linestring.xy)]


def to_geopandas_row(df: gp.GeoDataFrame, k: int, points: List[Point]) -> gp.GeoDataFrame:
    """
    Convert a row of a GeoDataFrame and a list of Points to a new GeoDataFrame.

    Parameters
    ----------
    df : GeoDataFrame
        The original GeoDataFrame.
    k : int
        The index of the row to convert.
    points : List[Point]
        The list of Points.

    Returns
    -------
    GeoDataFrame
        The new GeoDataFrame.

    Examples
    --------
    >>> gdf = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    >>> points = [Point(1, 1), Point(2, 2)]
    >>> new_gdf = to_geopandas_row(gdf, 0, points)
    >>> new_gdf.geometry.type.unique()
    array(['Point'], dtype=object)
    """
    return pd.concat([gp.GeoDataFrame(df.loc[[k]].drop(columns="geometry"), geometry=[pt]) for pt in points])


def convert_geopands_row_geometry_to_point(df: gp.GeoDataFrame, k: int) -> gp.GeoDataFrame:
    """
    Convert a row of a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

    Parameters
    ----------
    df : GeoDataFrame
        The original GeoDataFrame.
    k : int
        The index of the row to convert.

    Returns
    -------
    GeoDataFrame
        The new GeoDataFrame with "Point" geometry.

    Raises
    ------
    ValueError
        If the geometry type is not supported.

    Examples
    --------
    >>> gdf = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    >>> gdf = gdf.set_geometry(gdf.geometry.centroid)
    >>> gdf.geometry.type.unique()
    array(['Point'], dtype=object)
    >>> new_gdf = convert_geopands_row_geometry_to_point(gdf, 0)
    >>> new_gdf.geometry.type.unique()
    array(['Point'], dtype=object)
    """
    geom_type = df.geom_type[k]
    geometry = df.geometry[k]

    if geom_type == "LineString":
        points = linestring_to_points(geometry)
    elif geom_type == "MultiLineString":
        points = multilinestring_to_points(geometry)
    elif geom_type == "Point":
        points = [geometry]
    else:
        raise ValueError(f"Unsupported geometry type: {geom_type}")

    return to_geopandas_row(df, k, points)


def convert_to_point_geometry_dataframe(df: gp.GeoDataFrame) -> gp.GeoDataFrame:
    """
    Convert a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

    Parameters
    ----------
    df : GeoDataFrame
        The original GeoDataFrame with "LineString" or "MultiLineString" geometry.

    Returns
    -------
    GeoDataFrame
        The new GeoDataFrame with "Point" geometry.

    Examples
    --------
    >>> gdf = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    >>> gdf = gdf.set_geometry(gdf.geometry.centroid)
    >>> gdf.geometry.type.unique()
    array(['Point'], dtype=object)
    >>> gdf = convert_to_point_geometry_dataframe(gdf)
    >>> gdf.geometry.type.unique()
    array(['Point'], dtype=object)
    """
    return pd.concat([convert_geopands_row_geometry_to_point(df, k) for k in df.index]).reset_index(drop=True)


def distance(a: ndarray, b: ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Parameters
    ----------
    a : ndarray
        The first point. It can be an array of any shape.
    b : ndarray
        The second point. It should be the same shape as `a`.

    Returns
    -------
    float
        The Euclidean distance between `a` and `b`.

    Examples
    --------
    >>> a = np.array([1, 2, 3])
    >>> b = np.array([4, 5, 6])
    >>> distance(a, b)
    5.196152422706632
    """
    return float(np.linalg.norm(a - b))


def distances(pts: ndarray) -> ndarray:
    """
    Calculate the Euclidean distances between consecutive points.

    Parameters
    ----------
    pts : ndarray
        The points. It should be a 2D array where each row represents a point.

    Returns
    -------
    ndarray
        The distances. It is a 1D array where each element is the distance between consecutive points. The first element is always 0.

    Examples
    --------
    >>> pts = np.array([[1, 2], [4, 6], [7, 8]])
    >>> distances(pts)
    array([0.        , 5.        , 3.60555128])
    """
    # Shift the pts array by one row
    pts_shifted = np.roll(pts, shift=1, axis=0)

    # Calculate the Euclidean distance between consecutive points
    distances_arr = np.linalg.norm(pts - pts_shifted, axis=1)

    # Set the first distance to 0
    distances_arr[0] = 0.0

    return distances_arr


@pd.api.extensions.register_dataframe_accessor("convert")
class GeometryConverter:  # pylint: disable=too-few-public-methods
    """
    The pandas.DataFrame Accessor to convert a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

    This class is used as an extension to pandas DataFrame objects, and can be accessed via the `.convert` attribute.

    Parameters
    ----------
    pandas_obj : GeoDataFrame
        The GeoDataFrame to be converted.

    Examples
    --------
    >>> gdf = geopandas.read_file(geopandas.datasets.get_path('nybb'))
    >>> gdf.convert.to_points()
    """

    def __init__(self, pandas_obj):
        """
        Initialize the GeometryConverter object.

        Parameters
        ----------
        pandas_obj : GeoDataFrame
            The GeoDataFrame to be converted.
        """
        self._obj = pandas_obj

    def to_points(self):
        """
        Convert a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

        Returns
        -------
        GeoDataFrame
            The new GeoDataFrame with "Point" geometry.

        Examples
        --------
        >>> gdf = geopandas.read_file(geopandas.datasets.get_path('nybb'))
        >>> gdf.convert.to_points()
        """
        return pd.concat([convert_geopands_row_geometry_to_point(self._obj, k) for k in self._obj.index]).reset_index(
            drop=True
        )
