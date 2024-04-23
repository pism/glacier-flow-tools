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
from shapely import get_coordinates
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


def geopandas_dataframe_shorten_lines(
    df: gp.GeoDataFrame, buffer: float = 200, segmentize: float = 100
) -> gp.GeoDataFrame:
    """
    Shorten all lines in a GeoDataFrame.

    This function shortens all lines in a GeoDataFrame by removing points from both ends until the cumulative distance from each end is greater than a specified buffer distance. The lines are represented by GeoSeries of points, and the function returns a GeoDataFrame that contains the shortened lines.

    Parameters
    ----------
    df : gp.GeoDataFrame
        A GeoDataFrame containing the lines to be shortened. Each line is represented by a GeoSeries of points.
    buffer : float, optional
        The buffer distance. Points will be removed from the ends of the lines until the cumulative distance from each end is greater than this buffer distance. The default is 200.
    segmentize : float, optional
        The maximum length of line segments. If specified, the lines will be divided into segments of this length before shortening. The default is 100.

    Returns
    -------
    gp.GeoDataFrame
        A GeoDataFrame that contains the shortened lines. Each line is represented by a GeoSeries of points.

    Examples
    --------
    >>> df = gp.GeoDataFrame(geometry=[LineString([(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)])])
    >>> geopandas_dataframe_shorten_lines(df, buffer=1, segmentize=0.5)
    <geopandas.geodataframe.GeoDataFrame object at 0x7f4c6c3acf40>
    """
    g = df.segmentize(segmentize)
    geom_df = gp.GeoDataFrame(geometry=g)
    return df.set_geometry(geom_df.apply(shorten_line, buffer=buffer, axis=1)).convert.to_points()


def shorten_line(series: gp.GeoSeries, buffer: float = 100):
    """
    Shorten a line represented by a GeoSeries.

    This function shortens a line by removing points from both ends until the cumulative distance from each end is greater than a specified buffer distance. The line is represented by a GeoSeries of points, and the function returns a LineString that represents the shortened line.

    Parameters
    ----------
    series : gp.GeoSeries
        A GeoSeries representing the line. Each point in the series is a vertex of the line.
    buffer : float, optional
        The buffer distance. Points will be removed from the ends of the line until the cumulative distance from each end is greater than this buffer distance. The default is 100.

    Returns
    -------
    shapely.geometry.LineString
        A LineString that represents the shortened line.

    Examples
    --------
    >>> series = gp.GeoSeries([Point(0, 0), Point(1, 1), Point(2, 2), Point(3, 3), Point(4, 4)])
    >>> shorten_line(series, buffer=1)
    <shapely.geometry.linestring.LineString object at 0x7f4c6c3acf40>
    """
    coords = get_coordinates(series)
    d = distances(coords)
    i_min, i_max = np.argmax(d.cumsum() > buffer), np.argmax(np.flip(np.cumsum(np.flip(d))) < buffer) - 1
    geom = LineString(coords[[i_min, i_max], :])
    return geom


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
