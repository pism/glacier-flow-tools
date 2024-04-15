# Copyright (C) 2024 Andy Aschwanden, Constantine Khroulev
#
# This file is part of pypism.
#
# PYPISM is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PYPISM is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Module geometry functions
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

    Args:
        multilinestring (MultiLineString): The MultiLineString to convert.

    Returns:
        List[Point]: The list of Points.
    """
    return [Point(x, y) for linestring in multilinestring.geoms for x, y in zip(*linestring.xy)]


def linestring_to_points(linestring: LineString) -> List[Point]:
    """
    Convert a LineString to a list of Points.

    Args:
        linestring (LineString): The LineString to convert.

    Returns:
        List[Point]: The list of Points.
    """
    return [Point(x, y) for x, y in zip(*linestring.xy)]


def to_geopandas_row(df: gp.GeoDataFrame, k: int, points: List[Point]) -> gp.GeoDataFrame:
    """
    Convert a row of a GeoDataFrame and a list of Points to a new GeoDataFrame.

    Args:
        df (GeoDataFrame): The original GeoDataFrame.
        k (int): The index of the row to convert.
        points (List[Point]): The list of Points.

    Returns:
        GeoDataFrame: The new GeoDataFrame.
    """
    return pd.concat([gp.GeoDataFrame(df.loc[[k]].drop(columns="geometry"), geometry=[pt]) for pt in points])


def convert_geopands_row_geometry_to_point(df: gp.GeoDataFrame, k: int) -> gp.GeoDataFrame:
    """
    Convert a row of a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

    Args:
        df (GeoDataFrame): The original GeoDataFrame.
        k (int): The index of the row to convert.

    Returns:
        GeoDataFrame: The new GeoDataFrame.

    Raises:
        ValueError: If the geometry type is not supported.
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

    Args:
        df (GeoDataFrame): The original GeoDataFrame.

    Returns:
        GeoDataFrame: The new GeoDataFrame.
    """
    return pd.concat([convert_geopands_row_geometry_to_point(df, k) for k in df.index]).reset_index(drop=True)


def distance(a: ndarray, b: ndarray) -> float:
    """
    Calculate the Euclidean distance between two points.

    Args:
        a (ndarray): The first point.
        b (ndarray): The second point.

    Returns:
        float: The distance.
    """
    return float(np.linalg.norm(a - b))


def distances(pts: ndarray) -> ndarray:
    """
    Calculate the Euclidean distances between consecutive points.

    Args:
        pts (ndarray): The points.

    Returns:
        ndarray: The distances.
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
    pandas.DataFrame Accessor
    """

    def __init__(self, pandas_obj):
        self._obj = pandas_obj

    def to_points(self):
        """
        Convert a GeoDataFrame with "LineString" or "MultiLineString" geometry to "Point" geometry.

        Args:
            df (GeoDataFrame): The original GeoDataFrame.

        Returns:
            GeoDataFrame: The new GeoDataFrame.
        """
        return pd.concat([convert_geopands_row_geometry_to_point(self._obj, k) for k in self._obj.index]).reset_index(
            drop=True
        )
