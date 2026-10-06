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
# along with glacier-flow-tools; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

"""
Tests for the geometry functions.
"""

import geopandas as gp
import pytest
from shapely.geometry import LineString, MultiLineString, Point

from glacier_flow_tools.geom import (
    densify_line,
    geopandas_dataframe_densify_lines,
    parse_distance,
)


@pytest.mark.parametrize(
    "text, expected",
    [("500", 500.0), ("500m", 500.0), ("0.5km", 500.0), (" 250 M ", 250.0), ("1e3", 1000.0)],
)
def test_parse_distance(text, expected):
    """
    Read distances with and without a unit.

    Parameters
    ----------
    text : str
        The distance as text.
    expected : float
        The distance in meters.
    """
    assert parse_distance(text) == expected


@pytest.mark.parametrize("text", ["abc", "", "m", "-5m", "0", "500ft"])
def test_parse_distance_rejects(text):
    """
    Text that is not a positive distance in m or km is rejected.

    Parameters
    ----------
    text : str
        The invalid text.
    """
    with pytest.raises(ValueError):
        parse_distance(text)


def test_densify_line():
    """
    Space points along the line and keep the end only on a multiple of the spacing.
    """
    line = LineString([(0, 0), (1200, 0)])
    assert [(p.x, p.y) for p in densify_line(line, 500)] == [(0, 0), (500, 0), (1000, 0)]
    assert [(p.x, p.y) for p in densify_line(line, 400)] == [(0, 0), (400, 0), (800, 0), (1200, 0)]
    # A spacing longer than the line leaves only the start.
    assert [(p.x, p.y) for p in densify_line(line, 5000)] == [(0, 0)]


def test_densify_line_follows_the_line():
    """
    The spacing is measured along the line, around a corner.
    """
    line = LineString([(0, 0), (750, 0), (750, 750)])
    assert [(p.x, p.y) for p in densify_line(line, 500)] == [(0, 0), (500, 0), (750, 250), (750, 750)]


def test_densify_line_rejects_bad_spacing():
    """
    A spacing of zero or less is rejected.
    """
    with pytest.raises(ValueError):
        densify_line(LineString([(0, 0), (1, 0)]), 0)


def test_geopandas_dataframe_densify_lines():
    """
    Convert lines to points, keeping points, attributes and the CRS.
    """
    df = gp.GeoDataFrame(
        {"id": [1, 2, 3], "name": ["a", "b", "c"]},
        geometry=[
            LineString([(0, 0), (1200, 0)]),
            Point(5, 5),
            MultiLineString([[(0, 0), (600, 0)], [(0, 10), (500, 10)]]),
        ],
        crs="EPSG:3413",
    )
    result = geopandas_dataframe_densify_lines(df, 500)

    assert isinstance(result, gp.GeoDataFrame)
    assert result.crs == df.crs
    assert set(result.geom_type) == {"Point"}
    assert list(result.index) == list(range(8))
    assert list(result["id"]) == [1, 1, 1, 2, 3, 3, 3, 3]
    assert list(result["name"]) == ["a", "a", "a", "b", "c", "c", "c", "c"]
    expected = [(0, 0), (500, 0), (1000, 0), (5, 5), (0, 0), (500, 0), (0, 10), (500, 10)]
    assert [(p.x, p.y) for p in result.geometry] == expected
